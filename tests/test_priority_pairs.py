"""Tests for the dashboard → 15m engine priority-pair channel."""
import time

import pytest

from src.core.priority_pairs import (
    MAX_PRIORITY_PAIRS,
    PRIORITY_TABLE,
    due_pairs,
    normalize_pairs,
    select_fresh,
)


def test_normalize_pairs_accepts_dashboard_dicts_and_tuples():
    pairs = normalize_pairs([
        {"db": "x", "ex": "bybit", "ccxt": "bybit", "sym": "0G/USDT:USDT"},
        ("bybit", "BTC/USDT"),
        {"ex": "bybit", "sym": "0G/USDT:USDT"},   # duplicate -> dropped
        {"ex": "", "sym": "ETH/USDT"},            # no exchange -> dropped
        {"ex": "bybit", "sym": "GARBAGE"},        # not a pair -> dropped
        None,
    ])
    assert pairs == [("bybit", "0G/USDT:USDT"), ("bybit", "BTC/USDT")]


def test_normalize_pairs_keeps_order_and_caps_the_set():
    # The open pair is published first, so the cap must never drop it.
    raw = [{"ex": "bybit", "sym": f"C{i}/USDT"} for i in range(30)]
    pairs = normalize_pairs(raw)
    assert len(pairs) == MAX_PRIORITY_PAIRS
    assert pairs[0] == ("bybit", "C0/USDT")


def test_select_fresh_drops_expired_publications():
    rows = [
        ("bybit", "0G/USDT:USDT", 2.0),      # dashboard is open on it
        ("bybit", "OLD/USDT", 950.0),        # tab closed 15 min ago
        ("bybit", "BAD/USDT", "nan-ish"),    # unparsable age
        ("", "NOEX/USDT", 1.0),
    ]
    assert select_fresh(rows, ttl_sec=90.0) == [("bybit", "0G/USDT:USDT")]


def test_due_pairs_respects_per_pair_interval():
    now = 1_790_000_000.0
    pairs = [("bybit", "A/USDT"), ("bybit", "B/USDT"), ("bybit", "C/USDT")]
    last_run = {
        ("bybit", "A/USDT"): now - 0.2,   # refreshed 200 ms ago -> not due
        ("bybit", "B/USDT"): now - 1.5,   # due
        # C never ran -> due
    }
    assert due_pairs(pairs, last_run, 1.0, now=now) == [
        ("bybit", "B/USDT"),
        ("bybit", "C/USDT"),
    ]


def test_priority_table_name_is_stable():
    # Both sides (dashboard publisher, engine reader) hardcode nothing else.
    assert PRIORITY_TABLE == "dashboard_priority_pairs"


class _FakeConn:
    """Minimal asyncpg-like connection recording the statements it receives."""

    def __init__(self, rows=None):
        self.executed = []
        self.many = []
        self._rows = rows or []

    async def execute(self, sql, *args):
        self.executed.append((" ".join(sql.split()), args))

    async def executemany(self, sql, args):
        self.many.append((" ".join(sql.split()), list(args)))

    async def fetch(self, sql, *args):
        self.executed.append((" ".join(sql.split()), args))
        return self._rows


@pytest.mark.asyncio
async def test_publish_creates_table_upserts_and_expires():
    from src.core.priority_pairs import publish_priority_pairs

    conn = _FakeConn()
    n = await publish_priority_pairs(
        conn, [{"ex": "bybit", "sym": "0G/USDT:USDT"}, ("bybit", "BTC/USDT")], ttl_sec=90.0
    )
    assert n == 2
    assert any("CREATE TABLE IF NOT EXISTS" in sql for sql, _ in conn.executed)
    assert any("DELETE FROM" in sql for sql, _ in conn.executed)
    sql, rows = conn.many[0]
    assert "ON CONFLICT (exchange, symbol) DO UPDATE" in sql
    assert rows == [("bybit", "0G/USDT:USDT"), ("bybit", "BTC/USDT")]


@pytest.mark.asyncio
async def test_read_priority_pairs_filters_by_age():
    from src.core.priority_pairs import read_priority_pairs

    conn = _FakeConn(rows=[
        {"exchange": "bybit", "symbol": "0G/USDT:USDT", "age": 1.0},
        {"exchange": "bybit", "symbol": "STALE/USDT", "age": 3600.0},
    ])
    assert await read_priority_pairs(conn, ttl_sec=90.0) == [("bybit", "0G/USDT:USDT")]


def test_due_pairs_defaults_to_wall_clock():
    pairs = [("bybit", "A/USDT")]
    assert due_pairs(pairs, {("bybit", "A/USDT"): time.time()}, 1.0) == []


# ---------------------------------------------------------------------------
# Engine-side lane worker (fetch tail -> write through the shared writer)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refresh_priority_pair_writes_through_the_shared_writer(monkeypatch):
    import src.core.updater_15m as u

    written = {}

    class _Ex:
        async def fetch_ohlcv(self, symbol, tf, since=None, limit=None):
            assert tf == "15m" and limit == 10
            written["since"] = since
            return [[1_790_000_000_000, 1, 2, 0.5, 1.5, 10]]

    async def _fake_persistent(name):
        return _Ex()

    async def _fake_find(tbl):
        written["tbl"] = tbl
        return "db_low_15m", 1_789_999_100, 1_700_000_000

    async def _fake_save(db, tbl, symbol, ccxt_id, cs):
        written["save"] = (db, tbl, symbol, ccxt_id, len(cs))
        return db

    monkeypatch.setattr(u, "get_persistent_exchange", _fake_persistent)
    monkeypatch.setattr(u, "find_table_in_dbs", _fake_find)
    monkeypatch.setattr(u, "save_candles_to_table", _fake_save)

    n = await u.refresh_priority_pair("bybit", "0G/USDT:USDT")

    assert n == 1
    assert written["tbl"] == "0g_usdt:usdt_on_bybit"
    assert written["save"][1:] == ("0g_usdt:usdt_on_bybit", "0G/USDT:USDT", "bybit", 1)


@pytest.mark.asyncio
async def test_refresh_priority_pair_skips_dead_and_filtered_symbols(monkeypatch):
    import src.core.updater_15m as u

    async def _boom(*a, **k):  # must never be reached
        raise AssertionError("exchange must not be touched")

    monkeypatch.setattr(u, "get_persistent_exchange", _boom)
    monkeypatch.setattr(u, "_DEAD_SYMBOLS", {("bybit", "GONE/USDT")})

    assert await u.refresh_priority_pair("bybit", "GONE/USDT") == 0
    assert await u.refresh_priority_pair("bybit", "BTC3L/USDT") == 0  # leveraged token


# ---------------------------------------------------------------------------
# Exchange-label resolution (15m tables say "gateio", 1D tables say "gate")
# ---------------------------------------------------------------------------

def test_resolve_exchange_alias_accepts_both_spellings():
    from src.core.priority_pairs import resolve_exchange_alias

    map_1d = {"bybit": "bybit", "gateio": "gate", "okx": "okx"}
    assert resolve_exchange_alias("gateio", map_1d) == ("gateio", "gate")
    assert resolve_exchange_alias("gate", map_1d) == ("gateio", "gate")
    assert resolve_exchange_alias("BYBIT", map_1d) == ("bybit", "bybit")
    assert resolve_exchange_alias("kucoin", map_1d) == ("kucoin", "kucoin")  # pass-through
    assert resolve_exchange_alias("", map_1d) == ("", "")


@pytest.mark.asyncio
async def test_lane_never_creates_a_missing_15m_table(monkeypatch):
    """A published pair with no table must be ignored, not backfilled into a
    fresh junk table (e.g. a 1D-only exchange label reaching the 15m lane)."""
    import src.core.updater_15m as u

    async def _no_table(tbl):
        return None, 0, 0

    async def _boom(*a, **k):
        raise AssertionError("must not write")

    monkeypatch.setattr(u, "find_table_in_dbs", _no_table)
    monkeypatch.setattr(u, "save_candles_to_table", _boom)

    class _Ex:
        async def fetch_ohlcv(self, *a, **k):
            return [[1_790_000_000_000, 1, 2, 0.5, 1.5, 10]]

    async def _persistent(name):
        return _Ex()

    monkeypatch.setattr(u, "get_persistent_exchange", _persistent)
    assert await u.refresh_priority_pair("bybit", "0G/USDT:USDT") == 0


@pytest.mark.asyncio
async def test_1d_engine_lane_refreshes_the_daily_bar(monkeypatch):
    """The 1D engine runs the same lane: the FORMING daily candle is rewritten
    in the database, so the dashboard does not have to aggregate 15m itself."""
    import src.core.updater as u1d

    engine = u1d.MarketDataEngine(timeframe="1d")

    class _Repo:
        def __init__(self):
            self.written = None

        async def find_table(self, tbl):
            return "db_low_1d", 1_789_900_000, 1_600_000_000

        async def upsert_candles(self, db, tbl, df, timeframe="1d"):
            self.written = (db, tbl, timeframe, len(df))

    class _Ex:
        async def fetch_ohlcv(self, symbol, tf, since=None, limit=None):
            assert tf == "1d"
            return [[1_789_948_800_000, 1, 2, 0.5, 1.5, 10]]

    async def _persistent(ccxt_id, ccxt_name):
        return _Ex()

    engine.repository = _Repo()
    monkeypatch.setattr(u1d, "get_persistent_exchange", _persistent)
    monkeypatch.setattr(engine, "get_configured_exchanges", lambda: ["gateio"])

    # published as "gate" (1D table suffix) -> resolves to the gateio engine name
    n = await engine.refresh_priority_pair("gate", "0G/USDT:USDT")

    assert n == 1
    assert engine.repository.written == ("db_low_1d", "0g_usdt:usdt_on_gate", "1d", 1)


# ---------------------------------------------------------------------------
# The forming candle must be REWRITTEN, never appended to
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_15m_lane_refetches_the_last_stored_bar(monkeypatch):
    """The newest stored bar is a forming candle frozen mid-flight; the lane
    must ask for it again (one step earlier) so the DELETE >= min_ts rewrite
    replaces it even where `since` is exclusive."""
    import src.core.updater_15m as u

    last_ts = 1_790_000_100  # inside the 1_790_000_100 // 900 bucket
    seen = {}

    class _Ex:
        async def fetch_ohlcv(self, symbol, tf, since=None, limit=None):
            seen["since"] = since
            return [[last_ts * 1000, 1, 2, 0.5, 1.5, 10]]

    async def _persistent(name):
        return _Ex()

    async def _find(tbl):
        return "db_low_15m", last_ts, 1_700_000_000

    async def _save(db, tbl, symbol, ccxt_id, cs):
        seen["saved_min_ts"] = min(int(c[0]) // 1000 for c in cs)
        return db

    monkeypatch.setattr(u, "get_persistent_exchange", _persistent)
    monkeypatch.setattr(u, "find_table_in_dbs", _find)
    monkeypatch.setattr(u, "save_candles_to_table", _save)

    await u.refresh_priority_pair("bybit", "0G/USDT:USDT")

    assert seen["since"] <= (last_ts - 900) * 1000
    # everything from the stale bar on is deleted before the COPY
    assert seen["saved_min_ts"] <= last_ts


@pytest.mark.asyncio
async def test_1d_lane_refetches_the_forming_daily_bar(monkeypatch):
    import src.core.updater as u1d

    engine = u1d.MarketDataEngine(timeframe="1d")
    last_ts = 1_789_948_800
    seen = {}

    class _Repo:
        async def find_table(self, tbl):
            return "db_low_1d", last_ts, 1_600_000_000

        async def upsert_candles(self, db, tbl, df, timeframe="1d"):
            seen["min_ts"] = int(df["Timestamp"].min())

    class _Ex:
        async def fetch_ohlcv(self, symbol, tf, since=None, limit=None):
            seen["since"] = since
            return [[last_ts * 1000, 1, 2, 0.5, 1.5, 10]]

    async def _persistent(ccxt_id, ccxt_name):
        return _Ex()

    engine.repository = _Repo()
    monkeypatch.setattr(u1d, "get_persistent_exchange", _persistent)
    monkeypatch.setattr(engine, "get_configured_exchanges", lambda: ["bybit"])

    await engine.refresh_priority_pair("bybit", "0G/USDT:USDT")

    assert seen["since"] <= (last_ts - 86400) * 1000
    assert seen["min_ts"] <= last_ts
