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
# The lane must BRIDGE a hole, not refresh around it
# ---------------------------------------------------------------------------

def test_lane_since_sec_anchors_at_the_hole():
    """The "33 h gap on the chart" rule: with the collector behind, the fetch has
    to start AT THE HOLE. A fixed tail of 10 bars near `now` refreshed around the
    gap forever — and did so for 15m and 1D alike."""
    from src.core.priority_pairs import lane_since_sec

    now, step = 1_800_000_000, 900
    # a bar or two behind (normal): one bar before the last row, small ask
    assert lane_since_sec(now - 2 * step, now, step) == (now - 3 * step, 2 + 3 + 1)
    # 33 h behind: start at the hole, and ask for the whole hole
    hole = 134
    since, want = lane_since_sec(now - hole * step, now, step)
    assert since == now - (hole + 1) * step and want == hole + 4
    # two YEARS behind: that is history, not a tail — the sweep owns it, and the
    # lane must not spend an hour paging the exchange for a pair someone glanced at
    since, want = lane_since_sec(now - 70_000 * step, now, step)
    assert since == now - 4 * step and want == 5
    # PRIORITY_LANE_CATCHUP_MAX_BARS=0 restores the old behaviour exactly
    assert lane_since_sec(now - hole * step, now, step, catchup_max_bars=0)[0] == now - 4 * step
    # empty table: the sweep creates it, the lane just keeps a tail anchor
    assert lane_since_sec(0, now, step)[0] == now - 4 * step
    # daily: two missing days are one fetch, same rule
    d = 86400
    assert lane_since_sec(now - 2 * d, now, d) == (now - 3 * d, 2 + 4)


@pytest.mark.asyncio
async def test_15m_lane_bridges_a_day_behind(monkeypatch):
    """End-to-end for the visible bug: a table 33 h behind must be fetched from
    its own last row, not from ~45 min ago, so the chart's hole gets closed."""
    import src.core.updater_15m as u

    last_ts = int(time.time()) - 33 * 3600
    seen = {}

    class _Ex:
        async def fetch_ohlcv(self, symbol, tf, since=None, limit=None):
            seen["since"], seen["limit"] = since, limit
            return [[last_ts * 1000 + i * 900_000, 1, 2, 0.5, 1.5, 10] for i in range(limit)]

    async def _persistent(name):
        return _Ex()

    async def _find(tbl):
        return "db_low_15m", last_ts, 1_700_000_000

    async def _save(db, tbl, symbol, ccxt_id, cs):
        seen["saved"] = len(cs)
        return db

    monkeypatch.setattr(u, "get_persistent_exchange", _persistent)
    monkeypatch.setattr(u, "find_table_in_dbs", _find)
    monkeypatch.setattr(u, "save_candles_to_table", _save)

    n = await u.refresh_priority_pair("bybit", "RARE/USDT")

    assert n > 1                                   # more than a tail rewrite
    assert seen["since"] <= (last_ts - 900) * 1000  # reached back to the hole
    assert seen["limit"] >= 130                    # ~33 h of 15m bars in one request
    assert seen["saved"] == n


@pytest.mark.asyncio
async def test_1d_lane_bridges_missing_days(monkeypatch):
    """Same defect on the daily timeframe: bars close once a day, so a two-day
    outage is two WHOLE missing candles that refreshing the forming bar never
    fixes. The daily lane must start at the table's own last row."""
    import src.core.updater as u1d

    now = int(time.time())
    last_ts = now - (now % 86400) - 3 * 86400      # three daily buckets behind
    engine = u1d.MarketDataEngine(timeframe="1d")
    seen = {}

    class _Repo:
        async def find_table(self, tbl):
            return "db_low_1d", last_ts, 1_600_000_000

        async def upsert_candles(self, db, tbl, df, timeframe="1d"):
            seen["written"] = len(df)
            seen["min_ts"] = int(df["Timestamp"].min())

    class _Ex:
        async def fetch_ohlcv(self, symbol, tf, since=None, limit=None):
            seen["since"], seen["limit"] = since, limit
            rows = [[(last_ts + i * 86400) * 1000, 1, 2, 0.5, 1.5, 10] for i in range(4)]
            return [r for r in rows if r[0] >= since]

    async def _persistent(ccxt_id, ccxt_name):
        return _Ex()

    engine.repository = _Repo()
    monkeypatch.setattr(u1d, "get_persistent_exchange", _persistent)
    monkeypatch.setattr(engine, "get_configured_exchanges", lambda: ["bybit"])

    n = await engine.refresh_priority_pair("bybit", "RARE/USDT")

    # The old code anchored at `now - 3*step`, i.e. at the LAST stored bucket and
    # never one before it; the fix must reach one full bar behind that.
    assert seen["since"] == (last_ts - 86400) * 1000
    # the stale forming bar plus the three closed days: rewritten, not appended
    assert n == 4 and seen["min_ts"] == last_ts      # oldest written row = the stale bar


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
# The heartbeat: "did an engine actually service the pair being watched?"
# ---------------------------------------------------------------------------

class _PulseConn:
    """Minimal asyncpg stand-in: records statements, replays a canned row."""

    def __init__(self, row=None):
        self.calls = []
        self._row = row

    async def execute(self, sql, *args):
        self.calls.append(("execute", sql, args))

    async def fetchrow(self, sql, *args):
        self.calls.append(("fetchrow", sql, args))
        return self._row


@pytest.mark.asyncio
async def test_mark_lane_service_stamps_only_the_pairs_actually_served():
    from src.core.priority_pairs import mark_lane_service

    conn = _PulseConn()
    n = await mark_lane_service(conn, [("bybit", "BTC/USDT:USDT"), ("", "junk"),
                                       ("gate", "0G/USDT:USDT")], candles=137)
    assert n == 2
    sqls = [c for c in conn.calls if c[0] == "execute" and "UPDATE" in c[1]]
    assert len(sqls) == 2
    assert "served_at = now()" in sqls[0][1] and "served_bars" in sqls[0][1]
    assert sqls[0][2] == ("bybit", "BTC/USDT:USDT", 137)


@pytest.mark.asyncio
async def test_lane_pulse_reads_watched_served_and_idle():
    from src.core.priority_pairs import lane_pulse

    row = {"watched": 11, "served": 0, "idle_sec": 12_000.0}
    conn = _PulseConn(row)
    pulse = await lane_pulse(conn)
    assert pulse == {"watched": 11, "served": 0, "idle_sec": 12000.0}
    # the aggregate is ONE query, and it is asked with the freshness window
    assert sum(1 for c in conn.calls if c[0] == "fetchrow") == 1
    assert "count(*) FILTER" in conn.calls[-1][1]

    # A connection that already has the table does not re-run DDL every second:
    # the lane reads this table once per tick.
    again = _PulseConn({"watched": 1, "served": 1, "idle_sec": 2.0})
    await lane_pulse(again)
    assert any(c[0] == "execute" and "CREATE TABLE" in c[1] for c in again.calls)
    await lane_pulse(again)
    assert sum(1 for c in again.calls if c[0] == "execute") == 3   # 1 create + 2 alters


def test_both_lane_loops_stamp_the_heartbeat():
    """The engines' loops must report what they served, or the dashboard cannot
    tell a busy collector from an absent one."""
    import pathlib as _pl
    for rel in ("src/core/updater_15m.py", "src/core/updater.py"):
        src = (_pl.Path(__file__).parent.parent / rel).read_text()
        assert "mark_lane_service(" in src, rel
        assert "served_pending" in src or "_LANE_SERVED" in src, rel


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


# ---------------------------------------------------------------------------
# The writer must never delete bars the exchange omitted from its response
# ---------------------------------------------------------------------------

class _RecordingConn:
    def __init__(self):
        self.statements = []
        self.copied = None

    async def fetch(self, sql, *args):
        # information_schema.columns lookup -> pretend every column exists
        import src.core.updater_15m as u
        return [{"column_name": c} for c in u.ALL_COLUMNS_SQL]

    async def execute(self, sql, *args):
        self.statements.append((" ".join(sql.split()), args))

    async def copy_records_to_table(self, tbl, records=None, columns=None):
        self.copied = (tbl, len(records or []), list(columns or []))

    def transaction(self):
        conn = self

        class _Tx:
            async def __aenter__(self_inner):
                return conn

            async def __aexit__(self_inner, *a):
                return False

        return _Tx()


class _RecordingPool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self, *a, **k):
        conn = self._conn

        class _Acq:
            async def __aenter__(self_inner):
                return conn

            async def __aexit__(self_inner, *a):
                return False

        return _Acq()


@pytest.mark.asyncio
async def test_writer_deletes_only_the_buckets_it_rewrites(monkeypatch):
    """Illiquid pairs come back without the intervals that had no trades. The
    old blanket `DELETE >= min_ts` wiped those stored bars and punched a
    one-candle hole into the chart (JUSUNG/USDT:USDT @gateio)."""
    import src.core.updater_15m as u

    conn = _RecordingConn()
    monkeypatch.setitem(u.db_pools, "db_low_15m", _RecordingPool(conn))

    t0 = 1_790_000_000 // 900 * 900
    # exchange returned bar 0 and bar 2 — bar 1 had no trades
    cs = [
        [t0 * 1000, 1, 2, 0.5, 1.5, 10],
        [(t0 + 1800) * 1000, 1, 2, 0.5, 1.5, 10],
    ]
    await u.save_candles_to_table("db_low_15m", "x_on_bybit", "X/USDT", "bybit", cs)

    deletes = [(sql, args) for sql, args in conn.statements if sql.startswith("DELETE FROM")]
    assert deletes, "the rewrite must still delete the buckets it re-inserts"
    sql, args = deletes[0]
    assert '("Timestamp" / 900) = ANY($1::bigint[])' in sql
    assert args[0] == [t0 // 900, (t0 + 1800) // 900]   # the skipped bucket is untouched
    assert conn.copied[1] == 2
