"""
Driven tests for the BACKWARD history prefill (history repair) wired into the
real engine `process_pair` coroutines.

Context: perp tables were truncated to the last few days because the initial
import never paginated. Commit f429313 added backward prefill, but its first
version (a) failed SILENTLY (fetch error -> permanent latch, zero log lines)
and (b) could never be told apart from a stale process in the user's logs.
These tests pin down the visible behaviour:

* a truncated table gets its history paged in down to the floor, with 📜 logs;
* a failing prefill page logs a WARNING and only cooldown-latches (never a
  process-lifetime mute);
* epoch-ms table timestamps are normalized so cursors stay sane.
"""
import asyncio
import logging
import time

import pandas as pd
import pytest

from src.core.history_prefill import (
    MS_EPOCH_THRESHOLD_SEC,
    normalize_epoch_sec,
    should_attempt_prefill,
)

NOW = int(time.time())
DAY = 86400
M15 = 900


# ---------------------------------------------------------------- pure helpers

def test_normalize_epoch_sec():
    assert normalize_epoch_sec(None) is None
    assert normalize_epoch_sec(1_700_000_000) == 1_700_000_000          # seconds stay
    assert normalize_epoch_sec(1_700_000_000_000) == 1_700_000_000      # ms -> seconds
    assert 1_700_000_000 < MS_EPOCH_THRESHOLD_SEC < 1_700_000_000_000


def test_should_attempt_prefill():
    now = 1_000_000.0
    # never attempted -> go
    assert should_attempt_prefill(None, min_ts_sec=500, now_sec=now) is True
    # latched on the same start, recently -> skip (don't hammer the exchange)
    assert should_attempt_prefill((500, now - 60), 500, now_sec=now, retry_after_sec=3600) is False
    # latched but the table start IMPROVED -> keep walking left immediately
    assert should_attempt_prefill((600, now - 60), 500, now_sec=now, retry_after_sec=3600) is True
    # latch older than the cooldown -> retry (transient failures recover)
    assert should_attempt_prefill((500, now - 7200), 500, now_sec=now, retry_after_sec=3600) is True


# ------------------------------------------------------------ fakes / fixtures

class FakeExchange:
    """Generates deterministic OHLCV rows: step-sized bars ascending from
    `since`, capped by `limit` and by the last closed bar.

    `fail_on_prefill` makes only DEEP-history page requests raise (like a
    flaky exchange on far-past `since`), while fresh catch-up requests stay
    fine — this is the exact split that made prefill failures invisible.
    """

    def __init__(self, step_ms, now_ms=None, fail_on_prefill=False, prefill_limit=1000):
        self.step_ms = step_ms
        self.now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        self.fail_on_prefill = fail_on_prefill
        self.prefill_limit = prefill_limit
        self.calls = []  # (since_ms, limit)

    async def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        limit = int(limit or 500)
        since_ms = int(since if since is not None else self.now_ms - limit * self.step_ms)
        self.calls.append((since_ms, limit))
        if (
            self.fail_on_prefill
            and limit >= self.prefill_limit
            and since_ms < self.now_ms - 2 * 86400 * 1000
        ):
            raise RuntimeError("simulated exchange failure on deep-history page")
        last_closed = self.now_ms - self.step_ms
        if since_ms > last_closed:
            await asyncio.sleep(0)
            return []
        rows = []
        t = since_ms
        while t <= last_closed and len(rows) < limit:
            rows.append([t, 1.0, 2.0, 0.5, 1.5, 100.0])
            t += self.step_ms
        await asyncio.sleep(0)
        return rows


class FakeRepo1D:
    """Minimal duck-type of HistoricalMarketRepository for process_pair."""

    def __init__(self, last_ts, first_ts):
        self.low_db = "low_db"
        self._last = last_ts
        self._first = first_ts
        self.upserts = []  # list of (table, df)

    async def find_table(self, table_name):
        return "low_db", self._last, self._first

    async def upsert_candles(self, db_name, table_name, df, timeframe="1d"):
        self.upserts.append((table_name, df))

    async def create_table_if_not_exists(self, db_name, table_name):
        return None

    async def check_volume_floor_and_move(self, table_name, current_db, **kwargs):
        return current_db

    async def cleanup_invalid_bitget_tables(self):
        return 0


def _settings_1d(monkeypatch, tmp_retention=0):
    from config.settings import settings
    monkeypatch.setattr(settings, "update_days", 5, raising=False)
    monkeypatch.setattr(settings, "data_retention_days", tmp_retention, raising=False)
    monkeypatch.setattr(settings, "backfill_start_date", "2018-01-01", raising=False)
    monkeypatch.setattr(settings, "backfill_max_iterations", 400, raising=False)
    monkeypatch.setattr(settings, "backfill_request_limit", 1000, raising=False)
    monkeypatch.setattr(settings, "backfill_request_limit_per_exchange", {}, raising=False)
    monkeypatch.setattr(settings, "history_prefill_max_pages", 10, raising=False)
    monkeypatch.setattr(settings, "history_prefill_retry_sec", 4 * 3600, raising=False)
    monkeypatch.setattr(settings, "check_and_fill_gaps", False, raising=False)
    monkeypatch.setattr(settings, "collect_orderbook", False, raising=False)
    return settings


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------ 1D engine driven

def test_1d_truncated_table_is_filled_back_to_floor(monkeypatch, caplog):
    """A perp table starting 5 days ago must be paged back to the 2018 floor
    (resumable: 10 pages/cycle × 1000 daily candles covers it in one cycle)."""
    from src.core import updater
    settings = _settings_1d(monkeypatch)

    table_start = NOW - 5 * DAY
    repo = FakeRepo1D(last_ts=NOW - DAY, first_ts=table_start)
    engine = updater.MarketDataEngine(timeframe="1d")
    engine.repository = repo
    exchange = FakeExchange(DAY * 1000)

    with caplog.at_level(logging.INFO, logger="engine"):
        count = _run(engine.process_pair(exchange, "BTC/USDT:USDT", "bybit"))

    assert count >= 1  # the fresh candle still lands
    saved = pd.concat([df for _, df in repo.upserts]) if repo.upserts else pd.DataFrame()
    assert not saved.empty
    floor = int(pd.Timestamp("2018-01-01", tz="UTC").timestamp())
    oldest_saved = int(saved["Timestamp"].min())
    # history must reach (essentially) the floor
    assert oldest_saved <= floor + 2 * DAY
    # ...and the log must SHOW the repair (this is what the user watches for)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "📜" in text and "history repair" in text
    assert ("✅" in text) or ("floor reached" in text)


def test_1d_prefill_failure_is_loud_and_only_cooldown_latched(monkeypatch, caplog):
    """A failing deep-history page must (a) log a WARNING, (b) NOT kill the
    pair's normal +1 update, (c) latch for the cooldown only — and the latch
    releases once the cooldown has passed."""
    from src.core import updater
    _settings_1d(monkeypatch)

    table_start = NOW - 5 * DAY
    repo = FakeRepo1D(last_ts=NOW - DAY, first_ts=table_start)
    engine = updater.MarketDataEngine(timeframe="1d")
    engine.repository = repo
    exchange = FakeExchange(DAY * 1000, fail_on_prefill=True)

    with caplog.at_level(logging.INFO, logger="engine"):
        count = _run(engine.process_pair(exchange, "BTC/USDT:USDT", "bybit"))
    assert count >= 1  # catch-up candle still saved
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "⚠️" in text and "history-prefill" in text and "simulated exchange failure" in text
    assert "📜" not in text

    tbl = "btc_usdt:usdt_on_bybit"
    assert tbl in engine._prefill_done

    # Second call within the cooldown: prefill must NOT hit the exchange again...
    exchange.calls.clear()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="engine"):
        _run(engine.process_pair(exchange, "BTC/USDT:USDT", "bybit"))
    prefill_calls = [c for c in exchange.calls if c[1] >= 1000]
    assert prefill_calls == []

    # ...but once the cooldown expired it retries (no process-lifetime mute).
    latched_min, attempt_ts = engine._prefill_done[tbl]
    engine._prefill_done[tbl] = (latched_min, attempt_ts - 5 * 3600)
    exchange.calls.clear()
    with caplog.at_level(logging.INFO, logger="engine"):
        _run(engine.process_pair(exchange, "BTC/USDT:USDT", "bybit"))
    assert any(c[1] >= 1000 for c in exchange.calls)


def test_find_table_normalizes_epoch_ms(monkeypatch, caplog):
    """repository.find_table must normalize epoch-ms MIN/MAX to seconds
    (legacy mixed-epoch tables otherwise freeze catch-up at '+0 forever'
    and send prefill to the year 57000)."""
    from src.db.repository import HistoricalMarketRepository

    class _Row(dict):
        pass

    class _Conn:
        async def fetchval(self, *a, **k):
            return True

        async def fetchrow(self, *a, **k):
            return _Row(mx=(NOW - DAY) * 1000, mn=(NOW - 5 * DAY) * 1000)  # epoch-ms!

    class _Pool:
        def acquire(self, *a, **k):
            pool = self

            class _A:
                async def __aenter__(self):
                    return _Conn()

                async def __aexit__(self, *exc):
                    return False

            return _A()

    repo = HistoricalMarketRepository({"high_db": _Pool(), "low_db": _Pool()}, "high_db", "low_db")
    with caplog.at_level(logging.WARNING, logger="repository"):
        db, mx, mn = _run(repo.find_table("btc_usdt_on_bybit"))
    assert db == "high_db"
    assert mx == NOW - DAY and mn == NOW - 5 * DAY  # seconds now
    assert any("EPOCH-FIX" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------ 15m engine driven

class _Acquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self):
        self.saved = []  # rows copied into the table

    async def execute(self, *args, **kwargs):
        return None

    async def fetch(self, *args, **kwargs):
        return []

    async def fetchval(self, *args, **kwargs):
        return None

    async def fetchrow(self, *args, **kwargs):
        return None

    async def copy_records_to_table(self, table, records, columns):
        self.saved.extend(records)

    def transaction(self):
        return _Acquire(self)


class _FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self, *args, **kwargs):
        return _Acquire(self.conn)


def test_15m_truncated_table_is_filled_toward_180d_floor(monkeypatch, caplog):
    from src.core import updater_15m as u15

    table_start = NOW - 5 * DAY
    conn = _FakeConn()
    monkeypatch.setattr(u15, "db_pools", {"db_high": _FakePool(conn), "db_low": _FakePool(_FakeConn())})

    async def fake_find_table(t):
        return "db_high", NOW - M15, table_start

    async def fake_gaps(*a, **k):
        return 0, 0, 0

    monkeypatch.setattr(u15, "find_table_in_dbs", fake_find_table)
    monkeypatch.setattr(u15, "check_and_fill_table_gaps", fake_gaps)
    monkeypatch.setattr(u15, "COLLECT_ORDERBOOK", False)
    monkeypatch.setattr(u15.settings, "history_prefill_retry_sec", 4 * 3600, raising=False)
    u15._PREFILL_DONE.pop(("bybit", "BTC/USDT:USDT"), None)

    exchange = FakeExchange(M15 * 1000)
    with caplog.at_level(logging.INFO, logger="updater_15m"):
        res = _run(u15.process_pair(exchange, "BTC/USDT:USDT", "bybit"))

    assert res[0] > 0
    assert conn.saved, "prefilled candles must reach the table"
    # rows are written with Timestamp in SECONDS in the first column
    min_ts = min(int(r[0]) for r in conn.saved)
    ten_pages_cover = u15.PREFILL_MAX_PAGES * u15.FETCH_LIMIT * M15  # 10*1000 bars
    assert min_ts <= table_start - ten_pages_cover + 2 * M15
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "📜" in text and "history repair" in text


def test_15m_prefill_failure_is_loud_and_cooldown_latched(monkeypatch, caplog):
    from src.core import updater_15m as u15

    table_start = NOW - 5 * DAY
    conn = _FakeConn()
    monkeypatch.setattr(u15, "db_pools", {"db_high": _FakePool(conn), "db_low": _FakePool(_FakeConn())})

    async def fake_find_table(t):
        return "db_high", NOW - M15, table_start

    async def fake_gaps(*a, **k):
        return 0, 0, 0

    monkeypatch.setattr(u15, "find_table_in_dbs", fake_find_table)
    monkeypatch.setattr(u15, "check_and_fill_table_gaps", fake_gaps)
    monkeypatch.setattr(u15, "COLLECT_ORDERBOOK", False)
    u15._PREFILL_DONE.pop(("bybit", "BTC/USDT:USDT"), None)

    exchange = FakeExchange(M15 * 1000, fail_on_prefill=True)
    with caplog.at_level(logging.INFO, logger="updater_15m"):
        res = _run(u15.process_pair(exchange, "BTC/USDT:USDT", "bybit"))

    assert res[0] >= 1  # normal candle still lands
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "⚠️" in text and "history-prefill" in text
    assert "📜" not in text
    assert ("bybit", "BTC/USDT:USDT") in u15._PREFILL_DONE

    # within cooldown -> no new DEEP-history calls (catch-up still happens
    # with fresh `since`, so distinguish by the requested window age)
    exchange.calls.clear()
    with caplog.at_level(logging.INFO, logger="updater_15m"):
        _run(u15.process_pair(exchange, "BTC/USDT:USDT", "bybit"))
    deep_calls = [c for c in exchange.calls if c[0] < (NOW - 2 * DAY) * 1000]
    assert deep_calls == []
