"""Open Interest + Funding Rate collection tests (src/core/oi_funding.py).

Covers snapshot parsing/tolerance, column DDL latching, snapshot writes to the
newest candle row, funding-history pagination/resume/latching (no hot retry
loop after a failed import), and the settings defaults that gate the feature
in both engines.
"""
import asyncio
import logging

import pytest

from config.settings import settings
from src.core import oi_funding
from src.core.oi_funding import (
    backfill_funding_history,
    ensure_oi_funding_columns,
    fetch_oi_funding_snapshot,
    warn_once,
    write_oi_funding_snapshot,
)


@pytest.fixture(autouse=True)
def _clear_latches():
    oi_funding._ENSURED.clear()
    oi_funding._WARNED.clear()
    oi_funding._BF_DONE.clear()
    yield


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- fakes

class FakeExchange:
    def __init__(self, oi=None, fr=None, history_pages=None, has=None, boom=False):
        self._oi = oi or {}
        self._fr = fr or {}
        self._pages = list(history_pages or [])
        self._boom = boom
        self.has = has if has is not None else {
            "fetchOpenInterest": True,
            "fetchFundingRate": True,
            "fetchFundingRateHistory": True,
        }
        self.history_calls = []

    async def fetch_open_interest(self, symbol):
        return dict(self._oi)

    async def fetch_funding_rate(self, symbol):
        return dict(self._fr)

    async def fetch_funding_rate_history(self, symbol, since=None, limit=None):
        self.history_calls.append((symbol, since, limit))
        if self._boom:
            raise RuntimeError("429 rate limit")
        if not self._pages:
            return []
        return self._pages.pop(0)


class FakeConn:
    def __init__(self, columns=(), max_funding_ts=None):
        self._columns = list(columns)
        self._max_funding_ts = max_funding_ts
        self.executed = []      # (sql, args)
        self.executemany_calls = []  # (sql, rows)

    async def fetch(self, sql, *args):
        return [{"column_name": c} for c in self._columns]

    async def fetchval(self, sql, *args):
        return self._max_funding_ts

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "OK"

    async def executemany(self, sql, rows):
        self.executemany_calls.append((sql, list(rows)))


class _Acquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self, timeout=None):
        return _Acquire(self.conn)


# ------------------------------------------------------------------- snapshot

def test_snapshot_parses_oi_and_funding_ms_to_sec():
    ex = FakeExchange(
        oi={"openInterestAmount": 12345.5, "timestamp": 1_756_000_000_000},
        fr={"fundingRate": 0.0001, "previousFundingTimestamp": 1_755_996_800_000},
    )
    snap = _run(fetch_oi_funding_snapshot(ex, "BTC/USDT:USDT", "bybit"))
    assert snap["open_interest"] == pytest.approx(12345.5)
    assert snap["oi_ts"] == 1_756_000_000_000 // 1000
    assert snap["funding_rate"] == pytest.approx(0.0001)
    assert snap["funding_ts"] == 1_755_996_800_000 // 1000


def test_snapshot_tolerates_unsupported_or_partial_metrics():
    ex = FakeExchange(has={"fetchOpenInterest": False, "fetchFundingRate": False})
    assert _run(fetch_oi_funding_snapshot(ex, "X/USDT:USDT", "mexc")) == {}

    ex2 = FakeExchange(oi={"openInterestAmount": None, "openInterestValue": 777.0},
                       has={"fetchOpenInterest": True, "fetchFundingRate": False})
    snap = _run(fetch_oi_funding_snapshot(ex2, "X/USDT:USDT", "okx"))
    assert snap["open_interest"] == pytest.approx(777.0)
    assert "funding_rate" not in snap


# ------------------------------------------------------------------ DB writes

def test_write_snapshot_alters_columns_once_and_updates_latest_row():
    conn = FakeConn(columns=())  # nothing exists yet
    pool = FakePool(conn)
    snap = {"open_interest": 1.0, "oi_ts": 10, "funding_rate": 0.001, "funding_ts": 9}
    _run(write_oi_funding_snapshot(pool, "btc_usdt:usdt_on_bybit", snap))
    _run(write_oi_funding_snapshot(pool, "btc_usdt:usdt_on_bybit", snap))

    alters = [s for s, _ in conn.executed if s.startswith("ALTER TABLE")]
    assert len(alters) == 4  # second call latched by _ENSURED — no new DDL
    updates = [s for s, _ in conn.executed if s.startswith("UPDATE")]
    assert len(updates) == 2
    assert 'SELECT MAX("Timestamp")' in updates[0]


def test_ensure_columns_adds_only_missing():
    conn = FakeConn(columns=("open_interest", "oi_ts", "funding_rate", "funding_ts"))
    _run(ensure_oi_funding_columns(conn, "t_on_okx"))
    assert conn.executed == []  # nothing missing -> no ALTERs


# ----------------------------------------------------------- history backfill

def _ev(ts_ms, rate=0.0001):
    return {"timestamp": ts_ms, "fundingRate": rate}


def test_backfill_pages_forward_and_latches():
    t0 = 1_700_000_000_000
    ex = FakeExchange(history_pages=[
        [_ev(t0), _ev(t0 + 28_800_000), _ev(t0 + 57_600_000)],
        [],  # end of history
    ])
    conn = FakeConn(columns=tuple(oi_funding.OI_FUNDING_COLUMNS_SQL), max_funding_ts=None)
    total = _run(backfill_funding_history(
        ex, FakePool(conn), "eth_usdt:usdt_on_okx", "ETH/USDT:USDT", "okx",
        since_ts=t0 // 1000 - 86_400,
    ))
    assert total == 3
    assert len(conn.executemany_calls) == 1
    sql, rows = conn.executemany_calls[0]
    assert 'WHERE "Timestamp" =' in sql and 'MAX("Timestamp")' in sql
    assert rows[0] == (pytest.approx(0.0001), t0 // 1000)

    # Latched: a repeat call in the same run fetches nothing.
    assert _run(backfill_funding_history(
        ex, FakePool(conn), "eth_usdt:usdt_on_okx", "ETH/USDT:USDT", "okx",
        since_ts=0,
    )) == 0
    assert len(ex.history_calls) == 2  # first page + empty page only


def test_backfill_resumes_from_stored_max():
    t0 = 1_700_000_000_000
    have_max = t0 // 1000 + 28_800  # second event already stored
    ex = FakeExchange(history_pages=[[_ev(t0 + 57_600_000)], []])
    conn = FakeConn(columns=tuple(oi_funding.OI_FUNDING_COLUMNS_SQL), max_funding_ts=have_max)
    total = _run(backfill_funding_history(
        ex, FakePool(conn), "sol_usdt:usdt_on_bybit", "SOL/USDT:USDT", "bybit",
        since_ts=t0 // 1000 - 86_400,
    ))
    assert total == 1
    # cursor moved past the stored max: first fetch since > have_max
    first_since = ex.history_calls[0][1]
    assert first_since > have_max * 1000


def test_backfill_up_to_date_short_circuits():
    now_s = int(oi_funding.time.time())
    ex = FakeExchange(history_pages=[[_ev(now_s * 1000)]])
    conn = FakeConn(columns=tuple(oi_funding.OI_FUNDING_COLUMNS_SQL), max_funding_ts=now_s)
    assert _run(backfill_funding_history(
        ex, FakePool(conn), "x_usdt:usdt_on_bingx", "X/USDT:USDT", "bingx", since_ts=0,
    )) == 0
    assert ex.history_calls == []


def test_backfill_fetch_failure_latches_and_warns(caplog):
    ex = FakeExchange(boom=True)
    conn = FakeConn(columns=tuple(oi_funding.OI_FUNDING_COLUMNS_SQL))
    with caplog.at_level(logging.WARNING, logger="oi_funding"):
        total = _run(backfill_funding_history(
            ex, FakePool(conn), "y_usdt:usdt_on_gateio", "Y/USDT:USDT", "gateio", since_ts=0,
        ))
    assert total == 0
    assert any("will retry on restart" in r.message for r in caplog.records)

    # No hot loop: second call in the same run makes ZERO fetch attempts.
    assert _run(backfill_funding_history(
        ex, FakePool(conn), "y_usdt:usdt_on_gateio", "Y/USDT:USDT", "gateio", since_ts=0,
    )) == 0
    assert len(ex.history_calls) == 1

    conn.executemany_calls.clear()
    assert conn.executemany_calls == []  # failed page wrote nothing


def test_backfill_skips_exchanges_without_history_support():
    ex = FakeExchange(has={"fetchOpenInterest": True, "fetchFundingRate": True,
                           "fetchFundingRateHistory": False})
    conn = FakeConn()
    assert _run(backfill_funding_history(
        ex, FakePool(conn), "z_usdt:usdt_on_mexc", "Z/USDT:USDT", "mexc", since_ts=0,
    )) == 0
    assert ex.history_calls == []


# ---------------------------------------------------------------------- misc

def test_warn_once_first_warning_then_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="oi_funding"):
        warn_once("bybit", "A/USDT:USDT", "boom 1")
        warn_once("bybit", "A/USDT:USDT", "boom 2")
    levels = [r.levelno for r in caplog.records]
    assert levels == [logging.WARNING, logging.DEBUG]


def test_settings_defaults(monkeypatch):
    assert settings.collect_oi_funding is True
    assert settings.funding_history_backfill is True
    assert settings.funding_history_max_pages == 100


def test_engines_wire_the_module():
    import src.core.updater as u1d
    import src.core.updater_15m as u15
    assert u1d.fetch_oi_funding_snapshot is fetch_oi_funding_snapshot
    assert u15.backfill_funding_history is backfill_funding_history
