"""
Regression test for the live incident (2026-08-29): every orderbook snapshot
write failed with `NameError: name 'datetime' is not defined` —
save_orderbook_snapshot referenced datetime.datetime in the `ob_snapshot_time_msk`
branch while src/db/repository.py never imported datetime. The engine side only
surfaced it after the once-per-pair WARNING instrumentation; before that ob_*
metrics silently stayed empty for ALL exchanges.
"""
import asyncio
import time

import pandas as pd


def _mk_repo():
    from src.db.repository import HistoricalMarketRepository

    executed = []

    class _Conn:
        async def fetchval(self, query, *args):
            return int(time.time())

        async def fetch(self, query, *args):
            return []

        async def fetchrow(self, query, *args):
            return None

        async def execute(self, query, *args):
            executed.append(query)
            return "OK"

        async def executemany(self, query, *args):
            return None

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *exc):
            return False

    class _Pool:
        def acquire(self, *a, **kw):
            return _Acquire()

    repo = HistoricalMarketRepository(pools={"db1": _Pool()}, high_db="db1", low_db="db2")

    # ensure_columns issues ALTERs against the fake conn — fine as-is.
    return repo, executed


def test_save_orderbook_snapshot_writes_msk_time_without_nameerror():
    """Before the fix this raised NameError: name 'datetime' is not defined."""
    from src.db.repository import ORDERBOOK_COLUMNS_SQL

    repo, executed = _mk_repo()
    now = int(time.time())
    df = pd.DataFrame(
        {
            "Timestamp": [now - 3 * 86400, now - 2 * 86400, now - 86400, now],
            "volume": [100.0, 200.0, 150.0, 50.0],
            "low": [10.0, 11.0, 9.0, 12.0],
        }
    )
    ob_col = next(c for c in ORDERBOOK_COLUMNS_SQL if c not in (
        "ob_snapshot_ts", "ob_snapshot_time_msk", "ob_min_7d_volume_usd"))
    snap = {ob_col: 1.5, "ob_extra_ignored": 9.9}

    asyncio.run(repo.save_orderbook_snapshot("db1", "btc_usdt_on_bybit", df, snap, timeframe="1d"))

    updates = [q for q in executed if q.startswith("UPDATE") and "ob_snapshot_time_msk" in q]
    assert updates, f"expected an UPDATE setting ob_snapshot_time_msk, got: {executed}"
    assert "ob_min_7d_volume_usd" in updates[0]
    assert "ob_extra_ignored" not in updates[0]


def test_save_orderbook_snapshot_without_snap_still_ok():
    """Snap=None path must not touch OB columns but still writes min_7d volume."""
    repo, executed = _mk_repo()
    now = int(time.time())
    df = pd.DataFrame(
        {
            "Timestamp": [now - 86400, now],
            "volume": [10.0, 5.0],
            "low": [1.0, 1.0],
        }
    )
    asyncio.run(repo.save_orderbook_snapshot("db1", "eth_usdt_on_bybit", df, None, timeframe="1d"))

    updates = [q for q in executed if q.startswith("UPDATE") and "ob_min_7d_volume_usd" in q]
    assert updates, f"expected an UPDATE setting ob_min_7d_volume_usd, got: {executed}"
    assert "ob_snapshot_time_msk" not in updates[0]


def test_import_order_db_repository_first_no_cycle():
    """The circular import repository -> src.core/__init__ -> updater ->
    repository must not blow up when src.db.repository is the FIRST import
    (production enters via src.core and never noticed it)."""
    import subprocess
    import sys

    for snippet in (
        "import src.db.repository",
        "from src.db.repository import HistoricalMarketRepository",
        "import src.core.updater",
        "import src.core.updater_15m",
    ):
        r = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, cwd=None,
        )
        assert r.returncode == 0, f"{snippet!r} failed: {r.stderr[-500:]}"
