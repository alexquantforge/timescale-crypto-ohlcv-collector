"""15m maintenance (retention DELETE + VACUUM) cadence and scope tests.

Regression: run_maintenance() used to execute a whole-database `VACUUM;`
after EVERY 5-minute cycle — a single postgres backend was observed at
4.3 GB RSS in state D (uninterruptible disk sleep), saturating the laptop's
disk 24/7. Maintenance must be dayly-gated and VACUUM must touch only tables
that actually lost rows.
"""
import asyncio
import logging

from src.core import updater_15m


class FakeConn:
    def __init__(self, tables, delete_counts):
        self._tables = list(tables)
        self._delete_counts = dict(delete_counts)  # table -> rows deleted
        self.executed = []  # every statement seen

    async def fetch(self, sql, *args):
        return [{"table_name": t} for t in self._tables]

    async def execute(self, sql, *args):
        self.executed.append(sql)
        if sql.startswith("DELETE"):
            for t, cnt in self._delete_counts.items():
                if f'"{t}"' in sql:
                    return f"DELETE {cnt}"
            return "DELETE 0"
        return "OK"


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

    def acquire(self):
        return _Acquire(self.conn)


def _setup(monkeypatch, db_map, interval_hours=24):
    """db_map: {db_name: (tables, delete_counts)} -> wires module state."""
    pools, conns = {}, {}
    for db_name, (tables, delete_counts) in db_map.items():
        conn = FakeConn(tables, delete_counts)
        pools[db_name] = FakePool(conn)
        conns[db_name] = conn
    monkeypatch.setattr(updater_15m, "db_pools", pools, raising=True)
    monkeypatch.setattr(updater_15m, "_LAST_MAINTENANCE_AT", 0.0, raising=True)
    from config.settings import settings
    monkeypatch.setattr(settings, "maintenance_interval_hours", interval_hours, raising=False)
    return conns


def _run(coro):
    return asyncio.run(coro)


def test_maintenance_gated_to_interval_and_vacuums_only_affected(monkeypatch):
    conns = _setup(monkeypatch, {
        "low": (["a_on_bybit", "b_on_okx"], {"a_on_bybit": 7}),
        "high": (["c_on_mexc"], {"c_on_mexc": 0}),
    })

    _run(updater_15m.run_maintenance())

    low_sql = conns["low"].executed
    # retention scan touched both tables...
    assert any(s.startswith('DELETE FROM "a_on_bybit"') for s in low_sql)
    assert any(s.startswith('DELETE FROM "b_on_okx"') for s in low_sql)
    # ...but VACUUM ran ONLY for the table that lost rows
    assert low_sql.count('VACUUM "a_on_bybit";') == 1
    assert not any(s.startswith("VACUUM") and "b_on_okx" in s for s in low_sql)
    # and never a whole-database vacuum
    assert "VACUUM;" not in low_sql
    # high db had zero deletions -> no VACUUM there at all
    assert not any(s.startswith("VACUUM") for s in conns["high"].executed)

    # Second call within the interval: no SQL at all (latched).
    for c in conns.values():
        c.executed.clear()
    _run(updater_15m.run_maintenance())
    assert all(c.executed == [] for c in conns.values())


def test_no_deletions_means_no_vacuum(monkeypatch, caplog):
    conns = _setup(monkeypatch, {"low": (["a_on_bybit"], {})})
    with caplog.at_level(logging.INFO, logger="updater_15m"):
        _run(updater_15m.run_maintenance())
    assert not any(s.startswith("VACUUM") for s in conns["low"].executed)
    assert any("VACUUM skipped — no old rows deleted" in r.message for r in caplog.records)


def test_zero_interval_restores_every_cycle_behavior(monkeypatch):
    conns = _setup(monkeypatch, {"low": (["a_on_bybit"], {"a_on_bybit": 3})}, interval_hours=0)
    _run(updater_15m.run_maintenance())
    _run(updater_15m.run_maintenance())
    deletes = [s for s in conns["low"].executed if s.startswith("DELETE")]
    assert len(deletes) == 2  # one scan per call — legacy cadence opt-in


def test_kill_mid_maintenance_does_not_hot_loop(monkeypatch):
    """Latch happens BEFORE the work: an interrupted run waits the interval."""
    conns = _setup(monkeypatch, {"low": (["a_on_bybit"], {"a_on_bybit": 1})})

    class ExplodingConn(FakeConn):
        async def execute(self, sql, *args):
            if sql.startswith("VACUUM"):
                raise RuntimeError("terminated")
            return await super().execute(sql, *args)

    conn = ExplodingConn(["a_on_bybit"], {"a_on_bybit": 1})
    updater_15m.db_pools["low"] = FakePool(conn)

    async def boom():
        try:
            await updater_15m.run_maintenance()
        except Exception:
            pass  # asyncpg errors propagate; latch must still hold

    _run(boom())
    conn.executed.clear()
    _run(updater_15m.run_maintenance())  # immediate retry — must be skipped
    assert conn.executed == []
