"""Column-type preservation/repair tests.

Regression: move_table (HIGH<->LOW) recreated tables with
`ALL_COLUMNS_SQL.get(col, 'TEXT')`, so every ob_*/oi column of a moved table
became TEXT — and each subsequent numeric snapshot write failed with
`DataError: invalid input for query argument $1: ... (expected str, got
float)` (ZK/USDT @mexc 1D, ZINC/USDT:USDT @mexc 15m in production logs).

move_table must now take real types from information_schema, and the
*_ensure_columns paths must cast TEXT-ified numeric columns back.
"""
import asyncio
import logging

import pytest

from src.db.repository import (
    ALL_COLUMNS_SQL,
    ORDERBOOK_COLUMNS_SQL,
    HistoricalMarketRepository,
    pg_ddl_type,
    repair_text_typed_columns,
)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- fakes

class FakeConn:
    def __init__(self, star_rows=(), col_types=None):
        self._star_rows = list(star_rows)
        self._col_types = dict(col_types or {})
        self.executed = []
        self.copied = None

    async def fetch(self, sql, *args):
        if "information_schema.columns" in sql:
            return [
                {"column_name": c, "data_type": t}
                for c, t in self._col_types.items()
            ]
        if "SELECT * FROM" in sql:
            return [dict(r) for r in self._star_rows]
        return []

    async def execute(self, sql, *args):
        self.executed.append(sql)
        return "OK"

    async def copy_records_to_table(self, table_name, records, columns):
        self.copied = (table_name, list(records), list(columns))

    async def fetchval(self, sql, *args):
        return None


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

    def acquire(self, *args, **kwargs):
        return _Acquire(self.conn)


TBL = "zk_usdt_on_mexc"
ROW = {"Timestamp": 1, "open": 5.0, "ob_cvd": 12.5, "open_interest": 42.0}


def _col_types(**overrides):
    base = {
        "Timestamp": "bigint",
        "open": "double precision",
        "ob_cvd": "double precision",
        "open_interest": "double precision",
    }
    base.update(overrides)
    return base


# ------------------------------------------------------ move_table typing

def test_repository_move_table_uses_real_column_types():
    src, dst = FakeConn(star_rows=[ROW], col_types=_col_types()), FakeConn()
    repo = HistoricalMarketRepository({"low": FakePool(src), "high": FakePool(dst)}, "high", "low")
    _run(repo.move_table(TBL, "low", "high"))

    creates = [s for s in dst.executed if s.startswith("CREATE TABLE")]
    assert len(creates) == 1
    assert '"ob_cvd" DOUBLE PRECISION' in creates[0]
    assert '"open_interest" DOUBLE PRECISION' in creates[0]
    assert '"ob_cvd" TEXT' not in creates[0]
    assert dst.copied is not None


def test_15m_move_table_uses_real_column_types(monkeypatch):
    from src.core import updater_15m

    src, dst = FakeConn(star_rows=[ROW], col_types=_col_types()), FakeConn()
    monkeypatch.setattr(
        updater_15m, "db_pools", {"low": FakePool(src), "high": FakePool(dst)}
    )
    _run(updater_15m.move_table(TBL, "low", "high"))

    creates = [s for s in dst.executed if s.startswith("CREATE TABLE")]
    assert len(creates) == 1
    assert '"ob_cvd" DOUBLE PRECISION' in creates[0]
    assert '"open_interest" DOUBLE PRECISION' in creates[0]


# ---------------------------------------------------------- repair logic

def test_repair_casts_only_known_numeric_text_columns(caplog):
    conn = FakeConn()
    existing_types = {
        "ob_cvd": "text",                       # garbled by a move -> repair
        "ob_is_barcode": "text",                # garbled boolean    -> repair
        "open_interest": "text",                # garbled, but not in this dict
        "ob_snapshot_time_msk": "text",         # legitimately TEXT -> keep
        "volume": "double precision",           # healthy -> keep
    }
    with caplog.at_level(logging.WARNING, logger="repository"):
        repaired = _run(repair_text_typed_columns(
            conn, TBL, existing_types, ORDERBOOK_COLUMNS_SQL,
        ))
    assert repaired == ["ob_cvd", "ob_is_barcode"]
    assert any('ALTER COLUMN "ob_cvd" TYPE DOUBLE PRECISION' in s for s in conn.executed)
    assert any('ALTER COLUMN "ob_is_barcode" TYPE BOOLEAN' in s for s in conn.executed)
    assert not any("ob_snapshot_time_msk" in s for s in conn.executed)
    assert any("was TEXT after a HIGH" in r.message for r in caplog.records)


def test_repository_ensure_columns_repairs_in_place():
    conn = FakeConn(col_types=_col_types(ob_cvd="text"))
    repo = HistoricalMarketRepository({"high": FakePool(conn)}, "high", "low")
    _run(repo.ensure_columns(FakePool(conn), TBL))
    assert any('ALTER COLUMN "ob_cvd" TYPE DOUBLE PRECISION' in s for s in conn.executed)
    # candle column types are healthy — no repair noise
    assert not any('ALTER COLUMN "open"' in s for s in conn.executed)


def test_15m_ensure_orderbook_columns_repairs(monkeypatch):
    from src.core import updater_15m

    conn = FakeConn(col_types=_col_types(**{c: "text" for c in ORDERBOOK_COLUMNS_SQL}))
    _run(updater_15m.ensure_orderbook_columns(conn, TBL))
    assert any('ALTER COLUMN "ob_cvd" TYPE DOUBLE PRECISION' in s for s in conn.executed)
    # TEXT-typed ob_snapshot_time_msk must stay TEXT
    assert not any("ob_snapshot_time_msk" in s for s in conn.executed)


def test_oi_ensure_repairs_text_columns():
    from src.core import oi_funding

    oi_funding._ENSURED.clear()
    conn = FakeConn(col_types=_col_types(open_interest="text"))
    _run(oi_funding.ensure_oi_funding_columns(conn, TBL))
    assert any('ALTER COLUMN "open_interest" TYPE DOUBLE PRECISION' in s for s in conn.executed)


def test_pg_ddl_type_mapping():
    assert pg_ddl_type("double precision") == "DOUBLE PRECISION"
    assert pg_ddl_type("bigint") == "BIGINT"
    assert pg_ddl_type("text") == "TEXT"
    assert pg_ddl_type(None) == "TEXT"
    assert pg_ddl_type("something exotic") == "TEXT"
