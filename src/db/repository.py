"""
Data Repository layer for individual symbol tables in the 4 historical TimescaleDB databases.
Supports table migration (move_table) and cleanup (drop_table, cleanup_invalid_bitget_tables).
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import asyncpg
import numpy as np
import pandas as pd
from pytz import timezone as pytz_timezone

from config.settings import settings
from src.core.history_prefill import normalize_epoch_sec

logger = logging.getLogger("repository")
MSK_TZ = pytz_timezone("Europe/Moscow")

# Hard bound on pool.acquire() waits. Without a timeout, a starving pool
# (e.g. a nested-acquire deadlock or a pile-up of slow queries) blocks the
# acquiring task FOREVER — client-side, where Postgres statement/lock
# timeouts cannot help — and silently freezes the whole collector cycle.
_ACQUIRE_TIMEOUT = 30.0

ALL_COLUMNS_SQL = {
    "Timestamp": "BIGINT",
    "open": "DOUBLE PRECISION",
    "high": "DOUBLE PRECISION",
    "low": "DOUBLE PRECISION",
    "close": "DOUBLE PRECISION",
    "volume": "DOUBLE PRECISION",
    "ticker": "TEXT",
    "exchange": "TEXT",
    "open_time_msk": "TEXT",
    "open_time_almaty": "TEXT",
    "volume_x_low": "DOUBLE PRECISION",
    "volume_x_close": "DOUBLE PRECISION",
    "asset_type": "TEXT",
    "url_of_trading_pair": "TEXT",
    "url_of_swap_contract_if_it_exists": "TEXT",
}

ORDERBOOK_COLUMNS_SQL: Dict[str, str] = {
    "ob_snapshot_ts": "BIGINT",
    "ob_snapshot_time_msk": "TEXT",
    "ob_last_trade_sec": "DOUBLE PRECISION",
    "ob_trades_per_min": "DOUBLE PRECISION",
    "ob_buy_pressure_pct": "DOUBLE PRECISION",
    "ob_cvd": "DOUBLE PRECISION",
    "ob_cvd_5m": "DOUBLE PRECISION",
    "ob_spread_abs": "DOUBLE PRECISION",
    "ob_spread_pct": "DOUBLE PRECISION",
    "ob_spread_atr_pct": "DOUBLE PRECISION",
    "ob_atr_no_paranormal": "DOUBLE PRECISION",
    "ob_best_bid": "DOUBLE PRECISION",
    "ob_best_ask": "DOUBLE PRECISION",
    "ob_bid_depth_usd": "DOUBLE PRECISION",
    "ob_ask_depth_usd": "DOUBLE PRECISION",
    "ob_total_depth_usd": "DOUBLE PRECISION",
    "ob_imbalance": "DOUBLE PRECISION",
    "ob_vitality_score": "DOUBLE PRECISION",
    "ob_vitality_grade": "TEXT",
    "ob_is_barcode": "BOOLEAN",
    "ob_min_7d_volume_usd": "DOUBLE PRECISION",
}

# Crypto exception symbols starting with R that should NOT be dropped from Bitget
BITGET_CRYPTO_EXCEPTIONS = {
    "rare_usdt", "ray_usdt", "ramp_usdt", "rau_usdt", "rave_usdt",
    "rndr_usdt", "rsr_usdt", "rune_usdt", "rvn_usdt", "rose_usdt", "req_usdt",
    "rare_usdt:usdt", "ray_usdt:usdt", "rndr_usdt:usdt", "rsr_usdt:usdt", "rune_usdt:usdt"
}


class HistoricalMarketRepository:
    def __init__(self, pools: Dict[str, asyncpg.Pool], high_db: str, low_db: str):
        self.pools = pools
        self.high_db = high_db
        self.low_db = low_db

    async def find_table(self, table_name: str) -> Tuple[Optional[str], int, int]:
        """
        Searches HIGH and LOW databases for table_name.
        Returns (db_name, max_timestamp, min_timestamp) or (None, 0, 0).
        """
        for db in [self.high_db, self.low_db]:
            pool = self.pools.get(db)
            if not pool:
                continue
            async with pool.acquire(timeout=_ACQUIRE_TIMEOUT) as conn:
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=$1)",
                    table_name,
                )
                if exists:
                    row = await conn.fetchrow(
                        f'SELECT MAX("Timestamp") as mx, MIN("Timestamp") as mn FROM "{table_name}"'
                    )
                    raw_mx = int(row["mx"]) if row and row["mx"] else 0
                    raw_mn = int(row["mn"]) if row and row["mn"] else 0
                    mx = normalize_epoch_sec(raw_mx) or 0
                    mn = normalize_epoch_sec(raw_mn) or 0
                    if (raw_mx and raw_mx != mx) or (raw_mn and raw_mn != mn):
                        # Legacy epoch-ms rows poison both catch-up ("+0
                        # forever") and history-prefill (absurd `since`).
                        logger.warning(
                            f"[EPOCH-FIX] '{table_name}' in '{db}' stores Timestamp in "
                            f"epoch-ms (MIN={raw_mn}, MAX={raw_mx}) — cursors normalized "
                            f"to seconds; the next save will rewrite the table in seconds."
                        )
                    return db, mx, mn
        return None, 0, 0

    async def drop_table(self, db_name: str, table_name: str) -> bool:
        """
        Drops table_name from db_name.
        """
        pool = self.pools.get(db_name)
        if not pool:
            return False
        try:
            async with pool.acquire(timeout=_ACQUIRE_TIMEOUT) as conn:
                await conn.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
            logger.info(f"🗑️ [DROP TABLE] Dropped table '{table_name}' from database '{db_name}'.")
            return True
        except Exception as e:
            logger.warning(f"Failed dropping table '{table_name}' from '{db_name}': {e}")
            return False

    async def cleanup_invalid_bitget_tables(self) -> int:
        """
        Scans HIGH and LOW databases for Bitget tokenized stock tables (starting with r%_on_bitget)
        and drops them. Returns number of dropped tables.
        """
        total_dropped = 0
        for db in [self.high_db, self.low_db]:
            pool = self.pools.get(db)
            if not pool:
                continue
            async with pool.acquire(timeout=_ACQUIRE_TIMEOUT) as conn:
                rows = await conn.fetch(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='public' AND table_name LIKE 'r%_on_bitget%'"
                )
                for r in rows:
                    tbl = r["table_name"].lower()
                    # Check if symbol is a legitimate crypto exception
                    pair_part = tbl.rsplit("_on_", 1)[0]
                    if pair_part not in BITGET_CRYPTO_EXCEPTIONS:
                        await conn.execute(f'DROP TABLE IF EXISTS "{tbl}" CASCADE')
                        total_dropped += 1
                        logger.info(f"🗑️ [DROP BITGET STOCK] Dropped tokenized stock table '{tbl}' from '{db}'.")
        if total_dropped > 0:
            logger.info(f"✓ Cleaned up {total_dropped} Bitget tokenized stock tables.")
        return total_dropped

    async def move_table(self, table_name: str, from_db: str, to_db: str) -> None:
        """
        Moves table_name from from_db to to_db when volume tier threshold changes.
        """
        from_pool = self.pools.get(from_db)
        to_pool = self.pools.get(to_db)
        if not from_pool or not to_pool:
            return

        async with from_pool.acquire(timeout=_ACQUIRE_TIMEOUT) as fc:
            rows = await fc.fetch(f'SELECT * FROM "{table_name}"')
            if not rows:
                return

        async with to_pool.acquire(timeout=_ACQUIRE_TIMEOUT) as tc:
            cols_sql = ", ".join(
                [f'"{k}" {ALL_COLUMNS_SQL.get(k, "TEXT")}' for k in rows[0].keys()]
            )
            await tc.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
            await tc.execute(f'CREATE TABLE "{table_name}" ({cols_sql})')
            try:
                await tc.execute(
                    f"SELECT create_hypertable('{table_name}', 'Timestamp', if_not_exists => TRUE)"
                )
            except Exception:
                pass
            await tc.copy_records_to_table(
                table_name,
                records=[tuple(r.values()) for r in rows],
                columns=list(rows[0].keys()),
            )

        async with from_pool.acquire(timeout=_ACQUIRE_TIMEOUT) as fc2:
            await fc2.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')

        logger.info(f"🔄 Moved table {table_name}: {from_db} -> {to_db}")

    async def ensure_columns(
        self,
        pool: asyncpg.Pool,
        table_name: str,
        conn: Optional[asyncpg.Connection] = None,
    ) -> None:
        """Adds missing columns (migrations) to individual symbol table.

        Pass an already-acquired `conn` when the caller holds one. Acquiring a
        second connection from the same pool while holding the first is a
        classic pool-starvation deadlock: once max_pool_size workers each hold
        one connection and all wait for another, the pool never recovers
        (pool.acquire() has no default timeout) and every subsequent query
        queues forever — this froze the whole 1D collector cycle.
        """
        if conn is not None:
            await self._ensure_columns_on(conn, table_name)
            return
        async with pool.acquire(timeout=_ACQUIRE_TIMEOUT) as own_conn:
            await self._ensure_columns_on(own_conn, table_name)

    async def _ensure_columns_on(
        self, conn: asyncpg.Connection, table_name: str
    ) -> None:
        existing = {
            r["column_name"]
            for r in await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name=$1",
                table_name,
            )
        }
        for col, typ in ALL_COLUMNS_SQL.items():
            if col not in existing:
                await conn.execute(
                    f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" {typ}'
                )
        for col, typ in ORDERBOOK_COLUMNS_SQL.items():
            if col not in existing:
                await conn.execute(
                    f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" {typ}'
                )

    async def create_table_if_not_exists(self, db_name: str, table_name: str) -> None:
        """Creates table_name in db_name with standard schema and TimescaleDB hypertable."""
        pool = self.pools.get(db_name)
        if not pool:
            return
        async with pool.acquire(timeout=_ACQUIRE_TIMEOUT) as conn:
            cols = [f'"{c}" {t}' for c, t in ALL_COLUMNS_SQL.items()]
            await conn.execute(
                f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(cols)})'
            )
            try:
                await conn.execute(
                    f"SELECT create_hypertable('{table_name}', 'Timestamp', if_not_exists => TRUE)"
                )
            except Exception:
                pass

    async def upsert_candles(
        self, db_name: str, table_name: str, df: pd.DataFrame, timeframe: str = "1d"
    ) -> None:
        """
        Deletes overlapping candles and inserts new candles into TimescaleDB table_name.
        """
        pool = self.pools.get(db_name)
        if not pool or df.empty:
            return

        await self.ensure_columns(pool, table_name)
        actual_cols = list(ALL_COLUMNS_SQL.keys())
        tuples = [
            tuple(None if pd.isna(x) else x for x in row)
            for row in df[actual_cols].to_numpy()
        ]
        min_new_ts = int(df["Timestamp"].min())

        divisor = 900 if timeframe == "15m" else 86400

        async with pool.acquire(timeout=_ACQUIRE_TIMEOUT) as conn:
            async with conn.transaction():
                await conn.execute(
                    f'DELETE FROM "{table_name}" WHERE "Timestamp" >= $1', min_new_ts
                )
                await conn.copy_records_to_table(
                    table_name, records=tuples, columns=actual_cols
                )

                # TimescaleDB hypertable compatible deduplication CTE
                await conn.execute(
                    f"""
                    WITH dups AS (
                        SELECT "Timestamp",
                               row_number() OVER (
                                   PARTITION BY ("Timestamp" / {divisor})
                                   ORDER BY COALESCE(volume, 0) DESC, "Timestamp" DESC
                               ) AS rn
                        FROM "{table_name}"
                    )
                    DELETE FROM "{table_name}" 
                    WHERE "Timestamp" IN (
                        SELECT "Timestamp" FROM dups WHERE rn > 1
                    )
                    """
                )

    async def get_stored_days(self, symbol: str, ccxt_id: str, timeframe: str = "1d") -> List[int]:
        """
        Returns sorted distinct candle bucket numbers ("Timestamp" // step)
        stored for the symbol's table across HIGH/LOW databases.
        Used by the gap detector (1 bucket = 1 day for 1d, 15 minutes for 15m).
        """
        step = 900 if timeframe == "15m" else 86400
        tbl_name = f"{symbol.replace('/', '_').replace('-', '_')}_on_{ccxt_id}".lower()
        db, _, _ = await self.find_table(tbl_name)
        if not db:
            return []
        pool = self.pools.get(db)
        if not pool:
            return []
        try:
            async with pool.acquire(timeout=_ACQUIRE_TIMEOUT) as conn:
                rows = await conn.fetch(
                    f'SELECT DISTINCT "Timestamp" / {int(step)} AS bucket '
                    f'FROM "{tbl_name}" ORDER BY bucket ASC'
                )
            return [int(r["bucket"]) for r in rows]
        except Exception as e:
            logger.warning(f"get_stored_days failed for {tbl_name}: {e}")
            return []

    async def upsert_ohlcv_batch(self, df: pd.DataFrame, timeframe: str = "1d") -> int:
        """
        Inserts gap-fill candles produced by gap_filler (frame with ts in
        milliseconds plus url_trading/url_swap columns), deleting any
        overlapping stored rows first. Returns the number of inserted rows.
        """
        if df is None or df.empty:
            return 0

        df = df.copy()
        df["Timestamp"] = df["ts"].astype("int64") // 1000
        df["volume_x_low"] = df["volume"] * df["low"]
        df["volume_x_close"] = df["volume"] * df["close"]
        df["url_of_trading_pair"] = df["url_trading"] if "url_trading" in df.columns else None
        df["url_of_swap_contract_if_it_exists"] = df["url_swap"] if "url_swap" in df.columns else None
        if "open_time_msk" not in df.columns:
            df["open_time_msk"] = None
            df["open_time_almaty"] = None
        if "asset_type" not in df.columns:
            df["asset_type"] = "swap" if ":" in str(df["ticker"].iloc[0]) else "spot"

        symbol = str(df["ticker"].iloc[0])
        ccxt_id = str(df["exchange"].iloc[0])
        tbl_name = f"{symbol.replace('/', '_').replace('-', '_')}_on_{ccxt_id}".lower()
        db, _, _ = await self.find_table(tbl_name)
        if not db:
            logger.warning(f"upsert_ohlcv_batch: table {tbl_name} not found, skipping")
            return 0
        pool = self.pools.get(db)
        if not pool:
            return 0

        cols = list(ALL_COLUMNS_SQL.keys())
        tuples = [
            tuple(None if pd.isna(x) else x for x in row)
            for row in df[cols].to_numpy()
        ]
        min_ts = int(df["Timestamp"].min())
        try:
            async with pool.acquire(timeout=_ACQUIRE_TIMEOUT) as conn:
                await conn.execute(
                    f'DELETE FROM "{tbl_name}" WHERE "Timestamp" >= $1', min_ts
                )
                await conn.copy_records_to_table(tbl_name, records=tuples, columns=cols)
            return len(tuples)
        except Exception as e:
            logger.warning(f"upsert_ohlcv_batch failed for {tbl_name}: {e}")
            return 0

    async def save_orderbook_snapshot(
        self,
        db_name: str,
        table_name: str,
        full_df: pd.DataFrame,
        snap: Optional[Dict[str, Any]],
        timeframe: str = "1d",
        min_days_check: int = 7,
    ) -> None:
        """Writes orderbook metrics to the latest row (MAX Timestamp) of table_name."""
        pool = self.pools.get(db_name)
        if not pool:
            return

        interval_sec = 900 if timeframe == "15m" else 86400
        bars_count = min_days_check * (96 if timeframe == "15m" else 1)

        async with pool.acquire(timeout=_ACQUIRE_TIMEOUT) as conn:
            # Pass the already-held connection: never acquire a second one from
            # the same pool here (nested-acquire deadlock — freezes the engine).
            await self.ensure_columns(pool, table_name, conn=conn)

            min_7d = None
            try:
                closed_cutoff = int(time.time()) - interval_sec
                closed = full_df[full_df["Timestamp"] <= closed_cutoff]
                tail = closed.tail(bars_count)
                if len(tail) >= 1:
                    min_7d = float((tail["volume"] * tail["low"]).min())
            except Exception:
                min_7d = None

            last_ts = await conn.fetchval(
                f'SELECT MAX("Timestamp") FROM "{table_name}"'
            )
            if last_ts is None:
                return

            set_parts, values = [], []
            idx = 1

            def add(col, val):
                nonlocal idx
                set_parts.append(f'"{col}" = ${idx}')
                values.append(val)
                idx += 1

            if min_7d is not None and np.isfinite(min_7d):
                add("ob_min_7d_volume_usd", round(min_7d, 2))

            if snap:
                now = int(time.time())
                add("ob_snapshot_ts", now)
                add(
                    "ob_snapshot_time_msk",
                    datetime.datetime.fromtimestamp(now, MSK_TZ).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                )
                for col in ORDERBOOK_COLUMNS_SQL:
                    if col in (
                        "ob_snapshot_ts",
                        "ob_snapshot_time_msk",
                        "ob_min_7d_volume_usd",
                    ):
                        continue
                    if col in snap:
                        add(col, snap[col])

            if not set_parts:
                return
            values.append(int(last_ts))
            sql = f'UPDATE "{table_name}" SET {", ".join(set_parts)} WHERE "Timestamp" = ${idx}'
            await conn.execute(sql, *values)

    async def check_volume_floor_and_move(
        self,
        table_name: str,
        current_db: str,
        symbol: str,
        hard_floor_usd: float,
        timeframe: str = "1d",
        min_days_check: int = 7,
    ) -> str:
        """
        Calculates minimum volume over min_days_check and moves table between HIGH and LOW DBs if needed.
        """
        pool = self.pools.get(current_db)
        if not pool:
            return current_db

        interval_sec = 900 if timeframe == "15m" else 86400
        closed_cutoff = int(time.time()) - interval_sec
        bars_count = min_days_check * (96 if timeframe == "15m" else 1)

        try:
            async with pool.acquire(timeout=_ACQUIRE_TIMEOUT) as conn:
                vrows = await conn.fetch(
                    f'SELECT low, volume FROM "{table_name}" '
                    f'WHERE "Timestamp" <= $1 '
                    f'ORDER BY "Timestamp" DESC LIMIT {bars_count}',
                    closed_cutoff,
                )
            if len(vrows) >= bars_count:
                vols = [
                    (float(r["volume"]) if r["volume"] is not None else 0.0)
                    * (float(r["low"]) if r["low"] is not None else 0.0)
                    for r in vrows
                ]
                min_vol = min(vols) if vols else 0.0
                target_db = self.high_db if min_vol >= hard_floor_usd else self.low_db
                if target_db != current_db:
                    logger.info(
                        f"    🔄 MOVE {symbol} -> {'HIGH' if target_db == self.high_db else 'LOW'} (${min_vol:,.0f})"
                    )
                    await self.move_table(table_name, current_db, target_db)
                    return target_db
        except Exception as e:
            logger.debug(f"Volume move check error for {symbol}: {e}")

        return current_db
