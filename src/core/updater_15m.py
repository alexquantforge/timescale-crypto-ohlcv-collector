"""
OHLCV Updater — 15 MINUTE — ONLY BYBIT/GATEIO/MEXC/OKX/BINGX
With 3x load_markets retries for maximum Gate.io connection stability.
"""

import asyncio
import datetime
import time
import re
import sys
import logging
import warnings
from typing import Optional, Dict, Any, List, Set, Tuple

import asyncpg
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async

from src.exchanges.symbol_selector import get_exchange_url, get_swap_url
from pytz import timezone as pytz_timezone

from config.settings import settings

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger("updater_15m")


def log(msg: str):
    logger.info(msg)


DB_HIGH = settings.db_high_15m
DB_LOW = settings.db_low_15m

ALLOWED_EXCHANGES = {"bybit", "gateio", "mexc", "okx", "bingx"}

EXCHANGE_MAP = {
    "bybit": "bybit",
    "gateio": "gate",
    "mexc": "mexc",
    "okx": "okx",
    "bingx": "bingx",
}

DELETE_NOT_ALLOWED_EXCHANGES_ON_START = True
HARD_FLOOR_USD = 125000
MIN_INTERVALS_VOLUME_CHECK = 672  # 7 days = 672 15-minute bars
CONCURRENT_PER_EXCHANGE = settings.concurrent_per_exchange
MSK_TZ = pytz_timezone("Europe/Moscow")
ALMATY_TZ = pytz_timezone("Asia/Almaty")
UPDATE_INTERVAL_SECONDS = 300  # 5 minutes

FETCH_LIMIT = 1000
MAX_PAGES = 10

DATA_RETENTION_DAYS = 180
SKIP_DOWNLOAD_OLDER_DAYS = 180


def get_cutoff_timestamp() -> int:
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=DATA_RETENTION_DAYS
    )
    return int(cutoff.timestamp())


def get_download_min_timestamp() -> int:
    min_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=SKIP_DOWNLOAD_OLDER_DAYS
    )
    return int(min_date.timestamp())


GAP_TOLERANCE_SEC = 900 * 2
GAP_MAX_LOOKBACK_DAYS = DATA_RETENTION_DAYS
GAP_FETCH_DELAY = 0.1

SKIP_PATTERNS = re.compile(
    r"(3L|3S|5L|5S|2L|2S|4L|4S|UP|DOWN|BULL|BEAR)/USDT$", re.IGNORECASE
)

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
    "ob_gerchik_atr": "DOUBLE PRECISION",
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

COLLECT_ORDERBOOK: bool = True
DEBUG_ORDERBOOK: bool = False

OB_FETCH_LIMIT: int = 50
OB_TRADES_LIMIT: int = 100
OB_TRADES_WINDOW_SEC: int = 300
OB_DEPTH_PCT: float = 1.0
OB_FALLBACK_LIMITS: List[int] = [20, 10, 5]

GC_ATR_PERIOD: int = 5
GC_BAR_SMALL_THRESHOLD: float = 0.5
GC_BAR_LARGE_THRESHOLD: float = 1.8

db_pools: Dict[str, asyncpg.Pool] = {}

PER_PAGE_LIMIT: Dict[str, int] = {
    "bitget": 200,
    "coinex": 1000,
    "htx": 2000,
}
DEFAULT_PER_PAGE = 1000


def should_skip_pair(symbol: str, exchange: str = "") -> bool:
    if not symbol:
        return True

    parts = symbol.split("/")
    base = parts[0].upper() if len(parts) > 0 else ""

    if base and "USD" in base and base != "USDT":
        return True

    if SKIP_PATTERNS.search(symbol):
        return True

    if exchange == "bitget" and base.startswith("R"):
        crypto_exceptions = {"RARE", "RAY", "RAMP", "RAU", "RAVE", "RNDR", "RSR", "RUNE", "RVN", "ROSE", "REQ"}
        if base not in crypto_exceptions:
            return True

    return False


async def init_db_pools():
    for db in [DB_HIGH, DB_LOW]:
        db_pools[db] = await asyncpg.create_pool(
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            database=db,
            min_size=2,
            max_size=10,
        )
    log("✓ DB pools initialized for 15M")


async def find_table_in_dbs(table_name: str) -> Tuple[Optional[str], int]:
    for db in [DB_HIGH, DB_LOW]:
        if db not in db_pools:
            continue
        async with db_pools[db].acquire() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=$1)",
                table_name,
            )
            if exists:
                last_ts = await conn.fetchval(
                    f'SELECT MAX("Timestamp") FROM "{table_name}"'
                )
                return db, int(last_ts) if last_ts else 1  # 1 indicates table exists even if empty
    return None, 0


async def move_table(table_name, from_db, to_db):
    if from_db not in db_pools or to_db not in db_pools:
        return
    async with db_pools[from_db].acquire() as fc:
        rows = await fc.fetch(f'SELECT * FROM "{table_name}"')
        if not rows:
            return
    async with db_pools[to_db].acquire() as tc:
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
    async with db_pools[from_db].acquire() as fc2:
        await fc2.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')


async def delete_old_data_from_db(pool: asyncpg.Pool, db_name: str) -> int:
    cutoff_ts = get_cutoff_timestamp()
    cutoff_date = datetime.datetime.now(
        datetime.timezone.utc
    ) - datetime.timedelta(days=DATA_RETENTION_DAYS)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")

    logger.info(
        f"🗑️  [{db_name}] Deleting data older than {DATA_RETENTION_DAYS} days (before {cutoff_str} UTC)..."
    )

    total_deleted = 0
    async with pool.acquire() as conn:
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name LIKE '%_on_%'"
        )

        for r in tables:
            tbl = r["table_name"]
            try:
                deleted = await conn.execute(
                    f'DELETE FROM "{tbl}" WHERE "Timestamp" < $1',
                    cutoff_ts,
                )
                if deleted and deleted != "DELETE 0":
                    try:
                        deleted_count = int(deleted.split()[1])
                        total_deleted += deleted_count
                    except (IndexError, ValueError):
                        pass
            except asyncpg.PostgresError as e:
                logger.warning(f"  ⚠️  [{db_name}] {tbl}: error during delete — {e}")

    logger.info(f"  ✅ [{db_name}] Deletion finished: {total_deleted} old records removed")
    return total_deleted


async def run_maintenance() -> None:
    logger.info("=" * 60)
    logger.info("🔧 RUNNING DATABASE MAINTENANCE (15M)")
    logger.info("=" * 60)

    total_deleted = 0
    for db_name, pool in db_pools.items():
        try:
            deleted = await delete_old_data_from_db(pool, db_name)
            total_deleted += deleted
        except Exception as e:
            logger.warning(f"  ⚠️  [{db_name}] Error deleting old data: {e}")

    for db_name, pool in db_pools.items():
        try:
            async with pool.acquire() as conn:
                await conn.execute("VACUUM;")
                logger.info(f"  ✅ [{db_name}] VACUUM completed")
        except Exception as e:
            logger.warning(f"  ⚠️  [{db_name}] VACUUM skipped: {e}")

    logger.info(f"✅ Maintenance finished: total deleted {total_deleted} records")
    logger.info("=" * 60)


def get_exchange_from_table_name(table_name: str) -> str:
    try:
        t = str(table_name or "").strip().lower()
        if "_on_" not in t:
            return ""
        return t.rsplit("_on_", 1)[-1].strip().lower()
    except Exception:
        return ""


async def drop_not_allowed_exchange_tables() -> None:
    if not DELETE_NOT_ALLOWED_EXCHANGES_ON_START:
        return

    logger.info("=" * 60)
    logger.info("🧹 DROPPING TABLES FOR NON-ALLOWED EXCHANGES (15M)")
    logger.info(f"   Allowed exchanges: {', '.join(sorted(ALLOWED_EXCHANGES))}")
    logger.info("=" * 60)

    total_dropped = 0
    for db_name, pool in db_pools.items():
        dropped = 0
        kept = 0
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name LIKE '%_on_%' "
                "ORDER BY table_name"
            )
            for r in rows:
                tbl = r["table_name"]
                exch = get_exchange_from_table_name(tbl)
                if not exch or exch in ALLOWED_EXCHANGES:
                    kept += 1
                    continue
                try:
                    await conn.execute(f'DROP TABLE IF EXISTS "{tbl}" CASCADE')
                    dropped += 1
                    total_dropped += 1
                except Exception as e:
                    logger.warning(f"  ⚠️  [{db_name}] Failed dropping {tbl}: {e}")
        logger.info(
            f"  ✅ [{db_name}] kept tables: {kept}, dropped tables: {dropped}"
        )

    logger.info(f"✅ Non-allowed exchange tables cleanup finished. Dropped: {total_dropped}")
    logger.info("=" * 60)


def compute_gerchik_atr(highs, lows, closes) -> float:
    try:
        H = np.asarray(highs, dtype=float)
        L = np.asarray(lows, dtype=float)
        C = np.asarray(closes, dtype=float)
    except Exception:
        return 0.0
    n = len(C)
    if n < 3:
        return 0.0
    prev_c = np.roll(C, 1)
    prev_c[0] = C[0]
    tr = np.maximum(H - L, np.maximum(np.abs(H - prev_c), np.abs(L - prev_c)))
    tr = np.where(np.isfinite(tr), tr, 0.0)
    window_tr = tr[max(0, n - GC_ATR_PERIOD) : n]
    if len(window_tr) == 0:
        return 0.0
    curr = float(np.mean(window_tr))
    if not np.isfinite(curr) or curr <= 0:
        return 0.0
    for _ in range(10):
        valid = window_tr[
            (window_tr >= GC_BAR_SMALL_THRESHOLD * curr)
            & (window_tr <= GC_BAR_LARGE_THRESHOLD * curr)
        ]
        if len(valid) == 0:
            break
        new_atr = float(np.mean(valid))
        if not np.isfinite(new_atr) or new_atr <= 0:
            break
        if abs(new_atr - curr) / max(abs(curr), 1e-12) < 0.01:
            curr = new_atr
            break
        curr = new_atr
    return max(float(curr), 0.0)


def _vitality_grade_from_score(score: float) -> str:
    if score >= 8:
        return "A"
    if score >= 6:
        return "B"
    if score >= 4:
        return "C"
    if score >= 2:
        return "D"
    return "F"


async def fetch_orderbook_snapshot(
    exchange, symbol: str, gerchik_atr: float
) -> Optional[Dict[str, Any]]:
    ex_id = getattr(exchange, "id", "unknown")
    limit = OB_FETCH_LIMIT

    ob = None
    for l in [limit] + [fb for fb in OB_FALLBACK_LIMITS if fb != limit]:
        try:
            ob = await asyncio.wait_for(
                exchange.fetch_order_book(symbol, limit=l), timeout=6.0
            )
            if ob:
                break
        except Exception:
            continue
    if not ob:
        return None
    bids, asks = ob.get("bids", []) or [], ob.get("asks", []) or []
    if not bids or not asks:
        return None
    try:
        best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
    except Exception:
        return None
    if best_bid <= 0 or best_ask <= 0:
        return None

    mid = (best_bid + best_ask) / 2.0
    spread_abs = best_ask - best_bid
    spread_pct = (spread_abs / best_bid) * 100.0 if best_bid > 0 else 0.0
    spread_atr_pct = (
        (spread_abs / gerchik_atr) * 100.0
        if gerchik_atr and gerchik_atr > 0
        else 0.0
    )

    lo, hi = mid * (1 - OB_DEPTH_PCT / 100.0), mid * (1 + OB_DEPTH_PCT / 100.0)
    bid_depth = ask_depth = 0.0
    for entry in bids:
        if len(entry) < 2:
            continue
        p, a = float(entry[0]), float(entry[1])
        if p >= lo:
            bid_depth += p * a
    for entry in asks:
        if len(entry) < 2:
            continue
        p, a = float(entry[0]), float(entry[1])
        if p <= hi:
            ask_depth += p * a
    total_depth = bid_depth + ask_depth
    imbalance = round(bid_depth / ask_depth, 4) if ask_depth > 0 else 99.0

    tpm, last_sec, buy_pct = 0.0, 999.0, 50.0
    cvd = cvd_5m = 0.0
    trades = []
    try:
        trades = (
            await asyncio.wait_for(
                exchange.fetch_trades(symbol, limit=OB_TRADES_LIMIT), timeout=6.0
            )
            or []
        )
    except Exception:
        trades = []
    if trades:
        now_ms = time.time() * 1000
        for t in trades:
            price = float(t.get("price", 0) or 0)
            amt = float(t.get("amount", 0) or 0)
            usd = price * amt
            side = t.get("side")
            signed = usd if side == "buy" else (-usd if side == "sell" else 0.0)
            cvd += signed
            ts = t.get("timestamp") or 0
            if ts and ts >= now_ms - OB_TRADES_WINDOW_SEC * 1000:
                cvd_5m += signed
        recent = [
            t
            for t in trades
            if t.get("timestamp") and t["timestamp"] >= now_ms - OB_TRADES_WINDOW_SEC * 1000
        ]
        if recent:
            tpm = len(recent) / (OB_TRADES_WINDOW_SEC / 60.0)
            buys = sum(1 for t in recent if t.get("side") == "buy")
            buy_pct = buys / len(recent) * 100.0
        valid_ts = [t.get("timestamp", 0) or 0 for t in trades if t.get("timestamp")]
        if valid_ts:
            last_sec = (now_ms - max(valid_ts)) / 1000.0

    is_barcode = False
    try:
        prices = [float(t["price"]) for t in trades if t.get("price")]
        if len(trades) >= 30 and len(set(prices)) <= 4:
            is_barcode = True
    except Exception:
        is_barcode = False

    score = 0
    if not is_barcode:
        if tpm >= OB_TRADES_MIN_BLAZING:
            score += 4
        elif tpm >= OB_TRADES_MIN_GOOD:
            score += 3
        elif tpm >= OB_TRADES_MIN_OK:
            score += 2
        elif tpm >= OB_TRADES_MIN_SLOW:
            score += 1
        if total_depth >= OB_DEPTH_MIN_GOOD:
            score += 3
        elif total_depth >= OB_DEPTH_MIN_OK:
            score += 2
        elif total_depth >= OB_DEPTH_MIN_THIN:
            score += 1
        if spread_pct < 0.1:
            score += 3
        elif spread_pct < 0.3:
            score += 2
        elif spread_pct < 1.0:
            score += 1
    score = max(0, min(10, score))
    grade = _vitality_grade_from_score(score) if not is_barcode else "F"

    return {
        "ob_last_trade_sec": round(last_sec, 1),
        "ob_trades_per_min": round(tpm, 2),
        "ob_buy_pressure_pct": round(buy_pct, 1),
        "ob_cvd": round(cvd, 2),
        "ob_cvd_5m": round(cvd_5m, 2),
        "ob_spread_abs": spread_abs,
        "ob_spread_pct": round(spread_pct, 4),
        "ob_spread_atr_pct": round(spread_atr_pct, 4),
        "ob_gerchik_atr": round(float(gerchik_atr or 0.0), 10),
        "ob_best_bid": best_bid,
        "ob_best_ask": best_ask,
        "ob_bid_depth_usd": round(bid_depth, 2),
        "ob_ask_depth_usd": round(ask_depth, 2),
        "ob_total_depth_usd": round(total_depth, 2),
        "ob_imbalance": imbalance,
        "ob_vitality_score": float(score),
        "ob_vitality_grade": grade,
        "ob_is_barcode": bool(is_barcode),
    }


async def ensure_orderbook_columns(conn, tbl: str) -> None:
    existing = {
        r["column_name"]
        for r in await conn.fetch(
            "SELECT column_name FROM information_schema.columns WHERE table_name=$1",
            tbl,
        )
    }
    for col, typ in ORDERBOOK_COLUMNS_SQL.items():
        if col not in existing:
            await conn.execute(f'ALTER TABLE "{tbl}" ADD COLUMN "{col}" {typ}')


async def save_orderbook_snapshot(
    db, tbl: str, full_df, snap: Optional[Dict[str, Any]]
) -> None:
    async with db_pools[db].acquire() as conn:
        await ensure_orderbook_columns(conn, tbl)

        min_7d = None
        try:
            closed_cutoff = int(time.time()) - 900
            closed = full_df[full_df["Timestamp"] <= closed_cutoff]
            tail = closed.tail(MIN_INTERVALS_VOLUME_CHECK)
            if len(tail) >= 1:
                min_7d = float((tail["volume"] * tail["low"]).min())
        except Exception:
            min_7d = None

        last_ts = await conn.fetchval(f'SELECT MAX("Timestamp") FROM "{tbl}"')
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
        sql = f'UPDATE "{tbl}" SET {", ".join(set_parts)} WHERE "Timestamp" = ${idx}'
        await conn.execute(sql, *values)


async def check_and_fill_table_gaps(
    exchange, symbol: str, tbl: str, db: str
) -> Tuple[int, int, int]:
    download_min_ts = get_download_min_timestamp()

    async with db_pools[db].acquire() as conn:
        rows = await conn.fetch(
            f'SELECT "Timestamp" FROM "{tbl}" WHERE "Timestamp" >= $1 ORDER BY "Timestamp" ASC',
            download_min_ts,
        )

    if len(rows) < 2:
        return (len(rows), 0, 0)

    timestamps = [int(r["Timestamp"]) for r in rows]

    gaps: List[Tuple[int, int]] = []
    for i in range(len(timestamps) - 1):
        diff = timestamps[i + 1] - timestamps[i]
        if diff > GAP_TOLERANCE_SEC:
            gaps.append((timestamps[i], timestamps[i + 1]))

    if not gaps:
        return (len(timestamps), 0, 0)

    gaps_filled = 0

    for gap_start, gap_end in gaps:
        gap_periods = (gap_end - gap_start) // 900

        strategies = [
            gap_start + 900,
            (gap_start + gap_end) // 2,
            max(gap_start + 900, gap_end - 1000 * 900),
        ]
        unique_since = list(dict.fromkeys(strategies))

        all_fill: List[list] = []
        seen_fill_ts: set = set()

        for start_ts in unique_since:
            if len(seen_fill_ts) >= gap_periods - 1:
                break

            cursor_ms = start_ts * 1000
            pages = 0
            while cursor_ms < gap_end * 1000 and pages < 20:
                try:
                    cs = await asyncio.wait_for(
                        exchange.fetch_ohlcv(
                            symbol, "15m", since=cursor_ms, limit=FETCH_LIMIT
                        ),
                        timeout=6.0,
                    )
                except Exception:
                    break
                if not cs:
                    break
                for c in cs:
                    ts_sec = c[0] // 1000
                    if gap_start < ts_sec < gap_end and ts_sec not in seen_fill_ts:
                        all_fill.append(c)
                        seen_fill_ts.add(ts_sec)
                    elif ts_sec >= gap_end:
                        break
                if cs[-1][0] >= gap_end * 1000:
                    break
                cursor_ms = cs[-1][0] + 900 * 1000
                pages += 1
                await asyncio.sleep(GAP_FETCH_DELAY)

        if not all_fill:
            continue

        df = pd.DataFrame(
            all_fill, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        df["Timestamp"] = df["ts"] // 1000
        df["ticker"], df["exchange"] = symbol, tbl.rsplit("_on_", 1)[-1]
        df["volume_x_low"] = df["volume"] * df["low"]
        df["volume_x_close"] = df["volume"] * df["close"]
        df["asset_type"] = "swap" if ":" in symbol else "spot"
        gap_ccxt_id = tbl.rsplit("_on_", 1)[-1]
        gap_spot_url = get_exchange_url(gap_ccxt_id, symbol)
        gap_swap_url = get_swap_url(gap_ccxt_id, symbol)
        df["url_of_trading_pair"] = gap_swap_url if ":" in symbol else gap_spot_url
        df["url_of_swap_contract_if_it_exists"] = None if ":" in symbol else gap_swap_url
        dt_utc = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
        df["open_time_msk"] = dt_utc.dt.tz_convert(MSK_TZ).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        df["open_time_almaty"] = dt_utc.dt.tz_convert(ALMATY_TZ).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        actual_cols = list(ALL_COLUMNS_SQL.keys())
        tuples = [
            tuple(None if pd.isna(x) else x for x in row)
            for row in df[actual_cols].to_numpy()
        ]

        try:
            async with db_pools[db].acquire() as conn:
                await conn.copy_records_to_table(
                    tbl, records=tuples, columns=actual_cols
                )
            gaps_filled += 1
        except Exception:
            pass

    return (len(timestamps), len(gaps), gaps_filled)


async def create_empty_symbol_table(db_name: str, tbl_name: str) -> None:
    """Creates empty table for 0-candle symbols so find_table_in_dbs finds it next time."""
    pool = db_pools.get(db_name)
    if not pool:
        return
    try:
        async with pool.acquire() as conn:
            cols = [f'"{c}" {t}' for c, t in ALL_COLUMNS_SQL.items()]
            await conn.execute(
                f'CREATE TABLE IF NOT EXISTS "{tbl_name}" ({", ".join(cols)})'
            )
            try:
                await conn.execute(
                    f"SELECT create_hypertable('{tbl_name}', 'Timestamp', if_not_exists => TRUE)"
                )
            except Exception:
                pass
    except Exception:
        pass


async def process_pair(exchange, symbol, ccxt_id):
    tbl = f"{symbol.replace('/', '_').replace('-', '_')}_on_{ccxt_id}".lower()
    current_db, last_ts = await find_table_in_dbs(tbl)

    download_min_ts = get_download_min_timestamp()

    try:
        all_candles: List[list] = []

        if last_ts > 0:
            if last_ts * 1000 < download_min_ts * 1000:
                last_ts = download_min_ts

            per_page = PER_PAGE_LIMIT.get(ccxt_id, DEFAULT_PER_PAGE)
            cursor_ms = last_ts * 1000
            pages = 0
            while pages < 20:
                try:
                    cs = await asyncio.wait_for(
                        exchange.fetch_ohlcv(
                            symbol, "15m", since=cursor_ms, limit=FETCH_LIMIT
                        ),
                        timeout=6.0,
                    )
                except Exception:
                    break
                if not cs:
                    break
                all_candles.extend(cs)
                if len(cs) < per_page:
                    break
                cursor_ms = cs[-1][0] + 900 * 1000
                pages += 1
                await asyncio.sleep(0.05)
            if not all_candles:
                return 0, 0, 0, 0
        else:
            # Auto-discover new symbol and download initial history
            log(f"  [15M] 🚀 New symbol detected: {symbol} ({ccxt_id}) -> Downloading initial history...")
            per_page = PER_PAGE_LIMIT.get(ccxt_id, DEFAULT_PER_PAGE)
            start_since = download_min_ts * 1000

            try:
                cs = await asyncio.wait_for(
                    exchange.fetch_ohlcv(
                        symbol, "15m", since=start_since, limit=FETCH_LIMIT
                    ),
                    timeout=6.0,
                )
            except Exception as e:
                cs = []

            if cs:
                all_candles.extend(cs)
                seen_ts = {c[0] for c in cs}
                oldest_ts = cs[0][0]

                page = 1
                while page < MAX_PAGES:
                    since_ms = oldest_ts - per_page * 900 * 1000
                    if since_ms < start_since:
                        since_ms = start_since

                    try:
                        prev_cs = await asyncio.wait_for(
                            exchange.fetch_ohlcv(
                                symbol, "15m", since=since_ms, limit=FETCH_LIMIT
                            ),
                            timeout=6.0,
                        )
                    except Exception:
                        break

                    if not prev_cs:
                        break

                    new_prev_cs = [c for c in prev_cs if c[0] not in seen_ts]
                    if not new_prev_cs:
                        break

                    all_candles.extend(new_prev_cs)
                    for c in new_prev_cs:
                        seen_ts.add(c[0])

                    new_prev_cs.sort(key=lambda x: x[0])
                    oldest_ts = new_prev_cs[0][0]

                    if since_ms <= start_since:
                        break

                    page += 1
                    await asyncio.sleep(0.05)

                all_candles.sort(key=lambda x: x[0])

        if not all_candles:
            # Create empty table for 0-candle symbol so find_table_in_dbs finds it next time
            if last_ts == 0:
                await create_empty_symbol_table(DB_LOW, tbl)
            return 0, 0, 0, 0

        cs = all_candles

        df = pd.DataFrame(cs, columns=["ts", "open", "high", "low", "close", "volume"])
        df["Timestamp"] = df["ts"] // 1000
        df["ticker"], df["exchange"] = symbol, ccxt_id
        df["volume_x_low"] = df["volume"] * df["low"]
        df["volume_x_close"] = df["volume"] * df["close"]
        df["asset_type"] = "swap" if ":" in symbol else "spot"

        dt_utc = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
        df["open_time_msk"] = dt_utc.dt.tz_convert(MSK_TZ).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        df["open_time_almaty"] = dt_utc.dt.tz_convert(ALMATY_TZ).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        if not current_db:
            current_db = DB_LOW
            async with db_pools[current_db].acquire() as conn:
                cols = [f'"{c}" {t}' for c, t in ALL_COLUMNS_SQL.items()]
                await conn.execute(f'CREATE TABLE "{tbl}" ({", ".join(cols)})')
                try:
                    await conn.execute(
                        f"SELECT create_hypertable('{tbl}', 'Timestamp', if_not_exists => TRUE)"
                    )
                except Exception as e:
                    logger.warning(
                        f"  [TIMESCALEDB] Failed creating hypertable for {tbl}: {e}"
                    )

        async with db_pools[current_db].acquire() as conn:
            actual_cols = list(ALL_COLUMNS_SQL.keys())
            existing_cols = {
                r["column_name"]
                for r in await conn.fetch(
                    "SELECT column_name FROM information_schema.columns WHERE table_name=$1",
                    tbl,
                )
            }
            for col in actual_cols:
                if col not in existing_cols:
                    await conn.execute(
                        f'ALTER TABLE "{tbl}" ADD COLUMN "{col}" {ALL_COLUMNS_SQL[col]}'
                    )

            tuples = [
                tuple(None if pd.isna(x) else x for x in row)
                for row in df[actual_cols].to_numpy()
            ]

            min_new_ts = int(df["Timestamp"].min())
            async with conn.transaction():
                await conn.execute(
                    f'DELETE FROM "{tbl}" WHERE "Timestamp" >= $1', min_new_ts
                )
                await conn.copy_records_to_table(
                    tbl, records=tuples, columns=actual_cols
                )

                await conn.execute(
                    f"""
                    WITH dups AS (
                        SELECT "Timestamp",
                               row_number() OVER (
                                   PARTITION BY ("Timestamp" / 900)
                                   ORDER BY COALESCE(volume, 0) DESC, "Timestamp" DESC
                               ) AS rn
                        FROM "{tbl}"
                    )
                    DELETE FROM "{tbl}" 
                    WHERE "Timestamp" IN (
                        SELECT "Timestamp" FROM dups WHERE rn > 1
                    )
                """
                )

        total_bars, gaps_found, gaps_filled = await check_and_fill_table_gaps(
            exchange, symbol, tbl, current_db
        )

        try:
            closed_cutoff = int(time.time()) - 900
            async with db_pools[current_db].acquire() as vconn:
                vrows = await vconn.fetch(
                    f'SELECT low, volume FROM "{tbl}" '
                    f'WHERE "Timestamp" <= $1 '
                    f'ORDER BY "Timestamp" DESC LIMIT {MIN_INTERVALS_VOLUME_CHECK}',
                    closed_cutoff,
                )
            if len(vrows) >= MIN_INTERVALS_VOLUME_CHECK:
                vols = [
                    (float(r["volume"]) if r["volume"] is not None else 0.0)
                    * (float(r["low"]) if r["low"] is not None else 0.0)
                    for r in vrows
                ]
                min_vol = min(vols) if vols else 0.0
                target_db = DB_HIGH if min_vol >= HARD_FLOOR_USD else DB_LOW
                if target_db != current_db:
                    log(
                        f"    🔄 MOVE {symbol} -> {'HIGH' if target_db == DB_HIGH else 'LOW'} (${min_vol:,.0f})"
                    )
                    await move_table(tbl, current_db, target_db)
                    current_db = target_db
        except Exception as e_mv:
            if DEBUG_ORDERBOOK:
                log(f"    ⚠️ [MOVE] {symbol}: volume check error: {e_mv}")

        if COLLECT_ORDERBOOK:
            try:
                async with db_pools[current_db].acquire() as rconn:
                    hist = await rconn.fetch(
                        f'SELECT "Timestamp", high, low, close, volume FROM "{tbl}" '
                        f'WHERE "Timestamp" >= $1 '
                        f'ORDER BY "Timestamp" ASC',
                        download_min_ts,
                    )
                if hist:
                    full_df = pd.DataFrame(
                        hist, columns=["Timestamp", "high", "low", "close", "volume"]
                    )
                    g_atr = compute_gerchik_atr(
                        full_df["high"].to_numpy(dtype=float),
                        full_df["low"].to_numpy(dtype=float),
                        full_df["close"].to_numpy(dtype=float),
                    )
                    snap = await fetch_orderbook_snapshot(exchange, symbol, g_atr)
                    await save_orderbook_snapshot(current_db, tbl, full_df, snap)
            except Exception as e_ob:
                if DEBUG_ORDERBOOK:
                    log(f"    ⚠️ [OB] {symbol}: orderbook snapshot error: {e_ob}")

        return len(cs), total_bars, gaps_found, gaps_filled
    except (ccxt.BadSymbol, ccxt.SymbolNotFound) as e:
        if current_db:
            async with db_pools[current_db].acquire() as cconn:
                await cconn.execute(f'DROP TABLE IF EXISTS "{tbl}" CASCADE')
            log(f"  🗑️ [DROP DELISTED] Dropped table {tbl} from {current_db}")
        return 0, 0, 0, 0
    except ccxt.ExchangeError as e:
        err_msg = str(e).lower()
        if any(
            term in err_msg
            for term in [
                "symbol is not found",
                "invalid symbol",
                "symbol_not_found",
                "100204",
                "48001",
            ]
        ):
            if current_db:
                async with db_pools[current_db].acquire() as cconn:
                    await cconn.execute(f'DROP TABLE IF EXISTS "{tbl}" CASCADE')
                log(f"  🗑️ [DROP DELISTED] Dropped table {tbl} from {current_db}")
        return 0, 0, 0, 0
    except Exception as e:
        return 0, 0, 0, 0


PROGRESS_LOG_EVERY: int = 5
PRECOUNT_PAIRS: bool = True


def _fmt_eta(seconds: float) -> str:
    try:
        s = int(max(0, round(seconds)))
    except Exception:
        return "?"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


class GlobalProgress:
    def __init__(self):
        self.total = 0
        self.done = 0
        self.start_ts = time.time()
        self.total_prefilled = False

    def reset(self, total: int = 0):
        self.total = total
        self.done = 0
        self.start_ts = time.time()
        self.total_prefilled = total > 0

    def start_timing(self):
        self.start_ts = time.time()

    def add_to_total(self, n: int):
        if self.total_prefilled:
            return
        self.total += int(n)

    def subtract_from_total(self, n: int):
        self.total = max(self.done, self.total - int(n))

    def tick(self) -> Tuple[int, int, str, float]:
        self.done += 1
        done = self.done
        total = max(self.total, done)
        elapsed = time.time() - self.start_ts
        if done > 5 and total > done and elapsed > 0:
            rate = done / elapsed
            remaining = (total - done) / rate if rate > 0 else 0
            eta_str = _fmt_eta(remaining)
        elif done >= total:
            eta_str = "0:00"
        else:
            eta_str = "?"
        pct = (done / total * 100.0) if total > 0 else 0.0
        return done, total, eta_str, pct


GLOBAL_PROGRESS: Optional["GlobalProgress"] = None


async def close_exchange_safely(exchange, name: str = "") -> None:
    sps = getattr(exchange, "socks_proxy_sessions", None)
    if sps:
        for url in list(sps):
            try:
                await sps[url].close()
            except Exception:
                pass
        exchange.socks_proxy_sessions = None

    for attr in ("connector", "tcp_connector", "aiohttp_socks_connector"):
        conn = getattr(exchange, attr, None)
        if conn is not None:
            try:
                if not conn.closed:
                    await conn.close()
            except Exception:
                pass
            setattr(exchange, attr, None)

    session = getattr(exchange, "session", None)
    if session is not None:
        try:
            if not session.closed:
                await session.close()
        except Exception:
            pass
        exchange.session = None

    try:
        await exchange.close()
    except Exception:
        pass

    exchange.session = None
    exchange.socks_proxy_sessions = None
    for attr in ("connector", "tcp_connector", "aiohttp_socks_connector"):
        setattr(exchange, attr, None)

    await asyncio.sleep(0.05)


def select_symbols_for_exchange(symbols, markets, exchange_name="") -> List[str]:
    spots = {}
    swaps = {}

    for symbol in symbols:
        if symbol not in markets:
            continue
        market = markets[symbol]

        if should_skip_pair(symbol, exchange_name):
            continue

        base = market.get("base")
        if not base:
            continue
        base = base.upper()

        if market.get("spot") and symbol.endswith("/USDT"):
            spots[base] = symbol
        elif market.get("swap") and symbol.endswith("/USDT:USDT"):
            swaps[base] = symbol

    selected_symbols = []
    all_bases = set(spots.keys()) | set(swaps.keys())

    for base in sorted(all_bases):
        if base in swaps:
            selected_symbols.append(swaps[base])
        elif base in spots:
            selected_symbols.append(spots[base])

    return selected_symbols


def _make_exchange(ccxt_id: str):
    config = {
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {},
    }
    if settings.socks5_proxy:
        config["socks_proxy"] = settings.socks5_proxy
    return getattr(ccxt_async, ccxt_id)(config)


async def count_pairs_for_exchange(ccxt_name: str) -> int:
    ccxt_id = EXCHANGE_MAP.get(ccxt_name, ccxt_name)
    exchange = _make_exchange(ccxt_id)
    try:
        await asyncio.wait_for(exchange.load_markets(), timeout=12.0)
        syms = select_symbols_for_exchange(
            exchange.symbols, exchange.markets, ccxt_name
        )
        return len(syms)
    except Exception as e:
        log(f"  [PRECOUNT] {ccxt_name}: failed counting pairs: {e}")
        return 0
    finally:
        await close_exchange_safely(exchange, ccxt_name)


async def process_exchange(ccxt_name):
    ccxt_id = EXCHANGE_MAP.get(ccxt_name, ccxt_name)
    log(f"--- Exchange: {ccxt_name} ---")

    exchange = _make_exchange(ccxt_id)

    try:
        try:
            await asyncio.wait_for(exchange.load_markets(), timeout=12.0)
        except Exception as e:
            log(f"  ⚠️ Failed loading markets for {ccxt_name}: {e}")
            if GLOBAL_PROGRESS is not None:
                GLOBAL_PROGRESS.subtract_from_total(2000)
            return

        syms = select_symbols_for_exchange(
            exchange.symbols, exchange.markets, ccxt_name
        )
        total = len(syms)
        sem = asyncio.Semaphore(CONCURRENT_PER_EXCHANGE)

        if GLOBAL_PROGRESS is not None:
            GLOBAL_PROGRESS.add_to_total(total)

        processed = 0

        async def worker(s):
            nonlocal processed
            async with sem:
                count, total_bars, gaps_found, gaps_filled = await process_pair(
                    exchange, s, ccxt_name
                )
                processed += 1
                total_days = round(total_bars / 96, 1)
                if gaps_found > 0:
                    gap_msg = f"⚠️ gaps: {gaps_found}, filled: {gaps_filled}"
                else:
                    gap_msg = "no gaps. OK"

                if GLOBAL_PROGRESS is not None:
                    g_done, g_total, eta_str, pct = GLOBAL_PROGRESS.tick()
                    if count > 0 or g_done % PROGRESS_LOG_EVERY == 0:
                        log(
                            f"  [15M] [ALL {g_done}/{g_total} · {pct:.1f}% · ETA {eta_str}] "
                            f"[{ccxt_name} {processed}/{total}] {s}: +{count} candles, "
                            f"{total_days}d stored, {gap_msg}"
                        )
                else:
                    if count > 0 or processed % 10 == 0:
                        log(
                            f"  [15M] [{ccxt_name}] {processed}/{total} | {s}: +{count} candles, "
                            f"{total_days}d stored, {gap_msg}"
                        )

        await asyncio.gather(*[worker(s) for s in syms])
    finally:
        await close_exchange_safely(exchange, ccxt_name)


async def main_15m_loop():
    await init_db_pools()
    await drop_not_allowed_exchange_tables()

    global GLOBAL_PROGRESS
    while True:
        GLOBAL_PROGRESS = GlobalProgress()

        if PRECOUNT_PAIRS:
            log("Pre-counting 15M trading pairs...")
            counts = await asyncio.gather(
                *[count_pairs_for_exchange(eid) for eid in EXCHANGE_MAP.keys()],
                return_exceptions=True,
            )
            grand_total = sum(c for c in counts if isinstance(c, int))
            GLOBAL_PROGRESS.reset(grand_total)
            log(f"Total 15M symbols to process (5 exchanges): {grand_total}")
        else:
            GLOBAL_PROGRESS.reset(0)

        if GLOBAL_PROGRESS:
            GLOBAL_PROGRESS.start_timing()

        cycle_start = time.time()

        async def _safe_process_exchange(eid: str):
            try:
                await process_exchange(eid)
            except Exception as e:
                log(f"  [CRITICAL] {eid} failed: {e}")

        await asyncio.gather(
            *[_safe_process_exchange(eid) for eid in EXCHANGE_MAP.keys()]
        )

        elapsed = time.time() - cycle_start
        log(
            f"15M Cycle complete. Processed {GLOBAL_PROGRESS.done}/{GLOBAL_PROGRESS.total} "
            f"symbols in {_fmt_eta(elapsed)}."
        )

        await run_maintenance()

        log(f"Sleeping {UPDATE_INTERVAL_SECONDS // 60} minutes until next 15M cycle.")
        await asyncio.sleep(UPDATE_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(main_15m_loop())
    except KeyboardInterrupt:
        sys.exit(0)
