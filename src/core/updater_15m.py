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
import ccxt  # FIX: ccxt.BadSymbol / ccxt.ExchangeError are referenced in exception handlers
import ccxt.async_support as ccxt_async

from src.core.history_prefill import (
    extract_older_rows,
    is_transient_fetch_error,
    normalize_epoch_sec,
    prefill_empty_action,
    prefill_needed,
    prefill_page_since_ms,
    should_attempt_prefill,
)
from src.core.oi_funding import (
    backfill_funding_history,
    fetch_oi_funding_snapshot,
    warn_once as oi_funding_warn_once,
    write_oi_funding_snapshot,
)
from src.db.repository import (
    fetch_column_types,
    pg_ddl_type,
    repair_text_typed_columns,
)
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

# Gate.io rejects kline queries whose `from` is older than ~10000 recent
# points ("Candlestick too long ago. Maximum 10000 points recently are
# allowed"). Pairs hitting that were skipped every cycle forever. Clamp any
# fetch start for these exchanges into their allowed window (with a margin).
EXCHANGE_MAX_LOOKBACK_CANDLES_15M: Dict[str, int] = {
    "gateio": 9900,  # of 10000 allowed
    "gate": 9900,
}


def clamp_ohlcv_since_ms(exchange_name: str, since_ms: int) -> int:
    """Clamps the OHLCV 'since' cursor into the exchange's allowed lookback
    window (Gate.io: ~10000 recent candles). No-op for other exchanges."""
    max_candles = EXCHANGE_MAX_LOOKBACK_CANDLES_15M.get(exchange_name)
    if not max_candles:
        return int(since_ms)
    return max(int(since_ms), int(time.time() * 1000) - max_candles * 900 * 1000)


def ohlcv_since_floor_ms(exchange_name: str):
    """Absolute oldest 'since' the exchange accepts right now, or None when unlimited."""
    max_candles = EXCHANGE_MAX_LOOKBACK_CANDLES_15M.get(exchange_name)
    if not max_candles:
        return None
    return int(time.time() * 1000) - max_candles * 900 * 1000


# Symbols the exchange itself reports as NOT FOUND (e.g. BingX code 100204 —
# typically delisted spot tokens still present in load_markets). Retrying them
# on every 5-minute cycle is pure noise and pure rate-limit burn: they are
# skipped for the rest of the process run (a restart re-tries them once).
_DEAD_SYMBOLS: set = set()  # {(ccxt_name, symbol)}
_OB_WARNED: set = set()  # {(ccxt_name, symbol)} — orderbook warning printed once per process

# Backward history prefill (repair of truncated table starts): pages fetched
# per pair per cycle, and pairs for which the exchange could not deliver
# anything older than the current table start (empty/lose-progress attempt) —
# skipped until their table start actually improves.
PREFILL_MAX_PAGES = 10
_PREFILL_DONE: dict = {}  # {(ccxt_id, symbol): (min_ts_at_latch, attempt_ts)} — see should_attempt_prefill


def _is_symbol_not_found_error(e: Exception) -> bool:
    """Delisted/unknown market — permanent until the next engine restart."""
    if isinstance(e, ccxt.BadSymbol):
        return True
    msg = str(e).lower()
    return "symbol is not found" in msg or '"code":100204' in msg.replace(" ", "")


def _mark_dead_symbol_if_gone(e: Exception, ccxt_name: str, symbol: str) -> None:
    if _is_symbol_not_found_error(e):
        _DEAD_SYMBOLS.add((ccxt_name, symbol))
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

# FIX: vitality score thresholds used by the orderbook scoring block below
# (were referenced but never defined -> NameError, silently swallowed by try/except)
OB_TRADES_MIN_SLOW: float = 3.0
OB_TRADES_MIN_OK: float = 15.0
OB_TRADES_MIN_GOOD: float = 45.0
OB_TRADES_MIN_BLAZING: float = 120.0
OB_DEPTH_MIN_THIN: float = 1000.0
OB_DEPTH_MIN_OK: float = 10000.0
OB_DEPTH_MIN_GOOD: float = 50000.0

db_pools: Dict[str, asyncpg.Pool] = {}


def should_skip_pair(symbol: str, exchange: str = "") -> bool:
    if not symbol:
        return True

    parts = symbol.split("/")
    base = parts[0].upper() if len(parts) > 0 else ""

    if base and "USD" in base and base != "USDT":
        return True

    if SKIP_PATTERNS.search(symbol):
        return True

    # MEXC-style synthetic *STOCK* tokens (CXMTSTOCK, AAOISTOCK, DXCMSTOCK...):
    # their klines carry garbage timestamps (28 years of "history"), poison
    # tables and charts. Never collect them.
    if base.endswith("STOCK"):
        return True

    # Withdrawn/delisted listings still present in some exchanges' markets
    # (BingX): '$'-prefixed and '*_OLD'-suffixed tickers never trade — their
    # kline endpoint just 404s. Digit-leading tickers (1CAT, 10SET...) stay
    # untouched: e.g. 1INCH is legit, and the graveyard covers the rest.
    if base.startswith("$") or base.endswith("_OLD"):
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
            # Respect the global low-resource knob (DB_MAX_POOL_SIZE); floor of
            # 2 keeps min_size <= max_size valid.
            max_size=max(2, settings.db_max_pool_size),
        )
    log("✓ DB pools initialized for 15M")


async def find_table_in_dbs(table_name: str) -> Tuple[Optional[str], int, int]:
    """Returns (db_name, max_ts, min_ts); max_ts=1 / min_ts=0 marks an existing
    but EMPTY table (legacy convention, keeps process_pair on the initial path)."""
    for db in [DB_HIGH, DB_LOW]:
        if db not in db_pools:
            continue
        async with db_pools[db].acquire() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=$1)",
                table_name,
            )
            if exists:
                row = await conn.fetchrow(
                    f'SELECT MAX("Timestamp") AS mx, MIN("Timestamp") AS mn FROM "{table_name}"'
                )
                raw_max = int(row["mx"]) if row and row["mx"] else None
                raw_min = int(row["mn"]) if row and row["mn"] else None
                last_ts = normalize_epoch_sec(raw_max) if raw_max else None
                min_ts = normalize_epoch_sec(raw_min) if raw_min else None
                if (raw_max and raw_max != last_ts) or (raw_min and raw_min != min_ts):
                    # Legacy epoch-ms rows poison catch-up ("+0 свечей вечно")
                    # and history-prefill (absurd `since`) — repair the cursors.
                    log(
                        f"  [15M] ⚠️ [EPOCH-FIX] '{table_name}' in '{db}' stores Timestamp "
                        f"in epoch-ms (MIN={raw_min}, MAX={raw_max}) — cursors normalized "
                        f"to seconds; next save rewrites the table in seconds."
                    )
                return (
                    db,
                    int(last_ts) if last_ts else 1,  # 1 indicates table exists even if empty
                    int(min_ts) if min_ts else 0,
                )
    return None, 0, 0


async def move_table(table_name, from_db, to_db):
    if from_db not in db_pools or to_db not in db_pools:
        return
    async with db_pools[from_db].acquire() as fc:
        rows = await fc.fetch(f'SELECT * FROM "{table_name}"')
        if not rows:
            return
        # Real types from information_schema: the ALL_COLUMNS_SQL.get(k, "TEXT")
        # fallback used to TEXT-ify every ob_*/oi column of a moved table, and
        # numeric snapshot writes then died on `DataError: expected str, got
        # float` (see ZINC/ZK @mexc in the logs) until repaired on next ensure.
        col_types = await fetch_column_types(fc, table_name)
    async with db_pools[to_db].acquire() as tc:
        cols_sql = ", ".join(
            [f'"{k}" {pg_ddl_type(col_types.get(k))}' for k in rows[0].keys()]
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


async def delete_old_data_from_db(pool: asyncpg.Pool, db_name: str) -> Tuple[int, List[str]]:
    """Deletes rows older than the retention window; returns (rows_deleted,
    names of tables that actually lost rows)."""
    cutoff_ts = get_cutoff_timestamp()
    cutoff_date = datetime.datetime.now(
        datetime.timezone.utc
    ) - datetime.timedelta(days=DATA_RETENTION_DAYS)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")

    logger.info(
        f"🗑️  [{db_name}] Deleting data older than {DATA_RETENTION_DAYS} days (before {cutoff_str} UTC)..."
    )

    total_deleted = 0
    affected: List[str] = []
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
                        if deleted_count > 0:
                            total_deleted += deleted_count
                            affected.append(tbl)
                    except (IndexError, ValueError):
                        pass
            except asyncpg.PostgresError as e:
                logger.warning(f"  ⚠️  [{db_name}] {tbl}: error during delete — {e}")

    logger.info(f"  ✅ [{db_name}] Deletion finished: {total_deleted} old records removed")
    return total_deleted, affected


# Wall-clock latch limiting retention maintenance to the configured cadence.
_LAST_MAINTENANCE_AT = 0.0


async def run_maintenance() -> None:
    global _LAST_MAINTENANCE_AT
    interval_sec = settings.maintenance_interval_hours * 3600
    now = time.time()
    if _LAST_MAINTENANCE_AT and now - _LAST_MAINTENANCE_AT < interval_sec:
        logger.debug(
            f"⏭️  [15M] maintenance skipped — last run "
            f"{(now - _LAST_MAINTENANCE_AT) / 3600:.1f}h ago "
            f"(< {settings.maintenance_interval_hours}h)"
        )
        return
    # Latch BEFORE the work: an interrupted/killed maintenance retries next
    # day instead of re-running full scans in a hot every-cycle loop.
    _LAST_MAINTENANCE_AT = now

    logger.info("=" * 60)
    logger.info(
        f"🔧 RUNNING DATABASE MAINTENANCE (15M) — cadence "
        f"{settings.maintenance_interval_hours}h"
    )
    logger.info("=" * 60)

    total_deleted = 0
    affected_per_db: Dict[str, List[str]] = {}
    for db_name, pool in db_pools.items():
        try:
            deleted, affected = await delete_old_data_from_db(pool, db_name)
            total_deleted += deleted
            affected_per_db[db_name] = affected
        except Exception as e:
            logger.warning(f"  ⚠️  [{db_name}] Error deleting old data: {e}")
            affected_per_db[db_name] = []

    for db_name, pool in db_pools.items():
        affected = affected_per_db.get(db_name) or []
        if not affected:
            # Steady state deletes nothing → no manual VACUUM at all; the
            # stock autovacuum daemon owns routine cleanup. A whole-database
            # `VACUUM;` here ran after EVERY 5-minute cycle, scanned all
            # hypertables and saturated disk 24/7 (single backend observed at
            # 4.3 GB RSS, state D — that was the laptop stutter).
            logger.info(f"  ⏭️  [{db_name}] VACUUM skipped — no old rows deleted")
            continue
        try:
            async with pool.acquire() as conn:
                for tbl in affected:
                    await conn.execute(f'VACUUM "{tbl}";')
            logger.info(f"  ✅ [{db_name}] VACUUM completed ({len(affected)} tables)")
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
    col_types = await fetch_column_types(conn, tbl)
    for col, typ in ORDERBOOK_COLUMNS_SQL.items():
        if col not in col_types:
            await conn.execute(f'ALTER TABLE "{tbl}" ADD COLUMN "{col}" {typ}')
    # Self-heal tables TEXT-ified by an old HIGH↔LOW move (DataError storms).
    await repair_text_typed_columns(conn, tbl, col_types, ORDERBOOK_COLUMNS_SQL, log=logger)


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
        # Latest TWO rows (see repository.save_orderbook_snapshot): the
        # per-cycle refetch wipes the freshest row's metrics, so the
        # second-newest closed row is what actually persists as history.
        sql = (
            f'UPDATE "{tbl}" SET {", ".join(set_parts)} '
            f'WHERE "Timestamp" IN ('
            f'SELECT "Timestamp" FROM "{tbl}" '
            f'ORDER BY "Timestamp" DESC LIMIT 2)'
        )
        await conn.execute(sql, *values)


async def check_and_fill_table_gaps(
    exchange, symbol: str, tbl: str, db: str, ccxt_name: str = ""
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

    # Gate.io & co. reject 'from' older than their recent-candle window —
    # gaps lying (even partly) behind it can never be fetched there. Clamp or
    # drop them so the engine does not retry (and re-log) the impossible
    # fetch on every 5-minute cycle.
    floor_ms = ohlcv_since_floor_ms(ccxt_name)
    if floor_ms is not None:
        clamped: List[Tuple[int, int]] = []
        for gap_start, gap_end in gaps:
            if gap_end * 1000 < floor_ms:
                continue  # wholly behind the window — permanently out of reach
            clamped.append((max(gap_start, floor_ms // 1000), gap_end))
        gaps = clamped

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
                except Exception as e:
                    _mark_dead_symbol_if_gone(e, ccxt_name, symbol)
                    log(
                        f"  [15M] ⚠️ {symbol} @{tbl.rsplit('_on_', 1)[-1]}: gap-fill fetch_ohlcv failed "
                        f"({type(e).__name__}: {e}) — gap left open"
                    )
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
    current_db, last_ts, min_ts = await find_table_in_dbs(tbl)

    download_min_ts = get_download_min_timestamp()
    floor_ms_ex = ohlcv_since_floor_ms(ccxt_id)
    target_floor_ts = max(
        download_min_ts,
        floor_ms_ex // 1000 if floor_ms_ex is not None else 0,
    )

    try:
        all_candles: List[list] = []
        seen_ts: Set[int] = set()
        live_edge_ms = (int(time.time()) - 900) * 1000  # forming bar excluded

        def _append_new(cs) -> Tuple[int, int]:
            """Dedupe-merge a fetched page; returns (added_count, newest_ts_ms)."""
            added = 0
            newest = 0
            for c in cs:
                try:
                    t = int(c[0])
                except (TypeError, ValueError, IndexError):
                    continue
                newest = max(newest, t)
                if t in seen_ts:
                    continue
                seen_ts.add(t)
                all_candles.append(c)
                added += 1
            return added, newest

        async def _fetch_page(since_ms, phase: str, timeout: float = 12.0):
            """One OHLCV page -> (rows, transient). rows=None on failure
            (already logged + dead-marked); transient=True means rate-limit /
            network flakiness — retry next cycle, never a cooldown latch.
            12s default: MEXC under repair load answers plain downloads
            slower than 6s and whole pairs used to lose their cycle."""
            try:
                return await asyncio.wait_for(
                    exchange.fetch_ohlcv(symbol, "15m", since=since_ms, limit=FETCH_LIMIT),
                    timeout=timeout,
                ), False
            except Exception as e:
                _mark_dead_symbol_if_gone(e, ccxt_id, symbol)
                transient = is_transient_fetch_error(e)
                if phase == "history-prefill":
                    hint = (
                        " — transient (rate limit/network), retries NEXT cycle"
                        if transient
                        else f" — history-prefill retries after {settings.history_prefill_retry_sec // 3600}h or on restart"
                    )
                else:
                    hint = ""
                log(
                    f"  [15M] ⚠️ {symbol} @{ccxt_id}: {phase} fetch_ohlcv failed "
                    f"({type(e).__name__}: {e}); since={pd.to_datetime(int(since_ms), unit='ms')}{hint}"
                )
                return None, transient

        async def _paged_forward_fill(start_ms: int, max_pages: int) -> None:
            """
            Progress-based FORWARD pagination (start -> now). The NEXT page is
            keyed off the newest timestamp actually received, and the loop stops
            when a page adds nothing new or reaches the live edge — never off
            `len(page) == limit`, which silently broke on exchanges whose kline
            page cap is smaller than FETCH_LIMIT (that is how perp tables ended
            up with only the latest few days).
            """
            cursor_ms = start_ms
            for _ in range(max_pages):
                cs, _transient = await _fetch_page(cursor_ms, "download")
                if not cs:
                    break
                added, newest = _append_new(cs)
                if added == 0 or newest >= live_edge_ms:
                    break
                cursor_ms = newest + 900 * 1000
                await asyncio.sleep(0.05)

        if last_ts > 0:
            # Catch-up: from the last stored candle forward to now.
            if last_ts < download_min_ts:
                last_ts = download_min_ts
            cursor_ms = clamp_ohlcv_since_ms(ccxt_id, last_ts * 1000)
            await _paged_forward_fill(cursor_ms, max_pages=40)
        else:
            # New symbol: full initial history FORWARD from the retention floor
            # (same unified path as catch-up — spot and perp behave identically).
            log(f"  [15M] 🚀 New symbol detected: {symbol} ({ccxt_id}) -> Downloading initial history...")
            start_since = clamp_ohlcv_since_ms(ccxt_id, target_floor_ts * 1000)
            if start_since > download_min_ts * 1000:
                log(
                    f"  [15M] ℹ️ {symbol} @{ccxt_id}: exchange kline window — "
                    f"initial history limited to the latest "
                    f"{EXCHANGE_MAX_LOOKBACK_CANDLES_15M.get(ccxt_id)} candles"
                )
            await _paged_forward_fill(start_since, max_pages=40)
            if all_candles:
                log(
                    f"  [15M] 📥 {symbol} @{ccxt_id}: initial history downloaded "
                    f"{len(all_candles)} candles (from "
                    f"{pd.to_datetime(min(seen_ts) // 1000, unit='s')})"
                )

        # --- Backward history prefill (repair of truncated table starts) ----
        # A table whose first candle sits notably LATER than the 180d floor is
        # missing its old history (e.g. perp initial imports that never
        # paginated). Resumable: each cycle pages further left until the floor.
        if current_db and prefill_needed(min_ts, target_floor_ts, GAP_TOLERANCE_SEC):
            key = (ccxt_id, symbol)
            # Cooldown-gated retry: a failed/terminal attempt latches the pair
            # only for history_prefill_retry_sec (not for the whole run), and
            # any table-start improvement re-arms an attempt immediately.
            if should_attempt_prefill(
                _PREFILL_DONE.get(key), min_ts,
                retry_after_sec=settings.history_prefill_retry_sec,
            ):
                log(
                    f"  [15M] 🔧 {symbol} @{ccxt_id}: history repair — "
                    f"table starts {pd.to_datetime(int(min_ts), unit='s')}, "
                    f"filling back to {pd.to_datetime(target_floor_ts, unit='s')}"
                )
                oldest = min_ts
                prefilled = 0
                terminal_at = None   # (reason, oldest_at_latch)
                failed = False
                retry_next_cycle = False  # transient fetch error — leave unlatched
                span = FETCH_LIMIT  # window in candles; shrinks geometrically
                # when the exchange answers an out-of-range `since` with an
                # EMPTY page instead of clamping to the listing (mexc/bingx
                # style) — one empty big page proves nothing.
                for _ in range(PREFILL_MAX_PAGES):
                    since_ms = prefill_page_since_ms(
                        oldest, 900, span,
                        exchange_floor_ms=floor_ms_ex,
                        target_floor_sec=target_floor_ts,
                    )
                    if since_ms is None:
                        terminal_at = ("floor / exchange window reached — history complete", oldest)
                        break
                    cs, transient = await _fetch_page(since_ms, "history-prefill")
                    if cs is None:                      # fetch error (already logged)
                        if transient:
                            # Rate limit / network flake: do NOT cooldown-latch
                            # for hours — the next cycle retries right away.
                            retry_next_cycle = True
                        else:
                            failed = True
                        break
                    older = extract_older_rows(cs, oldest, target_floor_ts) if cs else []
                    if not older:
                        action, payload = prefill_empty_action(cs, oldest, 900, span)
                        if action == "shrink":
                            span = payload
                            continue
                        terminal_at = (payload, oldest)
                        break
                    for c in older:
                        if int(c[0]) not in seen_ts:
                            seen_ts.add(int(c[0]))
                            all_candles.append(c)
                    prefilled += len(older)
                    oldest = int(older[0][0]) // 1000
                    # AIMD: grow the window back gradually (an exchange that
                    # empties on out-of-range `since` would otherwise cost
                    # log2 probes before EVERY successful page).
                    span = min(span * 2, FETCH_LIMIT)
                    await asyncio.sleep(0.05)
                    if oldest <= target_floor_ts:
                        terminal_at = ("floor / exchange window reached — history complete", oldest)
                        break
                if failed:
                    _PREFILL_DONE[key] = (int(min_ts), time.time())  # retry after cooldown
                elif retry_next_cycle:
                    # UNLATCHED on purpose: rate limits clear in minutes, and
                    # the next cycle continues the repair from the same (or
                    # improved) table start. Any prefilled rows were saved.
                    log(
                        f"  [15M] ⏳ {symbol} @{ccxt_id}: history repair paused "
                        f"by rate limit/network — continues next cycle"
                        + (f" (+{prefilled} older candles so far)" if prefilled else "")
                    )
                elif terminal_at is not None:
                    reason, latch_min = terminal_at
                    _PREFILL_DONE[key] = (int(latch_min), time.time())
                    mark = "✅" if "complete" in reason else "⛔"
                    log(
                        f"  [15M] {mark} {symbol} @{ccxt_id}: history repair stop — "
                        f"{reason} at {pd.to_datetime(int(latch_min), unit='s')}"
                        + (f" (+{prefilled} older candles this round)" if prefilled else "")
                    )
                # Progress without a terminal state: stay UNLATCHED so the next
                # cycle keeps walking left immediately.
                if prefilled:
                    log(
                        f"  [15M] 📜 {symbol} @{ccxt_id}: history repair "
                        f"+{prefilled} older candles (table now from "
                        f"{pd.to_datetime(oldest, unit='s')}, floor "
                        f"{pd.to_datetime(target_floor_ts, unit='s')})"
                    )

        if not all_candles:
            # Create empty table for 0-candle symbol so find_table_in_dbs finds it next time
            # (but not for symbols the exchange itself reports as non-existent)
            if last_ts == 0 and (ccxt_id, symbol) not in _DEAD_SYMBOLS:
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
            exchange, symbol, tbl, current_db, ccxt_name=ccxt_id
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
                # Once per pair per process (was: only behind DEBUG_ORDERBOOK —
                # invisible). A table missing ob_* columns also vanished from
                # the dashboard scan until SELECT * replaced the fixed column
                # list, so this warning is the engine-side trace of that bug.
                ob_key = (ccxt_id, symbol)
                if ob_key not in _OB_WARNED:
                    _OB_WARNED.add(ob_key)
                    log(
                        f"    ⚠️ [OB] {symbol} @{ccxt_id}: orderbook snapshot failed "
                        f"({type(e_ob).__name__}: {e_ob}) — ob_* metrics will stay empty "
                        f"for this pair (repeat errors for it are logged only in debug)"
                    )
                elif DEBUG_ORDERBOOK:
                    log(f"    ⚠️ [OB] {symbol}: orderbook snapshot error: {e_ob}")

        # --- Open Interest & Funding Rate (perpetuals only) ---
        if settings.collect_oi_funding and current_db and ":" in symbol:
            try:
                snap = await fetch_oi_funding_snapshot(exchange, symbol, ccxt_id)
                await write_oi_funding_snapshot(db_pools[current_db], tbl, snap)
                if settings.funding_history_backfill:
                    await backfill_funding_history(
                        exchange,
                        db_pools[current_db],
                        tbl,
                        symbol,
                        ccxt_id,
                        since_ts=download_min_ts,
                        max_pages=settings.funding_history_max_pages,
                    )
            except Exception as e_oi:
                oi_funding_warn_once(
                    ccxt_id, symbol,
                    f"OI/funding collect failed ({type(e_oi).__name__}: {e_oi})",
                )

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
        # NEVER silent: an uncaught error here used to return +0 with zero log
        # output — exactly how a crashing history-prefill would stay invisible.
        log(
            f"  [15M] ⚠️ UNCAUGHT {symbol} @{ccxt_id}: {type(e).__name__}: {e}"
        )
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


# --- Persistent exchange instances (memory fix) ----------------------------
# Previously each 5-minute cycle created a fresh ccxt instance per exchange
# (and count_pairs_for_exchange created a SECOND one), re-downloaded the full
# markets JSON (tens of MB on gate/mexc/bingx) and threw everything away.
# The create/close churn plus Python allocator fragmentation ratcheted the
# process RSS up until the machine hit swap. Now: one long-lived instance
# per exchange; markets reloaded at most every MARKETS_TTL_SECONDS; instance
# recreated at most every EXCHANGE_MAX_AGE_SECONDS.
MARKETS_TTL_SECONDS = 1800.0           # reload market lists at most every 30 min
EXCHANGE_MAX_AGE_SECONDS = 6 * 3600.0  # recreate each ccxt instance every 6 h
_EXCHANGES: Dict[str, dict] = {}       # ccxt_name -> {ex, born_at, markets_at}


async def get_persistent_exchange(ccxt_name: str):
    """One long-lived ccxt instance per exchange. None on hard failure
    (markets not loadable AND nothing cached)."""
    now = time.time()
    entry = _EXCHANGES.get(ccxt_name)
    if entry and now - entry["born_at"] > EXCHANGE_MAX_AGE_SECONDS:
        await close_exchange_safely(entry["ex"], ccxt_name)
        _EXCHANGES.pop(ccxt_name, None)
        entry = None
    if entry is None:
        entry = {"ex": _make_exchange(EXCHANGE_MAP.get(ccxt_name, ccxt_name)),
                 "born_at": now, "markets_at": 0.0}
        _EXCHANGES[ccxt_name] = entry
    ex = entry["ex"]
    if now - entry["markets_at"] > MARKETS_TTL_SECONDS or not getattr(ex, "markets", None):
        ok = await load_markets_retries(ex, ccxt_name, reload=True)
        if ok:
            entry["markets_at"] = time.time()
        elif not getattr(ex, "markets", None):
            await close_exchange_safely(ex, ccxt_name)  # never loaded — drop, retry next cycle
            _EXCHANGES.pop(ccxt_name, None)
            return None
    return ex


def release_memory() -> None:
    """Return freed heap arenas to the OS after a cycle. Python's allocator
    ratchets RSS to the cycle peak and NEVER hands it back on its own — that
    ratchet is what filled RAM + swap over hours. gc + malloc_trim(0) gives
    it back on Linux; a harmless no-op elsewhere."""
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


async def load_markets_retries(exchange, ccxt_name, attempts: int = 3, timeout: float = 30.0, reload: bool = False) -> bool:
    """Load markets with retries — gateio/htx occasionally time out on flaky networks."""
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            await asyncio.wait_for(exchange.load_markets(reload), timeout=timeout)
            return True
        except Exception as e:
            last_err = e
            if attempt < attempts:
                log(f"  [MARKETS] {ccxt_name} attempt {attempt}/{attempts} failed: {e!r} — retrying...")
                await asyncio.sleep(2.0)
    log(f"  ⚠️ Failed loading markets for {ccxt_name} after {attempts} attempts: {last_err!r}")
    return False


async def count_pairs_for_exchange(ccxt_name: str) -> int:
    exchange = await get_persistent_exchange(ccxt_name)  # shared instance, no per-cycle churn
    if exchange is None:
        return 0
    try:
        syms = select_symbols_for_exchange(
            exchange.symbols, exchange.markets, ccxt_name
        )
        return len(syms)
    except Exception as e:
        log(f"  [PRECOUNT] {ccxt_name}: failed counting pairs: {e!r}")
        return 0


async def process_exchange(ccxt_name):
    ccxt_id = EXCHANGE_MAP.get(ccxt_name, ccxt_name)
    log(f"--- Exchange: {ccxt_name} ---")

    exchange = await get_persistent_exchange(ccxt_name)  # persistent instance, no per-cycle churn

    try:
        if exchange is None:
            if GLOBAL_PROGRESS is not None:
                GLOBAL_PROGRESS.subtract_from_total(2000)
            return

        syms = select_symbols_for_exchange(
            exchange.symbols, exchange.markets, ccxt_name
        )
        live_syms = [s for s in syms if (ccxt_name, s) not in _DEAD_SYMBOLS]
        if len(live_syms) < len(syms):
            log(
                f"  [15M] 🪦 {ccxt_name}: skipping {len(syms) - len(live_syms)} symbol(s) "
                f"the exchange reports as not found (delisted) until restart"
            )
        syms = live_syms
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
        # instance stays alive across cycles — the registry rotates it by age
        pass


async def main_15m_loop():
    await init_db_pools()
    await drop_not_allowed_exchange_tables()

    log(
        f"[15M] ⚙️ BUILD 2026-08-26-history-repair-v2 ACTIVE: backward prefill "
        f"floor=now-{settings.data_retention_days}d, ≤{PREFILL_MAX_PAGES} pages/pair/cycle, "
        f"retry={settings.history_prefill_retry_sec // 3600}h. "
        f"Expect 🔧 (start), 📜 (progress), ✅/⛔ (stop), ⚠️ (fetch error) lines."
    )

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

        release_memory()  # hand freed arenas back to the OS before the sleep

        log(f"Sleeping {UPDATE_INTERVAL_SECONDS // 60} minutes until next 15M cycle.")
        await asyncio.sleep(UPDATE_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(main_15m_loop())
    except KeyboardInterrupt:
        sys.exit(0)
