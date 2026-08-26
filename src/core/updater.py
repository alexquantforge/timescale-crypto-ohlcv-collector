"""
Main Orchestration Engine for Crypto Market Data Collection across the 4 historical databases.
Enforces strict per-request network timeouts and clear backfill progress logging.
"""
import asyncio
import datetime
import logging
import time
from typing import Dict, List, Optional, Set
import ccxt
import pandas as pd
from pytz import timezone as pytz_timezone

from config.settings import settings
from src.analytics.atr_filtered import compute_atr_no_paranormal_bars
from src.analytics.orderbook import fetch_orderbook_snapshot
from src.core.progress import GlobalProgress
from src.utils.timeouts import hard_wait_for
from src.db.connection import get_db_pools, close_all_db_pools
from src.core.history_prefill import (
    extract_older_rows,
    prefill_empty_action,
    prefill_needed,
    prefill_page_since_ms,
    should_attempt_prefill,
)
from src.db.migrations import ensure_databases_exist
from src.db.repository import HistoricalMarketRepository
from src.exchanges.client import create_exchange, close_exchange_safely
from src.exchanges.gap_filler import fill_history_gaps, fetch_ohlcv_catch_up
from src.exchanges.symbol_selector import (
    get_exchange_url,
    get_swap_url,
    select_symbols_perp_first,
)

logger = logging.getLogger("engine")

# Known per-exchange kline lookback windows (1D engine, days). BingX
# hard-rejects any kline range older than ~380 days
# (code 100204: "The maximum query range for XXX K-lines is 380 days") —
# without the clamp, its initial import from backfill_start_date dies on the
# very first page and the table is created EMPTY ("+0 candles" forever),
# and history-prefill burns requests on guaranteed errors.
EXCHANGE_MAX_LOOKBACK_DAYS_1D: Dict[str, int] = {
    "bingx": 379,
}


def _lookback_floor_ms_1d(ccxt_id: str) -> Optional[int]:
    """Epoch-ms floor imposed by the exchange's kline lookback window, if any."""
    days = EXCHANGE_MAX_LOOKBACK_DAYS_1D.get(ccxt_id)
    if not days:
        return None
    return (int(time.time()) - int(days) * 86400) * 1000


def _candles_df_from_rows(cs, symbol: str, ccxt_id: str) -> pd.DataFrame:
    """Raw ccxt OHLCV rows -> fully-decorated candle frame (shared by the
    forward save path and the backward history-prefill repair)."""
    df = pd.DataFrame(cs, columns=["ts", "open", "high", "low", "close", "volume"])
    df["Timestamp"] = df["ts"] // 1000
    df["ticker"] = symbol
    df["exchange"] = ccxt_id
    df["volume_x_low"] = df["volume"] * df["low"]
    df["volume_x_close"] = df["volume"] * df["close"]

    is_swap = ":" in symbol
    df["asset_type"] = "swap" if is_swap else "spot"
    spot_url = get_exchange_url(ccxt_id, symbol)
    swap_url = get_swap_url(ccxt_id, symbol)
    df["url_of_trading_pair"] = swap_url if is_swap else spot_url
    df["url_of_swap_contract_if_it_exists"] = None if is_swap else swap_url

    dt_utc = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    df["open_time_msk"] = dt_utc.dt.tz_convert(MSK_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["open_time_almaty"] = dt_utc.dt.tz_convert(ALMATY_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df
MSK_TZ = pytz_timezone("Europe/Moscow")
ALMATY_TZ = pytz_timezone("Asia/Almaty")


def format_await_chain(task: asyncio.Task, max_depth: int = 12) -> str:
    """
    Builds the REAL suspension chain of a task by following coroutine
    `cr_await` links. `Task.get_stack()` only returns the outermost frame of
    a suspended task (e.g. `worker()` at updater.py:433), which says nothing
    about whether the task is actually stuck in pool.acquire(), a ccxt fetch
    or a semaphore — this walker reaches the true innermost await point.
    """
    parts: List[str] = []
    obj = task.get_coro()
    seen: Set[int] = set()
    depth = 0
    while obj is not None and depth < max_depth and id(obj) not in seen:
        seen.add(id(obj))
        frame = getattr(obj, "cr_frame", None) or getattr(obj, "gi_frame", None)
        if frame is not None:
            parts.append(
                f"{frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}()"
            )
        else:
            parts.append(repr(obj)[:80])
        awaited = getattr(obj, "cr_await", None)
        if awaited is None:
            awaited = getattr(obj, "gi_await", None)
        if awaited is None:
            break
        if hasattr(awaited, "cr_await") or hasattr(awaited, "gi_await"):
            obj = awaited
            depth += 1
        else:
            parts.append(f"awaiting {type(awaited).__name__}")
            break
    return " <- ".join(parts) if parts else "<no suspension point>"


async def load_markets_with_retry(
    exchange, name: str, attempts: int = 3, timeout: float = 30.0, reload: bool = False
) -> bool:
    """
    Loads exchange markets with retries — gateio/htx/kucoin occasionally
    time out on flaky networks (empty exception messages are asyncio
    timeouts), which used to skip the whole exchange for the cycle.
    """
    last_err: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            await hard_wait_for(exchange.load_markets(reload), timeout, label=f"{name}.load_markets")
            return True
        except Exception as e:
            last_err = e
            if attempt < attempts:
                logger.warning(
                    f"load_markets for {name} failed (attempt {attempt}/{attempts}): {e!r} — retrying..."
                )
                await asyncio.sleep(2.0)
    logger.warning(f"Failed to load markets for {name} after {attempts} attempts: {last_err!r}")
    return False


# --- Persistent exchange instances (memory fix) ----------------------------
# Same story as the 15m engine: per cycle the engine created a fresh ccxt
# instance per exchange (plus one more for the pre-count), re-downloaded full
# market lists and threw everything out — the allocator churn ratcheted RSS
# up until the host swapped. Keep one long-lived instance per exchange;
# reload markets at most every MARKETS_TTL_SECONDS; recreate instances at
# most every EXCHANGE_MAX_AGE_SECONDS.
MARKETS_TTL_SECONDS = 1800.0
EXCHANGE_MAX_AGE_SECONDS = 6 * 3600.0
_EXCHANGES: Dict[str, dict] = {}  # ccxt_id -> {ex, born_at, markets_at}


async def get_persistent_exchange(ccxt_id: str, ccxt_name: str):
    """One long-lived ccxt instance per exchange. None on hard failure
    (markets not loadable AND nothing cached)."""
    now = time.time()
    entry = _EXCHANGES.get(ccxt_id)
    if entry and now - entry["born_at"] > EXCHANGE_MAX_AGE_SECONDS:
        await close_exchange_safely(entry["ex"], ccxt_name)
        _EXCHANGES.pop(ccxt_id, None)
        entry = None
    if entry is None:
        entry = {"ex": create_exchange(ccxt_id), "born_at": now, "markets_at": 0.0}
        _EXCHANGES[ccxt_id] = entry
    ex = entry["ex"]
    if now - entry["markets_at"] > MARKETS_TTL_SECONDS or not getattr(ex, "markets", None):
        ok = await load_markets_with_retry(ex, ccxt_name, reload=True)
        if ok:
            entry["markets_at"] = time.time()
        elif not getattr(ex, "markets", None):
            await close_exchange_safely(ex, ccxt_name)  # never loaded — drop, retry next cycle
            _EXCHANGES.pop(ccxt_id, None)
            return None
    return ex


def release_memory() -> None:
    """Return freed heap arenas to the OS after a cycle: gc + malloc_trim(0).
    Python's allocator ratchets RSS to the cycle peak and never returns it on
    its own — that ratchet is what filled RAM + swap over hours."""
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


class MarketDataEngine:
    def __init__(self, timeframe: Optional[str] = None):
        self.repository: Optional[HistoricalMarketRepository] = None
        self.progress: Optional[GlobalProgress] = None
        self.timeframe = timeframe or settings.timeframe
        # Throttles: keep cycles fast on a large fleet
        self._empty_since: Dict[str, float] = {}   # tbl -> ts of last empty-symbol backfill attempt
        self._gap_since: Dict[str, float] = {}     # tbl -> ts of last gap check
        self._gap_budget_until = 0.0               # gap filling deadline for the current cycle
        self._gap_budget_logged = False
        self._prefill_done: Dict[str, tuple] = {}  # tbl -> (min_ts_at_latch, attempt_ts) — see should_attempt_prefill

    def _should_attempt_backfill(self, tbl_name: str) -> bool:
        """Empty symbols (0 candles: tokenized stocks, delisted) are retried
        at most once per settings.empty_symbol_retry_sec instead of every cycle."""
        now = time.time()
        last = self._empty_since.get(tbl_name, 0.0)
        if now - last < settings.empty_symbol_retry_sec:
            return False
        self._empty_since[tbl_name] = now
        return True

    async def _watchdog(self, stall_timeout: float = 180.0) -> None:
        """
        If global progress stalls (no symbol completes) for stall_timeout,
        dumps every pending asyncio task with its current await location to
        the log — turns a silent freeze into an exact diagnostic of where
        the engine is stuck.
        """
        last_done = -1
        stalled_since: Optional[float] = None
        try:
            while True:
                await asyncio.sleep(30.0)
                done = self.progress.done if self.progress else 0
                if done != last_done:
                    last_done = done
                    stalled_since = None
                    continue
                now = time.time()
                if stalled_since is None:
                    stalled_since = now
                elif now - stalled_since >= stall_timeout:
                    total = self.progress.total if self.progress else 0
                    logger.error(
                        f"[{self.timeframe.upper()}] 🐕 WATCHDOG: no progress for {stall_timeout:.0f}s "
                        f"({done}/{total}). Pending tasks:"
                    )
                    shown = 0
                    for task in asyncio.all_tasks():
                        if task.done() or task is asyncio.current_task():
                            continue
                        if shown >= 40:
                            logger.error("  ... (more tasks truncated)")
                            break
                        corr = task.get_coro()
                        name = getattr(corr, "__qualname__", str(corr))
                        logger.error(f"  task: {name} <- {format_await_chain(task)}")
                        shown += 1
                    stalled_since = now  # re-arm: dump again if still frozen later
        except asyncio.CancelledError:
            pass

    def _should_gap_check(self, tbl_name: str) -> bool:
        """Each table is gap-checked at most once per settings.gap_recheck_sec,
        and only while the per-cycle gap-filling time budget lasts."""
        if time.time() > self._gap_budget_until:
            if not self._gap_budget_logged:
                logger.info(
                    f"[{self.timeframe.upper()}] Gap-filler budget for this cycle exhausted "
                    f"({settings.gap_filler_budget_sec}s) — remaining tables will be checked next cycle."
                )
                self._gap_budget_logged = True
            return False
        now = time.time()
        last = self._gap_since.get(tbl_name, 0.0)
        if now - last < settings.gap_recheck_sec:
            return False
        self._gap_since[tbl_name] = now
        return True

    async def initialize(self) -> None:
        """Ensures databases exist and initializes connection pools."""
        await ensure_databases_exist()
        pool_dict = await get_db_pools(timeframe=self.timeframe)
        
        high_db = pool_dict["high_db_name"]
        low_db = pool_dict["low_db_name"]
        
        pools = {
            high_db: pool_dict["HIGH"],
            low_db: pool_dict["LOW"],
        }
        
        self.repository = HistoricalMarketRepository(pools, high_db, low_db)
        logger.info(
            f"Engine initialized for timeframe '{self.timeframe}' with databases: "
            f"HIGH='{high_db}', LOW='{low_db}'"
        )
        logger.info(
            f"[{self.timeframe.upper()}] ⚙️ BUILD 2026-08-26-history-repair-v2 ACTIVE: "
            f"backward prefill floor={settings.backfill_start_date}, "
            f"≤{settings.history_prefill_max_pages} pages/pair/cycle, "
            f"retry={settings.history_prefill_retry_sec // 3600}h. "
            f"Expect 🔧 (start), 📜 (progress), ✅/⛔ (stop), ⚠️ (fetch error) lines."
        )
        # Automatically clean up Bitget tokenized stock tables on startup
        await self.repository.cleanup_invalid_bitget_tables()

    def get_configured_exchanges(self) -> List[str]:
        """Returns list of allowed exchanges filtered by ALLOWED_EXCHANGES setting."""
        exchange_map = settings.exchange_map_15m if self.timeframe == "15m" else settings.exchange_map_1d
        ex_keys = list(exchange_map.keys())
        if settings.allowed_exchanges:
            allowed = set(settings.allowed_exchanges)
            ex_keys = [k for k in ex_keys if k in allowed]
        return ex_keys

    async def handle_delisted_pair_cleanup(self, current_db: Optional[str], tbl_name: str, symbol: str, ccxt_id: str):
        """Drops table from database if symbol is delisted or not found on exchange."""
        if current_db and self.repository:
            logger.info(f"🗑️ [{self.timeframe.upper()}] [AUTO-DROP DELISTED] Dropping table '{tbl_name}' from '{current_db}' (symbol '{symbol}' no longer valid on {ccxt_id}).")
            await self.repository.drop_table(current_db, tbl_name)

    async def process_pair(self, exchange, symbol: str, ccxt_id: str) -> int:
        """
        Processes a single trading pair for an exchange with strict network timeouts.
        """
        if not self.repository:
            return 0

        tbl_name = f"{symbol.replace('/', '_').replace('-', '_')}_on_{ccxt_id}".lower()
        current_db, last_ts, first_ts = await self.repository.find_table(tbl_name)

        since_ts = int(time.time() - (settings.update_days * 86400))
        if settings.data_retention_days and self.timeframe == "15m":
            retention_cutoff = int(time.time() - (settings.data_retention_days * 86400))
            if last_ts > 0 and last_ts < retention_cutoff:
                since_ts = retention_cutoff
            elif last_ts > 0:
                since_ts = last_ts
            else:
                since_ts = retention_cutoff
        elif last_ts > 0:
            since_ts = last_ts

        bf_limit = settings.backfill_request_limit_per_exchange.get(
            ccxt_id, settings.backfill_request_limit
        )

        try:
            # --- Fetch OHLCV Candles ---
            if last_ts == 0:
                if not settings.backfill_new_tables:
                    return 0
                # Empty/failed symbols (tokenized stocks, delisted) create empty
                # tables — without a cooldown they would be re-backfilled every cycle.
                if not self._should_attempt_backfill(tbl_name):
                    return 0

                logger.info(f"  [{self.timeframe.upper()}] ⏳ Backfilling new symbol history: {symbol} ({ccxt_id})...")
                cs = []
                backfill_ms = int(
                    datetime.datetime.strptime(
                        settings.backfill_start_date, "%Y-%m-%d"
                    )
                    .replace(tzinfo=datetime.timezone.utc)
                    .timestamp()
                    * 1000
                )
                if settings.data_retention_days and self.timeframe == "15m":
                    retention_ms = int(time.time() - (settings.data_retention_days * 86400)) * 1000
                    backfill_ms = max(backfill_ms, retention_ms)
                # Exchange kline lookback window (BingX: 380d): an initial
                # import starting BEFORE the window dies with 100204 on page
                # one and the table is created empty — clamp the cursor.
                _exw_ms = _lookback_floor_ms_1d(ccxt_id)
                if _exw_ms is not None and backfill_ms < _exw_ms:
                    logger.warning(
                        f"  [{self.timeframe.upper()}] ⚠️ {symbol} @{ccxt_id}: exchange kline "
                        f"window is ~{EXCHANGE_MAX_LOOKBACK_DAYS_1D[ccxt_id]}d — initial history "
                        f"starts at {pd.to_datetime(_exw_ms, unit='ms')} instead of "
                        f"{settings.backfill_start_date}"
                    )
                    backfill_ms = _exw_ms

                cursor_ms = backfill_ms
                seen_ts: Set[int] = set()

                for page_idx in range(settings.backfill_max_iterations):
                    try:
                        batch = await hard_wait_for(
                            exchange.fetch_ohlcv(symbol, self.timeframe, since=cursor_ms, limit=bf_limit),
                            6.0,
                            label=f"{symbol}@{ccxt_id} backfill",
                        )
                    except Exception:
                        break

                    if not batch:
                        break

                    new_rows = [c for c in batch if int(c[0]) not in seen_ts]
                    if not new_rows:
                        break
                    for c in new_rows:
                        seen_ts.add(int(c[0]))
                    cs.extend(new_rows)
                    # Progress-based pagination: the next page is keyed off the
                    # newest timestamp actually received, never off
                    # `len(page) == limit` — exchanges whose kline page cap is
                    # smaller than bf_limit silently broke that signal and
                    # truncated perp initial imports to the latest few days.
                    newest_ms = max(int(c[0]) for c in new_rows)
                    live_edge_ms = (int(time.time()) - (900 if self.timeframe == "15m" else 86400)) * 1000
                    if newest_ms >= live_edge_ms:
                        break
                    cursor_ms = newest_ms + (900000 if self.timeframe == "15m" else 86400000)
            else:
                # Paged catch-up from the last stored candle up to now:
                # fully synchronises lagging tables in one cycle (a single
                # limit=50 request covers only 50 candles and left tables
                # crawling forward or gapped after long downtime).
                cs = await fetch_ohlcv_catch_up(
                    exchange, symbol, self.timeframe, since_sec=since_ts,
                    page_limit=50, max_pages=40,
                )

            if not cs and last_ts == 0:
                # Create empty table for 0-candle symbol so find_table finds it next time
                await self.repository.create_table_if_not_exists(self.repository.low_db, tbl_name)
                return 0

            # --- Save Candles ---
            if cs:
                df = _candles_df_from_rows(cs, symbol, ccxt_id)

                if not current_db:
                    current_db = self.repository.low_db
                    await self.repository.create_table_if_not_exists(current_db, tbl_name)

                await self.repository.upsert_candles(
                    current_db, tbl_name, df, timeframe=self.timeframe
                )

            # --- Backward history prefill (repair of truncated table starts) ---
            # Perp tables created by a non-paginating initial import start only
            # a few days back while the exchange has full history. Page OLDER
            # candles in, resumably: every cycle walks the table start left
            # until backfill_start_date (1D keeps the FULL available history).
            if current_db and first_ts:
                floor_sec = int(
                    datetime.datetime.strptime(settings.backfill_start_date, "%Y-%m-%d")
                    .replace(tzinfo=datetime.timezone.utc)
                    .timestamp()
                )
                step_sec = 900 if self.timeframe == "15m" else 86400
                if prefill_needed(first_ts, floor_sec, slack_sec=step_sec * 2) and should_attempt_prefill(
                    self._prefill_done.get(tbl_name),
                    first_ts,
                    retry_after_sec=settings.history_prefill_retry_sec,
                ):
                    logger.info(
                        f"  [{self.timeframe.upper()}] 🔧 {symbol} @{ccxt_id}: history repair — "
                        f"table starts {pd.to_datetime(int(first_ts), unit='s')}, "
                        f"filling back to {settings.backfill_start_date}"
                    )
                    oldest = int(first_ts)
                    prefilled = 0
                    terminal_at = None   # (reason, oldest_at_latch)
                    failed = False
                    span = bf_limit  # window in candles; shrinks geometrically
                    # when the exchange answers an out-of-range `since` with an
                    # EMPTY page instead of clamping to the listing (see
                    # prefill_empty_action) — one empty big page proves nothing.
                    ex_floor_ms = _lookback_floor_ms_1d(ccxt_id)
                    for _ in range(settings.history_prefill_max_pages):
                        since_ms = prefill_page_since_ms(
                            oldest, step_sec, span,
                            exchange_floor_ms=ex_floor_ms,
                            target_floor_sec=floor_sec,
                        )
                        if since_ms is None:
                            terminal_at = ("floor / exchange window reached — history complete", oldest)
                            break
                        try:
                            batch = await hard_wait_for(
                                exchange.fetch_ohlcv(symbol, self.timeframe, since=since_ms, limit=bf_limit),
                                12.0,
                                label=f"{symbol}@{ccxt_id} history-prefill",
                            )
                        except Exception as e:
                            # NEVER silent: a failed page used to latch the pair
                            # for the whole process run with zero log output —
                            # that is how "везде +1 свеча, история не качается".
                            logger.warning(
                                f"  [{self.timeframe.upper()}] ⚠️ {symbol} @{ccxt_id}: "
                                f"history-prefill page fetch failed "
                                f"({type(e).__name__}: {e}); since="
                                f"{pd.to_datetime(since_ms, unit='ms')} — will retry "
                                f"after {settings.history_prefill_retry_sec // 3600}h or on restart"
                            )
                            failed = True
                            break
                        older = extract_older_rows(batch, oldest, floor_sec) if batch else []
                        if not older:
                            action, payload = prefill_empty_action(batch, oldest, step_sec, span)
                            if action == "shrink":
                                span = payload
                                continue
                            terminal_at = (payload, oldest)
                            break
                        try:
                            await self.repository.upsert_candles(
                                current_db, tbl_name,
                                _candles_df_from_rows(older, symbol, ccxt_id),
                                timeframe=self.timeframe,
                            )
                        except Exception as e:
                            logger.warning(
                                f"  [{self.timeframe.upper()}] ⚠️ {symbol} @{ccxt_id}: "
                                f"history-prefill DB save failed ({type(e).__name__}: {e}) — "
                                f"will retry after {settings.history_prefill_retry_sec // 3600}h or on restart"
                            )
                            failed = True
                            break
                        prefilled += len(older)
                        oldest = int(older[0][0]) // 1000
                        # AIMD: grow the window back gradually (an exchange
                        # that empties on out-of-range `since` would otherwise
                        # cost log2 probes before EVERY successful page).
                        span = min(span * 2, bf_limit)
                        await asyncio.sleep(0.05)
                        if oldest <= floor_sec:
                            terminal_at = ("floor / exchange window reached — history complete", oldest)
                            break
                    if failed:
                        # Cooldown-latch on the (unchanged) table start so we
                        # don't hammer the exchange every cycle, but DO retry.
                        self._prefill_done[tbl_name] = (int(first_ts), time.time())
                    elif terminal_at is not None:
                        reason, latch_min = terminal_at
                        self._prefill_done[tbl_name] = (int(latch_min), time.time())
                        mark = "✅" if "floor" in reason else "⛔"
                        logger.info(
                            f"  [{self.timeframe.upper()}] {mark} {symbol} @{ccxt_id}: "
                            f"history repair stop — {reason} at "
                            f"{pd.to_datetime(int(latch_min), unit='s')}"
                            + (f" (+{prefilled} older candles this round)" if prefilled else "")
                        )
                    # Success with progress and more pages likely available:
                    # leave UNLATCHED so the next cycle continues left at once.
                    if prefilled:
                        logger.info(
                            f"  [{self.timeframe.upper()}] 📜 {symbol} @{ccxt_id}: history repair "
                            f"+{prefilled} older candles (table now from "
                            f"{pd.to_datetime(oldest, unit='s')}, floor {settings.backfill_start_date})"
                        )

            # --- Check Gap Filling ---
            if settings.check_and_fill_gaps and current_db and self._should_gap_check(tbl_name):
                try:
                    await fill_history_gaps(
                        exchange,
                        symbol,
                        ccxt_id,
                        self.repository,
                        bf_limit=bf_limit,
                        max_pages=settings.gap_max_pages_per_range,
                        timeframe=self.timeframe,
                    )
                except Exception as e:
                    logger.warning(f"Gap filling failed for {symbol} ({ccxt_id}): {e!r}")

            # --- Check Volume Floor & Move Table (HIGH <-> LOW) ---
            if current_db:
                floor_usd = (
                    settings.hard_floor_usd_15m
                    if self.timeframe == "15m"
                    else settings.hard_floor_usd_1d
                )
                current_db = await self.repository.check_volume_floor_and_move(
                    table_name=tbl_name,
                    current_db=current_db,
                    symbol=symbol,
                    hard_floor_usd=floor_usd,
                    timeframe=self.timeframe,
                    min_days_check=settings.min_days_volume_check,
                )

            # --- Compute ATR без паранормальных баров & Orderbook Snapshot ---
            if settings.collect_orderbook and current_db:
                try:
                    pool = self.repository.pools.get(current_db)
                    if pool:
                        async with pool.acquire() as conn:
                            hist = await conn.fetch(
                                f'SELECT "Timestamp", high, low, close, volume FROM "{tbl_name}" ORDER BY "Timestamp" ASC'
                            )
                        if hist:
                            full_df = pd.DataFrame(
                                hist, columns=["Timestamp", "high", "low", "close", "volume"]
                            )
                            atr_val = compute_atr_no_paranormal_bars(
                                highs=full_df["high"].to_numpy(dtype=float),
                                lows=full_df["low"].to_numpy(dtype=float),
                                closes=full_df["close"].to_numpy(dtype=float),
                                period=settings.atr_period,
                                small_threshold=settings.atr_small_threshold,
                                large_threshold=settings.atr_large_threshold,
                            )

                            snap = await fetch_orderbook_snapshot(
                                exchange=exchange,
                                symbol=symbol,
                                atr_no_paranormal=atr_val,
                                fetch_limit=settings.ob_fetch_limit,
                                trades_limit=settings.ob_trades_limit,
                                trades_window_sec=settings.ob_trades_window_sec,
                                depth_pct=settings.ob_depth_pct,
                                timeout_sec=6.0,
                            )

                            await self.repository.save_orderbook_snapshot(
                                db_name=current_db,
                                table_name=tbl_name,
                                full_df=full_df,
                                snap=snap,
                                timeframe=self.timeframe,
                                min_days_check=settings.min_days_volume_check,
                            )
                except Exception as e:
                    logger.debug(f"Orderbook snapshot error for {symbol}: {e}")

            return len(cs)
        except asyncio.TimeoutError:
            logger.debug(f"Timeout (6s) fetching {symbol} on {ccxt_id}, moving on...")
            return 0
        except ccxt.BadSymbol as e:
            await self.handle_delisted_pair_cleanup(current_db, tbl_name, symbol, ccxt_id)
            return 0
        except ccxt.ExchangeError as e:
            err_msg = str(e).lower()
            if any(term in err_msg for term in ["symbol is not found", "invalid symbol", "symbol_not_found", "100204", "48001"]):
                await self.handle_delisted_pair_cleanup(current_db, tbl_name, symbol, ccxt_id)
            else:
                logger.debug(f"Exchange error for {symbol} on {ccxt_id}: {e}")
            return 0
        except Exception as e:
            # Loud, not debug: this catch-all also intercepts errors escaping
            # the history-prefill block — invisible at debug level.
            logger.warning(
                f"  [{self.timeframe.upper()}] ⚠️ UNCAUGHT error processing {symbol} on {ccxt_id}: "
                f"{type(e).__name__}: {e}"
            )
            return 0

    async def precount_exchange_pairs(self, ccxt_name: str) -> int:
        """Counts symbols for exchange using Perp-First selection."""
        exchange_map = settings.exchange_map_15m if self.timeframe == "15m" else settings.exchange_map_1d
        ccxt_id = exchange_map.get(ccxt_name, ccxt_name)
        exchange = await get_persistent_exchange(ccxt_id, ccxt_name)  # shared, no per-cycle churn
        try:
            if exchange is None:
                return 0
            syms = select_symbols_perp_first(exchange.symbols, exchange.markets, exchange_name=ccxt_name)
            return len(syms)
        except Exception as e:
            logger.warning(f"Failed precount for {ccxt_name}: {e!r}")
            return 0

    async def process_exchange(self, ccxt_name: str) -> None:
        """Processes all selected symbols for an exchange in parallel."""
        exchange_map = settings.exchange_map_15m if self.timeframe == "15m" else settings.exchange_map_1d
        ccxt_id = exchange_map.get(ccxt_name, ccxt_name)
        logger.info(f"--- Processing Exchange: {ccxt_name} ({ccxt_id}) [{self.timeframe}] ---")

        exchange = await get_persistent_exchange(ccxt_id, ccxt_name)  # persistent, no per-cycle churn
        try:
            if exchange is None:
                return

            syms = select_symbols_perp_first(exchange.symbols, exchange.markets, exchange_name=ccxt_name)
            total = len(syms)
            sem = asyncio.Semaphore(settings.concurrent_per_exchange)

            if self.progress:
                self.progress.add_to_total(total)

            processed = 0

            async def worker(s: str):
                nonlocal processed
                async with sem:
                    count = await self.process_pair(exchange, s, ccxt_name)
                    processed += 1
                    if self.progress:
                        g_done, g_total, eta_str, pct = self.progress.tick()
                        if count > 0 or g_done % settings.progress_log_every == 0:
                            logger.info(
                                f"  [{self.timeframe.upper()}] [ALL {g_done}/{g_total} · {pct:.1f}% · ETA {eta_str}] "
                                f"[{ccxt_name} {processed}/{total}] {s}: +{count} candles"
                            )

            await asyncio.gather(*[worker(s) for s in syms])
        finally:
            # instance stays alive across cycles — the registry rotates it by age
            pass

    async def run_cycle(self) -> None:
        """Executes one full iteration cycle over all configured exchanges."""
        self.progress = GlobalProgress()
        self._gap_budget_until = time.time() + settings.gap_filler_budget_sec
        self._gap_budget_logged = False
        active_exchanges = self.get_configured_exchanges()

        if settings.precount_pairs:
            logger.info(f"[{self.timeframe.upper()}] Pre-counting trading pairs across configured exchanges...")
            counts = await asyncio.gather(
                *[self.precount_exchange_pairs(e) for e in active_exchanges]
            )
            total_pairs = sum(counts)
            self.progress.reset(total_pairs)
            logger.info(f"[{self.timeframe.upper()}] Total symbols to process across exchanges: {total_pairs}")

        # Reset progress timer right when processing begins
        if self.progress:
            self.progress.start_timing()

        start_time = time.time()
        watchdog = asyncio.create_task(self._watchdog())
        try:
            results = await asyncio.gather(
                *[self.process_exchange(e) for e in active_exchanges],
                return_exceptions=True,
            )
        finally:
            watchdog.cancel()

        for ex_name, res in zip(active_exchanges, results):
            if isinstance(res, Exception):
                logger.error(f"[{self.timeframe.upper()}] [CRITICAL] Exchange {ex_name} failed cycle: {res}")

        elapsed = time.time() - start_time
        logger.info(
            f"Cycle finished ({self.timeframe.upper()}). Processed {self.progress.done}/{self.progress.total} "
            f"symbols in {elapsed/60.0:.2f} mins."
        )
        release_memory()  # hand freed arenas back to the OS

    async def start_loop(self) -> None:
        """Starts continuous execution loop."""
        await self.initialize()
        interval = (
            settings.update_interval_seconds_15m
            if self.timeframe == "15m"
            else settings.update_interval_seconds_1d
        )
        while True:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error(f"[{self.timeframe.upper()}] Error during execution cycle: {e}")

            logger.info(f"[{self.timeframe.upper()}] Waiting {interval}s until next cycle...")
            await asyncio.sleep(interval)
