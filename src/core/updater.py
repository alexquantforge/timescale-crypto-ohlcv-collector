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
from src.db.connection import get_db_pools, close_all_db_pools
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
MSK_TZ = pytz_timezone("Europe/Moscow")
ALMATY_TZ = pytz_timezone("Asia/Almaty")


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

    def _should_attempt_backfill(self, tbl_name: str) -> bool:
        """Empty symbols (0 candles: tokenized stocks, delisted) are retried
        at most once per settings.empty_symbol_retry_sec instead of every cycle."""
        now = time.time()
        last = self._empty_since.get(tbl_name, 0.0)
        if now - last < settings.empty_symbol_retry_sec:
            return False
        self._empty_since[tbl_name] = now
        return True

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

                cursor_ms = backfill_ms
                seen_ts: Set[int] = set()

                for page_idx in range(settings.backfill_max_iterations):
                    try:
                        batch = await asyncio.wait_for(
                            exchange.fetch_ohlcv(symbol, self.timeframe, since=cursor_ms, limit=bf_limit),
                            timeout=6.0,
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
                    if len(batch) < bf_limit:
                        break
                    cursor_ms = int(new_rows[-1][0]) + (900000 if self.timeframe == "15m" else 86400000)
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
                df = pd.DataFrame(
                    cs, columns=["ts", "open", "high", "low", "close", "volume"]
                )
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
                df["open_time_msk"] = dt_utc.dt.tz_convert(MSK_TZ).dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                df["open_time_almaty"] = dt_utc.dt.tz_convert(ALMATY_TZ).dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                if not current_db:
                    current_db = self.repository.low_db
                    await self.repository.create_table_if_not_exists(current_db, tbl_name)

                await self.repository.upsert_candles(
                    current_db, tbl_name, df, timeframe=self.timeframe
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
            logger.debug(f"Uncatched error processing {symbol} on {ccxt_id}: {e}")
            return 0

    async def precount_exchange_pairs(self, ccxt_name: str) -> int:
        """Counts symbols for exchange using Perp-First selection."""
        exchange_map = settings.exchange_map_15m if self.timeframe == "15m" else settings.exchange_map_1d
        ccxt_id = exchange_map.get(ccxt_name, ccxt_name)
        exchange = create_exchange(ccxt_id)
        try:
            await asyncio.wait_for(exchange.load_markets(), timeout=12.0)
            syms = select_symbols_perp_first(exchange.symbols, exchange.markets, exchange_name=ccxt_name)
            return len(syms)
        except Exception as e:
            logger.warning(f"Failed precount for {ccxt_name}: {e}")
            return 0
        finally:
            await close_exchange_safely(exchange, ccxt_name)

    async def process_exchange(self, ccxt_name: str) -> None:
        """Processes all selected symbols for an exchange in parallel."""
        exchange_map = settings.exchange_map_15m if self.timeframe == "15m" else settings.exchange_map_1d
        ccxt_id = exchange_map.get(ccxt_name, ccxt_name)
        logger.info(f"--- Processing Exchange: {ccxt_name} ({ccxt_id}) [{self.timeframe}] ---")

        exchange = create_exchange(ccxt_id)
        try:
            try:
                await asyncio.wait_for(exchange.load_markets(), timeout=12.0)
            except Exception as e:
                logger.warning(f"Failed to load markets for {ccxt_name}: {e}")
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
            await close_exchange_safely(exchange, ccxt_name)

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
        results = await asyncio.gather(
            *[self.process_exchange(e) for e in active_exchanges],
            return_exceptions=True,
        )

        for ex_name, res in zip(active_exchanges, results):
            if isinstance(res, Exception):
                logger.error(f"[{self.timeframe.upper()}] [CRITICAL] Exchange {ex_name} failed cycle: {res}")

        elapsed = time.time() - start_time
        logger.info(
            f"Cycle finished ({self.timeframe.upper()}). Processed {self.progress.done}/{self.progress.total} "
            f"symbols in {elapsed/60.0:.2f} mins."
        )

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
