"""
Settings and configuration management using Pydantic Settings and db_config.py fallback.
"""
import json
import os
import sys
from typing import Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Fallback to db_config.py if present in project root or sys.path
_default_user = "postgres"
_default_password = "postgres"
_default_host = "localhost"
_default_port = 5432

try:
    sys.path.insert(0, os.getcwd())
    import db_config  # type: ignore
    _default_user = getattr(db_config, "user", "postgres")
    _default_password = getattr(db_config, "password", "postgres")
    _default_host = getattr(db_config, "host", "localhost")
    _default_port = int(getattr(db_config, "port", 5432))
except Exception:
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Database Server Configuration (reads from .env or db_config.py)
    db_host: str = Field(default=_default_host, alias="DB_HOST")
    db_port: int = Field(default=_default_port, alias="DB_PORT")
    db_user: str = Field(default=_default_user, alias="DB_USER")
    db_password: str = Field(default=_default_password, alias="DB_PASSWORD")
    db_min_pool_size: int = Field(default=2, alias="DB_MIN_POOL_SIZE")
    db_max_pool_size: int = Field(default=10, alias="DB_MAX_POOL_SIZE")

    # Original 4 Specific Database Names
    db_high_1d: str = Field(
        default="ohlcv_1d_data_for_usdt_pairs_using_ccxt_and_direct_api1",
        alias="DB_HIGH_1D",
    )
    db_low_1d: str = Field(
        default="ohlcv_1d_data_for_low_vol_usdt_pairs_using_ccxt_and_direct_api1",
        alias="DB_LOW_1D",
    )
    db_high_15m: str = Field(
        default="ohlcv_15m_data_for_usdt_pairs_using_ccxt_and_direct_api1",
        alias="DB_HIGH_15M",
    )
    db_low_15m: str = Field(
        default="ohlcv_15m_low_vol_usdt_pairs_using_ccxt_and_direct_api1",
        alias="DB_LOW_15M",
    )

    # Proxy Configuration
    socks5_proxy: Optional[str] = Field(default="socks5://127.0.0.1:10808", alias="SOCKS5_PROXY")

    # Timeframe & Data Retention
    timeframe: str = Field(default="1d", alias="TIMEFRAME")  # "1d" or "15m"
    data_retention_days: int = Field(default=180, alias="DATA_RETENTION_DAYS")  # 180 days retention for 15m
    update_days: int = Field(default=10, alias="UPDATE_DAYS")  # how many days back to fetch per update
    # 15m retention cleanup cadence: a full-database VACUUM after every 5-minute
    # cycle kept the disk saturated 24/7 — maintenance now runs at most this
    # often, and VACUUM touches only tables that actually had rows deleted.
    maintenance_interval_hours: int = Field(default=24, alias="MAINTENANCE_INTERVAL_HOURS")

    # 15m engine: on startup, DROP tables of exchanges that are not in
    # ALLOWED_EXCHANGES. Destructive — set to false to keep such tables.
    delete_not_allowed_exchange_tables_on_start: bool = Field(
        default=True, alias="DELETE_NOT_ALLOWED_EXCHANGE_TABLES_ON_START"
    )

    # Exchange Mapping for 1D (all 9 exchanges)
    exchange_map_1d: Dict[str, str] = {
        "bybit": "bybit",
        "gateio": "gate",
        "mexc": "mexc",
        "okx": "okx",
        "bingx": "bingx",
        "bitget": "bitget",
        "kucoin": "kucoin",
        "htx": "htx",
        "coinex": "coinex",
    }

    # Exchange Mapping for 15M (ONLY the 5 working exchanges from 15m updater script!)
    exchange_map_15m: Dict[str, str] = {
        "bybit": "bybit",
        "gateio": "gate",
        "mexc": "mexc",
        "okx": "okx",
        "bingx": "bingx",
    }

    # Optional whitelist of exchanges to run (empty = all exchanges from the map)
    allowed_exchanges_raw: str = Field(default="", alias="ALLOWED_EXCHANGES")
    # Optional blacklist — applied AFTER the whitelist, so it wins on conflicts.
    # Handy when one exchange is banned/slow but you still want the rest,
    # without retyping the whole include-list.
    excluded_exchanges_raw: str = Field(default="", alias="EXCLUDED_EXCHANGES")

    @staticmethod
    def _parse_exchange_list(raw: str) -> List[str]:
        """
        Exchange list from .env. Accepts BOTH formats:
          bybit,okx,bitget              (comma-separated)
          ["bybit","okx","bitget"]      (JSON list)
        Empty/unset/garbage -> [] (i.e. "no filter").
        """
        raw = (raw or "").strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                return [str(x).strip().lower() for x in json.loads(raw) if str(x).strip()]
            except (ValueError, TypeError):
                return []
        return [x.strip().lower() for x in raw.split(",") if x.strip()]

    @property
    def allowed_exchanges(self) -> List[str]:
        """Exchange allow-list; empty/unset = all configured exchanges allowed."""
        return self._parse_exchange_list(self.allowed_exchanges_raw)

    @property
    def excluded_exchanges(self) -> List[str]:
        """Exchange deny-list; empty/unset = nothing excluded."""
        return self._parse_exchange_list(self.excluded_exchanges_raw)

    def filter_exchange_ids(self, candidates) -> List[str]:
        """
        Apply ALLOWED_EXCHANGES / EXCLUDED_EXCHANGES to a list of ccxt ids,
        PRESERVING the input order (the maps carry per-exchange tuning order,
        e.g. EXCHANGE_MAX_LOOKBACK_DAYS_1D). Returns [] when the filters leave
        nothing — callers decide whether that means "idle" or "programmer error".
        """
        allowed = set(self.allowed_exchanges)
        denied = set(self.excluded_exchanges)
        out = []
        for eid in candidates:
            if allowed and eid not in allowed:
                continue
            if eid in denied:
                continue
            out.append(eid)
        return out

    # Volume & Liquidity Tiering Thresholds
    hard_floor_usd_1d: float = Field(default=500000.0, alias="HARD_FLOOR_USD_1D")  # $500k USD for 1d
    hard_floor_usd_15m: float = Field(default=125000.0, alias="HARD_FLOOR_USD_15M")  # $125k USD for 15m
    min_days_volume_check: int = Field(default=7, alias="MIN_DAYS_VOLUME_CHECK")
    concurrent_per_exchange: int = Field(default=5, alias="CONCURRENT_PER_EXCHANGE")
    # Dashboard: how many neighbours EACH SIDE of the current pair get
    # pre-warmed (candle loads + chart prebuilds) in background threads.
    # On low-RAM machines (16 GB, local Postgres) these bursts stutter the
    # whole desktop — set DASH_WARM_NEIGHBORS=0..2 to tame them. 5 = legacy.
    # How many pairs on each side get their charts prefetched. 2 (was 5): the
    # neighbours beyond that were never clicked often enough to pay for 9 × the
    # database and exchange traffic they generated.
    dash_warm_neighbors: int = Field(default=2, alias="DASH_WARM_NEIGHBORS")
    # Delay before a prefetch starts, so the click that scheduled it is served
    # first. The app felt SLOWER while warming was unthrottled: the prefetch was
    # competing with the render for the same connections.
    dash_warm_delay_sec: float = Field(default=1.5, alias="DASH_WARM_DELAY_SEC")
    # Pairs whose collector stopped writing this long ago (the dead spot tables
    # left by a spot→perp migration) are pre-built from the database only — no
    # exchange round trips for hundreds of missing candles nobody will watch.
    dash_warm_stale_skip_sec: float = Field(default=172800.0, alias="DASH_WARM_STALE_SKIP_SEC")
    update_interval_seconds_1d: int = Field(default=3600, alias="UPDATE_INTERVAL_SECONDS_1D")
    update_interval_seconds_15m: int = Field(default=300, alias="UPDATE_INTERVAL_SECONDS_15M")  # 5 minutes for 15m

    # Backfill & Gap Filling Parameters
    backfill_new_tables: bool = Field(default=True, alias="BACKFILL_NEW_TABLES")
    backfill_older_existing_tables: bool = Field(default=True, alias="BACKFILL_OLDER_EXISTING_TABLES")
    backfill_start_date: str = Field(default="2018-01-01", alias="BACKFILL_START_DATE")
    backfill_max_iterations: int = Field(default=400, alias="BACKFILL_MAX_ITERATIONS")
    backfill_request_limit: int = Field(default=1000, alias="BACKFILL_REQUEST_LIMIT")
    # Backward history prefill (repair of truncated table starts — e.g. perp
    # tables whose initial import never paginated): pages of older candles
    # fetched per pair per cycle; resumable across cycles until
    # backfill_start_date / the retention floor is reached.
    history_prefill_max_pages: int = Field(default=10, alias="HISTORY_PREFILL_MAX_PAGES")
    # How long a terminal/failed prefill attempt suppresses retries for the
    # same unchanged table start (a failed fetch must NOT mute the pair for
    # the whole process run — that was the silent "nothing ever downloaded").
    history_prefill_retry_sec: int = Field(default=4 * 3600, alias="HISTORY_PREFILL_RETRY_SEC")

    backfill_request_limit_per_exchange: Dict[str, int] = {
        "bybit": 1000,
        "bitget": 200,
        "mexc": 1000,
        "kucoin": 1500,
        "gate": 1000,
        "gateio": 1000,
        "bingx": 1000,
        "htx": 2000,
        "coinex": 1000,
        "okx": 100,
    }

    gap_max_pages_per_range: int = Field(default=50, alias="GAP_MAX_PAGES_PER_RANGE")  # max OHLCV pages per gap fill
    gap_filler_budget_sec: int = Field(default=600, alias="GAP_FILLER_BUDGET_SEC")  # max seconds of gap filling per engine cycle
    gap_recheck_sec: int = Field(default=21600, alias="GAP_RECHECK_SEC")  # min seconds between gap checks of the same table (6h)
    empty_symbol_retry_sec: int = Field(default=86400, alias="EMPTY_SYMBOL_RETRY_SEC")  # retry backfill of empty symbols at most once/day

    check_and_fill_gaps: bool = Field(default=True, alias="CHECK_AND_FILL_GAPS")
    gap_tolerance_sec_15m: int = Field(default=1800, alias="GAP_TOLERANCE_SEC_15M")  # 30 min tolerance

    # Open Interest & Funding Rate (perpetual contracts only — symbols with ':')
    collect_oi_funding: bool = Field(default=True, alias="COLLECT_OI_FUNDING")
    funding_history_backfill: bool = Field(default=True, alias="FUNDING_HISTORY_BACKFILL")
    funding_history_max_pages: int = Field(default=100, alias="FUNDING_HISTORY_MAX_PAGES")  # 100 pages x 100 events ≈ 9y of 8h fundings

    # Orderbook Analytics
    collect_orderbook: bool = Field(default=True, alias="COLLECT_ORDERBOOK")
    debug_orderbook: bool = Field(default=False, alias="DEBUG_ORDERBOOK")
    ob_fetch_limit: int = Field(default=50, alias="OB_FETCH_LIMIT")
    ob_trades_limit: int = Field(default=100, alias="OB_TRADES_LIMIT")
    ob_trades_window_sec: int = Field(default=300, alias="OB_TRADES_WINDOW_SEC")
    ob_depth_pct: float = Field(default=1.0, alias="OB_DEPTH_PCT")
    ob_fallback_limits: List[int] = [20, 10, 5]

    # ATR without Paranormal Bars Parameters
    atr_period: int = Field(default=5, alias="ATR_PERIOD")
    atr_small_threshold: float = Field(default=0.5, alias="ATR_SMALL_THRESHOLD")
    atr_large_threshold: float = Field(default=1.8, alias="ATR_LARGE_THRESHOLD")

    progress_log_every: int = Field(default=25, alias="PROGRESS_LOG_EVERY")
    precount_pairs: bool = Field(default=True, alias="PRECOUNT_PAIRS")

    # Priority lane: 1-second refresh of the pairs the dashboard is showing.
    # The dashboard publishes the open pair ±5 neighbours into
    # `dashboard_priority_pairs`; the 15m engine refreshes exactly those
    # tables in parallel with the full sweep, so the dashboard never has to
    # download or compute anything itself.
    # Wall-clock budget for the dashboard's in-memory gap stitching. It runs
    # while the user waits for a chart, so it is bounded by time, not only
    # by page count.
    dash_stitch_budget_sec: float = Field(default=4.0, alias="DASH_STITCH_BUDGET_SEC")
    # Wall-clock budget for the whole-database summary scan. Whatever came
    # back in time is rendered — the dashboard must stay usable while the
    # collector writes.
    dash_scan_budget_sec: float = Field(default=25.0, alias="DASH_SCAN_BUDGET_SEC")
    # Scan shape: all tables of a timeframe are read in CHUNKS of
    # dash_scan_chunk_size, each chunk being ONE UNION ALL query executed by
    # one of dash_scan_pool_size pooled connections. Bigger chunks = fewer
    # round trips (the scan is round-trip bound, not CPU bound); a chunk that
    # PostgreSQL rejects for any reason retries as an all-TEXT query, so a
    # legacy TEXT-typed column no longer costs the whole batch its batching.
    dash_scan_chunk_size: int = Field(default=120, alias="DASH_SCAN_CHUNK_SIZE")
    dash_scan_pool_size: int = Field(default=6, alias="DASH_SCAN_POOL_SIZE")
    # How long the sweep stays out of the way after a pair switch. There is no
    # query priority in PostgreSQL, so this is the mechanism by which a click
    # beats a 69-chunk catalog sweep; 0 disables it.
    dash_scan_yield_gap_sec: float = Field(default=1.2, alias="DASH_SCAN_YIELD_GAP_SEC")
    # Databases scanned at the same time INSIDE one summary scan. 1 (default)
    # walks HIGH then LOW; the two timeframes never overlap either, because the
    # whole scan holds a process-wide gate. Parallel sweeps looked faster on an
    # idle box and were several times slower on a loaded one: each extra
    # connection is a query the collector has to wait for.
    dash_scan_max_parallel_dbs: int = Field(default=1, alias="DASH_SCAN_MAX_PARALLEL_DBS")
    # When a chunk query fails on a SCHEMA/type problem it is retried table by
    # table; this caps how many tables of that chunk may be probed that way. A
    # chunk that merely TIMED OUT is skipped instead (the tables are just as
    # slow one by one, and 120 extra queries per chunk is what turned a 25 s
    # scan into a 300 s one). 0 = never fall back to per-table reads.
    dash_scan_recovery_max_tables: int = Field(default=24, alias="DASH_SCAN_RECOVERY_MAX_TABLES")
    # How often the table/column inventory of a database is re-read from
    # pg_catalog. New listings and engine column migrations land here, so this
    # is a freshness knob for the PAIR LIST only (candles and live data are
    # unaffected). 0 = re-read on every scan (the pre-fix behaviour, and on a
    # 14k-table database it is what made startup slow).
    dash_scan_inventory_ttl_sec: float = Field(
        default=600.0, alias="DASH_SCAN_INVENTORY_TTL_SEC"
    )
    # Keep the last good summary on disk and paint it instantly on startup,
    # refreshing in the background (stale-while-revalidate). A scan cut short
    # by the budget is NOT persisted, so a busy collector cannot shrink the
    # pair list from launch to launch.
    dash_snapshot_enabled: bool = Field(default=True, alias="DASH_SNAPSHOT_ENABLED")
    dash_snapshot_dir: str = Field(default="", alias="DASH_SNAPSHOT_DIR")
    dash_snapshot_max_age_sec: float = Field(default=86400.0, alias="DASH_SNAPSHOT_MAX_AGE_SEC")
    # Minimum gap between two background rescans. Without it EVERY rerun
    # (every pair click, every 60 s auto-reload) launched a full 4-database
    # scan, which on a busy machine is a self-sustaining load loop: the scans
    # slow the database, the slower scans hit their budget and return a
    # shorter pair list, and that list is re-scanned just as eagerly.
    dash_snapshot_refresh_sec: float = Field(default=120.0, alias="DASH_SNAPSHOT_REFRESH_SEC")
    # Ceiling for that gap once scans keep coming back truncated: the delay
    # doubles per consecutive partial scan (120 -> 240 -> 480 … ) so a busy
    # database is retried a few times and then left alone, instead of every
    # retry making the next scan truncate as well.
    dash_scan_retry_max_sec: float = Field(default=1800.0, alias="DASH_SCAN_RETRY_MAX_SEC")

    priority_lane_enabled: bool = Field(default=True, alias="PRIORITY_LANE_ENABLED")
    priority_lane_interval_sec: float = Field(default=1.0, alias="PRIORITY_LANE_INTERVAL_SEC")
    # The daily bar moves with every trade too, but one refresh per second
    # per engine would double the request rate for zero visible gain — the
    # 1D lane ticks a bit slower by default.
    priority_lane_interval_sec_1d: float = Field(default=2.0, alias="PRIORITY_LANE_INTERVAL_SEC_1D")
    # A published set expires this fast, so closing the browser tab stops the
    # lane instead of pinning the engine to an abandoned pair.
    priority_lane_ttl_sec: float = Field(default=90.0, alias="PRIORITY_LANE_TTL_SEC")
    # Coordination database holding the tiny handshake table (empty = 15m HIGH).
    priority_lane_db: str = Field(default="", alias="PRIORITY_LANE_DB")


settings = Settings()
