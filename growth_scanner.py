#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot scanner for strong recent rises with live liquidity checks.

Growth is measured as the largest chronological 15m OHLCV low-to-high move
inside the requested lookback window. The active/forming 15m candle is included;
its low and high are treated as the candle's range because OHLCV cannot reveal
which extreme occurred first.

A candidate must also have:
  * min(volume * low) >= the threshold on the last 7 CLOSED daily candles;
  * a daily Gerchik ATR computed from closed daily candles;
  * a fresh CCXT order-book/trades snapshot with vitality grade A or B;
  * live spread <= the requested fraction of that daily Gerchik ATR.

The program only reads the OHLCV databases and public exchange endpoints. It
writes no results to the database and does not modify any existing scanner.

Example:
    python growth_scanner.py --hours 6 --pct 25
    python growth_scanner.py --hours 12 --pct 40 --min-7d-volume-usd 250000
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

import asyncpg

from config.settings import settings
from src.analytics.orderbook import fetch_orderbook_snapshot
from src.exchanges.client import close_exchange_safely, create_exchange
from src.utils.timeouts import hard_wait_for

LOGGER = logging.getLogger("growth_scanner")

BAR_15M_SEC = 15 * 60
BAR_1D_SEC = 24 * 60 * 60
DEFAULT_MIN_7D_VOLUME_USD = 100_000.0
DEFAULT_MAX_SPREAD_ATR_PCT = 5.0


@dataclass(frozen=True)
class PairTable:
    db_name: str
    table: str
    symbol: str
    exchange: str
    ccxt_id: str
    market_type: str


@dataclass(frozen=True)
class GrowthMove:
    start_ts: int
    peak_ts: int
    low_price: float
    high_price: float
    pct: float


@dataclass
class Candidate:
    pair: PairTable
    move: GrowthMove
    min_7d_volume_usd: float = 0.0
    daily_gerchik_atr: float = 0.0
    live_snapshot: Optional[dict[str, Any]] = None


def _quote_ident(identifier: str) -> str:
    """Quote a PostgreSQL identifier obtained from the catalog."""
    return '"' + str(identifier).replace('"', '""') + '"'


def parse_pair_table(table: str, db_name: str) -> Optional[PairTable]:
    """Parse collector names such as ``btc_usdt:usdt_on_bybit``."""
    name = str(table or "").strip().lower()
    pair_part, marker, exchange_part = name.rpartition("_on_")
    if not marker or not pair_part or not exchange_part:
        return None

    table_to_ccxt = {str(k).lower(): str(v).lower()
                     for k, v in settings.exchange_map_15m.items()}
    ccxt_to_table = {v: k for k, v in table_to_ccxt.items()}
    if exchange_part in table_to_ccxt:
        exchange = exchange_part
        ccxt_id = table_to_ccxt[exchange]
    elif exchange_part in ccxt_to_table:
        ccxt_id = exchange_part
        exchange = ccxt_to_table[ccxt_id]
    else:
        return None

    symbol = pair_part.replace("_", "/").upper()
    if "/" not in symbol:
        return None
    market_type = "swap" if ":" in pair_part else "spot"
    return PairTable(
        db_name=db_name,
        table=name,
        symbol=symbol,
        exchange=exchange,
        ccxt_id=ccxt_id,
        market_type=market_type,
    )


def find_max_low_to_high_move(
    bars: Iterable[tuple[int, float, float]],
) -> Optional[GrowthMove]:
    """Return the strongest chronological low-to-high move in ordered bars.

    ``bars`` contains ``(timestamp_seconds, low, high)``. The running minimum
    is updated before evaluating the same bar's high; this allows a single
    15-minute candle's full range to count, as the source data cannot reveal the
    intrabar sequence.
    """
    min_low = math.inf
    min_low_ts: Optional[int] = None
    best: Optional[GrowthMove] = None

    for raw_ts, raw_low, raw_high in bars:
        try:
            ts = int(raw_ts)
            low = float(raw_low)
            high = float(raw_high)
        except (TypeError, ValueError, OverflowError):
            continue
        if not (math.isfinite(low) and math.isfinite(high)) or low <= 0 or high <= 0:
            continue
        if low < min_low:
            min_low = low
            min_low_ts = ts
        if min_low_ts is None or high < min_low:
            continue
        pct = (high / min_low - 1.0) * 100.0
        if not math.isfinite(pct):
            continue
        if best is None or pct > best.pct:
            best = GrowthMove(
                start_ts=min_low_ts,
                peak_ts=ts,
                low_price=min_low,
                high_price=high,
                pct=pct,
            )
    return best


def compute_daily_gerchik_atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    *,
    period: int,
    small_threshold: float = 0.5,
    large_threshold: float = 1.8,
) -> float:
    """Apply the collector's trimmed Gerchik-style true-range mean to 1D bars.

    Pass at least ``period + 1`` closed daily bars so the first true range in
    the selected ATR window has the preceding close available.
    """
    try:
        h = [float(x) for x in highs]
        l = [float(x) for x in lows]
        c = [float(x) for x in closes]
    except (TypeError, ValueError, OverflowError):
        return 0.0
    n = min(len(h), len(l), len(c))
    if period < 2 or n < period + 1:
        return 0.0
    h, l, c = h[-n:], l[-n:], c[-n:]
    if any(not math.isfinite(x) for seq in (h, l, c) for x in seq):
        return 0.0

    true_ranges: list[float] = []
    for i in range(n):
        prev_close = c[i - 1] if i > 0 else c[0]
        tr = max(h[i] - l[i], abs(h[i] - prev_close), abs(l[i] - prev_close))
        if not math.isfinite(tr):
            tr = 0.0
        true_ranges.append(tr)

    window = true_ranges[-period:]
    current = sum(window) / len(window)
    if not math.isfinite(current) or current <= 0:
        return 0.0
    for _ in range(10):
        valid = [
            tr for tr in window
            if small_threshold * current <= tr <= large_threshold * current
        ]
        if not valid:
            break
        new_atr = sum(valid) / len(valid)
        if not math.isfinite(new_atr) or new_atr <= 0:
            break
        if abs(new_atr - current) / max(abs(current), 1e-12) < 0.01:
            current = new_atr
            break
        current = new_atr
    return max(float(current), 0.0)


def calculate_daily_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    now_ts: int,
    atr_period: int,
    small_threshold: float = 0.5,
    large_threshold: float = 1.8,
) -> Optional[dict[str, float]]:
    """Compute min 7d dollar volume and daily Gerchik ATR from closed bars.

    ``rows`` may be ascending or descending and may include the currently
    forming daily candle; the latter is always excluded. A full set of seven
    closed bars is required for the volume filter.
    """
    cutoff = int(now_ts) - BAR_1D_SEC
    closed: list[Mapping[str, Any]] = []
    for row in rows:
        try:
            ts = int(row["Timestamp"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if ts <= cutoff:
            closed.append(row)
    closed.sort(key=lambda row: int(row["Timestamp"]))

    required = max(7, int(atr_period) + 1)
    if atr_period < 2 or len(closed) < required:
        return None

    daily_dollar_volumes: list[float] = []
    for row in closed[-7:]:
        try:
            value = float(row["volume"]) * float(row["low"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value) or value < 0:
            return None
        daily_dollar_volumes.append(value)

    atr_rows = closed[-required:]
    try:
        atr = compute_daily_gerchik_atr(
            [float(row["high"]) for row in atr_rows],
            [float(row["low"]) for row in atr_rows],
            [float(row["close"]) for row in atr_rows],
            period=int(atr_period),
            small_threshold=small_threshold,
            large_threshold=large_threshold,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(atr) or atr <= 0:
        return None
    return {
        "min_7d_volume_usd": min(daily_dollar_volumes),
        "daily_gerchik_atr": atr,
    }


async def list_public_tables(pool: asyncpg.Pool) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )
    return [str(row["table_name"]) for row in rows]


async def fetch_growth_move(
    pool: asyncpg.Pool,
    table: str,
    *,
    window_start_ts: int,
    now_ts: int,
) -> Optional[GrowthMove]:
    sql = (
        f'SELECT "Timestamp", "low", "high" FROM {_quote_ident(table)} '
        'WHERE "Timestamp" >= $1 AND "Timestamp" <= $2 '
        'ORDER BY "Timestamp" ASC'
    )
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, int(window_start_ts), int(now_ts))
    return find_max_low_to_high_move(
        (row["Timestamp"], row["low"], row["high"]) for row in rows
    )


async def fetch_daily_rows(
    pool: asyncpg.Pool,
    table: str,
    *,
    closed_cutoff_ts: int,
    limit: int,
) -> list[Mapping[str, Any]]:
    sql = (
        f'SELECT "Timestamp", "high", "low", "close", "volume" '
        f'FROM {_quote_ident(table)} WHERE "Timestamp" <= $1 '
        'ORDER BY "Timestamp" DESC LIMIT $2'
    )
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, int(closed_cutoff_ts), int(limit))
    return list(rows)


async def open_pools(db_names: Sequence[str], *, max_size: int) -> dict[str, asyncpg.Pool]:
    pools: dict[str, asyncpg.Pool] = {}
    for db_name in dict.fromkeys(name for name in db_names if name):
        try:
            pools[db_name] = await asyncpg.create_pool(
                user=settings.db_user,
                password=settings.db_password,
                host=settings.db_host,
                port=settings.db_port,
                database=db_name,
                min_size=1,
                max_size=max(1, int(max_size)),
            )
        except Exception as exc:
            LOGGER.error("Не удалось подключиться к БД %s: %r", db_name, exc)
    return pools


class LiveExchangeRegistry:
    """Lazily create one rate-limited CCXT client per exchange."""

    def __init__(self, markets_timeout_sec: float) -> None:
        self.markets_timeout_sec = max(1.0, float(markets_timeout_sec))
        self.clients: dict[str, Any] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.failed: set[str] = set()

    async def get(self, ccxt_id: str) -> Any:
        ccxt_id = str(ccxt_id).lower()
        if ccxt_id in self.failed:
            return None
        if ccxt_id in self.clients:
            return self.clients[ccxt_id]
        lock = self.locks.setdefault(ccxt_id, asyncio.Lock())
        async with lock:
            if ccxt_id in self.failed:
                return None
            if ccxt_id in self.clients:
                return self.clients[ccxt_id]

            try:
                exchange = create_exchange(ccxt_id)
            except Exception as exc:
                LOGGER.warning("Не удалось создать CCXT-клиент %s: %r", ccxt_id, exc)
                self.failed.add(ccxt_id)
                return None

            loaded = False
            last_error: Optional[BaseException] = None
            for attempt in range(2):
                try:
                    await hard_wait_for(
                        exchange.load_markets(reload=bool(attempt)),
                        self.markets_timeout_sec,
                        label=f"{ccxt_id}.load_markets",
                    )
                    loaded = bool(getattr(exchange, "markets", None))
                    if loaded:
                        break
                    last_error = RuntimeError("load_markets returned an empty market list")
                except Exception as exc:
                    last_error = exc
                if attempt == 0:
                    await asyncio.sleep(2.0)

            if not loaded:
                LOGGER.warning("Не удалось загрузить рынки %s: %r", ccxt_id, last_error)
                await close_exchange_safely(exchange, ccxt_id)
                self.failed.add(ccxt_id)
                return None

            self.clients[ccxt_id] = exchange
            return exchange

    async def close(self) -> None:
        clients, self.clients = list(self.clients.items()), {}
        await asyncio.gather(
            *(close_exchange_safely(exchange, name) for name, exchange in clients),
            return_exceptions=True,
        )


async def _collect_move_candidates(
    pair_tables: Sequence[PairTable],
    pools: Mapping[str, asyncpg.Pool],
    *,
    window_start_ts: int,
    now_ts: int,
    min_growth_pct: float,
    concurrency: int,
    stats: Counter,
) -> list[Candidate]:
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    tasks: list[asyncio.Task] = []

    async def worker(pair: PairTable) -> Optional[Candidate]:
        pool = pools.get(pair.db_name)
        if pool is None:
            stats["missing_15m_database"] += 1
            return None
        async with semaphore:
            try:
                move = await fetch_growth_move(
                    pool,
                    pair.table,
                    window_start_ts=window_start_ts,
                    now_ts=now_ts,
                )
            except Exception as exc:
                stats["ohlcv_read_error"] += 1
                LOGGER.debug("Ошибка чтения %s.%s: %r", pair.db_name, pair.table, exc)
                return None
        if move is None:
            stats["no_usable_15m_bars"] += 1
            return None
        if move.pct <= min_growth_pct:
            stats["growth_below_threshold"] += 1
            return None
        stats["growth_pass"] += 1
        return Candidate(pair=pair, move=move)

    tasks = [asyncio.create_task(worker(pair)) for pair in pair_tables]
    candidates: list[Candidate] = []
    total = len(tasks)
    for done, task in enumerate(asyncio.as_completed(tasks), start=1):
        candidate = await task
        if candidate is not None:
            candidates.append(candidate)
        if done % 500 == 0 or done == total:
            LOGGER.info("15m-пары: проверено %d/%d, рост прошли: %d",
                        done, total, len(candidates))
    return candidates


async def _daily_metrics_for_pair(
    pair: PairTable,
    daily_pools: Mapping[str, asyncpg.Pool],
    daily_databases_by_table: Mapping[str, Sequence[str]],
    *,
    now_ts: int,
    atr_period: int,
    daily_limit: int,
) -> Optional[dict[str, float]]:
    databases = daily_databases_by_table.get(pair.table, ())
    if not databases:
        return None
    cutoff = int(now_ts) - BAR_1D_SEC
    best_rows: list[Mapping[str, Any]] = []
    best_latest = -1
    for db_name in databases:
        pool = daily_pools.get(db_name)
        if pool is None:
            continue
        try:
            rows = await fetch_daily_rows(
                pool,
                pair.table,
                closed_cutoff_ts=cutoff,
                limit=daily_limit,
            )
        except Exception as exc:
            LOGGER.debug("Ошибка чтения дневных баров %s.%s: %r", db_name, pair.table, exc)
            continue
        latest = int(rows[0]["Timestamp"]) if rows else -1
        if latest > best_latest:
            best_rows, best_latest = rows, latest
    if not best_rows:
        return None
    return calculate_daily_metrics(
        best_rows,
        now_ts=now_ts,
        atr_period=atr_period,
        small_threshold=float(settings.atr_small_threshold),
        large_threshold=float(settings.atr_large_threshold),
    )


async def _add_daily_metrics(
    candidates: Sequence[Candidate],
    daily_pools: Mapping[str, asyncpg.Pool],
    daily_databases_by_table: Mapping[str, Sequence[str]],
    *,
    now_ts: int,
    atr_period: int,
    min_7d_volume_usd: float,
    concurrency: int,
    stats: Counter,
) -> list[Candidate]:
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    daily_limit = max(7, int(atr_period) + 1)

    async def worker(candidate: Candidate) -> Optional[Candidate]:
        async with semaphore:
            metrics = await _daily_metrics_for_pair(
                candidate.pair,
                daily_pools,
                daily_databases_by_table,
                now_ts=now_ts,
                atr_period=atr_period,
                daily_limit=daily_limit,
            )
        if metrics is None:
            stats["missing_or_insufficient_daily_data"] += 1
            return None
        volume = metrics["min_7d_volume_usd"]
        if volume < min_7d_volume_usd:
            stats["min_7d_volume_below_threshold"] += 1
            return None
        candidate.min_7d_volume_usd = volume
        candidate.daily_gerchik_atr = metrics["daily_gerchik_atr"]
        stats["daily_filters_pass"] += 1
        return candidate

    tasks = [asyncio.create_task(worker(candidate)) for candidate in candidates]
    result: list[Candidate] = []
    for task in asyncio.as_completed(tasks):
        candidate = await task
        if candidate is not None:
            result.append(candidate)
    return result


async def _apply_live_filters(
    candidates: Sequence[Candidate],
    registry: LiveExchangeRegistry,
    *,
    max_spread_atr_pct: float,
    concurrency: int,
    request_timeout_sec: float,
    stats: Counter,
) -> list[Candidate]:
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def worker(candidate: Candidate) -> Optional[Candidate]:
        pair = candidate.pair
        async with semaphore:
            exchange = await registry.get(pair.ccxt_id)
            if exchange is None:
                stats["exchange_unavailable"] += 1
                return None
            markets = getattr(exchange, "markets", None) or {}
            if not markets.get(pair.symbol):
                stats["symbol_not_listed_on_exchange"] += 1
                return None
            try:
                snapshot = await fetch_orderbook_snapshot(
                    exchange,
                    pair.symbol,
                    candidate.daily_gerchik_atr,
                    fetch_limit=int(settings.ob_fetch_limit),
                    trades_limit=int(settings.ob_trades_limit),
                    trades_window_sec=int(settings.ob_trades_window_sec),
                    depth_pct=float(settings.ob_depth_pct),
                    fallback_limits=list(settings.ob_fallback_limits),
                    timeout_sec=float(request_timeout_sec),
                )
            except Exception as exc:
                LOGGER.debug("Live OB ошибка %s %s: %r", pair.exchange, pair.symbol, exc)
                snapshot = None

        if not snapshot:
            stats["live_orderbook_unavailable"] += 1
            return None

        bid = _finite_positive(snapshot.get("ob_best_bid"))
        ask = _finite_positive(snapshot.get("ob_best_ask"))
        if bid is None or ask is None or ask < bid:
            stats["invalid_live_orderbook"] += 1
            return None

        grade = str(snapshot.get("ob_vitality_grade") or "").strip().upper()
        if grade not in {"A", "B"}:
            stats["vitality_not_a_or_b"] += 1
            return None

        spread_atr_pct = _finite_nonnegative(snapshot.get("ob_spread_atr_pct"))
        if spread_atr_pct is None or spread_atr_pct > max_spread_atr_pct:
            stats["spread_over_atr_limit"] += 1
            return None

        candidate.live_snapshot = snapshot
        stats["live_filters_pass"] += 1
        return candidate

    tasks = [asyncio.create_task(worker(candidate)) for candidate in candidates]
    result: list[Candidate] = []
    for task in asyncio.as_completed(tasks):
        candidate = await task
        if candidate is not None:
            result.append(candidate)
    return result


def _finite_positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _finite_nonnegative(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _parse_exchange_filter(raw: Optional[str]) -> Optional[set[str]]:
    if raw is None or not raw.strip():
        return None
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _pair_universe(
    tables_by_db: Mapping[str, Sequence[str]],
    *,
    requested_exchanges: Optional[set[str]],
) -> list[PairTable]:
    mapping = {str(k).lower(): str(v).lower() for k, v in settings.exchange_map_15m.items()}
    enabled = set(settings.filter_exchange_ids(mapping.keys()))
    pairs: list[PairTable] = []
    seen: set[tuple[str, str, str]] = set()

    for db_name, tables in tables_by_db.items():
        for table in tables:
            pair = parse_pair_table(table, db_name)
            if pair is None or pair.exchange not in enabled:
                continue
            if requested_exchanges and not ({pair.exchange, pair.ccxt_id} & requested_exchanges):
                continue
            key = (pair.exchange, pair.symbol, pair.market_type)
            if key in seen:
                # A pair should live in only one volume-tier database. If stale
                # duplicate tables exist, scan the first catalog occurrence once.
                continue
            seen.add(key)
            pairs.append(pair)
    return pairs


def _format_ts(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _print_results(results: Sequence[Candidate], stats: Counter, args: argparse.Namespace) -> None:
    print(
        f"\nРост: max low→high за {args.hours:g} ч > {args.pct:g}% | "
        f"min 7d vol×low ≥ ${args.min_7d_volume_usd:,.0f} | "
        f"живой vitality A/B | spread ≤ {args.max_spread_atr_pct:g}% 1D Gerchik ATR"
    )
    print(f"Найдено пар: {len(results)}")
    print("=" * 110)
    if not results:
        print("Подходящих пар нет.")
    else:
        for candidate in sorted(results, key=lambda item: item.move.pct, reverse=True):
            pair, move = candidate.pair, candidate.move
            snap = candidate.live_snapshot or {}
            grade = str(snap.get("ob_vitality_grade") or "?")
            score = float(snap.get("ob_vitality_score") or 0.0)
            depth = float(snap.get("ob_total_depth_usd") or 0.0)
            tpm = float(snap.get("ob_trades_per_min") or 0.0)
            spread_atr = float(snap.get("ob_spread_atr_pct") or 0.0)
            print(
                f"{pair.exchange:<8} {pair.market_type:<5} {pair.symbol:<24} "
                f"+{move.pct:8.2f}%  low {move.low_price:.10g} @ {_format_ts(move.start_ts)} "
                f"→ high {move.high_price:.10g} @ {_format_ts(move.peak_ts)}\n"
                f"           min7d ${candidate.min_7d_volume_usd:,.0f}  "
                f"1D-Gerchik ATR {candidate.daily_gerchik_atr:.10g}  "
                f"spread {spread_atr:.2f}% ATR  vitality {grade} ({score:.1f}/10)  "
                f"depth±{settings.ob_depth_pct:g}% ${depth:,.0f}  tape {tpm:.1f}/min"
            )
    print("=" * 110)
    if stats:
        print("Промежуточная статистика: " + ", ".join(
            f"{name}={count}" for name, count in sorted(stats.items())
        ))


async def run_scan(args: argparse.Namespace) -> int:
    db_15m = [settings.db_high_15m, settings.db_low_15m]
    db_1d = [settings.db_high_1d, settings.db_low_1d]
    pool_names = list(dict.fromkeys(db_15m + db_1d))
    max_pool_size = min(max(1, args.db_concurrency), max(1, int(settings.db_max_pool_size)))
    pools = await open_pools(pool_names, max_size=max_pool_size)
    if not any(name in pools for name in db_15m):
        LOGGER.error("Не удалось подключиться ни к одной 15m OHLCV-базе.")
        for pool in pools.values():
            await pool.close()
        return 1

    registry = LiveExchangeRegistry(float(settings.exchange_markets_load_sec))
    stats: Counter = Counter()
    try:
        tables_by_db: dict[str, list[str]] = {}
        for db_name in db_15m:
            pool = pools.get(db_name)
            if pool is None:
                continue
            try:
                tables_by_db[db_name] = await list_public_tables(pool)
            except Exception as exc:
                LOGGER.error("Не удалось прочитать каталог %s: %r", db_name, exc)
                tables_by_db[db_name] = []

        requested_exchanges = _parse_exchange_filter(args.exchanges)
        pair_tables = _pair_universe(
            tables_by_db,
            requested_exchanges=requested_exchanges,
        )
        LOGGER.info("Таблицы 15m: %d пар(ы) в HIGH/LOW, окно %g ч.",
                    len(pair_tables), args.hours)
        if not pair_tables:
            _print_results([], stats, args)
            return 0

        now_ts = int(time.time())
        window_start = int(now_ts - args.hours * 3600.0)
        growth_candidates = await _collect_move_candidates(
            pair_tables,
            pools,
            window_start_ts=window_start,
            now_ts=now_ts,
            min_growth_pct=args.pct,
            concurrency=args.db_concurrency,
            stats=stats,
        )
        LOGGER.info("Порог роста прошли: %d из %d.", len(growth_candidates), len(pair_tables))
        if not growth_candidates:
            _print_results([], stats, args)
            return 0

        daily_tables_by_db: dict[str, list[str]] = {}
        for db_name in db_1d:
            pool = pools.get(db_name)
            if pool is None:
                continue
            try:
                daily_tables_by_db[db_name] = await list_public_tables(pool)
            except Exception as exc:
                LOGGER.error("Не удалось прочитать каталог %s: %r", db_name, exc)
                daily_tables_by_db[db_name] = []
        daily_databases_by_table: dict[str, list[str]] = {}
        for db_name, tables in daily_tables_by_db.items():
            for table in tables:
                if "_on_" in table:
                    daily_databases_by_table.setdefault(table.lower(), []).append(db_name)

        daily_candidates = await _add_daily_metrics(
            growth_candidates,
            pools,
            daily_databases_by_table,
            now_ts=now_ts,
            atr_period=args.atr_period,
            min_7d_volume_usd=args.min_7d_volume_usd,
            concurrency=args.db_concurrency,
            stats=stats,
        )
        LOGGER.info("Проверку дневных объёма и ATR прошли: %d из %d.",
                    len(daily_candidates), len(growth_candidates))
        if not daily_candidates:
            _print_results([], stats, args)
            return 0

        live_candidates = await _apply_live_filters(
            daily_candidates,
            registry,
            max_spread_atr_pct=args.max_spread_atr_pct,
            concurrency=args.live_concurrency,
            request_timeout_sec=args.request_timeout_sec,
            stats=stats,
        )
        _print_results(live_candidates, stats, args)
        return 0
    finally:
        await registry.close()
        await asyncio.gather(
            *(pool.close() for pool in pools.values()),
            return_exceptions=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Найти отслеживаемые пары с максимальным ростом low→high "
            "за последние N часов, затем проверить 7d $объём и живой стакан."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hours", type=float, required=True,
                        help="Окно поиска роста в часах (например, 6 или 12)")
    parser.add_argument("--pct", type=float, required=True,
                        help="Строгий порог: рост должен быть больше этого процента")
    parser.add_argument(
        "--min-7d-volume-usd", type=float, default=DEFAULT_MIN_7D_VOLUME_USD,
        help="Минимум из volume×low за последние 7 закрытых дневных баров, USD",
    )
    parser.add_argument(
        "--atr-period", type=int, default=int(settings.atr_period),
        help="Период дневного Gerchik ATR; по умолчанию берётся из ATR_PERIOD",
    )
    parser.add_argument(
        "--max-spread-atr-pct", type=float, default=DEFAULT_MAX_SPREAD_ATR_PCT,
        help="Максимальный live spread в процентах от дневного Gerchik ATR",
    )
    parser.add_argument(
        "--exchanges", default=None,
        help="Необязательный список бирж через запятую (например, bybit,gateio)",
    )
    parser.add_argument("--db-concurrency", type=int, default=8,
                        help="Максимум одновременных чтений таблиц PostgreSQL")
    parser.add_argument("--live-concurrency", type=int, default=4,
                        help="Максимум одновременных live-проверок стакана")
    parser.add_argument("--request-timeout-sec", type=float, default=8.0,
                        help="Timeout каждого live-запроса стакана/ленты, секунд")
    parser.add_argument("--verbose", action="store_true", help="Подробные логи в stderr")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.hours <= 0:
        parser.error("--hours должен быть больше 0")
    if args.pct < 0:
        parser.error("--pct не может быть отрицательным")
    if args.min_7d_volume_usd < 0:
        parser.error("--min-7d-volume-usd не может быть отрицательным")
    if args.atr_period < 2:
        parser.error("--atr-period должен быть не меньше 2")
    if args.max_spread_atr_pct < 0:
        parser.error("--max-spread-atr-pct не может быть отрицательным")
    if args.db_concurrency < 1 or args.live_concurrency < 1:
        parser.error("--db-concurrency и --live-concurrency должны быть не меньше 1")
    if args.request_timeout_sec <= 0:
        parser.error("--request-timeout-sec должен быть больше 0")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return asyncio.run(run_scan(args))
    except KeyboardInterrupt:
        LOGGER.warning("Сканирование прервано пользователем.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
