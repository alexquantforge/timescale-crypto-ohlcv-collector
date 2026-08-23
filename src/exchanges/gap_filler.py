"""
Gap filling module for detecting and backfilling missing candle buckets
(1 day for the 1d timeframe, 15 minutes for the 15m timeframe).
"""
import asyncio
import logging
import time
import ccxt
from typing import List, Set, Tuple
import numpy as np
import pandas as pd
from pytz import timezone as pytz_timezone

from src.exchanges.symbol_selector import get_exchange_url, get_swap_url

logger = logging.getLogger("gap_filler")
MSK_TZ = pytz_timezone("Europe/Moscow")
ALMATY_TZ = pytz_timezone("Asia/Almaty")


async def fill_history_gaps(
    exchange,
    symbol: str,
    ccxt_id: str,
    repository,
    bf_limit: int = 300,
    max_pages: int = 50,
    timeframe: str = "1d",
) -> int:
    """
    Checks stored candle buckets in DB for missing intervals and fetches the
    gap ranges from the exchange with strict network timeouts. One bucket =
    one candle of the given timeframe (1 day for '1d', 15 minutes for '15m').
    """
    step_sec = 900 if timeframe == "15m" else 86400
    step_ms = step_sec * 1000

    existing_days = await repository.get_stored_days(symbol, ccxt_id, timeframe=timeframe)
    if len(existing_days) < 2:
        return 0

    days_arr = np.array(existing_days, dtype=np.int64)
    full_range = np.arange(days_arr[0], days_arr[-1] + 1, dtype=np.int64)
    missing = np.setdiff1d(full_range, days_arr, assume_unique=True)

    if len(missing) == 0:
        return 0

    boundaries = np.nonzero(np.diff(missing) > 1)[0]
    gap_ranges: List[Tuple[int, int]] = []
    start_idx = 0
    for b in boundaries:
        gap_ranges.append((int(missing[start_idx]), int(missing[b])))
        start_idx = b + 1
    gap_ranges.append((int(missing[start_idx]), int(missing[-1])))

    gap_days = set(int(d) for d in missing)
    all_gap_cs: List[list] = []
    seen_days: Set[int] = set()

    for r0, r1 in gap_ranges:
        cursor_day = r0
        for _ in range(max_pages):
            try:
                batch = await asyncio.wait_for(
                    exchange.fetch_ohlcv(symbol, timeframe, since=cursor_day * step_ms, limit=bf_limit),
                    timeout=8.0,
                )
            except Exception as e:
                logger.debug(f"Notice fetching gap OHLCV for {symbol}: {e}")
                break

            if not batch:
                break

            for c in batch:
                day = int(c[0]) // step_ms
                if day in gap_days and day not in seen_days:
                    seen_days.add(day)
                    all_gap_cs.append(c)

            last_day = int(batch[-1][0]) // step_ms
            if last_day >= r1 or len(batch) < bf_limit:
                break
            cursor_day = last_day + 1

    if not all_gap_cs:
        return 0

    is_swap = ":" in symbol
    spot_url = get_exchange_url(ccxt_id, symbol)
    swap_url = get_swap_url(ccxt_id, symbol)

    df = pd.DataFrame(
        all_gap_cs, columns=["ts", "open", "high", "low", "close", "volume"]
    )
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["ticker"] = symbol
    df["exchange"] = ccxt_id
    df["asset_type"] = "swap" if is_swap else "spot"
    df["url_trading"] = swap_url if is_swap else spot_url
    df["url_swap"] = None if is_swap else swap_url

    dt_utc = pd.to_datetime(df["ts"] // 1000, unit="s", utc=True)
    df["open_time_msk"] = dt_utc.dt.tz_convert(MSK_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["open_time_almaty"] = dt_utc.dt.tz_convert(ALMATY_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")

    inserted = await repository.upsert_ohlcv_batch(df, timeframe=timeframe)
    logger.info(f"Filled {inserted}/{len(missing)} missing {timeframe} gap buckets for {symbol} ({ccxt_id})")
    return inserted


async def fetch_ohlcv_catch_up(
    exchange,
    symbol: str,
    timeframe: str,
    since_sec: int,
    page_limit: int = 50,
    max_pages: int = 40,
    timeout: float = 6.0,
) -> List[list]:
    """
    Paged catch-up fetch: pulls ALL candles from `since_sec` up to now, page by
    page, so a lagging table synchronises fully in a single collector cycle
    (a single limit=50 request covers only 50 daily candles and used to leave
    the table crawling forward — or gapped — for months of downtime).
    """
    step_ms = (900 if timeframe == "15m" else 86400) * 1000
    now_ms = int(time.time() * 1000)
    cursor_ms = max(0, min(int(since_sec), now_ms // 1000)) * 1000

    collected: List[list] = []
    seen: Set[int] = set()

    for _ in range(max_pages):
        try:
            batch = await asyncio.wait_for(
                exchange.fetch_ohlcv(symbol, timeframe, since=cursor_ms, limit=page_limit),
                timeout=timeout,
            )
        except ccxt.BadSymbol:
            raise  # let the engine drop the delisted table
        except ccxt.ExchangeError as e:
            msg = str(e).lower()
            if any(t in msg for t in ("symbol is not found", "invalid symbol", "symbol_not_found", "100204", "48001")):
                raise  # delisted -> engine cleanup
            break
        except Exception:
            break
        if not batch:
            break

        for c in batch:
            ts = int(c[0])
            if ts not in seen:
                seen.add(ts)
                collected.append(c)

        last_ms = int(batch[-1][0])
        if last_ms >= now_ms or len(batch) < page_limit:
            break
        cursor_ms = last_ms + step_ms

    collected.sort(key=lambda c: c[0])
    return collected
