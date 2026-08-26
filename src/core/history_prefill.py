"""
Pure helpers for BACKWARD history prefill (history repair).

Spot tables accumulated full history over time, but perp tables in this
collector were often created by an initial import that did not paginate
(or broke on exchanges whose kline page size differs from FETCH_LIMIT),
so their tables start only a few days back while the exchange itself has
much deeper data. The engines therefore need a resumable way to fetch
candles OLDER than the current table start:

* `prefill_needed(min_ts, target_floor)` — is there anything to repair?
* `prefill_page_since_ms(...)` — `since` cursor for the next older page,
  None when the retention floor / exchange kline window has been reached
  (stop fetching — older candles are permanently out of scope/reach);
* `extract_older_rows(...)` — from a fetched batch, keep only rows that are
  strictly older than the table start and inside the retention window
  (deduped, ascending). An empty result means "no progress possible" —
  the caller marks the pair done instead of retrying forever.

The functions are pure (no I/O) so the pagination/stop logic is fully
unit-testable without an exchange or a database.
"""
from typing import List, Optional


def prefill_needed(min_ts_sec: Optional[int], target_floor_sec: int, slack_sec: int = 1800) -> bool:
    """True when the table starts noticeably LATER than the floor we want
    (15m: now-180d; 1D: backfill_start_date), i.e. older candles are missing."""
    if not min_ts_sec:
        return False
    return int(min_ts_sec) > int(target_floor_sec) + int(slack_sec)


def prefill_page_since_ms(
    oldest_ts_sec: int,
    step_sec: int,
    fetch_limit: int,
    exchange_floor_ms: Optional[int] = None,
    target_floor_sec: int = 0,
) -> Optional[int]:
    """
    `since` cursor (ms) for the next OLDER page: one full page of `step_sec`
    candles before the current table start, clamped to:
      * the exchange's kline window (Gate.io: ~10000 recent candles — older
        `from` values are REJECTED by the exchange), and
      * the retention/backfill floor (nothing older is stored anyway).
    Returns None when clamping collapses the page to zero width — the table
    is as deep as it's ever going to get.
    """
    oldest_ms = int(oldest_ts_sec) * 1000
    since_ms = oldest_ms - int(fetch_limit) * int(step_sec) * 1000
    if exchange_floor_ms is not None:
        if int(exchange_floor_ms) >= oldest_ms:
            return None
        since_ms = max(since_ms, int(exchange_floor_ms))
    since_ms = max(since_ms, int(target_floor_sec) * 1000)
    if since_ms >= oldest_ms:
        return None
    return since_ms


def extract_older_rows(
    batch,
    oldest_ts_sec: int,
    target_floor_sec: int = 0,
) -> List[list]:
    """
    From a fetched OHLCV batch keep rows STRICTLY older than `oldest_ts_sec`
    (the current table start) and not older than `target_floor_sec`.
    Deduped by timestamp and sorted ascending (ccxt already sorts ascending,
    but belt-and-braces against sloppy exchanges).
    """
    oldest_ms = int(oldest_ts_sec) * 1000
    floor_ms = int(target_floor_sec) * 1000
    seen = {}
    for c in batch or []:
        try:
            ts = int(c[0])
        except (TypeError, ValueError, IndexError):
            continue
        if floor_ms <= ts < oldest_ms:
            seen[ts] = c
    return [seen[k] for k in sorted(seen)]
