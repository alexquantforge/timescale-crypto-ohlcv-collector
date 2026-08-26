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
import time
from typing import List, Optional, Tuple

# A "Timestamp" column stores epoch SECONDS (~1.7e9 today). Values above
# 1e11 can only be epoch MILLISECONDS written by a legacy buggy import
# (that is the mixed-epoch data the dashboard has to special-case). Such
# rows poison every cursor derived from MIN()/MAX(): catch-up asks the
# exchange for candles of the year 57000 (gets nothing -> "+0 forever")
# and history-prefill computes an absurd `since`, then latches as if the
# exchange had nothing older. Normalizing on READ repairs the cursors,
# and the next save (`DELETE WHERE "Timestamp" >= min_new_ts`) physically
# replaces the ms-rows with proper seconds rows.
MS_EPOCH_THRESHOLD_SEC = 100_000_000_000


def normalize_epoch_sec(ts: Optional[int]) -> Optional[int]:
    """None -> None; epoch-ms -> epoch-sec; epoch-sec -> unchanged."""
    if ts is None:
        return None
    t = int(ts)
    if t > MS_EPOCH_THRESHOLD_SEC:
        return t // 1000
    return t


def is_transient_fetch_error(e: Exception) -> bool:
    """
    True for RATE-LIMIT / transient network failures (ccxt.NetworkError
    family: RateLimitExceeded, DDoSProtection, ExchangeNotAvailable,
    RequestTimeout... plus local timeouts): the exchange asked us to slow
    down, not "no data" — the caller must NOT cooldown-latch the pair for
    hours, it should simply retry on the next engine cycle. Hard failures
    (BadRequest, "symbol is not found", BadSymbol) return False.
    Lazily imports ccxt so this module keeps zero import-time deps.
    """
    import asyncio

    import ccxt

    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
        return True
    return isinstance(e, ccxt.NetworkError)


def prefill_empty_action(
    batch,
    oldest_ts_sec: int,
    step_sec: int,
    span: int,
) -> Tuple[str, object]:
    """
    Decide what to do when a prefill page yielded ZERO usable older rows.

    Exchanges disagree on out-of-range `since`:
      * most CLAMP pre-listing `since` to the listing date (the batch then
        starts right at/after the table start -> nothing older exists);
      * some return an EMPTY array whenever `since` is before their first
        candle (bingx/mexc style). One empty big-window page then proves
        NOTHING about candles between the listing and the table start —
        the window must be PROBED smaller and smaller before declaring the
        history exhausted;
      * some ignore a too-old `since` and return the LATEST candles (batch
        disconnected from the table start).

    Returns:
      ("terminal", reason:str) — history exhausted, latch the pair;
      ("shrink", new_span:int) — retry the same table start with a
      geometrically smaller window (span candles), bounded by max_pages.
    """
    if batch:
        try:
            batch_min_ms = min(int(c[0]) for c in batch)
        except (TypeError, ValueError, IndexError):
            batch_min_ms = None
        older_ms = (int(oldest_ts_sec) + 2 * int(step_sec)) * 1000
        if batch_min_ms is not None and batch_min_ms <= older_ms:
            # The batch visibly runs right up to the table start — the
            # exchange's data starts there (clamped pre-listing response).
            return "terminal", "exchange has nothing older"
    if int(span) > 1:
        return "shrink", max(1, int(span) // 2)
    if batch:
        return "terminal", "exchange has nothing older (verified by a 1-candle probe)"
    return "terminal", "exchange returned nothing (window probed down to 1 candle)"


def should_attempt_prefill(
    done_record: Optional[Tuple[int, float]],
    min_ts_sec: int,
    now_sec: Optional[float] = None,
    retry_after_sec: float = 4 * 3600,
) -> bool:
    """
    Resumable/cooldown gate for the backward history prefill.

    `done_record` is what the engine latched at the end of the previous
    attempt: (table_min_ts_seen_at_latch, unix_time_of_attempt) or None.
    Attempt again when:
      * never attempted (`done_record` is None);
      * the table start IMPROVED since the latch (progress happened — keep
        walking left immediately, do not wait out the cooldown);
      * the last attempt is older than `retry_after_sec` (transient fetch
        failures must not suppress the pair for the whole process run).
    A fresh latch on the SAME min_ts means "terminal state confirmed
    recently" (floor reached / exchange has nothing older / fetch keeps
    failing) — skip so we don't hammer the exchange every cycle.
    """
    if not done_record:
        return True
    latched_min, attempt_ts = done_record
    if int(latched_min) != int(min_ts_sec):
        return True
    now = time.time() if now_sec is None else float(now_sec)
    return (now - float(attempt_ts)) >= float(retry_after_sec)


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
