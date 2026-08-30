"""Open Interest + Funding Rate collection for perpetual contracts (15m & 1D).

Spot pairs have no OI/funding — both engines gate these calls on `":" in symbol`
(a ccxt unified perp like BTC/USDT:USDT).

Storage model (columns live on the same per-pair candle table, alongside the
ob_* metrics):

- ``open_interest`` / ``oi_ts`` — a point-in-time OI snapshot written onto the
  LATEST candle row every cycle. Over time this builds an OI time series at
  cycle frequency (exchange-native units: base amount or contracts, as ccxt
  returns them).
- ``funding_rate`` / ``funding_ts`` — realized funding events are backfilled
  ONCE per table per engine run (fetch_funding_rate_history, paged from the
  retention/backfill floor) onto the candle row at-or-before each 8h event;
  the current rate also lands on the latest row with the per-cycle snapshot.

Failure policy (same principle as the orderbook path): one loud WARNING per
pair per process, repeats go to debug. Column DDL is latched per table per
process — information_schema is queried at most once.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Set, Tuple

import asyncpg

logger = logging.getLogger("oi_funding")

ACQUIRE_TIMEOUT = 30.0
FETCH_TIMEOUT_SEC = 6.0

OI_FUNDING_COLUMNS_SQL: Dict[str, str] = {
    "open_interest": "DOUBLE PRECISION",
    "oi_ts": "BIGINT",
    "funding_rate": "DOUBLE PRECISION",
    "funding_ts": "BIGINT",
}

_ENSURED: Set[str] = set()   # table names whose columns were verified this run
_WARNED: Set[Tuple[str, str]] = set()  # (ccxt_id, symbol) — loud warn once per run
_BF_DONE: Set[str] = set()   # funding-history backfill latch per table per run


async def ensure_oi_funding_columns(conn: asyncpg.Connection, table_name: str) -> None:
    """Adds the OI/funding columns once per table per process; also casts them
    back from TEXT when an old HIGH↔LOW move garbled the types."""
    if table_name in _ENSURED:
        return
    from src.db.repository import fetch_column_types, repair_text_typed_columns

    col_types = await fetch_column_types(conn, table_name)
    for col, typ in OI_FUNDING_COLUMNS_SQL.items():
        if col not in col_types:
            await conn.execute(
                f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" {typ}'
            )
    await repair_text_typed_columns(
        conn, table_name, col_types, OI_FUNDING_COLUMNS_SQL, log=logger
    )
    _ENSURED.add(table_name)


def warn_once(ccxt_id: str, symbol: str, detail: str) -> None:
    """First failure of a pair is a visible WARNING; repeats are debug-only."""
    key = (ccxt_id, symbol)
    if key in _WARNED:
        logger.debug(f"[OI/FUNDING] {symbol} @{ccxt_id}: {detail}")
        return
    _WARNED.add(key)
    logger.warning(
        f"⚠️ [OI/FUNDING] {symbol} @{ccxt_id}: {detail} "
        f"(repeat errors for it are logged only in debug)"
    )


async def fetch_oi_funding_snapshot(
    exchange, symbol: str, ccxt_id: str
) -> Dict[str, Any]:
    """Fetches current OI and funding rate; per-metric tolerance — whatever a
    given exchange supports is captured, the rest is simply absent. Raises the
    first fetch error to the caller (it owns the loud logging)."""
    out: Dict[str, Any] = {}
    if exchange.has.get("fetchOpenInterest"):
        oi = await asyncio.wait_for(
            exchange.fetch_open_interest(symbol), FETCH_TIMEOUT_SEC
        )
        val = oi.get("openInterestAmount")
        if val is None:
            val = oi.get("openInterestValue")
        if val is not None:
            out["open_interest"] = float(val)
            ts = oi.get("timestamp")
            out["oi_ts"] = int(ts // 1000) if ts else int(time.time())
    if exchange.has.get("fetchFundingRate"):
        fr = await asyncio.wait_for(
            exchange.fetch_funding_rate(symbol), FETCH_TIMEOUT_SEC
        )
        rate = fr.get("fundingRate")
        if rate is not None:
            out["funding_rate"] = float(rate)
            ts = (
                fr.get("previousFundingTimestamp")
                or fr.get("fundingTimestamp")
                or fr.get("timestamp")
            )
            out["funding_ts"] = int(ts // 1000) if ts else int(time.time())
    return out


async def write_oi_funding_snapshot(
    pool: asyncpg.Pool, table_name: str, snap: Dict[str, Any]
) -> None:
    """Writes the snapshot onto the newest candle row (MAX Timestamp)."""
    if not snap:
        return
    async with pool.acquire(timeout=ACQUIRE_TIMEOUT) as conn:
        await ensure_oi_funding_columns(conn, table_name)
        sets = ", ".join(f'"{col}" = ${i + 1}' for i, col in enumerate(snap))
        # Latest TWO rows: the next cycle's refetch (DELETE+COPY of the fresh
        # tail) re-creates the max row with NULLs, erasing a max-row-only
        # snapshot — the just-closed second row is what survives as history.
        await conn.execute(
            f'UPDATE "{table_name}" SET {sets} '
            f'WHERE "Timestamp" IN ('
            f'SELECT "Timestamp" FROM "{table_name}" '
            f'ORDER BY "Timestamp" DESC LIMIT 2)',
            *snap.values(),
        )


async def backfill_funding_history(
    exchange,
    pool: asyncpg.Pool,
    table_name: str,
    symbol: str,
    ccxt_id: str,
    since_ts: int,
    max_pages: int = 25,
    limit: int = 100,  # OKX funding-history caps pages at 100
) -> int:
    """Imports realized funding events since ``since_ts`` (epoch sec).

    Runs at most once per table per engine run: a failed attempt is latched
    too (loud warning, retry on restart) — a rate-limited pair must not enter
    a hot every-cycle retry loop. Pagination is progress-based off the newest
    event ts, never off page size. The cursor resumes from MAX(funding_ts)
    already in the table, so a page-capped import continues on the next run
    instead of starting over.

    Each event lands on the candle row AT-OR-BEFORE its timestamp; funding
    events at 8h boundaries align with 15m candles exactly, and on 1D tables
    the day's last event wins (row keys are candle opens — events are not new
    rows).
    """
    if table_name in _BF_DONE or not exchange.has.get("fetchFundingRateHistory"):
        return 0
    _BF_DONE.add(table_name)

    cursor_ms = int(since_ts) * 1000
    live_edge_ms = int(time.time() * 1000) - 60_000
    total = 0
    async with pool.acquire(timeout=ACQUIRE_TIMEOUT) as conn:
        await ensure_oi_funding_columns(conn, table_name)
        max_have = await conn.fetchval(
            f'SELECT MAX("funding_ts") FROM "{table_name}"'
        )
        if max_have is not None:
            if int(max_have) >= live_edge_ms // 1000:
                return 0  # already at the live edge — nothing to resume
            cursor_ms = max(cursor_ms, (int(max_have) + 1) * 1000)
        for _ in range(max_pages):
            try:
                page = await asyncio.wait_for(
                    exchange.fetch_funding_rate_history(
                        symbol, since=cursor_ms, limit=limit
                    ),
                    FETCH_TIMEOUT_SEC * 2,
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ [OI/FUNDING] {symbol} @{ccxt_id}: funding history fetch "
                    f"failed ({type(e).__name__}: {e}) — {total} events imported so far, "
                    f"will retry on restart"
                )
                return total
            if not page:
                break
            events: List[Tuple[float, int]] = []
            newest_ms = cursor_ms
            for rec in page:
                ts = rec.get("timestamp")
                rate = rec.get("fundingRate")
                if ts is None or rate is None:
                    continue
                ts = int(ts)
                events.append((float(rate), ts // 1000))
                if ts > newest_ms:
                    newest_ms = ts
            if not events:
                break
            await conn.executemany(
                f'UPDATE "{table_name}" SET funding_rate = $1, funding_ts = $2 '
                f'WHERE "Timestamp" = ('
                f'SELECT MAX("Timestamp") FROM "{table_name}" WHERE "Timestamp" <= $2)',
                events,
            )
            total += len(events)
            if newest_ms >= live_edge_ms or newest_ms <= cursor_ms:
                break
            cursor_ms = newest_ms + 1
            await asyncio.sleep(0.05)
    if total:
        logger.info(
            f"📜 [OI/FUNDING] {symbol} @{ccxt_id}: funding history imported — "
            f"{total} events (since {time.strftime('%Y-%m-%d', time.gmtime(since_ts))})"
        )
    return total
