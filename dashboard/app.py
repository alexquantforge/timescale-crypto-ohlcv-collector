"""
Streamlit Web Dashboard for Timescale Crypto OHLCV Collector.
Visualizes live market liquidity, configurable ATR without paranormal bars,
live orderbook depth, CVD, Plotly charts, and native TradingView Lightweight Charts
from the 4 historical databases.

Performance-first design:
* Table summaries and candle frames are cached (`st.cache_data`), so switching
  pairs is instant instead of re-scanning every table in the 4 databases.
* The summary scan runs tables in parallel through an asyncpg connection pool.
* Charts load only the last N candles (configurable), never the full history.
* Live orderbook data never blocks the charts: charts render first, then metrics.

Layout: the Charts tab is first — 15m chart on top, 1D chart below, with
⏪ Prev / Next ⏩ buttons flanking each chart for instant pair cycling.
"""
import os
import re
import inspect
import sys
import json
import time
import asyncio
import threading
from typing import Optional

import asyncpg
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from src.analytics.atr_filtered import compute_atr_no_paranormal_bars
from src.analytics.orderbook import fetch_orderbook_snapshot
from src.exchanges.client import create_exchange, close_exchange_safely
from src.core.priority_pairs import publish_priority_pairs
from dashboard.helpers import (
    shift_option,
    exchanges_for_ticker,
    find_table_row,
    generate_demo_summary,
    generate_demo_candles,
    build_health_strip_html,
    build_pair_links_html,
    find_perp_ticker,
    sanitize_candle_frame,
    filter_sane_summary_rows,
    drop_stale_spot_duplicates,
    merge_intraday_into_daily,
    build_live_poller_js,
    build_history_loader_js,
    build_lightweight_chart_html,
    build_metric_chart_html,
    sanitize_metric_points,
    HIST_STATUS_HEIGHT,
    build_series_arrays,
    find_missing_bucket_ranges,
    rows_to_compact_candles,
    stitch_candle_gaps,
    fill_missing_bars,
    build_summary_union_sql,
    chart_render_plan,
    coerce_summary_types,
    feed_should_use,
    snapshot_refresh_due,
    scan_failure_is_transient,
    scan_retry_delay_sec,
    scan_pause_sec,
    chunked,
    snapshot_path,
    save_summary_snapshot,
    load_summary_snapshot,
)

st.set_page_config(
    page_title="Timescale Crypto OHLCV Collector",
    page_icon="📈",
    layout="wide",
)


# Whether the installed Streamlit's st.iframe still takes `scrolling`. Probed
# once (see _html_component); None = not probed yet.
_IFRAME_SCROLLING: Optional[bool] = None


def _html_component(html: str, height: int):
    """
    Renders raw HTML: st.iframe on new Streamlit, components.html on older ones.

    Current `st.iframe` has no `scrolling` parameter, so calling it with one
    raised TypeError on EVERY render, and the except-clause silently kept using
    the deprecated `components.v1.html` — a wall of "will be removed after
    2026-06-01" warnings per rerun (hundreds of them while browsing pairs).
    Ask the signature once instead of using exceptions as control flow: a
    TypeError thrown from inside a render must not double-render through the
    legacy path, and each component still needs ITS OWN height (only the
    keyword support is cached, never the kwargs).
    """
    global _IFRAME_SCROLLING
    iframe = getattr(st, "iframe", None)
    if iframe is None:
        components.html(html, height=height)
        return
    if _IFRAME_SCROLLING is None:
        try:
            allowed = set(inspect.signature(iframe).parameters)
        except (TypeError, ValueError):  # builtins without a signature
            allowed = set()
        _IFRAME_SCROLLING = "scrolling" in allowed
    if _IFRAME_SCROLLING:
        iframe(html, height=int(height), scrolling=False)
    else:
        iframe(html, height=int(height))

SUMMARY_COLUMNS = """
    ticker, exchange, asset_type,
    "Timestamp" as max_ts, close, volume,
    ob_vitality_score, ob_vitality_grade,
    ob_spread_abs, ob_spread_pct, ob_spread_atr_pct, ob_atr_no_paranormal,
    ob_best_bid, ob_best_ask, ob_bid_depth_usd, ob_ask_depth_usd,
    ob_cvd_5m, ob_total_depth_usd, ob_min_7d_volume_usd,
    ob_imbalance, ob_trades_per_min, ob_buy_pressure_pct, ob_is_barcode
"""

# Keys every summary row must expose. The scan reads `SELECT *` (NEVER a fixed
# column list): tables whose orderbook snapshot never landed simply LACK the
# ob_* columns, and a fixed SELECT used to raise UndefinedColumn -> the row
# was silently dropped -> "No 15m table for X" while the engine was happily
# storing 180 days of candles into that very table (PIXEL/USDT:USDT @bybit).
_EXPECTED_SUMMARY_KEYS = [
    "ticker", "exchange", "asset_type", "close", "volume",
    "ob_vitality_score", "ob_vitality_grade",
    "ob_spread_abs", "ob_spread_pct", "ob_spread_atr_pct", "ob_atr_no_paranormal",
    "ob_best_bid", "ob_best_ask", "ob_bid_depth_usd", "ob_ask_depth_usd",
    "ob_cvd_5m", "ob_total_depth_usd", "ob_min_7d_volume_usd",
    "ob_imbalance", "ob_trades_per_min", "ob_buy_pressure_pct", "ob_is_barcode",
]


async def _summary_row_for_table(
    pool, tbl: str, sem: asyncio.Semaphore, errors: list, timeout_sec: float = None,
    acquire_timeout: float = 10.0,
):
    """Last row of one candle table normalized to the summary schema; missing
    columns are padded with None. Returns None for empty/broken tables (broken
    ones are REPORTED into `errors`, never silently skipped)."""
    async with sem:
        try:
            # The wait for a connection is bounded by the SAME budget as the
            # query. With a hard 10 s here and 120 tables to walk, the recovery
            # loop kept queueing doomed work long after the scan's deadline:
            # the budget cut the sweep but not its tail, and that tail is the
            # load the collector had to compete with.
            async with pool.acquire(timeout=max(0.2, float(acquire_timeout))) as conn:
                fetch = conn.fetchrow(
                    f'SELECT * FROM "{tbl}" ORDER BY "Timestamp" DESC LIMIT 1'
                )
                row = await fetch if not timeout_sec else await asyncio.wait_for(
                    fetch, timeout=max(0.5, float(timeout_sec))
                )
        except Exception as e:
            errors.append((tbl, f"{type(e).__name__}: {e}"))
            return None
    if not row:
        return None
    d = dict(row)
    # historical alias used by every downstream consumer
    d.setdefault("max_ts", d.get("Timestamp"))
    for key in _EXPECTED_SUMMARY_KEYS:
        d.setdefault(key, None)
    return d


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

# Per-database / per-timeframe scan telemetry, filled in by the scan coroutine
# and read when deciding whether a snapshot may be persisted. Keyed by db name
# and by timeframe; a module-level dict because `asyncio.run` + the background
# refresh thread share one process, while st.cache_data would pickle away
# anything attached to the frame itself.
_SCAN_META: dict = {}
# When a timeframe was last (re)scanned — the throttle for the revalidation,
# so a page that reruns every second does not rescan every second. Written
# before the thread starts, not after it finishes: a slow scan must not invite
# a second one on top of it.
_LAST_SCAN_AT: dict = {}
# Consecutive TRUNCATED scans per timeframe, driving the retry backoff (see
# scan_retry_delay_sec). A pair list cut short by the budget has to be retried
# or the dashboard never shows the full list — but at a FIXED interval every
# retry adds load and guarantees the next scan is truncated too.
_SCAN_ATTEMPTS: dict = {}
# Last scan result per timeframe, in memory: {"df":…, "meta":…, "at":…}.
# A partial scan lives HERE and never on disk (the "last good snapshot" file
# must stay complete, or a busy collector shrinks the pair list from launch to
# launch). Serving it is what keeps a rerun off the database entirely.
_SUMMARY_STORE: dict = {}
# One scan per process. Before this, a scan of "15m" and a scan of "1d" each
# fanned out over both of their databases, i.e. up to 8 concurrent pools ×
# `dash_scan_pool_size` connections of UNION-ALL MAX() queries — against the
# same Postgres the collector is writing to. The scans then timed out, retried,
# and slowed each other down: measured on a production run, the same 116-chunk
# sweep took 19.9 s alone and 105 s when three other sweeps overlapped it.
_SCAN_GATE = threading.BoundedSemaphore(1)
# When the user last asked for a different pair. The summary sweep yields for a
# moment after it (scan_pause_sec), because a click must not queue behind
# thousands of catalog queries.
_LAST_INTERACTION_AT = 0.0


def _mark_interaction() -> None:
    global _LAST_INTERACTION_AT
    _LAST_INTERACTION_AT = time.time()



# table -> {column: data_type}, per (host, port, database), read from
# pg_catalog and kept for dash_scan_inventory_ttl_sec (see _table_inventory).
# Keyed by the server too: the sidebar can point the dashboard at another
# Postgres, and an inventory cached against the old one must never answer.
_SCAN_INVENTORY: dict = {}


async def _table_inventory(pool, db_name: str, db_host: str = "", db_port: int = 0) -> dict:
    """
    Which pair tables exist and what their columns are typed as.

    Three things this deliberately does NOT do the obvious way:

    * It reads pg_class/pg_attribute/pg_type instead of information_schema.
      information_schema.columns is a view that evaluates has_table_privilege()
      per column per table: measured at 30…250 s on a 14k-table database
      while the collector was writing — i.e. the catalog lookup, not the data,
      was the startup cost (and it was inside the timing, which is why a
      75-table database "scanned" in 8…24 s for one chunk query).
    * It selects only the projected columns (~23 per table, not all 30+), so
      the result is 8k×23 rows instead of 8k×30 and never carries candle data.
    * It is CACHED per database. The list of tables and their types changes on
      the order of minutes (a new listing, an engine migration adding ob_*
      columns), while the scan itself runs on every revalidation — reading the
      catalog that often is pure pressure on the same Postgres the collector is
      writing to.

    Types matter because a chunk of 120 heterogeneous tables is UNION ALL'ed:
    one TEXT-typed ob_* column (legacy HIGH<->LOW move) used to make PostgreSQL
    reject the whole chunk — see resolve_summary_union_casts.
    """
    ttl = float(settings.dash_scan_inventory_ttl_sec)
    cache_key = (db_host, db_port, db_name)
    ent = _SCAN_INVENTORY.get(cache_key)
    if ent and time.time() - ent["at"] < ttl:
        return ent["tables"]

    async with pool.acquire(timeout=20.0) as conn:
        col_rows = await conn.fetch(
            "SELECT c.relname AS table_name, a.attname AS column_name, "
            "       t.typname AS data_type "
            "FROM pg_catalog.pg_class c "
            "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid "
            "JOIN pg_catalog.pg_type t ON t.oid = a.atttypid "
            "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') "
            "  AND a.attnum > 0 AND NOT a.attisdropped "
            # '\_' = a literal underscore: '%_on_%' on its own lets '_' match
            # any character and pulls unrelated tables into the pair list.
            "  AND c.relname LIKE '%\\_on\\_%' "
            "  AND (a.attname = $1 OR a.attname = ANY($2::text[]))",
            "Timestamp", list(_EXPECTED_SUMMARY_KEYS),
        )

    tables: dict = {}
    for r in col_rows:
        tables.setdefault(r["table_name"], {})[r["column_name"]] = r["data_type"]
    _SCAN_INVENTORY[cache_key] = {"at": time.time(), "tables": tables}
    return tables


async def _scan_database(
    db_name: str, tier_label: str, db_host, db_port, db_user, db_pass,
    pool_size: int = None, chunk_size: int = None, budget_sec: float = None,
):
    """
    Last-row summary of every %_on_% table of one database.

    Batched on purpose: one query per `chunk_size` tables instead of one per
    table. With ~7.5k pairs the old per-table scan fired ~15k round trips at
    dashboard startup, which is why the page could spin for minutes while the
    collector was writing — the queries themselves are trivial, the round
    trips under concurrent write load are not.

    Batching only pays off while the batch query actually RUNS: every chunk is
    a UNION ALL over 120 heterogeneous tables, so a single TEXT-typed ob_*
    column (legacy HIGH<->LOW move, old import) used to kill the whole chunk
    with DatatypeMismatchError and silently drop it back to 120 round trips —
    the exact cost the batching was written to remove. Hence the scan reads
    information_schema TYPES as well, so `build_summary_union_sql` can flatten
    only the offending column, and a chunk that still fails retries once as an
    all-TEXT query before any per-table recovery is attempted.

    Bounded by `budget_sec`: whatever chunks answered in time are returned and
    the page renders, instead of the user staring at a spinner. The result is
    flagged in `_SCAN_META` so a truncated scan is never cached as the last
    good snapshot.

    A budget that only decides when to STOP LOGGING is not a budget: the sweep
    used to keep going through retries and per-table recovery for minutes after
    it expired (250–310 s per database on a loaded box), because a chunk already
    in flight waited 15 s for a pool connection and 30 s for its query. So the
    remaining budget is now threaded through every wait, a chunk that fails on
    LOAD is skipped instead of retried table by table, and recovery is capped by
    `dash_scan_recovery_max_tables`.
    """
    started = time.time()
    budget_sec = settings.dash_scan_budget_sec if budget_sec is None else budget_sec
    pool_size = settings.dash_scan_pool_size if pool_size is None else pool_size
    chunk_size = settings.dash_scan_chunk_size if chunk_size is None else chunk_size
    catalog_secs = 0.0
    try:
        pool = await asyncpg.create_pool(
            host=db_host, port=db_port, user=db_user, password=db_pass,
            database=db_name, min_size=1, max_size=pool_size, command_timeout=30,
            timeout=15,
        )
    except Exception as e:
        # A database the dashboard cannot even connect to is not an empty
        # database: say so, and mark the scan partial so nothing downstream
        # treats the shorter list as the truth.
        why = f"{type(e).__name__}: {e}"
        print(f"[scan] {db_name}: connect failed — {why}", flush=True)
        _SCAN_META[db_name] = {"tables": 0, "rows": 0, "chunks": 0, "skipped_chunks": 0,
                               "fallback_chunks": 0, "seconds": time.time() - started,
                               "sweep_seconds": 0.0, "catalog_seconds": 0.0,
                               "partial": True, "error": why}
        return []

    # asyncpg opens pool connections LAZILY, inside the first acquire().
    # The sweep cancels an acquire as soon as its budget is gone, and a
    # cancellation that lands mid-TLS-handshake is the asyncpg/uvloop race
    # behind "Fatal error on transport TCPTransport / InvalidStateError"
    # in the dashboard log. Opening the pool up front (uncancelled) means
    # the budget can only ever interrupt a QUERY. A warm-up failure is not
    # fatal: the connections are then simply created lazily as before.
    try:
        conns = await asyncio.wait_for(
            asyncio.gather(*[pool.acquire() for _ in range(max(1, int(pool_size)))],
                           return_exceptions=True),
            timeout=15.0,
        )
        for c in conns:
            if not isinstance(c, BaseException):
                await pool.release(c)
    except Exception:
        pass      # lazy creation as before — a warm-up is an optimization

    try:
        t_catalog = time.time()
        tables = await _table_inventory(pool, db_name, db_host, db_port)
        catalog_secs = max(catalog_secs, time.time() - t_catalog)
        if not tables:
            return []

        sem = asyncio.Semaphore(max(1, int(pool_size)))
        # The budget covers the DATA sweep. The catalog read above is cached
        # and was charged to the sweep before, which made the "25s budget
        # exhausted … in 76.0s" lines unreadable (76 s of catalog, 25 s of
        # budget, no contradiction after all).
        deadline = time.time() + float(budget_sec)
        errors: list = []
        skipped: list = []
        out: list = []
        recovery_cap = max(0, int(settings.dash_scan_recovery_max_tables))
        # Total time this sweep may spend yielding to clicks. A cap, so
        # interactivity never turns into "the pair list never completes".
        yield_left = [min(3.0, 0.2 * max(1.0, float(budget_sec)))]

        async def _fetch_union(sql: str) -> list:
            # Bounded by the SCAN budget, not only by the pool's
            # command_timeout: with 45 slow chunks and 6 connections,
            # 30s-per-query stalls add up to minutes while the page renders
            # nothing (asyncpg implements command_timeout the same way, so
            # cancelling here is a supported path, not a hack). A chunk that
            # cannot fit in what is left is not queued at all — a 0.5 s query
            # against a loaded server is a guaranteed TimeoutError that still
            # occupies a connection while it fails.
            #
            # The allowance is measured AFTER the semaphore, twice. With 69
            # chunks behind 6 connections, a chunk authorized when it was
            # queued used to run a 20 s query a minute and a half later: the
            # "25 s" sweep then took 142 s, and every chart query on that
            # database waited behind it — which is exactly the latency this
            # round is trying to remove.
            async with sem:
                left = deadline - time.time()
                if left < 1.0:
                    raise asyncio.TimeoutError("scan budget exhausted (queueing)")
                async with pool.acquire(timeout=min(5.0, left)) as conn:
                    left = deadline - time.time()
                    if left < 0.5:
                        raise asyncio.TimeoutError("scan budget exhausted (acquire)")
                    rows = await asyncio.wait_for(conn.fetch(sql), timeout=left)
            return [dict(r) for r in rows]

        async def run_chunk(chunk: dict):
            if time.time() > deadline:
                skipped.append(len(chunk))
                return []
            pause = scan_pause_sec(
                time.time(), _LAST_INTERACTION_AT,
                settings.dash_scan_yield_gap_sec, yield_left[0],
            )
            if pause > 0.0:
                yield_left[0] -= pause
                await asyncio.sleep(pause)
                if time.time() > deadline:
                    skipped.append(len(chunk))
                    return []
            sql = build_summary_union_sql(chunk, _EXPECTED_SUMMARY_KEYS)
            if not sql:
                return []
            try:
                return await _fetch_union(sql)
            except Exception as e:
                errors.append((f"chunk[{len(chunk)}]", f"{type(e).__name__}: {e}"))
                if scan_failure_is_transient(e):
                    skipped.append(len(chunk))
                    # LOADED, not broken. The all-TEXT retry and the per-table
                    # recovery would ask the same saturated pool for 120 more
                    # queries and answer nothing; the tables that did not make
                    # it stay one pass behind, which is visible in the
                    # "pair list incomplete" badge.
                    return []
                # A type mismatch (or a table the collector just dropped mid
                # -scan) must not cost the chunk its batching: retry ONCE with
                # every column flattened to TEXT. That query is type-stable by
                # construction, and the values are converted back in Python.
                retry_sql = build_summary_union_sql(
                    chunk, _EXPECTED_SUMMARY_KEYS, force_text=True
                )
                try:
                    return await _fetch_union(retry_sql)
                except Exception as e2:
                    errors.append((f"chunk[{len(chunk)}]:text", f"{type(e2).__name__}: {e2}"))
                    if scan_failure_is_transient(e2):
                        return []
            # Last resort: per-table reads, bounded by the remaining budget AND
            # by a table cap — an unbounded 120-query recovery is what turned a
            # slow startup into a minutes-long one.
            recovered = []
            for tbl in list(chunk)[:recovery_cap]:
                left = deadline - time.time()
                if left < 1.0:
                    break
                d = await _summary_row_for_table(
                    pool, tbl, sem, errors,
                    timeout_sec=min(8.0, max(0.5, left)),
                    acquire_timeout=min(3.0, max(0.2, left)),
                )
                if d:
                    d["table_name"] = tbl
                    recovered.append(d)
            return recovered

        chunks = list(chunked(tables, chunk_size))
        for result in await asyncio.gather(*[run_chunk(c) for c in chunks]):
            out.extend(result)

        for d in out:
            d["db_name"] = db_name
            d["volume_tier"] = tier_label

        secs = time.time() - started
        sweep_secs = max(0.0, secs - catalog_secs)
        # 'partial' means "this frame is NOT the truth about the database":
        # either the budget cut the sweep short, or tables reported errors while
        # rows are missing. Empty tables alone never set it — they are normal.
        partial = bool(skipped) or time.time() > deadline or (
            bool(errors) and len(out) < len(tables)
        )
        _SCAN_META[db_name] = {
            "tables": len(tables),
            "rows": len(out),
            "chunks": len(chunks),
            "skipped_chunks": len(skipped),
            "fallback_chunks": len([t for t, _ in errors if t.startswith("chunk[")]),
            "seconds": secs,
            "sweep_seconds": sweep_secs,
            "catalog_seconds": catalog_secs,
            "partial": partial,
        }
        tail = f" (+catalog {catalog_secs:.1f}s)" if catalog_secs > 1.0 else ""
        if errors:
            # Two very different things used to print as one alarming line: a
            # chunk the server was too busy to answer (skipped on purpose, by
            # design) and a chunk whose tables are actually broken (retried as
            # TEXT, then recovered table by table). Only the second is a bug.
            preview = "; ".join(f"{t}: {e}" for t, e in errors[:3])
            busy = len({t for t, e in errors if scan_failure_is_transient(str(e))})
            broken = len({t for t, e in errors if not scan_failure_is_transient(str(e))})
            print(
                f"[scan] {db_name}: {sweep_secs:.1f}s — {busy} chunk(s) skipped (db busy), "
                f"{broken} chunk(s) needed retry/recovery: {preview}{tail}",
                flush=True,
            )
        elif sweep_secs > 3.0:
            print(
                f"[scan] {db_name}: {len(chunks)} chunk(s) / {len(out)} tables in "
                f"{sweep_secs:.1f}s{tail}",
                flush=True,
            )
        if partial:
            print(
                f"[scan] {db_name}: {budget_sec:.0f}s budget exhausted — rendering "
                f"{len(out)}/{len(tables)} tables ({len(skipped)} chunk(s) skipped on "
                f"budget/busy db); the list will be retried with backoff",
                flush=True,
            )
        return out
    finally:
        await pool.close()


async def _load_summary(db_host, db_port, db_user, db_pass, timeframe: str) -> pd.DataFrame:
    if timeframe == "15m":
        dbs = [("HIGH", settings.db_high_15m), ("LOW", settings.db_low_15m)]
    else:
        dbs = [("HIGH", settings.db_high_1d), ("LOW", settings.db_low_1d)]

    # `dash_scan_max_parallel_dbs` = 1 (the default) walks the two databases of
    # a timeframe one after the other. Fanning them out halved the wall time of
    # an idle scan and multiplied it on a loaded one — every sweep was competing
    # with the others for the same Postgres the collector writes to.
    par = max(1, int(settings.dash_scan_max_parallel_dbs))
    if par == 1 or len(dbs) == 1:
        scans = [
            await _scan_database(db, tier, db_host, db_port, db_user, db_pass)
            for tier, db in dbs
        ]
    else:
        gate = asyncio.Semaphore(par)

        async def _one(tier, db):
            async with gate:
                return await _scan_database(db, tier, db_host, db_port, db_user, db_pass)

        scans = await asyncio.gather(*[_one(tier, db) for tier, db in dbs])
    all_rows = [r for scan in scans for r in scan]
    metas = [_SCAN_META.get(db, {}) for _, db in dbs]
    _SCAN_META[timeframe] = {
        "partial": bool(metas) and any(m.get("partial") for m in metas),
        "seconds": max((m.get("seconds") or 0.0) for m in metas) if metas else 0.0,
        "rows": sum(m.get("rows", 0) for m in metas),
        "tables": sum(m.get("tables", 0) for m in metas),
    }
    if not all_rows:
        return pd.DataFrame()
    # A chunk flattened to TEXT (type-stable UNION) hands back strings; give
    # the frame its numeric dtypes before anything sorts or formats it.
    return coerce_summary_types(pd.DataFrame(all_rows))


def _rescan_delay_sec(timeframe: str) -> float:
    """Backoff for this timeframe right now, from its truncated-scan streak."""
    return scan_retry_delay_sec(
        settings.dash_snapshot_refresh_sec, _SCAN_ATTEMPTS.get(timeframe, 0),
        settings.dash_scan_retry_max_sec,
    )


def _rescan_due(timeframe: str, now: float) -> bool:
    return snapshot_refresh_due(
        _LAST_SCAN_AT.get(timeframe, 0.0), now, _rescan_delay_sec(timeframe)
    )


def _scan_summary_now(db_host, db_port, db_user, db_pass, timeframe: str,
                      gate_sec: float = 0.0) -> pd.DataFrame:
    """Full scan + snapshot persist (used by both the sync and the background path).

    A scan cut short by the time budget is NOT persisted to disk: caching a
    truncated pair list as the 'last good snapshot' is how a busy collector
    quietly shrinks the dashboard from one launch to the next. It IS kept in
    `_SUMMARY_STORE` for the session — serving an incomplete list beats
    re-asking the database every rerun, which is the loop that was hammering
    the collector.

    `_SCAN_GATE` holds one scan per process. `gate_sec` says how long a caller
    on the render path may wait for it; the background path passes 0 and walks
    away, because a scan that could not start is exactly the load the app is
    trying not to add.
    """
    waited_before = (_SUMMARY_STORE.get(timeframe) or {}).get("at", 0.0)
    if not _SCAN_GATE.acquire(timeout=max(0.0, float(gate_sec))):
        ent = _SUMMARY_STORE.get(timeframe)
        print(f"[scan] {timeframe}: skipped — another scan holds the database", flush=True)
        return ent["df"] if ent else pd.DataFrame()
    if gate_sec and (_SUMMARY_STORE.get(timeframe) or {}).get("at", 0.0) > waited_before:
        # Somebody finished a scan while we were queueing for the gate: that IS
        # the answer, and scanning again would only add the load we waited for.
        _SCAN_GATE.release()
        return _SUMMARY_STORE[timeframe]["df"]
    try:
        df = asyncio.run(_load_summary(db_host, db_port, db_user, db_pass, timeframe))
        meta = dict(_SCAN_META.get(timeframe, {}))
        _SUMMARY_STORE[timeframe] = {"df": df, "meta": meta, "at": time.time()}
        # The streak decides when we dare to try again.
        _SCAN_ATTEMPTS[timeframe] = _SCAN_ATTEMPTS.get(timeframe, 0) + 1 if meta.get("partial") else 0
        if settings.dash_snapshot_enabled and not meta.get("partial"):
            save_summary_snapshot(
                snapshot_path(settings.dash_snapshot_dir, timeframe), df
            )
        return df
    finally:
        _SCAN_GATE.release()


def _scan_is_better(timeframe: str, before: dict) -> bool:
    """Whether the scan just stored deserves a rerun of the cached summary.

    Clearing the cache after EVERY scan is what re-armed the storm: the next
    rerun found no cache, scanned on the render path, got truncated again and
    launched another background scan. Only progress may cost a rerun.
    """
    after = (_SUMMARY_STORE.get(timeframe) or {}).get("meta") or {}
    if not after:
        return False
    if (before or {}).get("rows", -1) < after.get("rows", 0):
        return True
    return bool((before or {}).get("partial")) and not after.get("partial")


def _refresh_summary_in_background(db_host, db_port, db_user, db_pass, timeframe: str) -> None:
    """Rescans off the UI thread and drops the cache so the next rerun picks it up."""
    def _run():
        before = (_SUMMARY_STORE.get(timeframe) or {}).get("meta") or {}
        try:
            _scan_summary_now(db_host, db_port, db_user, db_pass, timeframe)
        except Exception:
            return
        if not _scan_is_better(timeframe, before):
            return
        try:
            load_summary_cached.clear()
        except Exception:
            pass

    _bg(("summary-scan", timeframe), _run)


@st.cache_data(ttl=600, show_spinner="📡 Scanning TimescaleDB tables…")
def load_summary_cached(db_host, db_port, db_user, db_pass, timeframe: str) -> pd.DataFrame:
    """
    Summary of the 2 databases for a timeframe (cached 10 min).

    Stale-while-revalidate: when a snapshot of the previous scan exists, it is
    returned IMMEDIATELY and the rescan runs in a daemon thread. A dashboard
    opened while the collector is mid-cycle then paints instantly instead of
    blocking on a full scan of every table.
    """
    now = time.time()

    def _serve(df, meta, age):
        """Paint what we have; retry later, always throttled."""
        st.session_state[f"_snap_age_{timeframe}"] = age
        if (meta or {}).get("partial"):
            st.session_state[f"_partial_scan_{timeframe}"] = (now, dict(meta))
        if _rescan_due(timeframe, now):
            # Marked at LAUNCH: a scan that takes 20 s must not invite a second
            # one 20 s later. This gate used to guard only the snapshot branch,
            # so on a box where scans kept coming back truncated (the usual
            # case while the collector writes) every rerun re-scanned.
            _LAST_SCAN_AT[timeframe] = now
            _refresh_summary_in_background(db_host, db_port, db_user, db_pass, timeframe)
        return df

    ent = _SUMMARY_STORE.get(timeframe)
    # An EMPTY frame is still an answer as long as the scan completed: without
    # this, a database that genuinely holds no pair tables would be re-scanned
    # on every rerun instead of rendering the "no tables" diagnostic.
    if ent is not None and (not ent["df"].empty or not (ent.get("meta") or {}).get("partial")):
        return _serve(ent["df"], ent.get("meta") or {}, now - float(ent.get("at", now)))

    if settings.dash_snapshot_enabled:
        snap, age = load_summary_snapshot(
            snapshot_path(settings.dash_snapshot_dir, timeframe),
            settings.dash_snapshot_max_age_sec,
        )
        if snap is not None and not snap.empty:
            _SUMMARY_STORE[timeframe] = {"df": snap, "meta": {}, "at": now - float(age or 0)}
            return _serve(snap, {}, age)

    # Nothing in memory and nothing on disk: this first scan is the one case
    # where the render path does wait for the database, bounded by its own
    # budget and by whoever holds the gate.
    try:
        df = _scan_summary_now(
            db_host, db_port, db_user, db_pass, timeframe,
            # The cold path may wait for an in-flight scan — for at most the
            # budget that scan itself was given, and then it serves whatever
            # exists rather than starting a third sweep.
            gate_sec=float(settings.dash_scan_budget_sec),
        )
        return df
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=15, show_spinner=False)
def _probe_db_error(db_host, db_port, db_user, db_pass):
    """Quick connect probe used by the 'no tables' diagnostic: returns the
    actual connection/auth error text instead of a generic warning."""

    async def _probe():
        try:
            conn = await asyncpg.connect(
                host=db_host, port=db_port, user=db_user, password=db_pass,
                database=settings.db_high_15m, timeout=10,
            )
            await conn.close()
            return None
        except Exception as e:
            return str(e) or type(e).__name__

    try:
        return asyncio.run(_probe())
    except Exception as e:
        return str(e) or type(e).__name__


async def _load_candles(db_name: str, table_name: str, limit: int, db_host, db_port, db_user, db_pass) -> pd.DataFrame:
    """Fetches only the last `limit` candles (DESC + reverse) instead of the full history."""
    conn = await asyncpg.connect(
        host=db_host, port=db_port, user=db_user, password=db_pass, database=db_name,
        timeout=15,
    )
    try:
        rows = await conn.fetch(
            f"""
            SELECT "Timestamp" AS ts, open, high, low, close, volume
            FROM "{table_name}"
            ORDER BY "Timestamp" DESC
            LIMIT {int(limit)}
            """
        )
    finally:
        await conn.close()

    return candle_rows_to_frame(rows)


def candle_rows_to_frame(rows) -> pd.DataFrame:
    """Rows (ts/open/high/low/close/volume, newest first) → the candle frame.

    Shared by the direct path and the pooled one so both sanitize identically
    (future/garbage timestamps dropped, ms tables converted).
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df = df.drop_duplicates(subset="ts", keep="first")
    df = sanitize_candle_frame(df)  # drop future/garbage timestamps (e.g. 2031), ms->s fix
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_candles_cached(db_host, db_port, db_user, db_pass, db_name: str, table_name: str, limit: int) -> pd.DataFrame:
    """Cached (60 s) candle frame per table, so pair switching is instant.

    Prefers the live API's PERSISTENT pool when it is up (`submit_recent`): the
    first paint of a pair then costs no TCP+TLS+auth round trip per query, which
    on a loaded server is 100–500 ms of what the user experiences as the switch.
    Anything that goes wrong there — infra not started, table dropped mid-click,
    timeout — falls through to the direct path and reports as it always did, so
    a pool problem can never look like an empty table.
    """
    infra = _live_infra_or_none()
    if infra is not None:
        res = None
        try:
            res = infra["submit_recent"](db_name, table_name, limit)
        except Exception:
            res = None
        if isinstance(res, dict) and "rows" in res:
            return candle_rows_to_frame(res["rows"])
        if isinstance(res, dict) and res.get("err"):
            print(f"[candles] pool path failed for {db_name}.{table_name}: {res['err']} "
                  f"— retrying directly", flush=True)
    try:
        return asyncio.run(_load_candles(db_name, table_name, limit, db_host, db_port, db_user, db_pass))
    except Exception as e:
        st.warning(f"Could not load candles for {table_name}: {e}")
        return pd.DataFrame()


async def _fetch_metric_points(db_name, table_name, column, db_host, db_port, db_user, db_pass) -> list:
    conn = await asyncpg.connect(
        host=db_host, port=db_port, user=db_user, password=db_pass,
        database=db_name, timeout=15,
    )
    try:
        rows = await conn.fetch(
            f'SELECT "Timestamp" AS ts, "{column}" AS v FROM "{table_name}" '
            f'WHERE "{column}" IS NOT NULL ORDER BY "Timestamp" ASC LIMIT 200000'
        )
    finally:
        await conn.close()
    return sanitize_metric_points(((r["ts"], r["v"]) for r in rows), int(time.time()))


@st.cache_data(ttl=45, show_spinner=False)
def metric_points_cached(db_host, db_port, db_user, db_pass, db_name, table_name, column):
    """Full OI/funding history of a pair for the panels under the charts.
    [] (not an error) while the table or column does not exist — the engines
    add them lazily on first write."""
    try:
        return asyncio.run(
            _fetch_metric_points(db_name, table_name, column, db_host, db_port, db_user, db_pass)
        )
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def _demo_summary_cached(timeframe: str) -> pd.DataFrame:
    return generate_demo_summary(timeframe)


@st.cache_data(ttl=3600, show_spinner=False)
def _demo_candles_cached(timeframe: str, ticker: str, exchange: str, limit: int) -> pd.DataFrame:
    df = generate_demo_candles(timeframe, ticker, exchange, n=2000)
    return df.tail(int(limit)).reset_index(drop=True)


async def _fetch_live_snapshot(ticker: str, exchange_name: str, ccxt_id: str, atr_val: float):
    exchange = None
    try:
        exchange = create_exchange(ccxt_id)
        return await fetch_orderbook_snapshot(
            exchange=exchange,
            symbol=ticker,
            atr_no_paranormal=atr_val,
            fetch_limit=settings.ob_fetch_limit,
            trades_limit=settings.ob_trades_limit,
            trades_window_sec=settings.ob_trades_window_sec,
            depth_pct=settings.ob_depth_pct,
        )
    except Exception as e:
        st.warning(f"Could not fetch live API data from {exchange_name}: {e}. Displaying DB snapshot.")
        return None
    finally:
        if exchange:
            await close_exchange_safely(exchange, exchange_name)


@st.cache_data(ttl=20, show_spinner=False)
def fetch_live_cached(ticker: str, exchange_name: str, ccxt_id: str, atr_val: float):
    """Cached (20 s) live orderbook snapshot keyed by primitives only."""
    return asyncio.run(_fetch_live_snapshot(ticker, exchange_name, ccxt_id, atr_val))


def get_candles(timeframe: str, row: dict, limit: int, demo: bool) -> pd.DataFrame:
    """Unified candle accessor for demo and DB modes."""
    if demo:
        return _demo_candles_cached(timeframe, row["ticker"], row["exchange"], limit)
    return load_candles_cached(db_host, db_port, db_user, db_pass, row["db_name"], row["table_name"], limit)


def _safe_max_ts(row: Optional[dict]) -> int:
    """Summary-row last-candle epoch as int; 0 for missing/NaN garbage
    (`int(float('nan'))` would raise, and `nan or 0` stays nan)."""
    try:
        return int(float((row or {}).get("max_ts") or 0))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Live ticker (server-side chips, cached ~1s)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_sync_exchange(ccxt_id: str):
    """Persistent synchronous ccxt instance (kept across reruns & sessions)."""
    import ccxt as _ccxt
    return getattr(_ccxt, ccxt_id)({"enableRateLimit": True, "timeout": 8000})


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_missing_candles_cached(ccxt_id: str, symbol: str, timeframe: str, r0: int, r1: int):
    """Fetches a missing candle range from the exchange (closed bars only -> cache 1h)."""
    step = 900 if timeframe == "15m" else 86400
    step_ms = step * 1000
    ex = _get_sync_exchange(ccxt_id)
    try:
        if not ex.markets:
            ex.load_markets()
    except Exception:
        return []

    # Pages needed for the requested span (+1 slack), bounded so a very stale
    # table (weeks behind) is still bridged completely instead of stopping
    # mid-way at a fixed 6-page cap.
    pages = min(40, max(2, (int(r1) - int(r0)) // 1000 + 2))
    # Hard wall-clock budget: this runs on the chart's critical path, and a
    # very stale table could otherwise page the exchange for a minute while
    # the user waits for a pair flip. Whatever arrived in time is drawn; the
    # rest streams in on the next render (the result is cached 1 h).
    deadline = time.time() + settings.dash_stitch_budget_sec
    out = []
    cursor = r0 * step_ms
    try:
        for _ in range(pages):  # paged catch-up over the requested gap
            if time.time() > deadline:
                break
            batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
            if not batch:
                break
            for c in batch:
                b = int(c[0]) // step_ms
                if r0 <= b < r1:
                    out.append(c)
            if batch[-1][0] >= (r1 - 1) * step_ms:
                break
            nxt = batch[-1][0] + step_ms
            if nxt <= cursor:  # exchange ignored `since` → no progress, stop
                break
            cursor = nxt
    except Exception:
        pass
    return out


@st.cache_data(ttl=1, show_spinner=False)
def _fetch_ticker_cached(ccxt_id: str, symbol: str):
    ex = _get_sync_exchange(ccxt_id)
    try:
        if not ex.markets:
            ex.load_markets()
        t = ex.fetch_ticker(symbol)
        return {
            "last": t.get("last"),
            "bid": t.get("bid"),
            "ask": t.get("ask"),
            "pct": t.get("percentage"),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Exchange feeds: serve-then-refresh. The UI thread NEVER waits on REST.
# ---------------------------------------------------------------------------

_FEED_TTL_SEC = 1.0        # fresher than this -> no refresh needed
_FEED_MAX_AGE_SEC = 20.0   # older than this -> do not present it as "LIVE"
_FEEDS: dict = {}          # (kind, ccxt_id, symbol) -> {"value":…, "at":…}


def _feed_refresh(kind: str, ccxt_id: str, symbol: str) -> None:
    """Runs one feed fetch (in a daemon thread) and remembers the result —
    including a None result, so a dead pair is not hammered every second."""
    fn = {
        "ticker": _fetch_ticker_cached,
        "orderbook": _fetch_orderbook_top,
        "tape": _fetch_trade_tape,
    }[kind]
    try:
        value = fn(ccxt_id, symbol)
    except BaseException:      # CancelledError included: a dead exchange must
        value = None           # cost a None, not the whole warm thread
    _FEEDS[(kind, ccxt_id, symbol)] = {"value": value, "at": time.time()}


def _feed_entry(kind: str, ccxt_id: str, symbol: str):
    """Latest known payload of a feed + schedules a background refresh."""
    key = (kind, ccxt_id, symbol)
    ent = _FEEDS.get(key)
    if not ent or time.time() - ent["at"] >= _FEED_TTL_SEC:
        _bg(("feed",) + key, lambda: _feed_refresh(kind, ccxt_id, symbol))
    return ent


def _feed_value(kind: str, ccxt_id: str, symbol: str, max_age_sec: float = _FEED_MAX_AGE_SEC):
    """Value of a feed if one is known and fresh enough, else None (the callers
    already fall back to the collector's DB snapshot).

    Wrapping the raw fetchers in `st.cache_data(ttl=1)` looked like a cache but
    behaved like a BLOCKING one: on the first view of a pair — whose live DB
    row the writer daemon has not produced yet — the health strip ran the
    orderbook and trade-tape fetches inline and the LIVE panel ran the ticker
    fetch, each up to the 8 s ccxt timeout. That is the multi-second stall on
    every Prev/Next click. Serving the previous value (or nothing) and patching
    a second later is how the rest of this dashboard already works.
    """
    ent = _feed_entry(kind, ccxt_id, symbol)
    if not feed_should_use(ent, time.time(), max_age_sec):
        return None
    return ent["value"]


def _render_live_panel(ticker: str, exchange: str, demo: bool, db_name: str = None):
    """One-line LIVE chips: price, bid/ask, spread, 24h change.

    Reads the live row the background daemon keeps writing to TimescaleDB
    every second; falls back to a direct exchange ticker fetch only when the
    DB row is missing/stale.
    """
    if demo:
        import random as _random
        key = f"demo_px_{ticker}_{exchange}"
        px = st.session_state.get(key)
        if px is None:
            px = 100.0
        px = max(px * (1.0 + _random.uniform(-0.0012, 0.0012)), 1e-9)
        st.session_state[key] = px
        data = {"last": px, "bid": px * 0.99995, "ask": px * 1.00005,
                "pct": st.session_state.get(f"demo_pct_{ticker}_{exchange}", 2.4)}
    else:
        data = None
        live_row = _db_live_read(db_name, exchange, ticker) if db_name else None
        if live_row and live_row.get("last"):
            data = {
                "last": live_row["last"],
                "bid": live_row.get("bid"),
                "ask": live_row.get("ask"),
                "pct": live_row.get("pct"),
            }
        if not data:
            # Serve-then-refresh: ask what the background feed thread last saw
            # instead of fetching here, so a pair switch never pays a ticker
            # round trip before the first paint.
            exchange_map = settings.exchange_map_1d
            ccxt_id = exchange_map.get(exchange, exchange)
            data = _feed_value("ticker", ccxt_id, ticker)

    if not data or not data.get("last"):
        st.caption("🔴 LIVE: waiting for the first tick…")
        return

    last = float(data["last"])
    bid, ask = data.get("bid"), data.get("ask")
    spread_html = ""
    if bid and ask:
        abs_sp = ask - bid
        pct_sp = abs_sp / ((ask + bid) / 2.0) * 100.0 if ask + bid else 0.0
        spread_html = f" &nbsp;·&nbsp; spread {abs_sp:.6g} ({pct_sp:.3f}%)"
    ba_html = f" &nbsp;·&nbsp; bid {bid:.6g} / ask {ask:.6g}" if bid and ask else ""
    pct = data.get("pct")
    chg_html = ""
    if pct is not None:
        color = "#66bb6a" if pct >= 0 else "#ef5350"
        sign = "+" if pct >= 0 else ""
        chg_html = f" &nbsp;·&nbsp; 24h <b style='color:{color}'>{sign}{pct:.2f}%</b>"

    st.markdown(
        f"<div style='font-size:13px;padding:0 0 6px 6px;'>"
        f"<b style='color:#42a5f5;'>🔴 LIVE ${last:.6g}</b>{ba_html}{spread_html}{chg_html}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Live snapshot pipeline — TimescaleDB is the single source of truth for live
# data. A process-wide background daemon writes the live price / orderbook top /
# spread / trade-tape stats of the CURRENT pair and its ±5 neighbours into the
# `dashboard_live_ticks` table every second. Every live widget in the UI — the
# 🔴 LIVE price line, the health strip, and even the in-chart poller (via a
# tiny local JSON endpoint) — READS those rows, so the page never blocks on
# exchange latency and charts stay live even when the browser itself cannot
# reach the exchange (CORS / geo-block). Direct exchange fetches remain only
# as a fallback for when the writer has not produced a fresh row yet.
# ---------------------------------------------------------------------------
_LIVE_TABLE = "dashboard_live_ticks"
_LIVE_TICK_PORTS = (8511, 8512, 8513, 8514, 8515)
_LIVE_TARGET_TTL = 90.0   # writer idles when the pair page stops refreshing its target
_LIVE_ROW_MAX_AGE = 30.0  # UI treats older rows as stale → direct fallback

_LIVE_TARGET_LOCK = threading.Lock()
_LIVE_TARGET: dict = {"ts": 0.0, "pairs": []}

_KNOWN_DBS = {
    getattr(settings, "db_high_1d", ""),
    getattr(settings, "db_low_1d", ""),
    getattr(settings, "db_high_15m", ""),
    getattr(settings, "db_low_15m", ""),
} - {""}

# Guards for the tiny live JSON endpoints (/tick, /candles). Table names are
# built as `symbol.replace('/', '_')_on_{exchange}.lower()`, so PERP symbols
# legitimately carry a ':' (PIXEL/USDT:USDT -> pixel_usdt:usdt_on_bybit).
# The previous charset [A-Za-z0-9_] REJECTED every perp table, /candles
# answered {"c": []}, and the chart's history loader read that as "start of
# history" — perp charts silently stopped paging older chunks (stuck at the
# initial window) while spot scrolled infinitely. ':' is safe here: the name
# is only interpolated as a double-quoted identifier, and '"' stays banned.
_LIVE_EX_RE = re.compile(r"[A-Za-z0-9_\-]{1,32}")
_LIVE_SYM_RE = re.compile(r"[A-Za-z0-9/:\._\-]{1,64}")
_LIVE_TBL_RE = re.compile(r"[A-Za-z0-9_:]{1,96}")


def _candles_query_ok(db: str, table: str, to_ts: int) -> bool:
    """Parameter guard for /candles (kept module-level for unit tests)."""
    return bool(
        db in _KNOWN_DBS
        and _LIVE_TBL_RE.fullmatch(table or "")
        and "_on_" in (table or "")
        and to_ts > 0
    )


def _set_live_target(pairs: list) -> None:
    """Replaces the writer's working set (current pair ± 5 neighbours).

    Also PUBLISHES that set to the 15m engine through
    `dashboard_priority_pairs`: the engine's priority lane then refreshes
    exactly these tables in TimescaleDB every second, which is what keeps the
    dashboard a pure renderer of stored rows.
    """
    with _LIVE_TARGET_LOCK:
        _LIVE_TARGET["ts"] = time.time()
        _LIVE_TARGET["pairs"] = list(pairs or [])
    _publish_priority_pairs_async(pairs)


_LAST_PUBLISH = {"ts": 0.0, "key": None}
_PUBLISH_MIN_INTERVAL = 5.0  # TTL is 90 s — re-announcing more often is waste


def _publish_priority_pairs_async(pairs: list) -> None:
    """Hands the displayed pair set to the engine without blocking the rerun.

    Skips the call entirely when the same set was announced moments ago, so
    holding down Prev/Next does not queue a write per keystroke.
    """
    key = tuple((p.get("ex"), p.get("sym")) for p in (pairs or []) if isinstance(p, dict))
    now = time.time()
    if key == _LAST_PUBLISH["key"] and now - _LAST_PUBLISH["ts"] < _PUBLISH_MIN_INTERVAL:
        return

    infra = _live_infra_or_none()
    if not infra or not infra.get("submit_publish"):
        return
    _LAST_PUBLISH["key"] = key
    _LAST_PUBLISH["ts"] = now
    try:
        infra["submit_publish"](pairs)
    except Exception:
        pass


def _get_live_target():
    with _LIVE_TARGET_LOCK:
        return _LIVE_TARGET["ts"], list(_LIVE_TARGET["pairs"])


@st.cache_resource(show_spinner=False)
def _live_infra() -> dict:
    """
    Process-wide live pipeline (started once, survives Streamlit reruns):
      1. one asyncio loop + asyncpg pools shared by the DB writer and readers;
      2. the background snapshot WRITER thread (ccxt → TimescaleDB, every 1 s
         for the current pair ± 5 neighbours);
      3. a tiny local JSON endpoint /tick for the in-chart live pollers, which
         SELECTs the freshest row straight from the table.
    """
    import queue as _queue

    ready = _queue.Queue()
    port_ready = threading.Event()

    def pg_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        infra = {"loop": loop, "pools": {}, "pool_lock": asyncio.Lock()}

        async def get_pool(db_name: str):
            async with infra["pool_lock"]:
                pool = infra["pools"].get(db_name)
                if pool is None:
                    pool = await asyncpg.create_pool(
                        host=db_host, port=db_port, user=db_user, password=db_pass,
                        database=db_name, min_size=1, max_size=4, command_timeout=15,
                    )
                    async with pool.acquire() as conn:
                        await conn.execute(
                            f'CREATE TABLE IF NOT EXISTS {_LIVE_TABLE} ('
                            ' exchange text NOT NULL, symbol text NOT NULL,'
                            ' payload jsonb NOT NULL,'
                            ' updated_at timestamptz NOT NULL DEFAULT now(),'
                            ' PRIMARY KEY (exchange, symbol))'
                        )
                    infra["pools"][db_name] = pool
            return pool

        async def upsert(db_name, exchange, symbol, payload):
            pool = await get_pool(db_name)
            async with pool.acquire() as conn:
                await conn.execute(
                    f'INSERT INTO {_LIVE_TABLE} (exchange, symbol, payload, updated_at)'
                    ' VALUES ($1, $2, $3::jsonb, now())'
                    ' ON CONFLICT (exchange, symbol) DO UPDATE'
                    ' SET payload = EXCLUDED.payload, updated_at = EXCLUDED.updated_at',
                    exchange, symbol, json.dumps(payload),
                )

        async def select(db_name, exchange, symbol):
            try:
                pool = await get_pool(db_name)
                async with pool.acquire() as conn:
                    row = await conn.fetchrow(
                        f'SELECT payload, EXTRACT(EPOCH FROM (now() - updated_at)) AS age'
                        f' FROM {_LIVE_TABLE} WHERE exchange=$1 AND symbol=$2',
                        exchange, symbol,
                    )
            except Exception:
                return None
            if not row:
                return None
            try:
                age = float(row["age"] if row["age"] is not None else 1e9)
            except (TypeError, ValueError):
                age = 1e9
            if age > _LIVE_ROW_MAX_AGE:
                return None  # writer stopped → caller falls back to direct fetch
            payload = row["payload"]
            if isinstance(payload, str):  # asyncpg returns jsonb as text by default
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = None
            data = dict(payload or {})
            data["age"] = age
            return data

        async def select_candles(db_name, table_name, to_ts, limit):
            """Older-history chunk for the chart's infinite left-scroll:
            `limit` rows strictly older than `to_ts` (SECONDS), ascending.

            Some tables store epochs in MILLISECONDS — a plain
            `WHERE "Timestamp" < to_sec` would return ZERO rows for them
            (every ms value is larger than any seconds cursor), which the
            chart then mistook for 'start of history'. The predicate accepts
            BOTH units: seconds rows directly, ms rows via to_sec*1000; the
            ms->s conversion + garbage filtering happens in
            rows_to_compact_candles afterwards.

            Errors are REPORTED, not swallowed: an exception yields
            {"c": None, "err": ...} (the chart shows a red badge instead of
            'start of history') and is printed to the dashboard console.
            An empty-but-valid chunk includes the table's true MIN/MAX dates
            so the badge can say when the table actually starts."""
            try:
                pool = await get_pool(db_name)
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        f'SELECT "Timestamp" AS ts, open, high, low, close, volume'
                        f' FROM "{table_name}"'
                        f' WHERE "Timestamp" < $1'
                        f'    OR ("Timestamp" >= 100000000000 AND "Timestamp" < $1 * 1000)'
                        f' ORDER BY "Timestamp" DESC LIMIT $2',
                        int(to_ts), int(limit),
                    )
            except Exception as e:
                print(
                    f"[candles] ERROR {db_name}.{table_name} to={to_ts}: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                return {"c": None, "err": f"{type(e).__name__}: {e}"}

            out = rows_to_compact_candles([dict(r) for r in rows])
            print(
                f"[candles] {db_name}.{table_name} to={to_ts} limit={limit}"
                f" -> {len(out)} rows",
                flush=True,
            )
            resp = {"c": out}
            if not out:
                # 'no rows' is ambiguous: tell the badge when the table really
                # starts/ends, so 'no older data' is verifiable at a glance.
                try:
                    async with pool.acquire() as conn:
                        mm = await conn.fetchrow(
                            f'SELECT MIN("Timestamp") AS mn, MAX("Timestamp") AS mx'
                            f' FROM "{table_name}"'
                        )

                    def _iso(v):
                        try:
                            v = int(v)
                        except (TypeError, ValueError):
                            return None
                        if v > 1e11:  # ms epoch table
                            v //= 1000
                        return time.strftime("%Y-%m-%d", time.gmtime(v))

                    if mm:
                        resp["mn"] = _iso(mm["mn"])
                        resp["mx"] = _iso(mm["mx"])
                except Exception:
                    pass
            return resp

        async def publish_pairs(pairs):
            """Hands the displayed pair set to the engine's priority lane."""
            db_name = getattr(settings, "priority_lane_db", "") or getattr(
                settings, "db_high_15m", ""
            )
            if not db_name:
                return 0
            pool = await get_pool(db_name)
            async with pool.acquire() as conn:
                return await publish_priority_pairs(
                    conn, pairs, ttl_sec=getattr(settings, "priority_lane_ttl_sec", 90.0)
                )

        def submit_publish(pairs):
            """Schedules the publish and returns IMMEDIATELY.

            Waiting for the round-trip here would put a database write on the
            critical path of every rerun — i.e. of every Prev/Next flip. The
            engine only needs the set within its 90 s TTL, so nothing is lost
            by letting it land a few milliseconds later.
            """
            try:
                asyncio.run_coroutine_threadsafe(publish_pairs(pairs), loop)
            except Exception:
                pass
            return None

        def submit_upsert_nowait(db, ex, sym, payload):
            try:
                asyncio.run_coroutine_threadsafe(upsert(db, ex, sym, payload), loop)
            except Exception:
                pass

        def submit_upsert(db, ex, sym, payload, timeout=5.0):
            try:
                return asyncio.run_coroutine_threadsafe(
                    upsert(db, ex, sym, payload), loop
                ).result(timeout=timeout)
            except Exception:
                return None

        def submit_select(db, ex, sym, timeout=2.5):
            try:
                return asyncio.run_coroutine_threadsafe(
                    select(db, ex, sym), loop
                ).result(timeout=timeout)
            except Exception:
                return None

        def submit_candles(db, table, to_ts, limit, timeout=8.0):
            try:
                return asyncio.run_coroutine_threadsafe(
                    select_candles(db, table, to_ts, limit), loop
                ).result(timeout=timeout)
            except Exception:
                return None

        async def select_recent(db_name, table_name, limit):
            """Newest `limit` rows of a table, newest-last, on the live pool.

            Same SQL as `_load_candles`, just without a fresh connection per
            call. Errors come back as {"err": …} rather than as an empty list:
            the chart reads an empty chunk as 'start of history' and must never
            be told that by a transport problem.
            """
            try:
                pool = await get_pool(db_name)
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        f'SELECT "Timestamp" AS ts, open, high, low, close, volume'
                        f' FROM "{table_name}"'
                        f' ORDER BY "Timestamp" DESC LIMIT $1',
                        int(limit),
                    )
            except Exception as e:
                return {"err": f"{type(e).__name__}: {e}"}
            return {"rows": list(rows)[::-1]}

        def submit_recent(db, table, limit, timeout=8.0):
            """Runs select_recent on the live loop and waits for it — this IS
            the render path, so unlike the publish helpers it must not return
            before the rows are here. A timeout yields None and the caller
            falls back to a direct connection."""
            try:
                return asyncio.run_coroutine_threadsafe(
                    select_recent(db, table, limit), loop
                ).result(timeout=timeout)
            except Exception as e:
                return {"err": f"{type(e).__name__}: {e}"}

        infra["submit_recent"] = submit_recent
        infra["submit_publish"] = submit_publish
        infra["submit_upsert_nowait"] = submit_upsert_nowait
        infra["submit_upsert"] = submit_upsert
        infra["submit_select"] = submit_select
        infra["submit_candles"] = submit_candles
        ready.put(infra)
        loop.run_forever()

    threading.Thread(target=pg_thread, daemon=True, name="live-pg").start()
    infra = ready.get(timeout=10)

    # --- tiny JSON endpoint for the in-chart pollers (/tick?db&ex&sym) -------
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse as _urlparse, parse_qs as _parse_qs

    class _TickHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def _send(self, obj, status=200, allow_gzip=False):
            body = json.dumps(obj).encode()
            gz = None
            if allow_gzip and len(body) > 4096:
                if "gzip" in (self.headers.get("Accept-Encoding") or ""):
                    import gzip as _gzip

                    gz = _gzip.compress(body, compresslevel=6)
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                if gz is not None:
                    self.send_header("Content-Encoding", "gzip")
                    self.send_header("Vary", "Accept-Encoding")
                    self.send_header("Content-Length", str(len(gz)))
                else:
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(gz if gz is not None else body)
            except Exception:
                pass

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.end_headers()

        def do_GET(self):
            try:
                u = _urlparse(self.path)
                if u.path == "/healthz":
                    # Liveness/probe: which routes this process actually serves.
                    # curl http://localhost:8511/healthz -> {"ok": true, "routes": [...]}
                    self._send({"ok": True, "routes": ["/tick", "/candles"], "ts": int(time.time())})
                    return
                if u.path == "/candles":
                    # Older-history chunks for the chart's infinite left-scroll.
                    q = _parse_qs(u.query)
                    db = (q.get("db") or [""])[0]
                    table = (q.get("table") or [""])[0]
                    try:
                        to_ts = int(float((q.get("to") or ["0"])[0]))
                        limit = int(float((q.get("limit") or ["1200"])[0]))
                    except (TypeError, ValueError):
                        to_ts, limit = 0, 1200
                    limit = max(1, min(3000, limit))
                    if not _candles_query_ok(db, table, to_ts):
                        # NEVER answer a guard rejection with {"c": []}: the
                        # chart reads an empty chunk as 'start of history' and
                        # stops paging — that is exactly how perp charts (':'
                        # in table names) got stuck at their initial window.
                        # Be loud instead: red badge in the chart + a line in
                        # the dashboard console.
                        print(
                            f"[candles] REJECTED db={db!r} table={table!r} to={to_ts}",
                            flush=True,
                        )
                        self._send({"c": None, "err": f"rejected /candles params (table={table!r})"})
                        return
                    data = infra["submit_candles"](db, table, to_ts, limit)
                    self._send(
                        data if data is not None else {"c": None, "err": "endpoint loop timeout"},
                        allow_gzip=True,
                    )
                    return
                if u.path != "/tick":
                    self._send({}, status=404)
                    return
                q = _parse_qs(u.query)
                db = (q.get("db") or [""])[0]
                ex = (q.get("ex") or [""])[0]
                sym = (q.get("sym") or [""])[0]
                if (
                    db not in _KNOWN_DBS
                    or not _LIVE_EX_RE.fullmatch(ex or "")
                    or not _LIVE_SYM_RE.fullmatch(sym or "")
                ):
                    self._send({})
                    return
                data = infra["submit_select"](db, ex, sym, timeout=2.5)
                self._send(data or {})
            except Exception:
                self._send({})

    def tick_thread():
        for port in _LIVE_TICK_PORTS:
            try:
                srv = ThreadingHTTPServer(("0.0.0.0", port), _TickHandler)
            except OSError:
                continue
            srv.daemon_threads = True
            infra["tick_port"] = port
            print(f"[live-api] serving /tick /candles /healthz on 0.0.0.0:{port}", flush=True)
            port_ready.set()
            srv.serve_forever()
            return
        port_ready.set()  # no free port — charts fall back to direct exchange REST

    threading.Thread(target=tick_thread, daemon=True, name="live-tick").start()
    port_ready.wait(timeout=1.5)  # first render already gets the tick port

    # --- background writer: ccxt → dashboard_live_ticks, every second --------
    writer_ex: dict = {}
    writer_ex_lock = threading.Lock()

    def writer_exchange(ccxt_id: str):
        import ccxt as _ccxt

        with writer_ex_lock:
            ex = writer_ex.get(ccxt_id)
            if ex is None:
                ex = getattr(_ccxt, ccxt_id)(
                    {"enableRateLimit": True, "timeout": 6000}
                )
                writer_ex[ccxt_id] = ex
        if not ex.markets:
            with writer_ex_lock:
                if not ex.markets:
                    try:
                        ex.load_markets()
                    except Exception:
                        pass
        return ex

    def writer_fetch_pair(entry: dict):
        """Live payload for one pair.

        FULL (ticker + orderbook + trade tape) only for the pair actually on
        screen; the ±5 neighbours get the ticker alone. Fetching everything
        for all 11 pairs meant ~33 blocking HTTP calls per second inside the
        Streamlit process — enough GIL and socket pressure to make the page
        itself feel stuck while the collector was also running.
        """
        ex = writer_exchange(entry["ccxt"])
        sym = entry["sym"]
        full = bool(entry.get("cur"))

        last = bid = ask = pct = None
        try:
            t = ex.fetch_ticker(sym)
            last, bid, ask, pct = (
                t.get("last"), t.get("bid"), t.get("ask"), t.get("percentage")
            )
        except Exception:
            pass

        depth = None
        try:
            if not full:
                raise StopIteration  # neighbour: ticker is enough
            ob = ex.fetch_order_book(sym, limit=50)
            bids, asks = ob.get("bids") or [], ob.get("asks") or []
            if bids and asks:
                ob_bid, ob_ask = float(bids[0][0]), float(asks[0][0])
                bid = bid if bid else ob_bid
                ask = ask if ask else ob_ask
                if last is None and ob_bid > 0 and ob_ask > 0:
                    last = (ob_bid + ob_ask) / 2.0
                mid = (ob_bid + ob_ask) / 2.0
                lo, hi = mid * 0.99, mid * 1.01
                depth = (
                    sum(float(p) * float(a) for p, a in bids if float(p) >= lo)
                    + sum(float(p) * float(a) for p, a in asks if float(p) <= hi)
                )
        except Exception:
            pass

        tpm = None
        barcode = False
        try:
            if not full:
                raise StopIteration  # neighbour: ticker is enough
            trades = ex.fetch_trades(sym, limit=200) or []
            now_ms = time.time() * 1000.0
            recent = [
                tr for tr in trades
                if tr.get("timestamp") and float(tr["timestamp"]) >= now_ms - 300_000
            ]
            tpm = len(recent) / 5.0
            prices = [float(tr["price"]) for tr in trades if tr.get("price")]
            if len(trades) >= 30 and len(set(prices)) <= 4:
                barcode = True
        except Exception:
            pass

        if last is None and depth is None and tpm is None:
            return None

        def _fin(x):
            """Non-finite floats would produce invalid JSONB — map to None."""
            try:
                f = float(x)
            except (TypeError, ValueError):
                return None
            return f if f == f and abs(f) != float("inf") else None

        return {
            "last": _fin(last), "bid": _fin(bid), "ask": _fin(ask), "pct": _fin(pct),
            "depth_usd": _fin(depth), "trades_per_min": _fin(tpm),
            "is_barcode": bool(barcode), "ts": time.time(),
        }

    def writer_loop():
        from concurrent.futures import ThreadPoolExecutor

        pool_exec = ThreadPoolExecutor(max_workers=6)
        while True:
            tgt_ts, pairs = _get_live_target()
            if not pairs or (time.time() - tgt_ts) > _LIVE_TARGET_TTL:
                time.sleep(1.0)
                continue
            started = time.time()
            futs = [(e, pool_exec.submit(writer_fetch_pair, e)) for e in pairs]
            for e, f in futs:
                try:
                    payload = f.result(timeout=10)
                except Exception:
                    payload = None
                if payload:
                    # fire-and-forget: waiting per pair serialised the whole
                    # second-long round on database latency
                    infra["submit_upsert_nowait"](e["db"], e["ex"], e["sym"], payload)
            time.sleep(max(0.2, 1.0 - (time.time() - started)))

    threading.Thread(target=writer_loop, daemon=True, name="live-writer").start()
    return infra


def _live_infra_or_none():
    try:
        return _live_infra()
    except Exception:
        return None


def _live_db_for(row_1d, row_15m) -> Optional[str]:
    """Database the live snapshot for this pair is written to / read from."""
    row = row_1d or row_15m
    if row and row.get("db_name"):
        return row["db_name"]
    return getattr(settings, "db_high_1d", None)


@st.cache_data(ttl=1, show_spinner=False)
def _db_live_read(db_name: str, exchange: str, symbol: str):
    """Latest live snapshot row for the pair (None when missing/stale)."""
    infra = _live_infra_or_none()
    if not infra or not db_name:
        return None
    try:
        return infra["submit_select"](db_name, exchange, symbol, timeout=2.5)
    except Exception:
        return None


def _live_tick_path(db_name: Optional[str], exchange: str, symbol: str) -> Optional[str]:
    """URL path for the dashboard's live-tick endpoint serving this pair."""
    from urllib.parse import urlencode

    infra = _live_infra_or_none()
    if not infra or not infra.get("tick_port") or not db_name:
        return None
    return "/tick?" + urlencode({"db": db_name, "ex": exchange, "sym": symbol})


# ---------------------------------------------------------------------------
# Live health-strip chips (live DB snapshot first, exchange REST fallback,
# ~1s refresh)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=1, show_spinner=False)
def _fetch_orderbook_top(ccxt_id: str, symbol: str, limit: int = 50):
    """Top-of-book (cached ~1s) for the live Depth/Spread chips."""
    ex = _get_sync_exchange(ccxt_id)
    try:
        if not ex.markets:
            ex.load_markets()
        ob = ex.fetch_order_book(symbol, limit=limit)
        return {"bids": ob.get("bids") or [], "asks": ob.get("asks") or []}
    except Exception:
        return None


@st.cache_data(ttl=1, show_spinner=False)
def _fetch_trade_tape(ccxt_id: str, symbol: str, limit: int = 200):
    """Recent trades (cached ~1s) for the live Trades/min chip."""
    ex = _get_sync_exchange(ccxt_id)
    try:
        if not ex.markets:
            ex.load_markets()
        trades = ex.fetch_trades(symbol, limit=limit) or []
        return [
            {
                "timestamp": t.get("timestamp"),
                "price": t.get("price"),
                "amount": t.get("amount"),
                "side": t.get("side"),
            }
            for t in trades
        ]
    except Exception:
        return None


def _compute_live_health_row(sym_ticker, sym_ex, hist_1d, atr_val, db_row, db_name: str = None):
    """
    Builds the health-strip row from the LIVE snapshot the background daemon
    writes to TimescaleDB every second (orderbook top + trade tape for the
    pair). When no fresh DB row exists yet, falls back to direct exchange
    fetches, and finally to the latest collector snapshot values in `db_row`.
    """
    row = dict(db_row or {})
    exchange_map = settings.exchange_map_1d
    ccxt_id = exchange_map.get(sym_ex, sym_ex)

    live_row = _db_live_read(db_name, sym_ex, sym_ticker) if db_name else None

    # --- Depth ±1% & Spread % ATR (DB live row, else direct orderbook) ---
    if live_row and live_row.get("depth_usd") is not None:
        row["ob_total_depth_usd"] = live_row["depth_usd"]
        bid, ask = live_row.get("bid"), live_row.get("ask")
        if bid and ask:
            try:
                bid, ask = float(bid), float(ask)
                row["ob_spread_abs"] = ask - bid
                row["ob_best_bid"] = bid
                row["ob_best_ask"] = ask
                if atr_val and atr_val > 0:
                    row["ob_spread_atr_pct"] = (ask - bid) / atr_val * 100.0
            except (TypeError, ValueError):
                pass
    else:
        # Serve-then-refresh (see _feed_value): no REST call on the render path.
        ob = _feed_value("orderbook", ccxt_id, sym_ticker)
        if ob and ob["bids"] and ob["asks"]:
            try:
                bid = float(ob["bids"][0][0])
                ask = float(ob["asks"][0][0])
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0
                    lo, hi = mid * 0.99, mid * 1.01
                    depth = sum(float(p) * float(a) for p, a in ob["bids"] if float(p) >= lo) + sum(
                        float(p) * float(a) for p, a in ob["asks"] if float(p) <= hi
                    )
                    row["ob_total_depth_usd"] = depth
                    row["ob_spread_abs"] = ask - bid
                    row["ob_best_bid"] = bid
                    row["ob_best_ask"] = ask
                    if atr_val and atr_val > 0:
                        row["ob_spread_atr_pct"] = (ask - bid) / atr_val * 100.0
            except (TypeError, ValueError, IndexError):
                pass

    # --- Trades/min over a 300s window (DB live row, else direct tape) ---
    if live_row and live_row.get("trades_per_min") is not None:
        row["ob_trades_per_min"] = live_row["trades_per_min"]
        if live_row.get("is_barcode"):
            row["ob_is_barcode"] = True  # barcode market → DEAD tape chip
    else:
        trades = _feed_value("tape", ccxt_id, sym_ticker)
        if trades:
            now_ms = time.time() * 1000.0
            recent = [
                t for t in trades
                if t.get("timestamp") and float(t["timestamp"]) >= now_ms - 300_000
            ]
            row["ob_trades_per_min"] = len(recent) / 5.0
            prices = [float(t["price"]) for t in trades if t.get("price")]
            if len(trades) >= 30 and len(set(prices)) <= 4:
                row["ob_is_barcode"] = True  # barcode market → DEAD tape chip

    # --- Min 7d $Vol: min(vol×low) over the last 7 CLOSED daily bars ---
    if hist_1d is not None and not hist_1d.empty:
        closed = hist_1d[hist_1d["ts"] <= int(time.time()) - 86400]
        tail = closed.tail(7)
        if len(tail) >= 1:
            row["ob_min_7d_volume_usd"] = float((tail["volume"] * tail["low"]).min())

    return row


# ---------------------------------------------------------------------------
# Background cache warming — near-instant Prev/Next pair switching.
# Everything a pair view touches (DB candle frames, exchange gap/tail stitch
# fetches, live ticker/orderbook/trade-tape chips, full orderbook snapshot)
# is pre-fetched for the ±5 neighbouring pairs in daemon threads while the
# user looks at the current pair. The UI thread NEVER waits on these.
# ---------------------------------------------------------------------------
_BG_LOCK = threading.Lock()
_BG_RUNNING: set = set()
_LIVE_SNAP: dict = {}  # (ticker, exchange, atr) -> (snapshot_or_None, fetched_at)


def _bg(key, fn, *args, **kwargs) -> None:
    """Runs fn(*args) in a daemon thread, deduplicated per key while running."""
    with _BG_LOCK:
        if key in _BG_RUNNING:
            return
        _BG_RUNNING.add(key)

    def _run():
        try:
            fn(*args, **kwargs)
        except BaseException as e:
            # BaseException, not Exception: `asyncio.run` re-raises the
            # CancelledError of its inner task, and ccxt's `load_markets` lets
            # one escape — so a neighbour whose exchange call was cut off by a
            # hard timeout used to print a traceback and die mid-way, taking the
            # chart-page priming with it. The click still rendered; the warming
            # that was supposed to make the NEXT click instant silently did not
            # happen, which is what "не быстрее" looked like from the outside.
            print(f"[bg] {key}: {type(e).__name__}: {e}", flush=True)
        finally:
            with _BG_LOCK:
                _BG_RUNNING.discard(key)

    threading.Thread(target=_run, daemon=True).start()


def _warm_live_snapshot(ticker: str, exchange_name: str, ccxt_id: str, atr_val: float) -> None:
    """Background fill of the live orderbook snapshot used by metrics cards."""
    key = (ticker, exchange_name, round(float(atr_val or 0.0), 10))
    try:
        snap = fetch_live_cached(ticker, exchange_name, ccxt_id, float(atr_val or 0.0))
    except BaseException:      # CancelledError included — see _bg
        snap = None
    _LIVE_SNAP[key] = (snap, time.time())


_WARMED_AT: dict = {}


def _warm_pair_due(ticker: str, exchange_name: str, now: float) -> bool:
    """Whether this pair deserves a background warm right now.

    Warming used to be re-armed by EVERY rerun of the script (pair click, 60 s
    auto-reload, fragment tick): the `_bg` key only dedupes while a warm is
    running, so a warm that finished in 2 s started again a second later, and
    ten neighbours × {2 candle queries, up to 20 exchange range fetches, 3 feed
    calls, 2 chart builds} became the dominant load on the machine — the app got
    slower to click, exactly because of the code meant to make clicking instant.
    One warm per pair per page lifetime (CHART_PAGE_TTL_SEC) is all the
    freshness a pre-built page needs: after that the entry is expired anyway.
    """
    key = (ticker, exchange_name)
    if now - float(_WARMED_AT.get(key, 0.0)) < CHART_PAGE_TTL_SEC:
        return False
    _WARMED_AT[key] = now
    return True


def _warm_yield_to_clicks(ticker: str = "", exchange_name: str = "") -> bool:
    """Delay a warm so the click that scheduled it wins, and abort it when the
    user has clicked again meanwhile (the next run warms what matters now).
    Returns False when the caller should stop — and un-marks the pair in that
    case, because nothing was actually prepared for it."""
    seen = _LAST_INTERACTION_AT
    left = float(settings.dash_warm_delay_sec)
    while left > 0.0:
        step = min(0.25, left)
        time.sleep(step)
        left -= step
        if _LAST_INTERACTION_AT > seen:
            if ticker:
                _WARMED_AT.pop((ticker, exchange_name), None)
            return False
    return True


def _pair_is_frozen(row_15m, row_1d, now: float = None) -> bool:
    """True when the collector stopped writing this pair long ago.

    These are the leftovers of the spot→perp migration: pairs whose dead spot
    table still holds hundreds of missing candles. Fetching that history in the
    background is pure waste — nobody is about to look at it, and the exchange
    round trips stall the live pool the charts need. The pair still opens
    normally (its DB data, its own on-click stitch), it is just not speculatively
    prepared.
    """
    now = time.time() if now is None else float(now)
    ts = 0
    for row in (row_15m, row_1d):
        if row:
            try:
                ts = max(ts, int(row.get("max_ts") or 0))
            except (TypeError, ValueError):
                pass
    if ts <= 0:
        return True
    if ts > 1e11:
        ts //= 1000            # milliseconds table
    return now - ts > float(settings.dash_warm_stale_skip_sec)


def _warm_pair_caches(
    ticker, exchange_name, ccxt_id, row_15m, row_1d, lim15, lim1d, atr_period, full_live: bool,
    chart_ctx: dict = None,
) -> None:
    """
    Warms ALL data a pair view needs so Prev/Next renders instantly:
    candle frames (15m+1D), the exchange-side gap/tail stitch fetches,
    the live chip feeds (ticker/orderbook/trade tape), and — for the
    immediate ±2 neighbours — the full live orderbook snapshot.

    With `chart_ctx` (style/height/volume/stitch/tick-port of the current
    view) it also pre-builds BOTH chart pages — the stitched one into
    `_STITCHED_PAGES`, the DB-only one into `_render_chart_html_cached` — so
    the very first flip to a neighbour skips the DB round-trip, the exchange
    stitch AND the JSON/HTML build: the render path becomes a memory lookup.
    """
    if not _warm_yield_to_clicks(ticker, exchange_name):
        return

    # A frozen pair gets its chart page primed from the DB (cheap, and it is the
    # one thing that makes opening it instant) but no exchange traffic at all.
    frozen = _pair_is_frozen(row_15m, row_1d)
    for tf, row, lim in (("15m", row_15m, lim15), ("1d", row_1d, lim1d)):
        if not row or frozen:
            continue
        frame = load_candles_cached(
            db_host, db_port, db_user, db_pass, row["db_name"], row["table_name"], lim
        )
        if frame is None or frame.empty:
            continue
        step = 900 if tf == "15m" else 86400
        buckets = sorted(set(int(t) // step for t in frame["ts"]))
        ranges = find_missing_bucket_ranges(buckets, step)
        now_bucket = int(time.time()) // step
        if buckets and buckets[-1] < now_bucket and now_bucket - buckets[-1] <= 2000:
            ranges.append((buckets[-1] + 1, now_bucket))  # closed tail
        # 3 ranges, not 10: this is a head start on the stitch, not the stitch.
        # Whatever is left is fetched when the pair is actually opened.
        for r0, r1 in ranges[:3]:
            _fetch_missing_candles_cached(ccxt_id, ticker, tf, r0, r1)
    if not frozen:
        _fetch_ticker_cached(ccxt_id, ticker)
        _fetch_orderbook_top(ccxt_id, ticker)
        _fetch_trade_tape(ccxt_id, ticker)
    if full_live and row_1d is not None and not frozen:
        atr_nb = 0.0
        try:
            fr = load_candles_cached(
                db_host, db_port, db_user, db_pass,
                row_1d["db_name"], row_1d["table_name"], max(int(lim1d), 60),
            )
            if fr is not None and not fr.empty and len(fr) >= 3:
                atr_nb = compute_atr_no_paranormal_bars(
                    highs=fr["high"].to_numpy(dtype=float),
                    lows=fr["low"].to_numpy(dtype=float),
                    closes=fr["close"].to_numpy(dtype=float),
                    period=atr_period,
                    small_threshold=settings.atr_small_threshold,
                    large_threshold=settings.atr_large_threshold,
                )
        except Exception:
            pass
        _warm_live_snapshot(ticker, exchange_name, ccxt_id, atr_nb)

    # skip_stitch for a frozen pair: building the PATCHED page means running the
    # gap stitch from the warm thread, i.e. the hundreds of exchange pages the
    # stale table is missing — the very thing being skipped above.
    _warm_chart_pages(
        chart_ctx, ticker, exchange_name, ccxt_id, row_15m, row_1d, lim15, lim1d,
        skip_stitch=frozen,
    )
    _WARMED_AT[(ticker, exchange_name)] = time.time()


def _warm_chart_pages(chart_ctx, ticker, exchange_name, ccxt_id,
                      row_15m, row_1d, lim15, lim1d, skip_stitch: bool = False) -> list:
    """Pre-builds both neighbour chart pages, with keys identical to the render path.

    The keys MUST match exactly (same primitives, same order), or the flip
    misses the cache and rebuilds — which is what "not instant" means here.
    Returns the store keys it primed, for tests and for the swap watcher.
    """
    if not chart_ctx:
        return []
    primed: list = []
    for tf_label, row, lim in (("15m", row_15m, lim15), ("1D", row_1d, lim1d)):
        if not row:
            continue
        step = 900 if tf_label == "15m" else 86400
        nb_live_db = _live_db_for(row_1d, row_15m)
        tick_path = (
            _live_tick_path(nb_live_db, exchange_name, ticker)
            if chart_ctx.get("interval_ms")
            else None
        )
        poller = build_live_poller_js(
            exchange_name, ticker, step,
            chart_ctx.get("interval_ms", 0),
            tick_path=tick_path,
            tick_port=chart_ctx.get("tick_port"),
        )
        hist = ""
        if chart_ctx.get("tick_port"):
            hist = build_history_loader_js(
                row["db_name"], row["table_name"], step,
                chart_ctx["tick_port"], chunk=1200 if tf_label == "15m" else 700,
            )
        m_db = m_tbl = ""
        m_lim = 0
        if tf_label == "1D" and row_15m:
            m_db, m_tbl = row_15m["db_name"], row_15m["table_name"]
            m_lim = max(int(lim15), 200)
        page_kwargs, store_key = _chart_page_args(
            row, tf_label, int(lim), m_db, m_tbl, m_lim,
            ccxt_id, ticker, exchange_name,
            chart_ctx["style"], chart_ctx["height"],
            chart_ctx["volume"], poller, hist,
            chart_ctx.get("flat_fill", True),
        )
        if chart_ctx["stitch"] and not skip_stitch:
            # The neighbour flip must find the PATCHED page ready in the store
            # (built here, off the render path). It must ALSO find the plain
            # page in st.cache_data: a flip that happens before the stitch
            # landed renders the DB-only page, and priming nothing for it meant
            # paying two fresh queries plus a full HTML build on the click.
            _render_chart_html_cached(**page_kwargs, stitch_enabled=False)
            _warm_stitched_page(store_key, page_kwargs)
        else:
            _render_chart_html_cached(**page_kwargs, stitch_enabled=False)
        primed.append(store_key)
    return primed


# Lifetime of a built chart page — shared by the st.cache_data entry of the
# plain (DB-only) page and the in-process store of the background-stitched
# one, so the patched page can never outlive the page it replaces.
CHART_PAGE_TTL_SEC = 45.0

# Stitched chart pages, built OFF the render path (see _chart_page_args).
# key -> {"html":…, "txt":…, "at":…, "hash":…}
_STITCHED_PAGES: dict = {}
# A chart-page key carries the pair, the timeframe and both JS blobs, and the
# user can browse hundreds of pairs per session — so these stores are trimmed
# to the most recent entries instead of holding an HTML page per visit.
_PAGE_STORE_LIMIT = 48
# Keys of the pair currently on screen: the watcher below reruns the app once
# when a background page lands for one of them and has not been displayed yet.
_CURRENT_CHART_KEYS: set = set()
# hash() of the HTML actually shown per chart slot. A background stitch that
# changed NOTHING must not repaint the iframe (that resets zoom/pan), so the
# swap is decided by comparing content hashes, never by "a page exists now".
_DISPLAYED_HASH: dict = {}
# When a key was last swapped, to rate-limit repaints: the keep-warm rebuild
# produces a slightly different page every cycle (live candles move), and an
# unsuppressed watcher would reset the user's zoom every ~27 s forever. One
# swap per CHART_SWAP_COOLDOWN_SEC is enough — the 60 s auto-reload repaints
# with whatever is newest anyway.
CHART_SWAP_COOLDOWN_SEC = 60.0
_LAST_SWAP_AT: dict = {}


def _chart_page_args(
    row, tf_label, limit, m_db, m_tbl, m_lim, ccxt_id, sym_ticker, sym_ex,
    chart_style, chart_height, show_volume, poller_js, hist_js, flat_fill,
):
    """
    (builder_kwargs, store_key) for one chart page, from ONE list of values.

    The neighbour warm and the render path must agree on the identity of a
    chart page bit for bit, or the flip misses and rebuilds — so both derive
    their arguments here instead of spelling the same 20 parameters twice.

    Everything is passed BY NAME on purpose. The builder takes 22 arguments and
    `stitch_enabled` sits in the MIDDLE of its signature, so a positional
    tuple here silently shifts `live_poller_js` onto it:
        TypeError: _render_chart_html_cached() got multiple values for
        argument 'stitch_enabled'
    — which is exactly what this helper introduced once, on a path no unit test
    covered (the crash needs a real DB row, demo mode takes the other branch).
    `_chart_page_args` is now bound against the real signature in the test
    suite, so the keyword names below are checked, not eyeballed.

    The store key is the same values WITHOUT the DB credentials (they are not
    part of a page's identity, and copying them into a long-lived dict would
    keep a password alive for the process lifetime).
    """
    params = {
        "db_name": row["db_name"],
        "table_name": row["table_name"],
        "max_ts": _safe_max_ts(row),
        "tf_label": tf_label,
        "limit": int(limit),
        "merge_db": m_db,
        "merge_table": m_tbl,
        "merge_limit": int(m_lim),
        "ccxt_id": ccxt_id,
        "sym_ticker": sym_ticker,
        "sym_ex": sym_ex,
        "chart_style": chart_style,
        "chart_height": int(chart_height),
        "show_volume": bool(show_volume),
        "live_poller_js": poller_js or "",
        "history_loader_js": hist_js or "",
        "flat_fill": bool(flat_fill),
    }
    # dict order == insertion order == the key: one source of truth, and a
    # missing/renamed parameter breaks the signature test instead of quietly
    # producing a key that never matches the warm.
    return dict(db_host=db_host, db_port=db_port, db_user=db_user,
                db_pass=db_pass, **params), tuple(params.values())


def _remember(store: dict, key, value, limit: int = _PAGE_STORE_LIMIT) -> None:
    """Store and keep only the `limit` most recent entries (dict order is
    insertion order, so re-storing a key must move it to the end)."""
    store.pop(key, None)
    store[key] = value
    while len(store) > limit:
        store.pop(next(iter(store)))


# Bumped by the chart builder itself: st.cache_data hides whether a page was
# built or looked up, and those two differ by a database round trip.
_CHART_BUILDS = 0


def _report_switch(ticker: str, tf_label: str, source: str,
                   started: float, build_started: float) -> None:
    """One console line per SLOW chart render, saying where the time went.

    "Switching is slow" has three causes that need opposite fixes — the warm did
    not happen (Streamlit), the candle query is slow (database), or the JSON/HTML
    build is (CPU). Guessing at this cost two rounds, so the render path now says
    which one it was: a "warmed page"/"cached page" that is still slow is NOT a
    database problem.
    """
    total_ms = (time.perf_counter() - started) * 1000.0
    if total_ms < float(settings.dash_switch_report_ms):
        return
    build_ms = (time.perf_counter() - build_started) * 1000.0
    print(
        f"[switch] {ticker} {tf_label}: {total_ms:.0f} ms — {build_ms:.0f} ms of it "
        f"query+build, source: {source}",
        flush=True,
    )


def _stitched_candle_count(txt: str) -> int:
    """How many candles the stitch caption reports (0 when there is no caption).
    The caption is produced a few lines below in this same module, so the
    pattern is intentionally loose: it counts, it does not validate."""
    if not txt or "\U0001fa79" not in txt:
        return 0
    m = re.search(r"(\d+)\s+missing", txt)
    try:
        return int(m.group(1)) if m else 0
    except (TypeError, ValueError):
        return 0


def _warm_stitched_page(store_key: tuple, page_kwargs: dict) -> int:
    """
    Builds the gap-stitched chart page in a BACKGROUND thread and stores it.

    Runs the builder UNCACHED (.__wrapped__): a keep-warm refresh that read its
    own 45 s cache entry would store a stale stitch and report it as fresh.
    Returns how many candles the stitch added — 0 means the DB page was
    already complete (the normal case for a healthy collector), which is also
    the case where swapping it in must NOT repaint the iframe and reset the
    user's zoom.
    """
    try:
        res = _render_chart_html_cached.__wrapped__(**page_kwargs, stitch_enabled=True)
    except Exception:
        return 0
    html, txt = (res or ("", ""))
    _remember(_STITCHED_PAGES, store_key, {
        "html": html, "txt": txt, "at": time.time(), "hash": hash(html or ""),
    })
    return _stitched_candle_count(txt)


@st.cache_data(ttl=CHART_PAGE_TTL_SEC, show_spinner=False)
def _render_chart_html_cached(
    db_host, db_port, db_user, db_pass,
    db_name: str, table_name: str, max_ts: int,
    tf_label: str, limit: int,
    merge_db: str, merge_table: str, merge_limit: int,
    ccxt_id: str, sym_ticker: str, sym_ex: str,
    chart_style: str, chart_height: int,
    show_volume: bool, stitch_enabled: bool,
    live_poller_js: str, history_loader_js: str,
    flat_fill: bool = True,
):
    """
    Fully-built chart page (candles → stitch → daily merge → compact JSON →
    HTML) cached ~45 s. Pair switching always re-runs the whole Streamlit
    script; with this cache — pre-warmed for the ±5 neighbours by
    `_warm_pair_caches` — the flip is a memory lookup plus an iframe refresh
    instead of a DB round-trip + JSON build for two charts. Returns
    (html, stitch_caption_text) or None when the table has no usable candles.
    """
    global _CHART_BUILDS
    _CHART_BUILDS += 1
    frame = load_candles_cached(db_host, db_port, db_user, db_pass, db_name, table_name, limit)
    if frame is None or frame.empty:
        return None

    tf_key = "15m" if tf_label == "15m" else "1d"
    step = 900 if tf_key == "15m" else 86400

    def _stitch(fr):
        if stitch_enabled and fr is not None and len(fr) > 1:
            return stitch_candle_gaps(
                fr,
                lambda r0, r1: _fetch_missing_candles_cached(ccxt_id, sym_ticker, tf_key, r0, r1),
                step,
            )
        return fr, 0

    frame, stitched = _stitch(frame)

    if tf_key == "1d" and merge_table:
        df15 = load_candles_cached(db_host, db_port, db_user, db_pass, merge_db, merge_table, merge_limit)
        df15, _ = _stitch(df15)
        frame = merge_intraday_into_daily(frame, df15)

    stale_hint = ""
    age_h = None
    if max_ts and max_ts > 0:
        mt = max_ts // 1000 if max_ts > 1e11 else max_ts
        age_h = (time.time() - mt) / 3600.0
    thr = 1.0 if tf_key == "15m" else 49.0
    stale = age_h is not None and age_h > thr
    if stitched and stale:
        stale_hint = f" · ⏳ collector {age_h:.1f}h behind"
    if stitched:
        stitch_txt = (
            f"🩹 {stitched} missing {tf_label} candles stitched from exchange "
            f"(in-memory){stale_hint}"
        )
    elif stale:
        # Nothing could be stitched while the table is stale → the chart would
        # silently end weeks before the live price line. Say so out loud.
        stitch_txt = (
            f"<span style='color:#ef5350'>⚠️ collector {age_h:.1f}h behind on this table "
            f"and the exchange returned no {tf_label} catch-up candles — "
            f"chart tail may be missing</span>"
        )
    else:
        stitch_txt = "&nbsp;"

    if flat_fill:
        # Intervals without trades: draw them flat like the exchange chart
        # does instead of leaving a hole (illiquid pairs).
        frame, filled = fill_missing_bars(frame, step)
        if filled and not stitched:
            stitch_txt = (
                f"🕳 {filled} empty {tf_label} interval(s) drawn flat "
                f"(no trades on the exchange)"
            )

    candles_arr, volume_arr = build_series_arrays(frame, with_volume=show_volume)
    dumps = lambda x: json.dumps(x, separators=(",", ":"))
    html = build_lightweight_chart_html(
        candles_json=dumps(candles_arr),
        volume_json=dumps(volume_arr) if show_volume else None,
        chart_height=chart_height,
        chart_style=chart_style,
        live_poller_js=live_poller_js,
        history_loader_js=history_loader_js,
    )
    return html, stitch_txt


def render_tradingview_lightweight_chart(
    hist_df: pd.DataFrame,
    ticker: str,
    exchange: str,
    tf_label: str,
    chart_height: int = 470,
    chart_style: str = "OHLCV Bars",
    show_volume: bool = False,
    live_poller_js: str = "",
    history_loader_js: str = "",
):
    """Renders a TradingView Lightweight Charts canvas with OHLCV Bars/Candles,
    volume histogram, crosshair tooltips, and fast rolling ATR channels."""
    if hist_df is None or hist_df.empty:
        st.info(f"No {tf_label} candles available for {ticker} ({exchange}).")
        return

    candles, volume_data = build_series_arrays(hist_df, with_volume=show_volume)

    dumps = lambda x: json.dumps(x, separators=(",", ":"))

    html_code = build_lightweight_chart_html(
        candles_json=dumps(candles),
        volume_json=dumps(volume_data) if show_volume else None,
        chart_height=chart_height,
        chart_style=chart_style,
        live_poller_js=live_poller_js,
        history_loader_js=history_loader_js,
    )
    _html_component(html_code, chart_height + 10 + HIST_STATUS_HEIGHT)


def render_tradingview_official_widget(ticker: str, exchange: str, interval: str = "D", style_code: str = "0"):
    """Renders TradingView's official Advanced Real-Time Chart Widget."""
    base_symbol = ticker.replace("/", "").replace(":", "")
    tv_symbol = f"{exchange.upper()}:{base_symbol}"

    html_code = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="height:520px;width:100%">
      <div id="tradingview_chart" style="height:calc(100% - 32px);width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "{interval}",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "{style_code}",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    _html_component(html_code, 550)


def render_plotly_chart(hist_df: pd.DataFrame, ticker: str, exchange: str, tf_label: str, chart_style: str):
    """Plotly dark price chart."""
    if hist_df is None or hist_df.empty:
        st.info(f"No {tf_label} candles available for {ticker} ({exchange}).")
        return

    fig = go.Figure()
    if chart_style == "OHLCV Bars":
        fig.add_trace(go.Ohlc(
            x=hist_df["time"], open=hist_df["open"], high=hist_df["high"],
            low=hist_df["low"], close=hist_df["close"], name="OHLCV Bars"
        ))
    else:
        fig.add_trace(go.Candlestick(
            x=hist_df["time"], open=hist_df["open"], high=hist_df["high"],
            low=hist_df["low"], close=hist_df["close"], name="Candlesticks"
        ))

    fig.update_layout(
        title=f"{ticker} ({exchange}) — {tf_label} — {chart_style}",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=500,
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🛠️ Settings")

demo_mode = st.sidebar.checkbox(
    "🧪 Demo mode (no database)",
    value=os.getenv("DASHBOARD_DEMO", "0") == "1",
    help="Render the dashboard from synthetic data — useful to preview the UI without TimescaleDB.",
)

if demo_mode:
    db_host = db_port = db_user = db_pass = ""
else:
    # Defaults come from config settings (db_config.py / .env priority), NOT a
    # bare "postgres" fallback — the shell environment often has no DB_USER
    # exported, and guessing "postgres/postgres" just fails authentication.
    db_host = st.sidebar.text_input("DB Host", value=os.getenv("DB_HOST", settings.db_host))
    db_port = st.sidebar.number_input("DB Port", value=int(os.getenv("DB_PORT", settings.db_port)))
    db_user = st.sidebar.text_input("DB User", value=os.getenv("DB_USER", settings.db_user))
    db_pass = st.sidebar.text_input("DB Password", value=os.getenv("DB_PASSWORD", settings.db_password), type="password")

# --- Exchange filter (which exchanges the dashboard is wired to) -------------
# Options respect the collector's ALLOWED_EXCHANGES / EXCLUDED_EXCHANGES
# (empty include-list = all) so the picker never offers a pair the engines
# deliberately stopped collecting. Filtering happens on the cached summary
# frames, so toggling is instant.
_all_configured_exs = sorted(settings.exchange_map_1d.keys())
_exchange_options = settings.filter_exchange_ids(_all_configured_exs) or _all_configured_exs
_default_exs = [e for e in ("bybit", "gateio", "okx", "mexc") if e in _exchange_options]
enabled_exs = st.sidebar.multiselect(
    "🌐 Exchanges",
    options=_exchange_options,
    default=_default_exs,
    help="Which exchanges the dashboard shows (charts, health strip, liquidity table). Default: Bybit / Gate.io / OKX / MEXC.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚡ Last N candles load instantly (chart pages cached ~45 s, neighbours pre-built); "
    "older history streams in automatically as you scroll the chart left."
)
limit_15m = st.sidebar.slider("Candles · 15m chart", min_value=100, max_value=3000, value=700, step=100)
limit_1d = st.sidebar.slider("Candles · 1D chart", min_value=100, max_value=1500, value=400, step=50)

table_tf = st.sidebar.selectbox("📊 Liquidity table timeframe", options=["15m", "1d"], index=0)

hide_spot_dupes = st.sidebar.checkbox(
    "🚫 Hide dead spot duplicates",
    value=True,
    help="Perp-first: once BASE/USDT:USDT exists, the collector stops writing the "
         "BASE/USDT spot table and it freezes forever. Hides such spot rows when a "
         "fresher perp table exists for the same base+exchange.",
)

if st.sidebar.button("🔄 Refresh data (clear caches)"):
    load_summary_cached.clear()
    load_candles_cached.clear()
    fetch_live_cached.clear()
    _render_chart_html_cached.clear()
    # module-level stores are not Streamlit caches, so the button has to
    # clear them by hand: without this, "show me the NEW pair" would keep
    # answering from a 10-minute-old pg_catalog snapshot.
    _SCAN_INVENTORY.clear()
    _STITCHED_PAGES.clear()
    _LAST_SCAN_AT.clear()
    _SUMMARY_STORE.clear()
    _SCAN_ATTEMPTS.clear()
    st.rerun()

# Compact layout: no big page title — charts come first with minimal top padding
st.markdown(
    """
    <style>
        .block-container {padding-top: 0.6rem !important; padding-bottom: 0.5rem !important;}
        .stTabs [data-baseweb="tab-list"] {gap: 6px; margin-bottom: 4px; height: 38px;}
        .stTabs [data-baseweb="tab"] {font-size: 14px; padding: 4px 12px;}
        h3 {margin-top: 0.2rem !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

if demo_mode:
    st.caption("🧪 Demo mode — synthetic data (uncheck in sidebar to connect to TimescaleDB)")
else:
    _snap_age = st.session_state.get("_snap_age_15m") or st.session_state.get("_snap_age_1d")
    if _snap_age:
        st.caption(
            f"📸 Pair list from the last scan snapshot ({_snap_age / 60:.0f} min old) — "
            f"rescanning in the background; charts and live data are unaffected."
        )

# ---------------------------------------------------------------------------
# Load summaries for BOTH timeframes (cached)
# ---------------------------------------------------------------------------

if demo_mode:
    df_15m = _demo_summary_cached("15m")
    df_1d = _demo_summary_cached("1d")
else:
    df_15m = load_summary_cached(db_host, db_port, db_user, db_pass, "15m")
    df_1d = load_summary_cached(db_host, db_port, db_user, db_pass, "1d")

    # The scan is bounded by a wall-clock budget so the page paints while the
    # collector writes; say so when it had to cut the pair list short, instead
    # of looking like the collector lost pairs.
    for _tf in ("15m", "1d"):
        # Stored with a timestamp and self-expiring: the flag is set inside a
        # cached function, so nothing would ever clear it on a later complete
        # scan, and mutating session_state while rendering is its own hazard.
        _entry = st.session_state.get(f"_partial_scan_{_tf}")
        _pm = _entry[1] if _entry and time.time() - _entry[0] < 180 else None
        if _pm:
            # Honest about WHEN the list gets better: the retry is backed off
            # while the database stays busy, so "refresh in a moment" would be
            # a promise the app does not keep.
            _since = time.time() - _LAST_SCAN_AT.get(_tf, 0.0)
            _delay = _rescan_delay_sec(_tf)
            if _since < 5.0:
                _when = "a full rescan is running now"
            elif not _LAST_SCAN_AT.get(_tf):
                _when = f"first retry within {int(_delay)} s"
            else:
                _when = f"retry in ~{max(0, int(_delay - _since))} s (backoff)"
            st.warning(
                f"⏳ {_tf} pair list is incomplete this frame: {_pm.get('rows', 0)}/"
                f"{_pm.get('tables', 0)} tables fit in the {settings.dash_scan_budget_sec:.0f}s "
                f"scan budget — {_when}. The charts are unaffected (they query the "
                f"tables directly). Raise DASH_SCAN_BUDGET_SEC if your collector "
                f"keeps the DB busy."
            )

# Keep only pairs of the exchanges enabled in the sidebar — charts tab,
# Prev/Next list, live writer target set and the liquidity table all inherit it.
if enabled_exs and not df_15m.empty:
    df_15m = df_15m[df_15m["exchange"].isin(enabled_exs)]
if enabled_exs and not df_1d.empty:
    df_1d = df_1d[df_1d["exchange"].isin(enabled_exs)]

# Drop ghost pairs: tables whose last-candle timestamp is garbage (corrupted
# future dates / ms epochs). The summary scan sees the raw row, but the chart
# sanitize drops every candle — such pairs must never enter the pair list.
df_15m = filter_sane_summary_rows(df_15m)
df_1d = filter_sane_summary_rows(df_1d)

# Perp-first leftovers: a spot table frozen at the day its perp was listed
# (0G/USDT @bybit: spot 983h behind, perp 0.7h behind). Opening such a pair
# looks like a data bug — candles end weeks before the live price line.
if hide_spot_dupes:
    df_15m = drop_stale_spot_duplicates(df_15m)
    df_1d = drop_stale_spot_duplicates(df_1d)

df_table = df_15m if table_tf == "15m" else df_1d

if not enabled_exs:
    st.warning("⬅️ Select at least one exchange in the sidebar (🌐 Exchanges).")
    st.stop()

if df_15m.empty and df_1d.empty:
    _probe_err = None if demo_mode else _probe_db_error(db_host, db_port, db_user, db_pass)
    if _probe_err:
        st.error(
            f"❌ Cannot connect to TimescaleDB at `{db_host}:{db_port}` as user "
            f"`{db_user}`: **{_probe_err}**\n\n"
            f"Check DB Host/User/Password in the sidebar — defaults now come from "
            f"db_config.py / .env (user `{settings.db_user}`)."
        )
    st.warning(
        "⚠️ No data tables found in historical databases. Ensure PostgreSQL/TimescaleDB is running "
        "or enable 🧪 Demo mode in the sidebar."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Tabs: charts first
# ---------------------------------------------------------------------------

tab_charts, tab_liquidity, tab_info = st.tabs(
    ["📈 Charts (15m + 1D)", "📊 Liquidity Monitor", "ℹ️ Methodology & Algorithms"]
)


def _unique_sorted(values):
    return sorted(set(v for v in values if v))


# Optional pair filter from the Chart options: only tickers that have a 15m
# table. Many pairs have a 1D table but no 15m data — with this enabled the
# pair list and Prev/Next skip them. The flag itself lives in a checkbox
# (key="only_with_15m") inside the Charts tab; reading session_state here is
# safe: on toggle reruns the new value is already set, default False.
_only_15m = bool(st.session_state.get("only_with_15m", False))
if _only_15m and not df_15m.empty:
    _15m_tickers = df_15m["ticker"].dropna().unique().tolist()
    _filtered_opts = _unique_sorted(_15m_tickers)
    if _filtered_opts:  # never lock the UI behind an empty list
        TICKER_OPTIONS = _filtered_opts
    else:
        TICKER_OPTIONS = _unique_sorted(
            ([] if df_15m.empty else df_15m["ticker"].dropna().tolist())
            + ([] if df_1d.empty else df_1d["ticker"].dropna().tolist())
        )
else:
    TICKER_OPTIONS = _unique_sorted(
        ([] if df_15m.empty else df_15m["ticker"].dropna().tolist())
        + ([] if df_1d.empty else df_1d["ticker"].dropna().tolist())
    )

if not TICKER_OPTIONS:
    st.warning(
        ("No 15m pairs for the exchanges enabled in the sidebar — "
         "disable '⏱ Only pairs with 15m data' in ⚙️ Chart options, enable more exchanges, "
         "or let the 15m collector build their tables.")
        if _only_15m
        else (
            "No pairs yet for the exchanges enabled in the sidebar — "
            "enable more exchanges or let the collector build their tables."
        )
    )
    st.stop()


def _nav(delta: int):
    """
    Queues a Prev/Next pair switch and reruns.
    The pending value is applied BEFORE the selectbox (key="sym_ticker") is
    instantiated on the next run — directly modifying st.session_state.sym_ticker
    after widget creation raises StreamlitAPIException.
    """
    _mark_interaction()
    st.session_state.nav_ticker = shift_option(TICKER_OPTIONS, st.session_state.get("sym_ticker"), delta)
    st.rerun()

with tab_charts:
    # --- Pair selection row with Prev / Next (compact, single row) -----------
    nav_l, sel_c, sel_e, lay_c, nav_r = st.columns([1, 3, 2, 2, 1])

    if "sym_ticker" not in st.session_state or st.session_state.sym_ticker not in TICKER_OPTIONS:
        st.session_state.sym_ticker = TICKER_OPTIONS[0]

    # Apply pending Prev/Next navigation BEFORE the pair selectbox is instantiated
    pending = st.session_state.pop("nav_ticker", None)
    if pending in TICKER_OPTIONS:
        st.session_state.sym_ticker = pending

    with nav_l:
        if st.button("◀", key="sel_prev", use_container_width=True, help="Previous pair"):
            _nav(-1)
    with nav_r:
        if st.button("▶", key="sel_next", use_container_width=True, help="Next pair"):
            _nav(+1)

    with sel_c:
        sym_ticker = st.selectbox("Pair", options=TICKER_OPTIONS, key="sym_ticker", label_visibility="collapsed")
        if st.session_state.get("_last_viewed_ticker") != sym_ticker:
            # Picking a pair from the list is as much an interruption as Prev/
            # Next; the sweep yields for a moment either way (scan_pause_sec).
            st.session_state["_last_viewed_ticker"] = sym_ticker
            _mark_interaction()
    with sel_e:
        ex_opts = _unique_sorted(exchanges_for_ticker(df_15m, sym_ticker) + exchanges_for_ticker(df_1d, sym_ticker))
        if not ex_opts or st.session_state.get("sym_ex") not in ex_opts:
            st.session_state.sym_ex = ex_opts[0] if ex_opts else None
        sym_ex = st.selectbox("Exchange", options=ex_opts, key="sym_ex", label_visibility="collapsed")

    with lay_c:
        stacked_layout = st.toggle(
            "⬓ Large stacked",
            value=False,
            key="stacked_layout",
            help="OFF: compact — 15m left, 1D right. ON: large — 15m top, 1D bottom.",
        )

    # --- Chart options (collapsed by default to save vertical space) ---------
    with st.expander("⚙️ Chart options", expanded=False):
        opt1, opt2, opt3, opt4 = st.columns([2, 2, 2, 1])
        atr_days = opt1.slider("🎯 ATR Period (bars)", min_value=1, max_value=30, value=5, step=1)
        chart_engine = opt2.selectbox(
            "📈 Chart Engine",
            options=["TradingView Lightweight Canvas", "TradingView Official Widget", "Plotly Dark"],
        )
        chart_style = opt3.selectbox("📊 Chart Style", options=["OHLCV Bars", "Candlesticks"], index=0)
        show_volume = opt4.checkbox("📊 Show volume bars", value=False, key="show_volume")
        opt5, opt6, opt7, opt8 = st.columns(4)
        live_refresh = opt5.selectbox(
            "🔴 Live refresh", options=["1s", "2s", "5s", "Off"], index=0, key="live_refresh",
        )
        auto_reload = opt6.checkbox("Auto-reload DB (60s)", value=True, key="auto_reload")
        stitch_gaps = opt7.checkbox("🩹 Stitch gaps", value=True, key="stitch_gaps",
                                    help="Fetch missing candles from the exchange into the chart (in-memory).")
        flat_fill = opt7.checkbox(
            "🕳 Flat-fill empty bars", value=True, key="flat_fill",
            help="Intervals in which nothing traded have no candle on the exchange API. "
                 "Draw them flat at the previous close (like the exchange's own chart) "
                 "instead of leaving a hole.",
        )
        opt8.checkbox(
            "⏱ Only pairs with 15m data", value=False, key="only_with_15m",
            help="Pair list and Prev/Next skip tickers that only have a 1D table "
                 "(no 15m candles). Default OFF.",
        )

    # Resolve table rows per timeframe (same exchange preferred)
    row_15m = find_table_row(df_15m, sym_ticker, sym_ex)
    row_1d = find_table_row(df_1d, sym_ticker, sym_ex)
    live_interval = 0.0 if live_refresh == "Off" else float(live_refresh[:-1])
    ccxt_id = settings.exchange_map_1d.get(sym_ex, sym_ex)
    # TimescaleDB table row all live widgets read from (written every second
    # by the background live daemon for the current pair ± 5 neighbours).
    live_db = None if demo_mode else _live_db_for(row_1d, row_15m)

    # Daily candles for the live health strip chips (ATR & Min 7d $Vol).
    # Reused by the ATR metrics below the charts — cache makes it free.
    hist_1d_live = None
    if row_1d is not None:
        hist_1d_live = get_candles("1d", row_1d, max(limit_1d, 60), demo_mode)
    atr_live = 0.0
    if hist_1d_live is not None and not hist_1d_live.empty and len(hist_1d_live) >= 3:
        atr_live = compute_atr_no_paranormal_bars(
            highs=hist_1d_live["high"].to_numpy(dtype=float),
            lows=hist_1d_live["low"].to_numpy(dtype=float),
            closes=hist_1d_live["close"].to_numpy(dtype=float),
            period=atr_days,
            small_threshold=settings.atr_small_threshold,
            large_threshold=settings.atr_large_threshold,
        )

    # --- Compact health strip — LIVE from the exchange every ~1s -------------
    # (Tape / Depth ±1% / Spread % ATR via orderbook & trade tape, Min 7d $Vol
    #  from fresh daily candles; falls back to the collector snapshot values
    #  only for fields the live fetch couldn't produce)
    if live_interval > 0 and hasattr(st, "fragment") and not demo_mode:
        @st.fragment(run_every=live_interval)
        def _strip_fragment():
            live_row = _compute_live_health_row(
                sym_ticker, sym_ex, hist_1d_live, atr_live, row_1d or row_15m,
                db_name=live_db,
            )
            st.markdown(build_health_strip_html(live_row), unsafe_allow_html=True)

        _strip_fragment()
    else:
        health_row = row_1d or row_15m
        if health_row:
            st.markdown(build_health_strip_html(health_row), unsafe_allow_html=True)

    # --- Spot/Swap links + shortability badge --------------------------------
    from src.exchanges.symbol_selector import split_symbol
    _base, _quote = split_symbol(sym_ticker or "")
    _perp = None if ":" in sym_ticker else find_perp_ticker([df_15m, df_1d], _base, sym_ex)
    st.markdown(build_pair_links_html(sym_ticker, sym_ex, _perp), unsafe_allow_html=True)

    # --- LIVE panel (auto-refreshing chips, ~1s) ------------------------------
    if live_interval > 0 and hasattr(st, "fragment"):
        @st.fragment(run_every=live_interval)
        def _live_fragment():
            _render_live_panel(sym_ticker, sym_ex, demo_mode, db_name=live_db)

        _live_fragment()
    else:
        _render_live_panel(sym_ticker, sym_ex, demo_mode, db_name=live_db)

    def _render_stitch_caption(stitch_txt: str) -> None:
        # Always render the stitch caption line at a fixed height (empty when
        # nothing was stitched) so the side-by-side 15m / 1D charts stay
        # perfectly level — previously the 15m chart was pushed ~20px down
        # whenever it had a stitched-gap caption and the 1D chart did not.
        st.markdown(
            f"<div style='font-size:12px;color:#808495;height:20px;line-height:20px;"
            f"margin:0 0 2px 4px;white-space:nowrap;overflow:hidden;'>{stitch_txt}</div>",
            unsafe_allow_html=True,
        )

    def render_chart(row, tf_label, limit, interval, chart_height=470):
        """Renders one timeframe chart into the current container."""
        if row is None:
            st.info(f"No {tf_label} table for {sym_ticker}.")
            return

        # --- FAST PATH: Lightweight Charts against the real DB --------------
        # The whole chart page (candles → stitch → merge → compact JSON →
        # HTML) is one st.cache_data entry pre-warmed for the ±5 neighbours,
        # so Prev/Next flipping costs a memory lookup + iframe refresh.
        if chart_engine == "TradingView Lightweight Canvas" and not demo_mode:
            step = 900 if tf_label == "15m" else 86400
            interval_ms = int(live_interval * 1000) if live_interval > 0 else 0
            # Chart poller reads the DB live row through the dashboard's own
            # /tick endpoint (host resolved in-browser); direct exchange REST
            # is kept only as fallback. Works for all 9 exchanges and never
            # gives up on errors. The same tiny endpoint also serves /candles
            # history chunks for infinite left-scroll.
            _infra = _live_infra_or_none() if live_db else None
            tick_path = _live_tick_path(live_db, sym_ex, sym_ticker) if (_infra and interval_ms > 0) else None
            poller_js = build_live_poller_js(
                sym_ex, sym_ticker, step, interval_ms,
                tick_path=tick_path,
                tick_port=(_infra or {}).get("tick_port"),
            )
            hist_js = ""
            if _infra and _infra.get("tick_port"):
                hist_js = build_history_loader_js(
                    row["db_name"], row["table_name"], step,
                    _infra["tick_port"], chunk=1200 if tf_label == "15m" else 700,
                )
            m_db = m_tbl = ""
            m_lim = 0
            if tf_label == "1D" and row_15m is not None:
                m_db, m_tbl = row_15m["db_name"], row_15m["table_name"]
                m_lim = max(limit_15m, 200)
            page_kwargs, store_key = _chart_page_args(
                row, tf_label, limit, m_db, m_tbl, m_lim,
                ccxt_id, sym_ticker, sym_ex,
                chart_style, chart_height, bool(show_volume),
                poller_js, hist_js, bool(flat_fill),
            )

            _t_start = time.perf_counter()
            _t_build = _t_start       # before this: Streamlit, widgets, the summary
            # Progressive paint. The DB page renders NOW (one query, no
            # network to the exchange); the gap stitch — which pages the
            # exchange for up to DASH_STITCH_BUDGET_SEC and was previously the
            # first thing the render path did — runs in a daemon thread and
            # swaps its page in when it lands.
            variant, should_warm = chart_render_plan(
                _STITCHED_PAGES.get(store_key), bool(stitch_gaps),
                time.time(), CHART_PAGE_TTL_SEC,
            )
            if should_warm:
                _bg(("stitch", store_key), _warm_stitched_page, store_key, page_kwargs)
                _CURRENT_CHART_KEYS.add(store_key)

            if variant == "stitched":
                entry = _STITCHED_PAGES[store_key]
                html_code, stitch_txt = entry["html"], entry["txt"]
                _remember(_DISPLAYED_HASH, store_key, entry.get("hash"))
                _report_switch(sym_ticker, tf_label, "warmed page", _t_start, _t_build)
            else:
                _built_before = _CHART_BUILDS
                _t_build = time.perf_counter()
                res = _render_chart_html_cached(**page_kwargs, stitch_enabled=False)
                _report_switch(
                    sym_ticker, tf_label,
                    "built from DB" if _CHART_BUILDS != _built_before else "cached page",
                    _t_start, _t_build,
                )
                if res is None:
                    st.info(f"No {tf_label} candles available for {sym_ticker} ({sym_ex}).")
                    return
                html_code, stitch_txt = res
                _remember(_DISPLAYED_HASH, store_key, hash(html_code or ""))
                if stitch_gaps:
                    # APPENDED, never replaced: the plain page may be carrying
                    # the red "collector Xh behind, chart tail may be missing"
                    # warning, and hiding it behind a progress note would be
                    # exactly the silence this app keeps having to unlearn.
                    stitch_txt = (stitch_txt or "&nbsp;") + (
                        " &nbsp;·&nbsp; 🩹 filling missing candles in the background"
                    )
            _render_stitch_caption(stitch_txt)
            _html_component(html_code, chart_height + 10 + HIST_STATUS_HEIGHT)
            return

        # --- LEGACY PATH (demo mode / TradingView widget / Plotly) ----------
        hist_df = get_candles(tf_label, row, limit, demo_mode)

        def _stitch(frame, timeframe):
            """Gap + closed-tail stitching from the exchange (in-memory)."""
            if stitch_gaps and not demo_mode and frame is not None and len(frame) > 1:
                step = 900 if timeframe == "15m" else 86400
                frame, added = stitch_candle_gaps(
                    frame,
                    lambda r0, r1: _fetch_missing_candles_cached(ccxt_id, sym_ticker, timeframe, r0, r1),
                    step,
                )
                return frame, added
            return frame, 0

        hist_df, stitched = _stitch(hist_df, "15m" if tf_label == "15m" else "1d")
        # When the stitch covers a stale DB tail, say how far the COLLECTOR
        # is behind, so a big '47 missing candles' hint is not mistaken for a
        # stitch bug: it means the engine has not persisted this table lately.
        stale_hint = ""
        if stitched and row is not None:
            try:
                _mt = int(row.get("max_ts"))
            except (TypeError, ValueError):
                _mt = None
            if _mt:
                if _mt > 1e11:  # ms epoch defence (frame-level filter already ran)
                    _mt //= 1000
                _age_h = (time.time() - _mt) / 3600.0
                _thr = 1.0 if tf_label == "15m" else 49.0  # 1D: last closed day is ~24-48h back, normal
                if _age_h > _thr:
                    stale_hint = f" · ⏳ collector {_age_h:.1f}h behind"
        stitch_txt = (
            f"🩹 {stitched} missing {tf_label} candles stitched from exchange (in-memory){stale_hint}"
            if stitched
            else "&nbsp;"
        )
        _render_stitch_caption(stitch_txt)

        # Keep the daily chart in sync: aggregate fresher 15m candles of today
        # (the 15m frame is stitched too, so today's daily bar is always fresh)
        if tf_label == "1D" and row_15m is not None:
            df15 = get_candles("15m", row_15m, max(limit_15m, 200), demo_mode)
            df15, _ = _stitch(df15, "15m")
            hist_df = merge_intraday_into_daily(hist_df, df15)

        if flat_fill:
            hist_df, _filled = fill_missing_bars(
                hist_df, 900 if tf_label == "15m" else 86400
            )

        if chart_engine == "TradingView Lightweight Canvas":
            step = 900 if tf_label == "15m" else 86400
            interval_ms = int(live_interval * 1000) if live_interval > 0 else 0
            # Chart poller reads the DB live row through the dashboard's own
            # /tick endpoint (host resolved in-browser); direct exchange REST
            # is kept only as fallback. Works for all 9 exchanges and never
            # gives up on errors.
            _infra = _live_infra_or_none() if (live_db and interval_ms > 0) else None
            tick_path = _live_tick_path(live_db, sym_ex, sym_ticker) if _infra else None
            poller_js = build_live_poller_js(
                sym_ex, sym_ticker, step, interval_ms,
                tick_path=tick_path,
                tick_port=(_infra or {}).get("tick_port"),
            )
            render_tradingview_lightweight_chart(
                hist_df, sym_ticker, sym_ex, tf_label,
                chart_height=chart_height,
                chart_style=chart_style,
                show_volume=show_volume,
                live_poller_js=poller_js,
            )
        elif chart_engine == "TradingView Official Widget":
            style_code = "0" if chart_style == "OHLCV Bars" else "1"
            render_tradingview_official_widget(sym_ticker, sym_ex, interval=interval, style_code=style_code)
        else:
            render_plotly_chart(hist_df, sym_ticker, sym_ex, tf_label, chart_style)

    def _slim_header(icon, tf_name):
        st.markdown(
            f"<div style='font-size:15px;margin:2px 0 2px 6px;'>{icon} <b>{sym_ticker}</b> · {tf_name} · {sym_ex}</div>",
            unsafe_allow_html=True,
        )

    def _side_nav_button(delta, key):
        if st.button("⏪" if delta < 0 else "⏭", key=key, use_container_width=True,
                     help="Previous pair" if delta < 0 else "Next pair"):
            _nav(delta)
        st.markdown("<div style='height:170px'></div>", unsafe_allow_html=True)

    # Which chart slots this run rendered, and whether a background page is
    # waiting to be swapped into them (see _stitch_swap_watcher below).
    _CURRENT_CHART_KEYS.clear()

    if stacked_layout:
        # LARGE MODE: 15m on top, 1D below, nav buttons flanking each chart
        _slim_header("⏱", "15m")
        left, center, right = st.columns([1, 30, 1])
        with left:
            _side_nav_button(-1, "nav_prev_15m")
        with right:
            _side_nav_button(+1, "nav_next_15m")
        with center:
            render_chart(row_15m, "15m", limit_15m, "15", chart_height=470)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        _slim_header("📅", "1D")
        left2, center2, right2 = st.columns([1, 30, 1])
        with left2:
            _side_nav_button(-1, "nav_prev_1D")
        with right2:
            _side_nav_button(+1, "nav_next_1D")
        with center2:
            render_chart(row_1d, "1D", limit_1d, "D", chart_height=470)
    else:
        # COMPACT MODE (default): 15m on the LEFT, 1D on the RIGHT, shared nav
        nvl, c15, c1d, nvr = st.columns([1, 15, 15, 1])
        with nvl:
            _side_nav_button(-1, "nav_prev_pair")
        with nvr:
            _side_nav_button(+1, "nav_next_pair")
        with c15:
            _slim_header("⏱", "15m")
            render_chart(row_15m, "15m", limit_15m, "15", chart_height=430)
        with c1d:
            _slim_header("📅", "1D")
            render_chart(row_1d, "1D", limit_1d, "D", chart_height=430)

    # --- Swap a background-stitched chart in, once, and only if it differs --
    # The stitch runs off the render path, so its page has to reach the screen
    # somehow: this fragment (1 s, and its body is two dict lookups) triggers
    # one app rerun when a landed page differs from what is displayed.
    # Three reasons NOT to repaint: the stitch filled nothing (the healthy
    # case — the DB page already matches), the page is byte-identical, or the
    # last swap was too recent (a repaint resets zoom/pan and panning is the
    # main thing a user does while looking at a chart).
    if stitch_gaps and not demo_mode and hasattr(st, "fragment"):
        @st.fragment(run_every=1.0)
        def _stitch_swap_watcher():
            now = time.time()
            for _k in list(_CURRENT_CHART_KEYS):
                _ent = _STITCHED_PAGES.get(_k)
                if not _ent or _ent.get("hash") == _DISPLAYED_HASH.get(_k):
                    continue
                if _stitched_candle_count(_ent.get("txt") or "") <= 0:
                    _CURRENT_CHART_KEYS.discard(_k)   # nothing to show, stop watching
                    continue
                if now - _LAST_SWAP_AT.get(_k, 0.0) < CHART_SWAP_COOLDOWN_SEC:
                    continue
                _remember(_LAST_SWAP_AT, _k, now)
                st.rerun(scope="app")
                return

        _stitch_swap_watcher()

    # --- Open Interest, Funding Rate & Spread history ----------------------
    # Line panels under the two candle charts. OI points accumulate per engine
    # cycle (15m table); funding is the realized 8h-event history backfilled
    # once per table (1D table, day's last event); spread comes from the
    # per-cycle orderbook snapshots (ob_spread_pct) — collected for BOTH perps
    # and spot, so the section works for spot pairs too (spread panel only).
    if not demo_mode and (row_15m is not None or row_1d is not None):
        _dense_row = row_15m or row_1d          # per-cycle snapshot history
        _is_perp = ":" in sym_ticker
        oi_points = []
        fr_points = []
        spread_points = []
        if _dense_row is not None:
            spread_points = metric_points_cached(
                db_host, db_port, db_user, db_pass,
                _dense_row["db_name"], _dense_row["table_name"], "ob_spread_pct",
            )
        if _is_perp:
            if row_15m is not None:
                oi_points = metric_points_cached(
                    db_host, db_port, db_user, db_pass,
                    row_15m["db_name"], row_15m["table_name"], "open_interest",
                )
            if row_1d is not None:
                fr_points = metric_points_cached(
                    db_host, db_port, db_user, db_pass,
                    row_1d["db_name"], row_1d["table_name"], "funding_rate",
                )

        def _span(points) -> str:
            n = len(points)
            if not n:
                return "no data points yet"
            f = time.strftime("%Y-%m-%d", time.localtime(points[0][0]))
            t = time.strftime("%Y-%m-%d", time.localtime(points[-1][0]))
            return f"{n} pts · {f} → {t}"

        _panels = []
        if _is_perp:
            _panels.append(("🔵 Open Interest", oi_points, f"OI {sym_ticker} · {sym_ex}",
                            "#4c9aff", 2, 0.01, "15m table"))
            _panels.append(("🟣 Funding Rate", fr_points, f"Funding {sym_ticker} · {sym_ex}",
                            "#a26bff", 6, 0.000001, "1D (last 8h event of the day)"))
        _panels.append(("🟠 Spread %", spread_points, f"Spread % {sym_ticker} · {sym_ex}",
                        "#ff9f43", 4, 0.0001, "orderbook snapshots"))
        _have_data = any(p[1] for p in _panels)

        st.markdown("---")
        st.markdown("#### 📊 Open Interest & Funding Rate" if _is_perp else "#### 📊 Spread History")
        if not _have_data:
            st.caption(
                "Empty for now — OI accumulates from every 15m engine cycle, "
                "funding history appears after the one-off backfill run at "
                "engine start, and spread comes from periodic orderbook "
                "snapshots (both need a few cycles after an engine restart)."
                if _is_perp else
                "Empty for now — spread accumulates from orderbook snapshots "
                "during the first cycles after an engine restart."
            )
        else:
            _dumps = lambda x: json.dumps(x, separators=(",", ":"))
            for _col, (_label, _pts, _title, _color, _prec, _mm, _src) in zip(
                st.columns(len(_panels)), _panels
            ):
                with _col:
                    st.markdown(
                        f"<div style='font-size:12px;color:#808495;margin:0 0 2px 4px'>"
                        f"{_label} · {_span(_pts)} · {_src}</div>",
                        unsafe_allow_html=True,
                    )
                    if _pts:
                        _html_component(
                            build_metric_chart_html(
                                _dumps(_pts), _title, _color, 230,
                                precision=_prec, min_move=_mm,
                            ),
                            240,
                        )

    # --- Live market & orderbook metrics (below charts, never blocks them) ---
    st.markdown("---")
    st.markdown("#### 🔴 Live Market & Orderbook Metrics")

    # Daily candles + ATR already loaded above for the live health strip —
    # reuse them here (the 60s candle cache makes the call free anyway).
    hist_1d = hist_1d_live
    atr_val = atr_live

    db_row = row_1d or row_15m or {}
    snap_key = (sym_ticker, sym_ex, round(float(atr_val or 0.0), 10))

    def _render_metrics(src):
        """Metric cards from `src` (live snapshot when ready, else DB snapshot)."""
        g = lambda k, d=0.0: float(src.get(k, d) or d)

        mid_price = (g("ob_best_bid") + g("ob_best_ask")) / 2.0
        if not mid_price:
            mid_price = g("close")
        spread_abs = g("ob_spread_abs")
        spread_pct = g("ob_spread_pct")
        spread_atr_pct = (spread_abs / atr_val * 100.0) if atr_val > 0 else 0.0

        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.metric(
            "Live Mid Price", f"${mid_price:,.4f}",
            delta=f"Bid: ${g('ob_best_bid', mid_price):,.4f} | Ask: ${g('ob_best_ask', mid_price):,.4f}",
        )
        mcol2.metric("Live Spread", f"${spread_abs:.4f}", delta=f"{spread_pct:.3f}%")
        mcol3.metric(f"ATR w/o Paranormal Bars ({atr_days}d)", f"${atr_val:.4f}")
        mcol4.metric("Spread % of ATR", f"{spread_atr_pct:.2f}%", delta=f"Relative to {atr_days}d ATR")
        mcol5.metric("Vitality Score", f"{g('ob_vitality_score'):.1f} / 10", delta=f"Grade {src.get('ob_vitality_grade', 'N/A')}")

        st.markdown("##### 📖 Live Orderbook & Trade Tape Metrics")
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        dcol1.metric(
            "Total Depth (±1% $)", f"${g('ob_total_depth_usd'):,.2f}",
            delta=f"Bid: ${g('ob_bid_depth_usd'):,.0f} | Ask: ${g('ob_ask_depth_usd'):,.0f}",
        )
        dcol2.metric("Orderbook Imbalance", f"{g('ob_imbalance'):.2f}", delta="Bid / Ask Depth Ratio")
        dcol3.metric("5m Cumulative Vol Delta (CVD)", f"${g('ob_cvd_5m'):,.2f}", delta=f"Buy Pressure: {g('ob_buy_pressure_pct'):.1f}%")
        dcol4.metric("Trade Activity", f"{g('ob_trades_per_min'):.1f} trades/min")

    # The full snapshot fetch (exchange ctor + markets + orderbook + trades)
    # takes seconds — never run it synchronously in the UI path. Show the best
    # available data instantly and refresh in the background.
    if demo_mode or not (sym_ticker and sym_ex):
        _render_metrics(db_row)
    elif hasattr(st, "fragment"):
        @st.fragment(run_every=3.0)
        def _metrics_fragment():
            entry = _LIVE_SNAP.get(snap_key)
            if entry is None or time.time() - entry[1] > 25:
                _bg(
                    ("live", snap_key),
                    lambda: _warm_live_snapshot(sym_ticker, sym_ex, ccxt_id, atr_val),
                )
            _render_metrics((entry[0] if entry else None) or db_row)

        _metrics_fragment()
    else:
        with st.spinner(f"Fetching live orderbook for {sym_ticker} on {sym_ex}…"):
            _warm_live_snapshot(sym_ticker, sym_ex, ccxt_id, atr_val)
        _render_metrics((_LIVE_SNAP.get(snap_key) or (None, None))[0] or db_row)

    # --- Background warming of ±5 neighbour pairs (instant Prev/Next) --------
    # Kicks daemon threads that pre-fill every cache the neighbour views need:
    # candle frames, exchange gap/tail stitch fetches, live chips feeds, and
    # (±2) the full orderbook snapshot. Zero blocking on pair switch.
    # The same ±5 set becomes the live writer's target: every second the live
    # daemon persists price / orderbook / spread / tape stats of exactly these
    # pairs into TimescaleDB, and all live widgets above just read those rows.
    if not demo_mode and sym_ticker and sym_ex and TICKER_OPTIONS:
        try:
            _infra_main = _live_infra()  # starts the writer daemon + /tick + /candles endpoints (runs once)
            # Current chart-rendering parameters shared by every neighbour
            # warm, so the pre-built HTML cache keys match the render path.
            _chart_ctx = {
                "interval_ms": int(live_interval * 1000) if live_interval > 0 else 0,
                "tick_port": (_infra_main or {}).get("tick_port"),
                "style": chart_style,
                "height": 470 if stacked_layout else 430,
                "volume": bool(show_volume),
                "stitch": bool(stitch_gaps),
                "flat_fill": bool(flat_fill),
            }
            live_pairs = []
            seen = set()
            cur_idx = TICKER_OPTIONS.index(sym_ticker)
            for delta in range(-5, 6):
                nb_ticker = TICKER_OPTIONS[(cur_idx + delta) % len(TICKER_OPTIONS)]
                if nb_ticker in seen:
                    continue
                seen.add(nb_ticker)
                nb_15 = find_table_row(df_15m, nb_ticker, sym_ex)
                nb_1d = find_table_row(df_1d, nb_ticker, sym_ex)
                if delta != 0 and not (nb_15 or nb_1d):
                    continue
                entry = {
                    "db": _live_db_for(nb_1d, nb_15) or live_db,
                    "ex": sym_ex, "ccxt": ccxt_id, "sym": nb_ticker,
                    "cur": delta == 0,
                }
                # the displayed pair leads the list: it gets the full live
                # payload here and survives the engine-side cap
                live_pairs.insert(0, entry) if delta == 0 else live_pairs.append(entry)
                if delta == 0:
                    continue
                if abs(delta) > settings.dash_warm_neighbors:
                    continue  # low-resource mode: fewer background warm bursts
                if not _warm_pair_due(nb_ticker, sym_ex, time.time()):
                    continue     # warmed within this page's lifetime already
                _bg(
                    ("pair", nb_ticker, sym_ex),
                    lambda t=nb_ticker, r15=nb_15, r1=nb_1d, fl=abs(delta) <= 2, ctx=_chart_ctx:
                        _warm_pair_caches(
                            t, sym_ex, ccxt_id, r15, r1,
                            limit_15m, limit_1d, atr_days, fl,
                            chart_ctx=ctx,
                        ),
                )
            _set_live_target([p for p in live_pairs if p["db"]])
        except Exception:
            pass
    else:
        _set_live_target([])  # demo mode / no pair selected → live writer idles

    # --- Auto-reload DB data (full app rerun every 60 s) ---------------------
    if auto_reload and hasattr(st, "fragment"):
        def _auto_reload():
            # Rerun only on scheduled ticks, never on the initial creation run
            if getattr(_auto_reload, "armed", False):
                st.rerun(scope="app")
            _auto_reload.armed = True

        st.fragment(run_every=60.0)(_auto_reload)()

with tab_liquidity:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Pair Tables (1D)", f"{len(df_1d):,}")
    col2.metric("Total Pair Tables (15M)", f"{len(df_15m):,}")
    col3.metric("Liquid HIGH Tier (1D)", f"{0 if df_1d.empty else len(df_1d[df_1d['volume_tier'] == 'HIGH']):,}")
    col4.metric("Active Exchanges (1D)", 0 if df_1d.empty else len(df_1d["exchange"].dropna().unique()))
    col5.metric(
        "Vitality Grade A/B (1D)",
        0 if df_1d.empty else len(df_1d[df_1d["ob_vitality_grade"].isin(["A", "B"])]),
    )

    st.subheader(f"Liquidity & Orderbook Metrics Table ({table_tf})")

    if df_table.empty:
        st.info(f"No tables found for timeframe '{table_tf}'.")
    else:
        fcol1, fcol2, fcol3 = st.columns(3)
        ex_options = sorted([e for e in df_table["exchange"].dropna().unique()])
        selected_ex = fcol1.multiselect("Filter by Exchange", options=ex_options, default=ex_options, key="liq_ex")
        selected_tier = fcol2.multiselect("Filter by Volume Tier", options=["HIGH", "LOW"], default=["HIGH", "LOW"], key="liq_tier")
        grade_options = [gr for gr in ["A", "B", "C", "D", "F"] if gr in df_table["ob_vitality_grade"].values]
        selected_grade = fcol3.multiselect(
            "Filter by Vitality Grade", options=["A", "B", "C", "D", "F"],
            default=grade_options or ["A", "B"], key="liq_grade",
        )

        filtered_df = df_table[
            (df_table["exchange"].isin(selected_ex))
            & (df_table["volume_tier"].isin(selected_tier))
            & (df_table["ob_vitality_grade"].isin(selected_grade))
        ]

        display_cols = [
            "ticker", "exchange", "asset_type", "volume_tier", "close",
            "ob_vitality_grade", "ob_vitality_score", "ob_spread_pct",
            "ob_spread_atr_pct", "ob_atr_no_paranormal", "ob_cvd_5m", "ob_min_7d_volume_usd",
        ]
        available_cols = [c for c in display_cols if c in filtered_df.columns]

        st.dataframe(
            filtered_df[available_cols].sort_values(by="ob_vitality_score", ascending=False),
            width="stretch",
            column_config={
                "close": st.column_config.NumberColumn("Close Price", format="$%.4f"),
                "ob_spread_pct": st.column_config.NumberColumn("Spread %", format="%.3f%%"),
                "ob_spread_atr_pct": st.column_config.NumberColumn("Spread % of ATR", format="%.2f%%"),
                "ob_atr_no_paranormal": st.column_config.NumberColumn("ATR w/o Paranormal Bars", format="%.4f"),
                "ob_cvd_5m": st.column_config.NumberColumn("CVD (5m $)", format="$%.2f"),
                "ob_min_7d_volume_usd": st.column_config.NumberColumn("Min 7d Vol $", format="$%,.0f"),
            },
        )

with tab_info:
    st.subheader("ℹ️ Methodology & Key Algorithms")
    st.markdown(r"""
    ### 1. ATR without Paranormal Bars (Filtered Robust ATR)
    Standard **ATR (Average True Range)** is highly sensitive to single abnormal candles — *paranormal bars* (news spikes, squeezes, false breakouts).

    Our algorithm filters out bars whose range falls outside the threshold window:
    $$\text{Bar Range} \notin [0.5 \times \text{ATR}, 1.8 \times \text{ATR}]$$
    and iteratively recalculates robust volatility reflecting true average daily asset movement over the **user-selected calculation period (N bars)**.

    ---

    ### 2. Spread % of ATR
    The ratio of current absolute orderbook spread to ATR without paranormal bars:
    $$\text{Spread \% of ATR} = \frac{\text{Ask} - \text{Bid}}{\text{ATR}_{\text{robust}}} \times 100\%$$

    ---

    ### 3. Fast Dual-Timeframe Charts
    The Charts tab renders the **15m chart on top and the 1D chart below** for the selected pair,
    with **⏪ Prev / Next ⏭** buttons flanking every chart. Table summaries are cached for 10 minutes,
    candle frames for 60 seconds, and each chart loads only the last N candles (configurable in the
    sidebar) — so switching pairs is effectively instant. The filtered ATR value itself is still
    computed live for the metrics cards below the charts.

    ---

    ### 4. Perp-First Selection Strategy
    For every base asset, the system prioritizes perpetual linear contracts (`BTC/USDT:USDT`). Spot markets are loaded only if a perpetual contract does not exist on that exchange.

    ---

    ### 5. Historical 4-Database Storage Architecture
    Connects directly to the 4 historical PostgreSQL / TimescaleDB databases:
    * **1D High Volume:** `ohlcv_1d_data_for_usdt_pairs_using_ccxt_and_direct_api1`
    * **1D Low Volume:** `ohlcv_1d_data_for_low_vol_usdt_pairs_using_ccxt_and_direct_api1`
    * **15M High Volume:** `ohlcv_15m_data_for_usdt_pairs_using_ccxt_and_direct_api1`
    * **15M Low Volume:** `ohlcv_15m_data_for_low_vol_usdt_pairs_using_ccxt_and_direct_api1`
    """)
