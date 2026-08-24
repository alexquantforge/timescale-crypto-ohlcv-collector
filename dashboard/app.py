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
import sys
import json
import time
import asyncio
import threading
from typing import Optional

import asyncpg
import numpy as np
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
    merge_intraday_into_daily,
    build_live_poller_js,
    build_lightweight_chart_html,
    find_missing_bucket_ranges,
    stitch_candle_gaps,
)

st.set_page_config(
    page_title="Timescale Crypto OHLCV Collector",
    page_icon="📈",
    layout="wide",
)


def _html_component(html: str, height: int):
    """Renders raw HTML: st.iframe on new Streamlit, components.html on older versions."""
    try:
        st.iframe(html, height=height, scrolling=False)
    except (AttributeError, TypeError):
        components.html(html, height=height)

SUMMARY_COLUMNS = """
    ticker, exchange, asset_type,
    "Timestamp" as max_ts, close, volume,
    ob_vitality_score, ob_vitality_grade,
    ob_spread_abs, ob_spread_pct, ob_spread_atr_pct, ob_atr_no_paranormal,
    ob_best_bid, ob_best_ask, ob_bid_depth_usd, ob_ask_depth_usd,
    ob_cvd_5m, ob_total_depth_usd, ob_min_7d_volume_usd,
    ob_imbalance, ob_trades_per_min, ob_buy_pressure_pct, ob_is_barcode
"""


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

async def _scan_database(db_name: str, tier_label: str, db_host, db_port, db_user, db_pass, pool_size: int = 6):
    """Scans all %_on_% tables of one database in parallel and returns last-row summaries."""
    try:
        pool = await asyncpg.create_pool(
            host=db_host, port=db_port, user=db_user, password=db_pass,
            database=db_name, min_size=1, max_size=pool_size, command_timeout=30,
        )
    except Exception:
        return []

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name LIKE '%_on_%'"
            )
        tables = [r["table_name"] for r in rows]

        sem = asyncio.Semaphore(pool_size)

        async def fetch_one(tbl: str):
            async with sem:
                try:
                    async with pool.acquire() as conn:
                        row = await conn.fetchrow(
                            f'SELECT {SUMMARY_COLUMNS} FROM "{tbl}" ORDER BY "Timestamp" DESC LIMIT 1'
                        )
                except Exception:
                    return None
            if not row:
                return None
            d = dict(row)
            d["table_name"] = tbl
            d["db_name"] = db_name
            d["volume_tier"] = tier_label
            return d

        results = await asyncio.gather(*[fetch_one(t) for t in tables])
        return [r for r in results if r]
    finally:
        await pool.close()


async def _load_summary(db_host, db_port, db_user, db_pass, timeframe: str) -> pd.DataFrame:
    if timeframe == "15m":
        dbs = [("HIGH", settings.db_high_15m), ("LOW", settings.db_low_15m)]
    else:
        dbs = [("HIGH", settings.db_high_1d), ("LOW", settings.db_low_1d)]

    scans = await asyncio.gather(
        *[_scan_database(db, tier, db_host, db_port, db_user, db_pass) for tier, db in dbs]
    )
    all_rows = [r for scan in scans for r in scan]
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


@st.cache_data(ttl=600, show_spinner="📡 Scanning TimescaleDB tables…")
def load_summary_cached(db_host, db_port, db_user, db_pass, timeframe: str) -> pd.DataFrame:
    """Cached (10 min) summary of the 2 databases for a timeframe."""
    try:
        return asyncio.run(_load_summary(db_host, db_port, db_user, db_pass, timeframe))
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
    """Cached (60 s) candle frame per table, so pair switching is instant."""
    try:
        return asyncio.run(_load_candles(db_name, table_name, limit, db_host, db_port, db_user, db_pass))
    except Exception as e:
        st.warning(f"Could not load candles for {table_name}: {e}")
        return pd.DataFrame()


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

    out = []
    cursor = r0 * step_ms
    try:
        for _ in range(6):  # up to 6 pages per gap range
            batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
            if not batch:
                break
            for c in batch:
                b = int(c[0]) // step_ms
                if r0 <= b < r1:
                    out.append(c)
            if batch[-1][0] >= (r1 - 1) * step_ms:
                break
            cursor = batch[-1][0] + step_ms
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
            exchange_map = settings.exchange_map_1d
            ccxt_id = exchange_map.get(exchange, exchange)
            data = _fetch_ticker_cached(ccxt_id, ticker)

    if not data or not data.get("last"):
        st.caption("🔴 LIVE: exchange unreachable")
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


def _set_live_target(pairs: list) -> None:
    """Replaces the writer's working set (current pair ± 5 neighbours)."""
    with _LIVE_TARGET_LOCK:
        _LIVE_TARGET["ts"] = time.time()
        _LIVE_TARGET["pairs"] = list(pairs or [])


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
                        database=db_name, min_size=1, max_size=3, command_timeout=15,
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

        infra["submit_upsert"] = submit_upsert
        infra["submit_select"] = submit_select
        ready.put(infra)
        loop.run_forever()

    threading.Thread(target=pg_thread, daemon=True, name="live-pg").start()
    infra = ready.get(timeout=10)

    # --- tiny JSON endpoint for the in-chart pollers (/tick?db&ex&sym) -------
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse as _urlparse, parse_qs as _parse_qs
    import re as _re

    _EX_RE = _re.compile(r"[A-Za-z0-9_\-]{1,32}")
    _SYM_RE = _re.compile(r"[A-Za-z0-9/:\._\-]{1,64}")

    class _TickHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def _send(self, obj, status=200):
            body = json.dumps(obj).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
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
                if u.path != "/tick":
                    self._send({}, status=404)
                    return
                q = _parse_qs(u.query)
                db = (q.get("db") or [""])[0]
                ex = (q.get("ex") or [""])[0]
                sym = (q.get("sym") or [""])[0]
                if (
                    db not in _KNOWN_DBS
                    or not _EX_RE.fullmatch(ex or "")
                    or not _SYM_RE.fullmatch(sym or "")
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
        """Fetches ticker + orderbook top + trade tape; returns payload dict."""
        ex = writer_exchange(entry["ccxt"])
        sym = entry["sym"]

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
                    infra["submit_upsert"](e["db"], e["ex"], e["sym"], payload)
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
        ob = _fetch_orderbook_top(ccxt_id, sym_ticker)
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
        trades = _fetch_trade_tape(ccxt_id, sym_ticker)
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


def _bg(key, fn) -> None:
    """Runs fn() in a daemon thread, deduplicated per key while running."""
    with _BG_LOCK:
        if key in _BG_RUNNING:
            return
        _BG_RUNNING.add(key)

    def _run():
        try:
            fn()
        except Exception:
            pass
        finally:
            with _BG_LOCK:
                _BG_RUNNING.discard(key)

    threading.Thread(target=_run, daemon=True).start()


def _warm_live_snapshot(ticker: str, exchange_name: str, ccxt_id: str, atr_val: float) -> None:
    """Background fill of the live orderbook snapshot used by metrics cards."""
    key = (ticker, exchange_name, round(float(atr_val or 0.0), 10))
    try:
        snap = fetch_live_cached(ticker, exchange_name, ccxt_id, float(atr_val or 0.0))
    except Exception:
        snap = None
    _LIVE_SNAP[key] = (snap, time.time())


def _warm_pair_caches(
    ticker, exchange_name, ccxt_id, row_15m, row_1d, lim15, lim1d, atr_period, full_live: bool
) -> None:
    """
    Warms ALL data a pair view needs so Prev/Next renders instantly:
    candle frames (15m+1D), the exchange-side gap/tail stitch fetches,
    the live chip feeds (ticker/orderbook/trade tape), and — for the
    immediate ±2 neighbours — the full live orderbook snapshot.
    """
    for tf, row, lim in (("15m", row_15m, lim15), ("1d", row_1d, lim1d)):
        if not row:
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
        for r0, r1 in ranges[:10]:
            _fetch_missing_candles_cached(ccxt_id, ticker, tf, r0, r1)
    _fetch_ticker_cached(ccxt_id, ticker)
    _fetch_orderbook_top(ccxt_id, ticker)
    _fetch_trade_tape(ccxt_id, ticker)
    if full_live and row_1d is not None:
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

def build_series_payloads(hist_df: pd.DataFrame, with_volume: bool = True):
    """
    Builds lightweight-charts payloads (int-epoch times): candles + optional volume.
    """
    t = hist_df["ts"].to_numpy(dtype=np.int64)
    o = hist_df["open"].to_numpy(dtype=float)
    h = hist_df["high"].to_numpy(dtype=float)
    l = hist_df["low"].to_numpy(dtype=float)
    c = hist_df["close"].to_numpy(dtype=float)
    v = np.nan_to_num(hist_df["volume"].to_numpy(dtype=float))

    candles = [
        {"time": int(t[i]), "open": float(o[i]), "high": float(h[i]), "low": float(l[i]), "close": float(c[i])}
        for i in range(len(t))
    ]
    volume_data = []
    if with_volume:
        volume_data = [
            {
                "time": int(t[i]),
                "value": float(v[i]),
                "color": "rgba(38, 166, 154, 0.5)" if c[i] >= o[i] else "rgba(239, 83, 80, 0.5)",
            }
            for i in range(len(t))
        ]
    return candles, volume_data


def render_tradingview_lightweight_chart(
    hist_df: pd.DataFrame,
    ticker: str,
    exchange: str,
    tf_label: str,
    chart_height: int = 470,
    chart_style: str = "OHLCV Bars",
    show_volume: bool = False,
    live_poller_js: str = "",
):
    """Renders a TradingView Lightweight Charts canvas with OHLCV Bars/Candles,
    volume histogram, crosshair tooltips, and fast rolling ATR channels."""
    if hist_df is None or hist_df.empty:
        st.info(f"No {tf_label} candles available for {ticker} ({exchange}).")
        return

    candles, volume_data = build_series_payloads(hist_df, with_volume=show_volume)

    dumps = lambda x: json.dumps(x, separators=(",", ":"))

    html_code = build_lightweight_chart_html(
        candles_json=dumps(candles),
        volume_json=dumps(volume_data) if show_volume else None,
        chart_height=chart_height,
        chart_style=chart_style,
        live_poller_js=live_poller_js,
    )
    _html_component(html_code, chart_height + 10)


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
# Options respect the collector's ALLOWED_EXCHANGES whitelist (empty = all).
# Filtering happens on the cached summary frames, so toggling is instant.
_all_configured_exs = sorted(settings.exchange_map_1d.keys())
_allowed_exs = settings.allowed_exchanges
_exchange_options = (
    [e for e in _all_configured_exs if e in _allowed_exs] if _allowed_exs else _all_configured_exs
)
_default_exs = [e for e in ("bybit", "okx", "mexc") if e in _exchange_options]
enabled_exs = st.sidebar.multiselect(
    "🌐 Exchanges",
    options=_exchange_options,
    default=_default_exs,
    help="Which exchanges the dashboard shows (charts, health strip, liquidity table). Default: Bybit / OKX / MEXC.",
)

st.sidebar.markdown("---")
st.sidebar.caption("⚡ Charts load only the last N candles (cached 60 s) — instant pair switching.")
limit_15m = st.sidebar.slider("Candles · 15m chart", min_value=100, max_value=3000, value=700, step=100)
limit_1d = st.sidebar.slider("Candles · 1D chart", min_value=100, max_value=1500, value=400, step=50)

table_tf = st.sidebar.selectbox("📊 Liquidity table timeframe", options=["15m", "1d"], index=0)

if st.sidebar.button("🔄 Refresh data (clear caches)"):
    load_summary_cached.clear()
    load_candles_cached.clear()
    fetch_live_cached.clear()
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

# ---------------------------------------------------------------------------
# Load summaries for BOTH timeframes (cached)
# ---------------------------------------------------------------------------

if demo_mode:
    df_15m = _demo_summary_cached("15m")
    df_1d = _demo_summary_cached("1d")
else:
    df_15m = load_summary_cached(db_host, db_port, db_user, db_pass, "15m")
    df_1d = load_summary_cached(db_host, db_port, db_user, db_pass, "1d")

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


TICKER_OPTIONS = _unique_sorted(
    ([] if df_15m.empty else df_15m["ticker"].dropna().tolist())
    + ([] if df_1d.empty else df_1d["ticker"].dropna().tolist())
)

if not TICKER_OPTIONS:
    st.warning(
        "No pairs yet for the exchanges enabled in the sidebar — "
        "enable more exchanges or let the collector build their tables."
    )
    st.stop()


def _nav(delta: int):
    """
    Queues a Prev/Next pair switch and reruns.
    The pending value is applied BEFORE the selectbox (key="sym_ticker") is
    instantiated on the next run — directly modifying st.session_state.sym_ticker
    after widget creation raises StreamlitAPIException.
    """
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
        opt5, opt6, opt7 = st.columns(3)
        live_refresh = opt5.selectbox(
            "🔴 Live refresh", options=["1s", "2s", "5s", "Off"], index=0, key="live_refresh",
        )
        auto_reload = opt6.checkbox("Auto-reload DB (60s)", value=True, key="auto_reload")
        stitch_gaps = opt7.checkbox("🩹 Stitch gaps", value=True, key="stitch_gaps",
                                    help="Fetch missing candles from the exchange into the chart (in-memory).")

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

    def render_chart(row, tf_label, limit, interval, chart_height=470):
        """Renders one timeframe chart into the current container."""
        if row is None:
            st.info(f"No {tf_label} table for {sym_ticker}.")
            return
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
        # Always render the stitch caption line at a fixed height (empty when
        # nothing was stitched) so the side-by-side 15m / 1D charts stay
        # perfectly level — previously the 15m chart was pushed ~20px down
        # whenever it had a stitched-gap caption and the 1D chart did not.
        stitch_txt = (
            f"🩹 {stitched} missing {tf_label} candles stitched from exchange (in-memory){stale_hint}"
            if stitched
            else "&nbsp;"
        )
        st.markdown(
            f"<div style='font-size:12px;color:#808495;height:20px;line-height:20px;"
            f"margin:0 0 2px 4px;white-space:nowrap;overflow:hidden;'>{stitch_txt}</div>",
            unsafe_allow_html=True,
        )

        # Keep the daily chart in sync: aggregate fresher 15m candles of today
        # (the 15m frame is stitched too, so today's daily bar is always fresh)
        if tf_label == "1D" and row_15m is not None:
            df15 = get_candles("15m", row_15m, max(limit_15m, 200), demo_mode)
            df15, _ = _stitch(df15, "15m")
            hist_df = merge_intraday_into_daily(hist_df, df15)

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
            _live_infra()  # starts the writer daemon + /tick endpoint (runs once)
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
                live_pairs.append({
                    "db": _live_db_for(nb_1d, nb_15) or live_db,
                    "ex": sym_ex, "ccxt": ccxt_id, "sym": nb_ticker,
                })
                if delta == 0:
                    continue
                _bg(
                    ("pair", nb_ticker, sym_ex),
                    lambda t=nb_ticker, r15=nb_15, r1=nb_1d, fl=abs(delta) <= 2:
                        _warm_pair_caches(
                            t, sym_ex, ccxt_id, r15, r1,
                            limit_15m, limit_1d, atr_days, fl,
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
