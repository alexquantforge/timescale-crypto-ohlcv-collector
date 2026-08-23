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
import asyncio
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
    merge_intraday_into_daily,
    build_live_poller_js,
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


def _render_live_panel(ticker: str, exchange: str, demo: bool):
    """One-line LIVE chips: price, bid/ask, spread, 24h change."""
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
# Charts
# ---------------------------------------------------------------------------

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

    if chart_style == "OHLCV Bars":
        series_js_code = """
            const mainSeries = chart.addBarSeries({
                upColor: '#26a69a',
                downColor: '#ef5350',
            });
        """
    else:
        series_js_code = """
            const mainSeries = chart.addCandlestickSeries({
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderVisible: false,
                wickUpColor: '#26a69a',
                wickDownColor: '#ef5350',
            });
        """

    dumps = lambda x: json.dumps(x, separators=(",", ":"))

    price_formatter_js = """
            const fmtPrice = (p) => {
                const a = Math.abs(p);
                const trim = (x) => {
                    const ax = Math.abs(x);
                    let s;
                    if (ax >= 100) s = x.toFixed(0);
                    else if (ax >= 1) s = x.toFixed(2);
                    else s = x.toPrecision(4);
                    return parseFloat(s).toString();
                };
                if (a >= 1e9) return trim(p / 1e9) + 'B';
                if (a >= 1e6) return trim(p / 1e6) + 'M';
                if (a >= 1e5) return trim(p / 1e3) + 'K';
                return trim(p);
            };
    """

    volume_js = ""
    if show_volume:
        volume_js = f"""
            const volumeSeries = chart.addHistogramSeries({{
                color: '#26a69a',
                priceFormat: {{ type: 'volume' }},
                priceScaleId: '',
                scaleMargins: {{ top: 0.82, bottom: 0 }},
            }});
            volumeSeries.setData({dumps(volume_data)});
        """

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background-color: #131722; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; }}
            #tv-chart {{ width: 100%; height: {chart_height}px; position: relative; }}
            #live-badge {{ position: absolute; top: 8px; right: 70px; z-index: 10; display: none;
                background: rgba(66, 165, 245, 0.15); color: #42a5f5; border: 1px solid rgba(66, 165, 245, 0.4);
                border-radius: 8px; padding: 2px 8px; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div id="tv-chart"><div id="live-badge"></div></div>
        <script>
            {price_formatter_js}
            const chartElement = document.getElementById('tv-chart');
            const chart = LightweightCharts.createChart(chartElement, {{
                width: chartElement.clientWidth,
                height: {chart_height},
                layout: {{
                    background: {{ type: 'solid', color: '#131722' }},
                    textColor: '#d1d4dc',
                }},
                grid: {{
                    vertLines: {{ color: 'rgba(42, 46, 57, 0.5)' }},
                    horzLines: {{ color: 'rgba(42, 46, 57, 0.5)' }},
                }},
                crosshair: {{
                    mode: LightweightCharts.CrosshairMode.Normal,
                }},
                rightPriceScale: {{
                    borderColor: 'rgba(197, 203, 206, 0.8)',
                }},
                timeScale: {{
                    borderColor: 'rgba(197, 203, 206, 0.8)',
                    timeVisible: true,
                    rightOffset: 5,
                }},
                localization: {{
                    priceFormatter: fmtPrice,
                }},
            }});

            {series_js_code}
            mainSeries.setData({dumps(candles)});

            {volume_js}

            chart.timeScale().fitContent();

            let lastBar = candles.length ? Object.assign({{}}, candles[candles.length - 1]) : null;
            let liveLine = null;
            try {{
                liveLine = mainSeries.createPriceLine({{
                    color: '#42a5f5', lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dotted,
                    axisLabelVisible: true, title: 'LIVE',
                    price: lastBar ? lastBar.close : 0,
                }});
            }} catch (e) {{}}
{live_poller_js}

            window.addEventListener('resize', () => {{
                chart.applyOptions({{ width: chartElement.clientWidth }});
            }});
        </script>
    </body>
    </html>
    """
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
    db_host = st.sidebar.text_input("DB Host", value=os.getenv("DB_HOST", "localhost"))
    db_port = st.sidebar.number_input("DB Port", value=int(os.getenv("DB_PORT", 5432)))
    db_user = st.sidebar.text_input("DB User", value=os.getenv("DB_USER", "postgres"))
    db_pass = st.sidebar.text_input("DB Password", value=os.getenv("DB_PASSWORD", "postgres"), type="password")

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

df_table = df_15m if table_tf == "15m" else df_1d

if df_15m.empty and df_1d.empty:
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

    # --- Compact health strip (latest collector snapshot) --------------------
    health_row = row_1d or row_15m
    if health_row:
        st.markdown(build_health_strip_html(health_row), unsafe_allow_html=True)

    # --- Spot/Swap links + shortability badge --------------------------------
    from src.exchanges.symbol_selector import split_symbol
    _base, _quote = split_symbol(sym_ticker or "")
    _perp = None if ":" in sym_ticker else find_perp_ticker([df_15m, df_1d], _base, sym_ex)
    st.markdown(build_pair_links_html(sym_ticker, sym_ex, _perp), unsafe_allow_html=True)

    # --- LIVE panel (auto-refreshing chips, ~1s) ------------------------------
    live_interval = 0.0 if live_refresh == "Off" else float(live_refresh[:-1])
    if live_interval > 0 and hasattr(st, "fragment"):
        @st.fragment(run_every=live_interval)
        def _live_fragment():
            _render_live_panel(sym_ticker, sym_ex, demo_mode)

        _live_fragment()
    else:
        _render_live_panel(sym_ticker, sym_ex, demo_mode)

    def render_chart(row, tf_label, limit, interval, chart_height=470):
        """Renders one timeframe chart into the current container."""
        if row is None:
            st.info(f"No {tf_label} table for {sym_ticker}.")
            return
        hist_df = get_candles(tf_label, row, limit, demo_mode)

        exchange_map = settings.exchange_map_1d
        ccxt_id = exchange_map.get(sym_ex, sym_ex)

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
        if stitched:
            st.caption(f"🩹 {stitched} missing {tf_label} candles stitched from exchange (in-memory)")

        # Keep the daily chart in sync: aggregate fresher 15m candles of today
        # (the 15m frame is stitched too, so today's daily bar is always fresh)
        if tf_label == "1D" and row_15m is not None:
            df15 = get_candles("15m", row_15m, max(limit_15m, 200), demo_mode)
            df15, _ = _stitch(df15, "15m")
            hist_df = merge_intraday_into_daily(hist_df, df15)

        if chart_engine == "TradingView Lightweight Canvas":
            step = 900 if tf_label == "15m" else 86400
            interval_ms = int(live_interval * 1000) if live_interval > 0 else 0
            poller_js = build_live_poller_js(sym_ex, sym_ticker, step, interval_ms)
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

    hist_1d = None
    if row_1d is not None:
        hist_1d = get_candles("1d", row_1d, max(limit_1d, 60), demo_mode)

    atr_val = 0.0
    if hist_1d is not None and not hist_1d.empty and len(hist_1d) >= 3:
        atr_val = compute_atr_no_paranormal_bars(
            highs=hist_1d["high"].to_numpy(dtype=float),
            lows=hist_1d["low"].to_numpy(dtype=float),
            closes=hist_1d["close"].to_numpy(dtype=float),
            period=atr_days,
            small_threshold=settings.atr_small_threshold,
            large_threshold=settings.atr_large_threshold,
        )

    db_row = row_1d or row_15m or {}

    snap = None
    if not demo_mode and sym_ticker and sym_ex:
        with st.spinner(f"Fetching live orderbook for {sym_ticker} on {sym_ex}…"):
            exchange_map = settings.exchange_map_1d
            ccxt_id = exchange_map.get(sym_ex, sym_ex)
            snap = fetch_live_cached(sym_ticker, sym_ex, ccxt_id, atr_val)

    def _metric_source():
        if snap:
            return snap
        return db_row

    src = _metric_source()
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

    # --- Prefetch neighbour pairs (cache warming for instant flipping) -------
    # Warms the candle cache for the ±5 pairs around the current one so that
    # Prev/Next switching renders instantly. Cache hits are free; misses run
    # a single parallel-safe DB query per table, TTL-bound by the cache itself.
    if not demo_mode and len(TICKER_OPTIONS) > 1:
        try:
            cur_idx = TICKER_OPTIONS.index(sym_ticker)
            for delta in [d for d in range(-5, 6) if d != 0]:
                nb_ticker = TICKER_OPTIONS[(cur_idx + delta) % len(TICKER_OPTIONS)]
                for df_tf, lim in ((df_15m, limit_15m), (df_1d, limit_1d)):
                    nb_row = find_table_row(df_tf, nb_ticker, sym_ex)
                    if nb_row:
                        load_candles_cached(
                            db_host, db_port, db_user, db_pass,
                            nb_row["db_name"], nb_row["table_name"], lim,
                        )
        except Exception:
            pass

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
