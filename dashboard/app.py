"""
Streamlit Web Dashboard for Timescale Crypto OHLCV Collector.
Visualizes live market liquidity, configurable ATR without paranormal bars,
live orderbook depth, CVD, Plotly charts, and native TradingView Lightweight Charts
from the 4 historical databases.
"""
import os
import sys
import json
import asyncio
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

st.set_page_config(
    page_title="Timescale Crypto OHLCV Collector",
    page_icon="📈",
    layout="wide",
)


async def load_historical_databases_summary(db_host, db_port, db_user, db_pass, timeframe="1d"):
    """
    Connects to the 4 historical databases corresponding to timeframe (1d or 15m)
    and fetches summary statistics for all symbol tables (%_on_%).
    """
    if timeframe == "15m":
        high_db = settings.db_high_15m
        low_db = settings.db_low_15m
    else:
        high_db = settings.db_high_1d
        low_db = settings.db_low_1d

    all_rows = []

    for tier_label, db_name in [("HIGH", high_db), ("LOW", low_db)]:
        try:
            conn = await asyncpg.connect(
                host=db_host, port=db_port, user=db_user, password=db_pass, database=db_name
            )
            tables = await conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name LIKE '%_on_%'"
            )
            for r in tables:
                tbl = r["table_name"]
                try:
                    row = await conn.fetchrow(
                        f"""
                        SELECT 
                            ticker, exchange, asset_type,
                            "Timestamp" as max_ts, open, high, low, close, volume,
                            ob_vitality_score, ob_vitality_grade,
                            ob_spread_abs, ob_spread_pct, ob_spread_atr_pct, ob_atr_no_paranormal,
                            ob_best_bid, ob_best_ask, ob_bid_depth_usd, ob_ask_depth_usd,
                            ob_cvd_5m, ob_total_depth_usd, ob_min_7d_volume_usd, ob_is_barcode,
                            ob_imbalance, ob_trades_per_min, ob_buy_pressure_pct
                        FROM "{tbl}"
                        ORDER BY "Timestamp" DESC LIMIT 1
                        """
                    )
                    if row:
                        d = dict(row)
                        d["table_name"] = tbl
                        d["db_name"] = db_name
                        d["volume_tier"] = tier_label
                        all_rows.append(d)
                except Exception:
                    continue
            await conn.close()
        except Exception:
            continue

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


async def load_table_history(db_host, db_port, db_user, db_pass, db_name, table_name):
    """Fetches full historical candles for a specific symbol table."""
    try:
        conn = await asyncpg.connect(
            host=db_host, port=db_port, user=db_user, password=db_pass, database=db_name
        )
        rows = await conn.fetch(
            f"""
            SELECT TO_TIMESTAMP("Timestamp") as time, open, high, low, close, volume
            FROM "{table_name}"
            ORDER BY "Timestamp" ASC
            """
        )
        await conn.close()
        return pd.DataFrame([dict(r) for r in rows])
    except Exception as e:
        st.error(f"Error fetching history for table {table_name}: {e}")
        return pd.DataFrame()


async def fetch_live_orderbook_data(ticker: str, exchange_name: str, atr_days: int, hist_df: pd.DataFrame, timeframe: str = "1d"):
    """
    Fetches live market data (price, orderbook depth, spread, trades) directly from exchange
    and calculates dynamic ATR without paranormal bars for user-selected period (atr_days).
    """
    exchange_map = settings.exchange_map_15m if timeframe == "15m" else settings.exchange_map_1d
    ccxt_id = exchange_map.get(exchange_name, exchange_name)
    
    atr_val = 0.0
    if not hist_df.empty and len(hist_df) >= 3:
        atr_val = compute_atr_no_paranormal_bars(
            highs=hist_df["high"].to_numpy(dtype=float),
            lows=hist_df["low"].to_numpy(dtype=float),
            closes=hist_df["close"].to_numpy(dtype=float),
            period=atr_days,
            small_threshold=settings.atr_small_threshold,
            large_threshold=settings.atr_large_threshold,
        )

    exchange = None
    snap = None
    try:
        exchange = create_exchange(ccxt_id)
        snap = await fetch_orderbook_snapshot(
            exchange=exchange,
            symbol=ticker,
            atr_no_paranormal=atr_val,
            fetch_limit=settings.ob_fetch_limit,
            trades_limit=settings.ob_trades_limit,
            trades_window_sec=settings.ob_trades_window_sec,
            depth_pct=settings.ob_depth_pct,
        )
    except Exception as e:
        st.warning(f"Could not fetch live API data directly from {exchange_name}: {e}. Displaying DB snapshot.")
    finally:
        if exchange:
            await close_exchange_safely(exchange, exchange_name)

    return snap, atr_val


def render_tradingview_lightweight_chart(
    hist_df: pd.DataFrame, ticker: str, exchange: str, atr_days: int, chart_style: str = "OHLCV Bars"
):
    """
    Renders a TradingView Lightweight Charts canvas in Streamlit with
    OHLCV Bars or Candlesticks (default: OHLCV Bars), volume histogram,
    crosshair tooltips, and dynamic ATR channels.
    """
    if hist_df.empty:
        return

    candles = []
    volume_data = []
    upper_atr_data = []
    lower_atr_data = []

    closes = hist_df["close"].to_numpy(dtype=float)
    highs = hist_df["high"].to_numpy(dtype=float)
    lows = hist_df["low"].to_numpy(dtype=float)

    for i, row in enumerate(hist_df.itertuples()):
        dt_str = pd.to_datetime(getattr(row, "time")).strftime("%Y-%m-%d %H:%M")
        c_open = float(getattr(row, "open"))
        c_high = float(getattr(row, "high"))
        c_low = float(getattr(row, "low"))
        c_close = float(getattr(row, "close"))
        c_vol = float(getattr(row, "volume"))

        candles.append({
            "time": dt_str,
            "open": c_open,
            "high": c_high,
            "low": c_low,
            "close": c_close,
        })

        volume_data.append({
            "time": dt_str,
            "value": c_vol,
            "color": "rgba(38, 166, 154, 0.5)" if c_close >= c_open else "rgba(239, 83, 80, 0.5)"
        })

        sub_h = highs[:i+1]
        sub_l = lows[:i+1]
        sub_c = closes[:i+1]
        if len(sub_c) >= 3:
            a = compute_atr_no_paranormal_bars(
                sub_h, sub_l, sub_c,
                period=atr_days,
                small_threshold=settings.atr_small_threshold,
                large_threshold=settings.atr_large_threshold,
            )
            upper_atr_data.append({"time": dt_str, "value": c_close + a})
            lower_atr_data.append({"time": dt_str, "value": max(0.0, c_close - a)})

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

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background-color: #131722; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; }}
            #chart-title {{ color: #d1d4dc; font-size: 16px; font-weight: 600; padding: 10px 15px 5px 15px; }}
            #tv-chart {{ width: 100%; height: 500px; }}
        </style>
    </head>
    <body>
        <div id="chart-title">📈 TradingView Canvas ({chart_style}): {ticker} ({exchange.upper()}) — ATR Channels ({atr_days} Days)</div>
        <div id="tv-chart"></div>
        <script>
            const chartElement = document.getElementById('tv-chart');
            const chart = LightweightCharts.createChart(chartElement, {{
                width: chartElement.clientWidth,
                height: 500,
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
                }},
            }});

            {series_js_code}
            mainSeries.setData({json.dumps(candles)});

            const volumeSeries = chart.addHistogramSeries({{
                color: '#26a69a',
                priceFormat: {{ type: 'volume' }},
                priceScaleId: '',
                scaleMargins: {{ top: 0.8, bottom: 0 }},
            }});
            volumeSeries.setData({json.dumps(volume_data)});

            const upperAtrSeries = chart.addLineSeries({{
                color: '#ff9800',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                title: 'Upper ATR Band',
            }});
            upperAtrSeries.setData({json.dumps(upper_atr_data)});

            const lowerAtrSeries = chart.addLineSeries({{
                color: '#ff9800',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                title: 'Lower ATR Band',
            }});
            lowerAtrSeries.setData({json.dumps(lower_atr_data)});

            window.addEventListener('resize', () => {{
                chart.applyOptions({{ width: chartElement.clientWidth }});
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=560)


def render_tradingview_official_widget(ticker: str, exchange: str, style_code: str = "0"):
    """
    Renders TradingView's official Advanced Real-Time Chart Widget.
    style_code "0" = Bars, "1" = Candlesticks.
    """
    base_symbol = ticker.replace("/", "").replace(":", "")
    tv_symbol = f"{exchange.upper()}:{base_symbol}"
    
    html_code = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="height:550px;width:100%">
      <div id="tradingview_chart" style="height:calc(100% - 32px);width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "D",
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
    components.html(html_code, height=560)


# Sidebar Configuration
st.sidebar.title("🛠️ Database & Timeframe Settings")
db_host = st.sidebar.text_input("DB Host", value=os.getenv("DB_HOST", "localhost"))
db_port = st.sidebar.number_input("DB Port", value=int(os.getenv("DB_PORT", 5432)))
db_user = st.sidebar.text_input("DB User", value=os.getenv("DB_USER", "postgres"))
db_pass = st.sidebar.text_input("DB Password", value=os.getenv("DB_PASSWORD", "postgres"), type="password")

selected_tf = st.sidebar.selectbox("Select Timeframe Mode", options=["1d", "15m"], index=0)

if st.sidebar.button("🔄 Refresh Dashboard Data"):
    st.rerun()

st.title("⚡ Timescale Crypto OHLCV Collector — Dashboard")
st.markdown(
    f"Connected to the **4 Historical Databases** in timeframe mode: **`{selected_tf}`**."
)

# Load data from the 4 historical databases
df = asyncio.run(load_historical_databases_summary(db_host, db_port, db_user, db_pass, timeframe=selected_tf))

if df.empty:
    st.warning(f"⚠️ No data tables found in historical databases for timeframe mode '{selected_tf}'. Ensure PostgreSQL/TimescaleDB is running.")
else:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Pair Tables", f"{len(df):,}")
    col2.metric("Liquid HIGH Tier", f"{len(df[df['volume_tier'] == 'HIGH']):,}")
    col3.metric("Low Volume LOW Tier", f"{len(df[df['volume_tier'] == 'LOW']):,}")
    col4.metric("Active Exchanges", len(df["exchange"].dropna().unique()))
    col5.metric("Vitality Grade A/B", len(df[df["ob_vitality_grade"].isin(["A", "B"])]))

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Liquidity Monitor", "⚡ Live Symbol Inspector", "ℹ️ Methodology & Algorithms"])

    with tab1:
        st.subheader(f"Liquidity & Orderbook Metrics Table ({selected_tf})")

        fcol1, fcol2, fcol3 = st.columns(3)
        ex_options = sorted([e for e in df["exchange"].dropna().unique()])
        selected_ex = fcol1.multiselect("Filter by Exchange", options=ex_options, default=ex_options)
        selected_tier = fcol2.multiselect("Filter by Volume Tier", options=["HIGH", "LOW"], default=["HIGH", "LOW"])
        grade_options = [g for g in ["A", "B", "C", "D", "F"] if g in df["ob_vitality_grade"].values]
        selected_grade = fcol3.multiselect("Filter by Vitality Grade", options=["A", "B", "C", "D", "F"], default=grade_options)

        filtered_df = df[
            (df["exchange"].isin(selected_ex)) &
            (df["volume_tier"].isin(selected_tier)) &
            (df["ob_vitality_grade"].isin(selected_grade))
        ]

        display_cols = [
            "ticker", "exchange", "asset_type", "volume_tier", "close",
            "ob_vitality_grade", "ob_vitality_score", "ob_spread_pct",
            "ob_spread_atr_pct", "ob_atr_no_paranormal", "ob_cvd_5m", "ob_min_7d_volume_usd"
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
            }
        )

    with tab2:
        st.subheader("⚡ Live Symbol & Orderbook Inspector")
        
        cc1, cc2, cc3, cc4, cc5 = st.columns([2, 2, 2, 2, 2])
        ticker_options = sorted([t for t in df["ticker"].dropna().unique()])
        sym_ticker = cc1.selectbox("Select Ticker", options=ticker_options)
        
        available_ex = df[df["ticker"] == sym_ticker]["exchange"].dropna().unique()
        sym_ex = cc2.selectbox("Select Exchange", options=available_ex)
        
        atr_days = cc3.slider("🎯 ATR Period (Days)", min_value=1, max_value=30, value=5, step=1)
        
        chart_engine = cc4.selectbox(
            "📈 Chart Engine",
            options=["TradingView Lightweight Canvas", "TradingView Official Widget", "Plotly Dark"]
        )

        chart_style = cc5.selectbox(
            "📊 Chart Style",
            options=["OHLCV Bars", "Candlesticks"],
            index=0
        )

        # Match exact table row from historical DBs
        db_match = df[(df["ticker"] == sym_ticker) & (df["exchange"] == sym_ex)]
        if not db_match.empty:
            db_row = db_match.iloc[0]
            db_name = db_row["db_name"]
            tbl_name = db_row["table_name"]

            hist_df = asyncio.run(load_table_history(db_host, db_port, db_user, db_pass, db_name, tbl_name))

            with st.spinner(f"Fetching Live Market Orderbook & Price for {sym_ticker} on {sym_ex}..."):
                snap, atr_val = asyncio.run(fetch_live_orderbook_data(sym_ticker, sym_ex, atr_days, hist_df, timeframe=selected_tf))

            if snap:
                best_bid = snap.get("ob_best_bid", 0.0)
                best_ask = snap.get("ob_best_ask", 0.0)
                mid_price = (best_bid + best_ask) / 2.0 if best_bid and best_ask else 0.0
                spread_abs = snap.get("ob_spread_abs", 0.0)
                spread_pct = snap.get("ob_spread_pct", 0.0)
                tot_depth = snap.get("ob_total_depth_usd", 0.0)
                bid_depth = snap.get("ob_bid_depth_usd", 0.0)
                ask_depth = snap.get("ob_ask_depth_usd", 0.0)
                imbalance = snap.get("ob_imbalance", 0.0)
                tpm = snap.get("ob_trades_per_min", 0.0)
                cvd_5m = snap.get("ob_cvd_5m", 0.0)
                buy_pct = snap.get("ob_buy_pressure_pct", 0.0)
                score = snap.get("ob_vitality_score", 0.0)
                grade = snap.get("ob_vitality_grade", "N/A")
            else:
                mid_price = float(db_row.get("close", 0.0))
                best_bid = float(db_row.get("ob_best_bid", mid_price))
                best_ask = float(db_row.get("ob_best_ask", mid_price))
                spread_abs = float(db_row.get("ob_spread_abs", best_ask - best_bid if best_ask and best_bid else 0.0))
                spread_pct = float(db_row.get("ob_spread_pct", 0.0))
                tot_depth = float(db_row.get("ob_total_depth_usd", 0.0))
                bid_depth = float(db_row.get("ob_bid_depth_usd", 0.0))
                ask_depth = float(db_row.get("ob_ask_depth_usd", 0.0))
                imbalance = float(db_row.get("ob_imbalance", 0.0))
                tpm = float(db_row.get("ob_trades_per_min", 0.0))
                cvd_5m = float(db_row.get("ob_cvd_5m", 0.0))
                buy_pct = float(db_row.get("ob_buy_pressure_pct", 0.0))
                score = float(db_row.get("ob_vitality_score", 0.0))
                grade = db_row.get("ob_vitality_grade", "N/A")

            spread_atr_pct = (spread_abs / atr_val * 100.0) if atr_val > 0 else 0.0

            st.markdown("### 🔴 Live Market & Orderbook Metrics")
            
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            mcol1.metric("Live Mid Price", f"${mid_price:,.4f}", delta=f"Bid: ${best_bid:,.4f} | Ask: ${best_ask:,.4f}")
            mcol2.metric("Live Spread", f"${spread_abs:.4f}", delta=f"{spread_pct:.3f}%")
            mcol3.metric(f"ATR w/o Paranormal Bars ({atr_days}d)", f"${atr_val:.4f}")
            mcol4.metric("Spread % of ATR", f"{spread_atr_pct:.2f}%", delta=f"Relative to {atr_days}d ATR")
            mcol5.metric("Vitality Score", f"{score:.1f} / 10", delta=f"Grade {grade}")

            st.markdown("#### 📖 Live Orderbook & Trade Tape Metrics")
            dcol1, dcol2, dcol3, dcol4 = st.columns(4)
            dcol1.metric("Total Depth (±1% $)", f"${tot_depth:,.2f}", delta=f"Bid: ${bid_depth:,.0f} | Ask: ${ask_depth:,.0f}")
            dcol2.metric("Orderbook Imbalance", f"{imbalance:.2f}", delta="Bid / Ask Depth Ratio")
            dcol3.metric("5m Cumulative Vol Delta (CVD)", f"${cvd_5m:,.2f}", delta=f"Buy Pressure: {buy_pct:.1f}%")
            dcol4.metric("Trade Activity", f"{tpm:.1f} trades/min")

            st.markdown("---")

            if chart_engine == "TradingView Lightweight Canvas":
                render_tradingview_lightweight_chart(hist_df, sym_ticker, sym_ex, atr_days, chart_style=chart_style)
            elif chart_engine == "TradingView Official Widget":
                style_code = "0" if chart_style == "OHLCV Bars" else "1"
                render_tradingview_official_widget(sym_ticker, sym_ex, style_code=style_code)
            else: # Plotly Dark
                if not hist_df.empty:
                    atr_series = []
                    closes = hist_df["close"].to_numpy(dtype=float)
                    highs = hist_df["high"].to_numpy(dtype=float)
                    lows = hist_df["low"].to_numpy(dtype=float)

                    for i in range(1, len(hist_df) + 1):
                        sub_h = highs[:i]
                        sub_l = lows[:i]
                        sub_c = closes[:i]
                        if len(sub_c) >= 3:
                            a = compute_atr_no_paranormal_bars(
                                sub_h, sub_l, sub_c,
                                period=atr_days,
                                small_threshold=settings.atr_small_threshold,
                                large_threshold=settings.atr_large_threshold,
                            )
                            atr_series.append(a)
                        else:
                            atr_series.append(0.0)

                    hist_df["dynamic_atr"] = atr_series

                    fig = go.Figure()
                    if chart_style == "OHLCV Bars":
                        fig.add_trace(go.Ohlc(
                            x=hist_df["time"],
                            open=hist_df["open"],
                            high=hist_df["high"],
                            low=hist_df["low"],
                            close=hist_df["close"],
                            name="OHLCV Bars"
                        ))
                    else:
                        fig.add_trace(go.Candlestick(
                            x=hist_df["time"],
                            open=hist_df["open"],
                            high=hist_df["high"],
                            low=hist_df["low"],
                            close=hist_df["close"],
                            name="Candlesticks"
                        ))

                    fig.add_trace(go.Scatter(
                        x=hist_df["time"],
                        y=hist_df["close"] + hist_df["dynamic_atr"],
                        mode="lines",
                        name=f"Upper ATR Channel ({atr_days} Days)",
                        line=dict(color="orange", dash="dash")
                    ))

                    fig.add_trace(go.Scatter(
                        x=hist_df["time"],
                        y=hist_df["close"] - hist_df["dynamic_atr"],
                        mode="lines",
                        name=f"Lower ATR Channel ({atr_days} Days)",
                        line=dict(color="orange", dash="dash")
                    ))

                    fig.update_layout(
                        title=f"{sym_ticker} ({sym_ex}) — Price Chart & ATR without Paranormal Bars ({atr_days} Days)",
                        xaxis_rangeslider_visible=False,
                        template="plotly_dark",
                        height=550,
                    )
                    st.plotly_chart(fig, width="stretch")

    with tab3:
        st.subheader("ℹ️ Methodology & Key Algorithms")
        st.markdown(r"""
        ### 1. ATR without Paranormal Bars (Filtered Robust ATR)
        Standard **ATR (Average True Range)** is highly sensitive to single abnormal candles — *paranormal bars* (news spikes, squeezes, false breakouts).
        
        Our algorithm filters out bars whose range falls outside the threshold window:
        $$\text{Bar Range} \notin [0.5 \times \text{ATR}, 1.8 \times \text{ATR}]$$
        and iteratively recalculates robust volatility reflecting true average daily asset movement over the **user-selected calculation period (N days)**.

        ---

        ### 2. Spread % of ATR
        The ratio of current absolute orderbook spread to ATR without paranormal bars:
        $$\text{Spread \% of ATR} = \frac{\text{Ask} - \text{Bid}}{\text{ATR}_{\text{robust}}} \times 100\%$$
        This metric normalizes spread against daily market volatility, allowing direct liquidity comparison across high-priced and low-priced assets.

        ---

        ### 3. Native TradingView Charts & OHLCV Bars Integration
        The dashboard supports **TradingView Lightweight Charts** (JS library by TradingView) as well as the **TradingView Advanced Real-Time Chart Widget**, allowing users to toggle between **OHLCV Bars (Default)** and **Candlesticks** with real-time zooming, panning, and volatility channels.

        ---

        ### 4. Perp-First Selection Strategy
        For every base asset, the system prioritizes perpetual linear contracts (`BTC/USDT:USDT`). Spot markets are loaded only if a perpetual contract does not exist on that exchange.

        ---

        ### 5. Historical 4-Database Storage Architecture
        Connects directly to the 4 historical PostgreSQL / TimescaleDB databases:
        * **1D High Volume:** `ohlcv_1d_data_for_usdt_pairs_using_ccxt_and_direct_api1`
        * **1D Low Volume:** `ohlcv_1d_data_for_low_vol_usdt_pairs_using_ccxt_and_direct_api1`
        * **15M High Volume:** `ohlcv_15m_data_for_usdt_pairs_using_ccxt_and_direct_api1`
        * **15M Low Volume:** `ohlcv_15m_low_vol_usdt_pairs_using_ccxt_and_dist_api1`
        """)
