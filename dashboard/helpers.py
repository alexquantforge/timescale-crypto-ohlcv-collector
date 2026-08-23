"""
Pure helper utilities for the Streamlit dashboard.

Kept free of any `streamlit` import so they can be unit-tested with plain pytest:
pair navigation (prev/next), table lookup across the timeframe summary frames,
and a synthetic demo-data generator used when no TimescaleDB is reachable.
"""
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pair navigation (Prev / Next buttons)
# ---------------------------------------------------------------------------

def shift_option(options: List[str], current: Optional[str], delta: int) -> Optional[str]:
    """
    Returns the option shifted by `delta` positions (with wrap-around) from
    `current` inside `options`. Unknown/None current maps to the first option.
    """
    if not options:
        return None
    try:
        idx = options.index(current)
    except ValueError:
        idx = 0
    return options[(idx + delta) % len(options)]


# ---------------------------------------------------------------------------
# Summary-frame lookups
# ---------------------------------------------------------------------------

def exchanges_for_ticker(df: pd.DataFrame, ticker: str) -> List[str]:
    """Sorted unique exchanges offering `ticker` in a summary frame."""
    if df is None or df.empty:
        return []
    return sorted(df.loc[df["ticker"] == ticker, "exchange"].dropna().unique().tolist())


def find_table_row(
    df: pd.DataFrame,
    ticker: str,
    exchange: Optional[str] = None,
) -> Optional[dict]:
    """
    Finds the best summary row for ticker:
    1) exact ticker+exchange match,
    2) any exchange for that ticker (HIGH volume tier preferred),
    3) None.
    Returns the row as a plain dict.
    """
    if df is None or df.empty:
        return None

    if exchange:
        m = df[(df["ticker"] == ticker) & (df["exchange"] == exchange)]
        if not m.empty:
            return m.iloc[0].to_dict()

    m = df[df["ticker"] == ticker]
    if m.empty:
        return None
    if "volume_tier" in m.columns:
        m = m.sort_values("volume_tier", ascending=True)  # HIGH before LOW
    return m.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Synthetic demo data (no database required)
# ---------------------------------------------------------------------------

_DEMO_TICKERS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "TON/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "ADA/USDT:USDT", "TRX/USDT:USDT", "NEAR/USDT:USDT",
]
_DEMO_EXCHANGES = ["bybit", "okx"]


def _demo_seed(timeframe: str, ticker: str, exchange: str) -> int:
    return abs(hash((timeframe, ticker, exchange))) % (2 ** 31)


def generate_demo_summary(timeframe: str) -> pd.DataFrame:
    """Builds a summary frame imitating the TimescaleDB scan output."""
    rows = []
    for ex in _DEMO_EXCHANGES:
        for t in _DEMO_TICKERS:
            rng = np.random.default_rng(_demo_seed(timeframe, t, ex))
            close = float(rng.uniform(0.05, 60000))
            spread_pct = float(rng.uniform(0.01, 0.15))
            rows.append({
                "ticker": t,
                "exchange": ex,
                "asset_type": "perp",
                "max_ts": int(1.8e9),
                "close": close,
                "ob_vitality_grade": str(rng.choice(["A", "B", "C", "D"])),
                "ob_vitality_score": float(rng.uniform(1, 10)),
                "ob_spread_abs": close * spread_pct / 100.0,
                "ob_spread_pct": spread_pct,
                "ob_best_bid": close * 0.9999,
                "ob_best_ask": close * 1.0001,
                "ob_bid_depth_usd": float(rng.uniform(1e5, 5e7)),
                "ob_ask_depth_usd": float(rng.uniform(1e5, 5e7)),
                "ob_total_depth_usd": float(rng.uniform(2e5, 1e8)),
                "ob_imbalance": float(rng.uniform(-0.4, 0.4)),
                "ob_trades_per_min": float(rng.uniform(5, 400)),
                "ob_buy_pressure_pct": float(rng.uniform(35, 65)),
                "ob_cvd_5m": float(rng.uniform(-5e5, 5e5)),
                "ob_min_7d_volume_usd": float(rng.uniform(1e6, 5e8)),
                "ob_spread_atr_pct": float(rng.uniform(0.5, 8)),
                "ob_atr_no_paranormal": close * 0.02,
                "table_name": f"demo_{t.replace('/', '_').replace(':', '').lower()}_on_{ex}",
                "db_name": "demo_db",
                "volume_tier": "HIGH" if rng.random() < 0.6 else "LOW",
            })
    return pd.DataFrame(rows)


def generate_demo_candles(timeframe: str, ticker: str, exchange: str, n: int = 1500) -> pd.DataFrame:
    """
    Generates a synthetic OHLCV random-walk (with a deliberate double-bottom
    near the end) so the charts and ATR channels render realistically.
    """
    step = 900 if timeframe == "15m" else 86400
    rng = np.random.default_rng(_demo_seed(timeframe, ticker, exchange))

    base = float(rng.uniform(0.05, 60000))
    drift = rng.normal(0, 0.004)
    rets = rng.normal(drift, 0.012, n)

    closes = base * np.cumprod(1.0 + rets)

    # Deliberate double-bottom at ~92% of the mid-series low, on the right side
    k = n // 3
    closes[k] = closes[k:k + 40].min() * 0.97
    closes[k + 25] = closes[k] * 1.002
    closes[k + 1:k + 25] = closes[k] * (1.0 + rng.uniform(0.0, 0.05, 24))

    highs = closes * (1.0 + np.abs(rng.normal(0, 0.005, n)))
    lows = closes * (1.0 - np.abs(rng.normal(0, 0.005, n)))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    vols = rng.uniform(1e3, 1e7, n)

    ts = int(1.78e9) + np.arange(n) * step
    return pd.DataFrame({
        "ts": ts,
        "time": pd.to_datetime(ts, unit="s"),
        "open": opens,
        "high": np.maximum.reduce([opens, closes, highs]),
        "low": np.minimum.reduce([opens, closes, lows]),
        "close": closes,
        "volume": vols,
    })


# ---------------------------------------------------------------------------
# Health strip (compact green→red indicators at the top of the Charts tab)
# ---------------------------------------------------------------------------

def _clamp01(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


def score_trades_per_min(tpm) -> float:
    """0 = dead tape (<3 trades/min), 1 = blazing (>=120 trades/min). Linear."""
    if tpm is None:
        return 0.0
    return _clamp01(float(tpm) / 120.0)


def score_depth_usd(depth) -> float:
    """0 = thin orderbook (<$1K within ±1%), 1 = deep (>=$50K). Log scale."""
    if depth is None or depth <= 0:
        return 0.0
    return _clamp01((np.log10(float(depth)) - 3.0) / (np.log10(50_000.0) - 3.0))


def score_spread_atr_pct(pct) -> float:
    """1 = spread below 5% of daily filtered ATR, 0 = 15% or wider."""
    if pct is None:
        return 0.0
    pct = float(pct)
    if pct <= 5.0:
        return 1.0
    if pct >= 15.0:
        return 0.0
    return (15.0 - pct) / 10.0


def score_min_volume_usd(v) -> float:
    """0 = <$100K/day (LOW tier floor), 1 = >=$500K/day (HIGH tier floor). Log scale."""
    if v is None or v <= 0:
        return 0.0
    return _clamp01((np.log10(float(v)) - 5.0) / (np.log10(500_000.0) - 5.0))


def fmt_usd_compact(v) -> str:
    """$1.5B / $1.2M / $850K / $950 / n/a."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "n/a"
    v = float(v)
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:,.0f}"


def _health_chip(label: str, value: str, score: float, hint: str = "") -> str:
    hue = int(140 * _clamp01(score))
    hint = hint.replace('"', "&quot;")
    return (
        f"<span title=\"{hint}\" style='display:inline-block;background:hsl({hue},62%,26%);"
        f"border:1px solid hsl({hue},62%,45%);color:#f1f3f6;border-radius:12px;"
        f"padding:3px 10px;white-space:nowrap;'>{label} <b>{value}</b></span>"
    )


def build_health_strip_html(row: dict) -> str:
    """
    Compact one-line health strip (green→red chips) from the latest collector
    snapshot row: trade tape activity, orderbook density, spread vs daily ATR,
    and min(vol×low) dollar volume over the last 7 days.
    """
    def _num(key):
        v = row.get(key)
        try:
            v = float(v)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    tpm = _num("ob_trades_per_min")
    depth = _num("ob_total_depth_usd")
    spread_pct = _num("ob_spread_atr_pct")
    minvol = _num("ob_min_7d_volume_usd")

    dead = bool(row.get("ob_is_barcode")) or (tpm is not None and tpm < 3.0)

    tpm_txt = f"{tpm:.0f}/min" if tpm is not None else "n/a"
    if dead:
        tpm_txt += " · DEAD"

    chips = [
        _health_chip("⚡ Tape", tpm_txt, 0.02 if dead else score_trades_per_min(tpm),
                     "Trades per minute: live ≥ 42/min, dead < 3/min (barcode market)"),
        _health_chip("🌊 Depth ±1%", fmt_usd_compact(depth), score_depth_usd(depth),
                     "Orderbook depth within ±1% of price: deep ≥ $50K, thin < $1K"),
        _health_chip("↔ Spread % ATR",
                     f"{spread_pct:.1f}%" if spread_pct is not None else "n/a",
                     score_spread_atr_pct(spread_pct),
                     "Spread as % of daily filtered ATR: green < 5%, red ≥ 15%"),
        _health_chip("💰 Min 7d $Vol", fmt_usd_compact(minvol), score_min_volume_usd(minvol),
                     "min(vol×low) over the last 7 days: HIGH tier ≥ $500K/day, LOW tier < $100K/day"),
    ]
    return (
        "<div style='display:flex;gap:8px;flex-wrap:wrap;align-items:center;"
        "padding:2px 0 6px 0;font-size:12.5px;font-family:-apple-system,Segoe UI,Roboto,sans-serif;'>"
        + "".join(chips) + "</div>"
    )


# ---------------------------------------------------------------------------
# Candle timestamp sanitization (fixes "year 2031" garbage on charts)
# ---------------------------------------------------------------------------

import time as _time


def sanitize_candle_frame(df: pd.DataFrame, min_ts: int = 1356998400) -> pd.DataFrame:
    """
    Cleans raw candle timestamps:
    * auto-detects milliseconds (> 1e11) and converts to seconds,
    * drops rows with garbage dates (before 2013 or more than 2 days in the future),
    * sorts ascending and resets the index.

    Some tables contain corrupted future timestamps (e.g. a daily ADA chart
    stretching to 2031); those rows are removed so the chart axis stays sane.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    if float(df["ts"].median()) > 1e11:  # whole table stored in milliseconds
        df["ts"] = df["ts"] // 1000
    now = _time.time()
    df = df[(df["ts"] >= min_ts) & (df["ts"] <= now + 2 * 86400)]
    return df.sort_values("ts").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pair links & shortability badge
# ---------------------------------------------------------------------------

def is_perp_symbol(symbol: str) -> bool:
    """True for linear perpetual symbols like ADA/USDT:USDT."""
    return ":" in (symbol or "")


def find_perp_ticker(dfs, base: str, exchange: str) -> Optional[str]:
    """
    Searches summary frames for a perpetual ticker of the same base asset on
    the given exchange (e.g. base ADA -> 'ADA/USDT:USDT'). Returns None if absent.
    """
    prefix = f"{base.upper()}/"
    for d in dfs:
        if d is None or d.empty or "ticker" not in d.columns:
            continue
        m = d[
            (d["exchange"] == exchange)
            & d["ticker"].astype(str).str.upper().str.startswith(prefix)
            & d["ticker"].astype(str).str.endswith(":USDT")
        ]
        if not m.empty:
            return str(m.iloc[0]["ticker"])
    return None


def build_pair_links_html(symbol: str, exchange: str, perp_ticker: Optional[str] = None) -> str:
    """
    Compact HTML line with Spot/Swap exchange links and a shortability badge:
    a pair is shortable when it IS a perp, or a perp variant exists on the exchange.
    """
    from src.exchanges.symbol_selector import get_exchange_url, get_swap_url

    spot_url = get_exchange_url(exchange, symbol)
    swap_symbol = symbol if is_perp_symbol(symbol) else (perp_ticker or symbol)
    swap_url = get_swap_url(exchange, swap_symbol)

    if is_perp_symbol(symbol):
        badge, color = "✅ Short: perp", "#66bb6a"
    elif perp_ticker:
        badge, color = f"✅ Short: perp {perp_ticker}", "#66bb6a"
    else:
        badge, color = "⚠️ Short: no perp — spot margin only", "#ffa726"

    spot_link = f"<a href='{spot_url}' target='_blank' rel='noopener' style='color:#4fc3f7;text-decoration:none;'>Spot&nbsp;↗</a>" if spot_url else "Spot n/a"
    swap_link = f"<a href='{swap_url}' target='_blank' rel='noopener' style='color:#4fc3f7;text-decoration:none;'>Swap&nbsp;↗</a>" if swap_url else "Swap n/a"
    return (
        f"<div style='font-size:13px;padding:0 0 6px 6px;'>"
        f"🔗 {spot_link} &nbsp;·&nbsp; {swap_link} &nbsp;·&nbsp; "
        f"<b style='color:{color};'>{badge}</b></div>"
    )


# ---------------------------------------------------------------------------
# Live data: intraday->daily aggregation & in-browser exchange poller
# ---------------------------------------------------------------------------

def merge_intraday_into_daily(df_1d: pd.DataFrame, df_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps the daily chart in sync with fresher 15m data: aggregates all 15m
    candles of the still-running UTC day into one daily bar and replaces the
    stale/appends the missing daily bar. Fixes '15m ahead of the daily chart'
    and 'daily chart missing the last day'.
    """
    if df_15m is None or df_15m.empty or df_1d is None or df_1d.empty:
        return df_1d

    last15 = int(df_15m["ts"].iloc[-1])
    day_ts = last15 - (last15 % 86400)
    day_bars = df_15m[df_15m["ts"] >= day_ts]
    if day_bars.empty:
        return df_1d

    last_1d_ts = int(df_1d["ts"].iloc[-1])
    if day_ts <= last_1d_ts - 86400:
        return df_1d  # 15m data is older than the daily chart — nothing to add

    new_bar = {
        "ts": day_ts,
        "time": pd.to_datetime(day_ts, unit="s"),
        "open": float(day_bars["open"].iloc[0]),
        "high": float(day_bars["high"].max()),
        "low": float(day_bars["low"].min()),
        "close": float(day_bars["close"].iloc[-1]),
        "volume": float(day_bars["volume"].fillna(0.0).sum()),
    }

    out = df_1d[df_1d["ts"] < day_ts].copy()
    out = pd.concat([out, pd.DataFrame([new_bar])], ignore_index=True)
    return out


_LIVE_POLLER_TEMPLATE = """
(function(){
  let fails = 0;
  const timer = setInterval(async () => {
    try {
      const r = await fetch('__URL__');
      const j = await r.json();
      let price = NaN, bid = NaN, ask = NaN;
      __PARSE__
      if (!isFinite(price) || price <= 0) { throw new Error('bad price'); }
      if (liveLine) { try { liveLine.applyOptions({ price: price }); } catch (e) {} }
      const badge = document.getElementById('live-badge');
      if (badge) {
        let sp = '';
        if (isFinite(bid) && isFinite(ask) && bid > 0 && ask > 0) {
          const pct = (ask - bid) / ((ask + bid) / 2) * 100;
          sp = ' \\u00b7 spread ' + pct.toFixed(3) + '%';
        }
        badge.style.display = 'block';
        badge.textContent = '\\u25cf LIVE ' + price + sp;
      }
      const now = Math.floor(Date.now() / 1000);
      const step = __STEP__;
      const barTs = now - (now % step);
      if (!lastBar || barTs > lastBar.time) {
        lastBar = { time: barTs, open: price, high: price, low: price, close: price };
      } else {
        lastBar.close = price;
        if (price > lastBar.high) { lastBar.high = price; }
        if (price < lastBar.low) { lastBar.low = price; }
      }
      mainSeries.update(lastBar);
      fails = 0;
    } catch (e) {
      fails += 1;
      if (fails >= 5) {
        clearInterval(timer);
        const b = document.getElementById('live-badge');
        if (b) { b.style.display = 'none'; }
      }
    }
  }, __MS__);
})();
"""


def build_live_poller_js(exchange: str, symbol: str, step_sec: int, interval_ms: int = 1000) -> str:
    """
    Returns browser-side JS that polls the exchange public REST ticker once per
    interval and live-updates the last chart bar + a LIVE price badge/line.
    Empty string for unsupported exchanges (chart simply stays DB-static).
    """
    if interval_ms <= 0:
        return ""
    from src.exchanges.symbol_selector import split_symbol

    base, quote = split_symbol(symbol or "")
    b, q = base.upper(), quote.upper()
    perp = ":" in (symbol or "")
    ex = (exchange or "").lower()

    if ex == "bybit":
        cat = "linear" if perp else "spot"
        url = f"https://api.bybit.com/v5/market/tickers?category={cat}&symbol={b}{q}"
        parse = "const it=((j.result||{}).list||[])[0]||{};price=+it.lastPrice;bid=+it.bid1Price;ask=+it.ask1Price;"
    elif ex == "okx":
        inst = f"{b}-{q}-SWAP" if perp else f"{b}-{q}"
        url = f"https://www.okx.com/api/v5/market/ticker?instId={inst}"
        parse = "const it=(j.data||[])[0]||{};price=+it.last;bid=+it.bidPx;ask=+it.askPx;"
    elif ex in ("gateio", "gate"):
        kind = "futures/usdt" if perp else "spot"
        url = f"https://api.gateio.ws/api/v4/{kind}/tickers"
        url += f"?contract={b}_{q}" if perp else f"?currency_pair={b}_{q}"
        parse = "const it=(j[0]||{});price=+it.last;bid=+it.highest_bid;ask=+it.lowest_ask;"
    elif ex == "kucoin":
        url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={b}-{q}"
        parse = "const it=(j.data||{});price=+it.price;bid=+it.bestBid;ask=+it.bestAsk;"
    elif ex == "mexc":
        url = f"https://api.mexc.com/api/v3/ticker/bookTicker?symbol={b}{q}"
        parse = "const it=(j[0]||{});bid=+it.bidPrice;ask=+it.askPrice;price=(bid+ask)/2;"
    elif ex == "bingx":
        if perp:
            url = f"https://open-api.bingx.com/openApi/swap/v2/quote/ticker?symbol={b}-{q}"
            parse = "const it=((j.data||[])[0])||{};price=+it.lastPrice;"
        else:
            url = f"https://open-api.bingx.com/openApi/spot/v1/ticker/24hr?symbol={b}-{q}"
            parse = "const it=(j.data||{});price=+it.closePrice;"
    else:
        return ""

    return (
        _LIVE_POLLER_TEMPLATE
        .replace("__URL__", url)
        .replace("__PARSE__", parse)
        .replace("__STEP__", str(int(step_sec)))
        .replace("__MS__", str(int(interval_ms)))
    )


# ---------------------------------------------------------------------------
# In-memory gap stitching (dashboard-side, DB untouched)
# ---------------------------------------------------------------------------

def find_missing_bucket_ranges(buckets, step: int):
    """
    Given sorted unique bucket numbers (ts // step), returns the half-open
    missing ranges (start_bucket, end_bucket) strictly between the first and
    the last stored bucket.
    """
    if buckets is None or len(buckets) < 2:
        return []
    ranges = []
    prev = buckets[0]
    for b in buckets[1:]:
        if b - prev > 1:
            ranges.append((prev + 1, b))
        prev = b
    return ranges


def stitch_candle_gaps(df: pd.DataFrame, fetcher, step: int, max_gap_buckets: int = 2000):
    """
    Fills gaps in a candle frame by fetching the missing bars through `fetcher`
    (a callable (start_bucket, end_bucket) -> [[ts_ms, o, h, l, c, v], ...]).
    Pure data operation (fully testable with a fake fetcher); returns
    (new_df, added_count). The database itself is NOT modified.
    """
    if df is None or df.empty:
        return df, 0

    ts_sorted = sorted(int(t) for t in df["ts"])
    buckets = sorted(set(t // step for t in ts_sorted))
    ranges = [r for r in find_missing_bucket_ranges(buckets, step) if r[1] - r[0] <= max_gap_buckets]
    if not ranges:
        return df, 0

    missing_set = set()
    for r0, r1 in ranges:
        missing_set.update(range(r0, r1))

    added_rows = []
    for r0, r1 in ranges:
        try:
            candles = fetcher(r0, r1) or []
        except Exception:
            candles = []
        for c in candles:
            b = int(c[0]) // (step * 1000)
            if b in missing_set:
                added_rows.append({
                    "ts": int(c[0]) // 1000,
                    "open": float(c[1]), "high": float(c[2]),
                    "low": float(c[3]), "close": float(c[4]),
                    "volume": float(c[5]) if c[5] is not None else 0.0,
                })

    if not added_rows:
        return df, 0

    add_df = pd.DataFrame(added_rows)
    add_df["time"] = pd.to_datetime(add_df["ts"], unit="s")
    merged = (
        pd.concat([df, add_df], ignore_index=True)
        .drop_duplicates(subset="ts", keep="first")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    return merged, len(add_df)
