"""
Pure helper utilities for the Streamlit dashboard.

Kept free of any `streamlit` import so they can be unit-tested with plain pytest:
pair navigation (prev/next), table lookup across the timeframe summary frames,
and a synthetic demo-data generator used when no TimescaleDB is reachable.
"""
import json
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
                "max_ts": int(_time.time()),  # anchor demo summaries to now so the sanity filter keeps them
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


def filter_sane_summary_rows(df: pd.DataFrame, min_ts: int = 1356998400) -> pd.DataFrame:
    """
    Drops summary rows whose last-candle timestamp is garbage, applying the
    SAME bounds as sanitize_candle_frame() (whole-frame ms→s detection, then
    min_ts <= max_ts <= now + 2 days).

    Without this, a pair whose table contains ONLY corrupted rows (e.g.
    future 2031 timestamps or ms-epochs) still entered the dashboard pair
    list — the summary scan saw the raw row — but its charts then rendered
    zero sanitized candles ("No candles available"). Now such ghost pairs
    never enter TICKER_OPTIONS at all.
    """
    if df is None or df.empty or "max_ts" not in df.columns:
        return df
    ts = pd.to_numeric(df["max_ts"], errors="coerce")
    med = ts.median(skipna=True)
    if pd.notna(med) and float(med) > 1e11:  # milliseconds table(s)
        ts = ts // 1000
    now = _time.time()
    ok = (ts >= min_ts) & (ts <= now + 2 * 86400)
    return df[ok.fillna(False)]


def drop_stale_spot_duplicates(
    df: pd.DataFrame, max_lag_sec: int = 3 * 86400
) -> pd.DataFrame:
    """
    Drops SPOT summary rows that are dead leftovers of the perp-first switch.

    When a base asset gets a perpetual contract on an exchange, the collector
    starts writing only `BASE/USDT:USDT` and the old `BASE/USDT` spot table is
    never touched again (nothing drops it). It still shows up in the pair list
    and opening it looks like a bug: candles end weeks ago while the live price
    line sits at the current price (0G/USDT @bybit — spot 983h behind, perp
    0.7h behind).

    A spot row is dropped only when BOTH hold for the same (base, exchange):
      * a perp row exists, and
      * the spot table's last candle is more than `max_lag_sec` older than the
        perp one (a spot market that is still actively collected stays).
    """
    if df is None or df.empty or "ticker" not in df.columns:
        return df

    ts = pd.to_numeric(df.get("max_ts"), errors="coerce").fillna(0)
    ts = ts.where(ts < 1e11, ts // 1000)  # ms-epoch rows → seconds

    perp_ts: dict = {}
    for (idx, ticker), exchange in zip(df["ticker"].items(), df.get("exchange", "")):
        if ":" in str(ticker):
            base = str(ticker).split("/")[0].upper()
            key = (base, exchange)
            perp_ts[key] = max(perp_ts.get(key, 0), float(ts.get(idx, 0)))

    if not perp_ts:
        return df

    keep = []
    for (idx, ticker), exchange in zip(df["ticker"].items(), df.get("exchange", "")):
        ticker = str(ticker)
        if ":" in ticker:
            keep.append(True)
            continue
        base = ticker.split("/")[0].upper()
        p_ts = perp_ts.get((base, exchange), 0)
        keep.append(not (p_ts and float(ts.get(idx, 0)) < p_ts - max_lag_sec))

    return df[pd.Series(keep, index=df.index)]


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
# Lightweight Charts page assembly (pure builder — no streamlit import)
# ---------------------------------------------------------------------------


def _sig10(x) -> float:
    """Round to 10 significant digits — shrinks JSON payloads a lot without
    any visible precision loss (prices span 1e-8 .. 1e5, so a fixed decimal
    round would destroy small-cap prices)."""
    return float(f"{float(x):.10g}")


def build_series_arrays(hist_df: pd.DataFrame, with_volume: bool = False):
    """
    Compact chart payloads as ARRAYS instead of objects:
      candles: [[ts, open, high, low, close], ...]
      volume:  [[ts, value, dir], ...]   dir = +1 up / -1 down
    Roughly half the JSON size of {time, open, ...} objects — faster to
    serialize on pair switch and faster for the iframe to JSON.parse.
    """
    t = hist_df["ts"].to_numpy(dtype=np.int64)
    o = hist_df["open"].to_numpy(dtype=float)
    h = hist_df["high"].to_numpy(dtype=float)
    l = hist_df["low"].to_numpy(dtype=float)
    c = hist_df["close"].to_numpy(dtype=float)

    candles = [
        [int(t[i]), _sig10(o[i]), _sig10(h[i]), _sig10(l[i]), _sig10(c[i])]
        for i in range(len(t))
    ]
    volume_data = None
    if with_volume:
        v = np.nan_to_num(hist_df["volume"].to_numpy(dtype=float))
        volume_data = [
            [int(t[i]), round(float(v[i]), 2), 1 if c[i] >= o[i] else -1]
            for i in range(len(t))
        ]
    return candles, volume_data


def rows_to_compact_candles(rows, min_ts: int = 1356998400, now_sec: int = None):
    """
    Server-side (history endpoint): asyncpg row dicts -> ascending compact
    [[ts, o, h, l, c, v], ...] arrays, applying the SAME garbage-timestamp
    policy as sanitize_candle_frame (per-row ms->s when the epoch looks like
    milliseconds, then sane min/max bounds). Returns [] for empty input.

    All numbers are guaranteed FINITE: a single NaN/Inf in a row would make
    json.dumps emit a bare `NaN` literal, which is invalid JSON — the
    browser's response.json() then throws and the chart's history loader
    silently retries forever. Rows with non-finite OHLC are skipped; a
    non-finite volume becomes 0.
    """
    if not rows:
        return []
    now = _time.time() if now_sec is None else now_sec
    out = []
    hi = now + 2 * 86400

    def _f(x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if v == v and abs(v) != float("inf") else None

    for r in rows:
        try:
            ts = int(r["ts"])
        except (KeyError, TypeError, ValueError):
            continue
        if ts > 1e11:  # millisecond epoch stored in the table
            ts //= 1000
        if ts < min_ts or ts > hi:
            continue
        o = _f(r.get("open"))
        h_ = _f(r.get("high"))
        l_ = _f(r.get("low"))
        c = _f(r.get("close"))
        if o is None or h_ is None or l_ is None or c is None:
            continue
        v = _f(r.get("volume"))
        out.append([ts, _sig10(o), _sig10(h_), _sig10(l_), _sig10(c),
                    round(v, 2) if v is not None else 0.0])
    out.sort(key=lambda a: a[0])
    return out


_HISTORY_LOADER_TEMPLATE = """
(function(){
  'use strict';
  const PORT = __PORT__;
  const BASE = __BASE__;      // '/candles?db=..&table=..' served by the dashboard
  const CHUNK = __CHUNK__;
  const statusEl = document.getElementById('hist-status');
  function showStatus(txt, color){
    if (!statusEl) { return; }
    statusEl.textContent = txt;
    statusEl.style.color = color || '#808495';
  }
  // Candidate dashboard endpoints, in order: the host the page was loaded
  // from (document.referrer), then plain localhost/127.0.0.1 — the referrer
  // may be empty in some iframe setups, and a pure localhost dashboard is
  // the overwhelmingly common case anyway. Probing /healthz picks the first
  // candidate that actually answers, so the loader works even when the
  // referrer is blank.
  const CANDS = [];
  try {
    const ref = document.referrer || ((location.ancestorOrigins && location.ancestorOrigins[0]) || '');
    if (ref) {
      const u = new URL(ref);
      CANDS.push(u.protocol + '//' + u.hostname + ':' + PORT);
    }
  } catch (e) {}
  CANDS.push('http://localhost:' + PORT, 'http://127.0.0.1:' + PORT);

  let SVC = null;
  let inflight = false;
  let exhausted = false;

  // Infinite history: when the user pans the chart to the left edge of the
  // loaded window, fetch the next OLDER chunk straight from the dashboard's
  // /candles endpoint (no Streamlit rerun!) and prepend it.
  async function loadOlder(){
    if (!SVC || inflight || exhausted || !allCandles || !allCandles.length) { return; }
    inflight = true;
    try {
      const first = allCandles[0].time;
      showStatus('… loading older history');
      const r = await fetch(SVC + BASE + '&to=' + first + '&limit=' + CHUNK, { cache: 'no-store' });
      const j = await r.json();
      if (j && j.err) {
        // Server-side error (SQL/pool) — reported, never mistaken for the
        // start of history; details are in the dashboard console.
        console.warn('[hist] /candles server error:', j.err);
        showStatus('⚠ history: DB error — see the dashboard console', '#ef5350');
        return;
      }
      if (!j || !Array.isArray(j.c)) {
        // Unexpected response shape — an OLD dashboard process (started
        // before git pull) is still serving this port without /candles.
        // NOT the start of history: do not set `exhausted`.
        console.warn('[hist] unexpected /candles response', j);
        showStatus('⚠ history: an old process owns this port — restart the dashboard', '#ef5350');
        return;
      }
      const rows = j.c;
      if (rows.length === 0) {
        exhausted = true;
        showStatus(j.mn
          ? ('⇤ table starts at ' + j.mn + ' — nothing older is stored')
          : '⇤ start of history — nothing older is stored');
        return;
      }
      const olderC = [];
      const olderV = [];
      for (let i = 0; i < rows.length; i++) {
        const a = rows[i];
        if (!a || a[0] >= first) { continue; }   // strictly older + ms-epoch defence
        olderC.push({ time: a[0], open: a[1], high: a[2], low: a[3], close: a[4] });
        olderV.push({ time: a[0], value: a[5] || 0, color: (a[4] >= a[1] ? UP_VOL_COLOR : DN_VOL_COLOR) });
      }
      if (olderC.length === 0) {
        exhausted = true;
        showStatus(j.mn
          ? ('⇤ table starts at ' + j.mn + ' — nothing older is stored')
          : '⇤ start of history — nothing older is stored');
        return;
      }
      allCandles = olderC.concat(allCandles);
      mainSeries.setData(allCandles);
      if (typeof volumeSeries !== 'undefined' && allVolume) {
        allVolume = olderV.concat(allVolume);
        volumeSeries.setData(allVolume);
      }
      const dt = new Date(allCandles[0].time * 1000).toISOString().slice(0, 10);
      showStatus('⇤ +' + olderC.length + ' bars, now from ' + dt, '#66bb6a');
      if (rows.length < CHUNK * 0.95) { exhausted = true; }   // short page = start of history
    } catch (e) {
      // transient endpoint failure — the next left-pan retries automatically
      console.warn('[hist] /candles fetch failed:', e);
      showStatus('⚠ history: /candles unavailable (retried on scroll)', '#ef5350');
    } finally { inflight = false; }
  }

  showStatus('⇤ history: looking for the /tick server…');   // sync: badge proves the loader JS ran
  (async function(){
    let sawAny = false;
    for (const b of CANDS) {
      try {
        const r = await fetch(b + '/healthz', { cache: 'no-store' });
        const j = await r.json();
        sawAny = true;
        if (j && j.ok) { SVC = b; break; }
      } catch (e) {}
    }
    if (!SVC) {
      showStatus(
        sawAny ? '⚠ history: an old process owns this port — restart the dashboard'
               : '⚠ history: the /tick server is unreachable from the browser',
        '#ef5350'
      );
      return;
    }
    showStatus('⇤ scroll left — older history loads on demand');
    try {
      const lr = chart.timeScale().getVisibleLogicalRange();
      if (lr && lr.from < 10) { loadOlder(); }   // already parked at the left edge
    } catch (e) {}
  })();

  chart.timeScale().subscribeVisibleLogicalRangeChange(function(range){
    if (range && range.from < 10) { loadOlder(); }
  });
})();
"""


def build_history_loader_js(
    db_name: Optional[str],
    table_name: Optional[str],
    step_sec: int,
    tick_port: Optional[int],
    chunk: int = 1200,
) -> str:
    """
    Browser-side JS that gives the chart INFINITE left-scroll history: on a
    visible-range change near the left edge it fetches the next older chunk
    from the dashboard's /candles HTTP endpoint (same tiny server as /tick)
    and prepends it — no Streamlit rerun, no slider bumping. Returns "" when
    no endpoint is available; the chart then behaves exactly as before.
    """
    if not tick_port or not db_name or not table_name:
        return ""
    from urllib.parse import urlencode

    base = "/candles?" + urlencode({"db": db_name, "table": table_name})
    return (
        _HISTORY_LOADER_TEMPLATE
        .replace("__BASE__", json.dumps(base))
        .replace("__PORT__", str(int(tick_port)))
        .replace("__CHUNK__", str(int(chunk)))
    )


# Height (px) of the history-loader badge line rendered BELOW the chart.
# The iframe height in app.py must reserve this much extra room, so keep
# this constant in sync with the #hist-status CSS height in
# build_lightweight_chart_html.
HIST_STATUS_HEIGHT = 16


def build_lightweight_chart_html(
    candles_json: str,
    volume_json: Optional[str],
    chart_height: int,
    chart_style: str,
    live_poller_js: str = "",
    history_loader_js: str = "",
) -> str:
    """
    Assembles the complete TradingView Lightweight Charts page as a pure
    string (unit-testable without streamlit).

    The candle/volume payloads are embedded ONCE as JS variables
    (`candlesData` / `volumeData`) and every consumer — setData, the lastBar
    initializer, the live price line and the browser-side live poller —
    references those variables.

    Regression note: a previous version built the lastBar line from a JS
    `candles` variable that was never declared in the page (the payload was
    inlined only inside setData()). The resulting `ReferenceError: candles is
    not defined` aborted the whole script tail: the LIVE price line and the
    1-second exchange poller were never installed, so charts looked frozen
    while the server-side LIVE chip kept updating.
    """
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

    volume_js = ""
    if volume_json is not None:
        volume_js = f"""
            const volumeData = ({volume_json}).map(function(r){{ return {{ time: r[0], value: r[1], color: (r[2] >= 0 ? UP_VOL_COLOR : DN_VOL_COLOR) }}; }});
            const volumeSeries = chart.addHistogramSeries({{
                color: '#26a69a',
                priceFormat: {{ type: 'volume' }},
                priceScaleId: '',
                scaleMargins: {{ top: 0.82, bottom: 0 }},
            }});
            volumeSeries.setData(volumeData);
            allVolume = volumeData;
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background-color: #131722; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; }}
            #tv-chart {{ width: 100%; height: {chart_height}px; position: relative; }}
            /* History-loader badge lives BELOW the chart, under the time/date
               axis line: absolutely positioned inside #tv-chart it overlapped
               the axis labels (the "table starts at …" note vs the dates) was
               unreadable. Static block with a fixed height = HIST_STATUS_HEIGHT. */
            #hist-status {{ height: {HIST_STATUS_HEIGHT}px; line-height: {HIST_STATUS_HEIGHT}px; padding-left: 8px; font-size: 11px; color: #808495; pointer-events: none; opacity: 0.85; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        </style>
    </head>
    <body>
        <div id="tv-chart"></div>
        <div id="hist-status"></div>
        <script>
            const UP_VOL_COLOR = 'rgba(38, 166, 154, 0.5)';
            const DN_VOL_COLOR = 'rgba(239, 83, 80, 0.5)';
            const fmtPrice = (p) => {{
                const a = Math.abs(p);
                const trim = (x) => {{
                    const ax = Math.abs(x);
                    let s;
                    if (ax >= 100) s = x.toFixed(0);
                    else if (ax >= 1) s = x.toFixed(2);
                    else s = x.toPrecision(4);
                    return parseFloat(s).toString();
                }};
                if (a >= 1e9) return trim(p / 1e9) + 'B';
                if (a >= 1e6) return trim(p / 1e6) + 'M';
                if (a >= 1e5) return trim(p / 1e3) + 'K';
                return trim(p);
            }};
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

            const candlesData = ({candles_json}).map(function(r){{ return {{ time: r[0], open: r[1], high: r[2], low: r[3], close: r[4] }}; }});
            {series_js_code}
            mainSeries.setData(candlesData);
            let allCandles = candlesData;   // grows leftwards via the history loader
            let allVolume = null;           // wired when the volume series exists

            {volume_js}

            chart.timeScale().fitContent();

            let lastBar = candlesData.length ? Object.assign({{}}, candlesData[candlesData.length - 1]) : null;
            let liveLine = null;
            try {{
                liveLine = mainSeries.createPriceLine({{
                    color: '#42a5f5', lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dotted,
                    axisLabelVisible: true,
                    price: lastBar ? lastBar.close : 0,
                }});
            }} catch (e) {{}}
{live_poller_js}
{history_loader_js}

            window.addEventListener('resize', () => {{
                chart.applyOptions({{ width: chartElement.clientWidth }});
            }});
        </script>
    </body>
    </html>
    """


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
  'use strict';
  const STEP = __STEP__;
  const TICK_PATH = __TICK_PATH__;    // '/tick?db=..&ex=..&sym=..' served by the dashboard, or ''
  const TICK_PORT = __TICK_PORT__;    // dashboard live-tick endpoint port, or 0
  const DIRECT_URL = __DIRECT_URL__;  // exchange public REST ticker URL, or ''
  function directParse(j){ let price = NaN, bid = NaN, ask = NaN; __PARSE__ return price; }
  // Candidate dashboard endpoints, in order: the host the page was loaded
  // from (document.referrer), then plain localhost/127.0.0.1 — the referrer
  // may be empty in some iframe setups, and a PURE localhost dashboard is
  // the overwhelmingly common case anyway.
  function dbTickUrls(){
    const urls = [];
    if (!TICK_PATH || !TICK_PORT) { return urls; }
    try {
      const ref = document.referrer || ((location.ancestorOrigins && location.ancestorOrigins[0]) || '');
      if (ref) {
        const u = new URL(ref);
        urls.push(u.protocol + '//' + u.hostname + ':' + TICK_PORT + TICK_PATH);
      }
    } catch (e) {}
    urls.push('http://localhost:' + TICK_PORT + TICK_PATH);
    urls.push('http://127.0.0.1:' + TICK_PORT + TICK_PATH);
    return urls;
  }
  const TURLS = dbTickUrls();
  let inflight = false;
  async function grabPrice(url, parse){
    const r = await fetch(url, { cache: 'no-store' });
    const j = await r.json();
    return parse(j);
  }
  // The poller NEVER gives up: a dead endpoint / CORS rejection just skips the
  // tick and the next interval tries again, so the chart recovers by itself.
  setInterval(function(){
    (async function(){
      if (inflight) { return; }
      inflight = true;
      try {
        let price = NaN;
        for (let i = 0; i < TURLS.length && (!isFinite(price) || price <= 0); i++) {
          try { price = await grabPrice(TURLS[i], function(j){ return +j.last; }); } catch (e) {}
        }
        if ((!isFinite(price) || price <= 0) && DIRECT_URL) {
          try { price = await grabPrice(DIRECT_URL, directParse); } catch (e) {}
        }
        if (!isFinite(price) || price <= 0) { return; }
        if (liveLine) { try { liveLine.applyOptions({ price: price }); } catch (e) {} }
        const now = Math.floor(Date.now() / 1000);
        const barTs = now - (now % STEP);
        if (!lastBar || barTs > lastBar.time) {
          lastBar = { time: barTs, open: price, high: price, low: price, close: price };
        } else {
          lastBar.close = price;
          if (price > lastBar.high) { lastBar.high = price; }
          if (price < lastBar.low) { lastBar.low = price; }
        }
        mainSeries.update(lastBar);
      } finally { inflight = false; }
    })();
  }, __MS__);
})();
"""


def build_live_poller_js(
    exchange: str,
    symbol: str,
    step_sec: int,
    interval_ms: int = 1000,
    tick_path: Optional[str] = None,
    tick_port: Optional[int] = None,
) -> str:
    """
    Returns browser-side JS that live-updates the last chart bar + the dotted
    live price line once per interval.

    Two data sources, tried in order on every tick:
    1) `tick_path`/`tick_port` — the dashboard's own live-tick JSON endpoint
       (host resolved in the browser from document.referrer, since these charts
       live in srcdoc iframes). The endpoint serves the DB row written every
       second by the dashboard's background writer, so charts stay live even
       when the browser itself cannot reach the exchange (CORS, geo-block).
    2) the exchange public REST ticker directly (fallback when the endpoint is
       unavailable), supported for all 9 collector exchanges.

    The poller never stops on errors (a failing source only skips that tick).
    Returns "" when live refresh is disabled or no data source is available.
    """
    if interval_ms <= 0:
        return ""
    if not tick_path and exchange is None:
        return ""
    from src.exchanges.symbol_selector import split_symbol

    base, quote = split_symbol(symbol or "")
    b, q = base.upper(), quote.upper()
    perp = ":" in (symbol or "")
    ex = (exchange or "").lower()

    url = None
    parse = None
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
    elif ex == "bitget":
        if perp:
            url = f"https://api.bitget.com/api/v2/mix/market/ticker?symbol={b}{q}&productType={q}-FUTURES"
        else:
            url = f"https://api.bitget.com/api/v2/spot/market/tickers?symbol={b}{q}"
        parse = "const it=(Array.isArray(j.data)?j.data[0]:j.data)||{};price=+it.lastPr;bid=+it.bidPr;ask=+it.askPr;"
    elif ex == "htx":
        if perp:
            url = f"https://api.hbdm.com/linear-swap-ex/market/detail/merged/{b}-{q}"
        else:
            url = f"https://api.huobi.pro/market/detail/merged?symbol={(b + q).lower()}"
        parse = "const it=j.tick||{};price=+it.close;bid=+((it.bid||[])[0]);ask=+((it.ask||[])[0]);"
    elif ex == "coinex":
        kind = "futures" if perp else "spot"
        url = f"https://api.coinex.com/v2/{kind}/ticker?market={b}{q}"
        parse = "const it=(Array.isArray(j.data)?j.data[0]:j.data)||{};price=+it.last;bid=+it.bid;ask=+it.ask;"

    if not tick_path and not url:
        return ""

    return (
        _LIVE_POLLER_TEMPLATE
        .replace("__TICK_PATH__", json.dumps(tick_path) if tick_path else "''")
        .replace("__TICK_PORT__", str(int(tick_port or 0)))
        .replace("__DIRECT_URL__", json.dumps(url) if url else "''")
        .replace("__PARSE__", parse or "price=NaN;")
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


def stitch_candle_gaps(
    df: pd.DataFrame,
    fetcher,
    step: int,
    max_gap_buckets: int = 2000,
    include_tail: bool = True,
    now_sec: Optional[int] = None,
    max_tail_buckets: int = 40_000,
):
    """
    Fills gaps in a candle frame by fetching the missing bars through `fetcher`
    (a callable (start_bucket, end_bucket) -> [[ts_ms, o, h, l, c, v], ...]).
    With include_tail=True it also fetches the CLOSED bars between the last
    stored candle and now (the still-forming bar is left to the live poller —
    closed bars are immutable and safe to cache). The TAIL has its own, much
    larger budget (`max_tail_buckets`, ~40k bars ≈ 1.1 years of 15m): an
    interior hole of 3 weeks is suspicious, but a 3-week-stale table tail is
    exactly the case that MUST be bridged — otherwise the chart ends weeks
    before the live price line and the user sees a big empty price gap.
    Pure data operation (fully
    testable with a fake fetcher); returns (new_df, added_count). The database
    itself is NOT modified.
    """
    if df is None or df.empty:
        return df, 0

    ts_sorted = sorted(int(t) for t in df["ts"])
    buckets = sorted(set(t // step for t in ts_sorted))
    ranges = [r for r in find_missing_bucket_ranges(buckets, step) if r[1] - r[0] <= max_gap_buckets]

    if include_tail and buckets:
        now = int(now_sec if now_sec is not None else _time.time())
        now_bucket = now // step
        if buckets[-1] < now_bucket and now_bucket - buckets[-1] <= max_tail_buckets:
            ranges.append((buckets[-1] + 1, now_bucket))  # half-open: forming bar excluded

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


def build_metric_chart_html(
    points_json: str,
    title: str,
    color: str,
    chart_height: int,
    precision: int = 2,
    min_move: float = 0.01,
) -> str:
    """Self-contained Lightweight-Charts LINE page for a scalar metric series
    (open interest / funding rate) rendered under the main candle charts.

    Pure string builder (unit-testable without streamlit). `points_json` is a
    compact array of [epoch_sec, value] pairs — same transport shape as the
    candle payloads. Shows the whole stored history of the metric ("as much as
    was collected"); an empty series still renders a proper empty chart with
    the title so panels never collapse visually.
    """
    import json as _json

    safe_title = _json.dumps(str(title))
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background-color: #131722; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; }}
            #metric-chart {{ width: 100%; height: {chart_height}px; position: relative; }}
            #legend {{ position: absolute; top: 6px; left: 10px; z-index: 3; font-size: 12px;
                       color: #d1d4dc; pointer-events: none; opacity: 0.9; }}
        </style>
    </head>
    <body>
        <div id="metric-chart"></div>
        <div id="legend">{title}</div>
        <script>
            const points = {points_json};
            const legendEl = document.getElementById('legend');
            const baseTitle = {safe_title};
            const chart = LightweightCharts.createChart(document.getElementById('metric-chart'), {{
                height: {chart_height},
                layout: {{ background: {{ color: '#131722' }}, textColor: '#d1d4dc' }},
                grid: {{ vertLines: {{ color: '#1e222d' }}, horzLines: {{ color: '#1e222d' }} }},
                timeScale: {{ borderColor: '#2a2e39', timeVisible: true, secondsVisible: false }},
                rightPriceScale: {{ borderColor: '#2a2e39' }},
                crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            }});
            const series = chart.addLineSeries({{
                color: '{color}',
                lineWidth: 2,
                priceFormat: {{ type: 'price', precision: {int(precision)}, minMove: {float(min_move)} }},
            }});
            const data = points.map(function(r) {{ return {{ time: r[0], value: r[1] }}; }});
            series.setData(data);
            chart.timeScale().fitContent();
            function fmt(v) {{ return v.toLocaleString('en-US', {{ maximumFractionDigits: {int(precision)} }}); }}
            if (data.length) {{
                legendEl.textContent = baseTitle + ' · ' + fmt(data[data.length - 1].value);
            }}
            chart.subscribeCrosshairMove(function(param) {{
                if (!param.time || !param.seriesData) {{ legendEl.textContent = baseTitle; return; }}
                const d = param.seriesData.get(series);
                if (d && d.value !== undefined) legendEl.textContent = baseTitle + ' · ' + fmt(d.value);
            }});
            new ResizeObserver(function() {{
                chart.applyOptions({{ width: document.getElementById('metric-chart').clientWidth }});
            }}).observe(document.getElementById('metric-chart'));
        </script>
    </body>
    </html>
    """


def sanitize_metric_points(rows, now_sec: int) -> list:
    """[epoch_sec, value] pairs from raw metric rows (OI / funding): ms→s
    normalization, None filtering, and dropping of pre-2010 / future
    timestamps — junk-min tables (the 1983 ZINC glitch) must not stretch the
    panel's x-axis across decades."""
    out = []
    for ts, v in rows:
        if ts is None or v is None:
            continue
        ts = int(ts)
        if ts > 1e11:
            ts //= 1000
        if ts < 1262304000 or ts > now_sec + 900:  # 2010-01-01 .. now+15m
            continue
        out.append([ts, float(v)])
    out.sort(key=lambda p: p[0])
    return out
