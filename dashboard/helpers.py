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
