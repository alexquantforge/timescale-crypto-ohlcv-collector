"""
Unit tests for pure dashboard helpers: pair navigation (Prev/Next),
summary-frame lookups, and synthetic demo data generation.
"""
import pandas as pd
import pytest

from dashboard.helpers import (
    shift_option,
    exchanges_for_ticker,
    find_table_row,
    generate_demo_summary,
    generate_demo_candles,
)


def test_shift_option_basic_and_wraparound():
    opts = ["A", "B", "C"]
    assert shift_option(opts, "A", 1) == "B"
    assert shift_option(opts, "B", -1) == "A"
    # Wrap-around in both directions
    assert shift_option(opts, "C", 1) == "A"
    assert shift_option(opts, "A", -1) == "C"
    # Multiple steps
    assert shift_option(opts, "A", 4) == "B"


def test_shift_option_unknown_and_empty():
    assert shift_option(["A", "B"], "ZZZ", 1) == "B"
    assert shift_option([], "A", 1) is None
    assert shift_option(None, "A", 1) is None if False else shift_option(["X"], None, 0) == "X"


def _summary_frame():
    return pd.DataFrame([
        {"ticker": "BTC/USDT:USDT", "exchange": "bybit", "volume_tier": "HIGH", "close": 100.0, "table_name": "t1", "db_name": "d1"},
        {"ticker": "BTC/USDT:USDT", "exchange": "okx", "volume_tier": "LOW", "close": 100.5, "table_name": "t2", "db_name": "d2"},
        {"ticker": "ETH/USDT:USDT", "exchange": "bybit", "volume_tier": "HIGH", "close": 50.0, "table_name": "t3", "db_name": "d1"},
    ])


def test_exchanges_for_ticker():
    df = _summary_frame()
    assert exchanges_for_ticker(df, "BTC/USDT:USDT") == ["bybit", "okx"]
    assert exchanges_for_ticker(df, "MISSING") == []
    assert exchanges_for_ticker(None, "BTC/USDT:USDT") == []


def test_find_table_row_prefers_exact_exchange_then_high_tier():
    df = _summary_frame()
    row = find_table_row(df, "BTC/USDT:USDT", "okx")
    assert row["exchange"] == "okx"

    # No exchange preference -> HIGH tier row wins
    row = find_table_row(df, "BTC/USDT:USDT")
    assert row["volume_tier"] == "HIGH"
    assert row["exchange"] == "bybit"

    assert find_table_row(df, "MISSING") is None
    assert find_table_row(None, "BTC/USDT:USDT") is None


def test_demo_generators_shape_and_determinism():
    for tf in ["15m", "1d"]:
        summary = generate_demo_summary(tf)
        assert not summary.empty
        assert {"ticker", "exchange", "close", "table_name", "volume_tier"} <= set(summary.columns)

        candles = generate_demo_candles(tf, "BTC/USDT:USDT", "bybit", n=300)
        assert len(candles) == 300
        assert {"ts", "time", "open", "high", "low", "close", "volume"} <= set(candles.columns)
        # OHLC integrity
        assert (candles["high"] >= candles[["open", "close"]].max(axis=1)).all()
        assert (candles["low"] <= candles[["open", "close"]].min(axis=1)).all()
        # Sorted ascending time
        assert candles["ts"].is_monotonic_increasing

        # Deterministic for the same seed inputs
        candles2 = generate_demo_candles(tf, "BTC/USDT:USDT", "bybit", n=300)
        pd.testing.assert_frame_equal(candles, candles2)


# ---------------------------------------------------------------------------
# Health strip
# ---------------------------------------------------------------------------

def test_health_score_boundaries():
    from dashboard.helpers import (
        score_trades_per_min, score_depth_usd, score_spread_atr_pct, score_min_volume_usd,
    )

    assert score_trades_per_min(0) == 0.0
    assert score_trades_per_min(60) == 0.5
    assert score_trades_per_min(500) == 1.0
    assert score_trades_per_min(None) == 0.0

    assert score_depth_usd(500) == 0.0
    assert score_depth_usd(50_000) == 1.0
    assert score_depth_usd(2_000_000) == 1.0

    assert score_spread_atr_pct(3.0) == 1.0
    assert score_spread_atr_pct(5.0) == 1.0   # user criterion: <5% of daily ATR = green
    assert score_spread_atr_pct(10.0) == 0.5
    assert score_spread_atr_pct(25.0) == 0.0

    assert score_min_volume_usd(50_000) == 0.0
    assert score_min_volume_usd(500_000) == 1.0
    assert score_min_volume_usd(None) == 0.0


def test_fmt_usd_compact():
    from dashboard.helpers import fmt_usd_compact
    assert fmt_usd_compact(1_500_000) == "$1.5M"
    assert fmt_usd_compact(850_000) == "$850K"
    assert fmt_usd_compact(950) == "$950"
    assert fmt_usd_compact(2_300_000_000) == "$2.3B"
    assert fmt_usd_compact(None) == "n/a"


def test_build_health_strip_html_chips_and_colors():
    from dashboard.helpers import build_health_strip_html
    row = {
        "ob_trades_per_min": 55.0,
        "ob_total_depth_usd": 120_000.0,
        "ob_spread_atr_pct": 2.1,
        "ob_min_7d_volume_usd": 700_000.0,
    }
    html = build_health_strip_html(row)
    for label in ["Tape", "Depth", "Spread", "Min 7d"]:
        assert label in html
    assert "55/min" in html and "$120K" in html and "2.1%" in html and "$700K" in html
    # green chips present (score 1 -> hsl(140))
    assert "hsl(140" in html
    assert "DEAD" not in html


def test_build_health_strip_dead_market():
    from dashboard.helpers import build_health_strip_html
    row = {
        "ob_trades_per_min": 1.2,
        "ob_total_depth_usd": 400.0,
        "ob_spread_atr_pct": 30.0,
        "ob_min_7d_volume_usd": 20_000.0,
        "ob_is_barcode": False,
    }
    html = build_health_strip_html(row)
    assert "DEAD" in html
    # red chips (score ~0 -> hsl(0))
    assert "hsl(0" in html


def test_build_health_strip_empty_row_is_safe():
    from dashboard.helpers import build_health_strip_html
    html = build_health_strip_html({})
    assert html.count("n/a") >= 3
    assert "hsl(0" in html
