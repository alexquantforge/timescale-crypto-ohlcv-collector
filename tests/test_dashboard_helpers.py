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


# ---------------------------------------------------------------------------
# Candle sanitization & pair links
# ---------------------------------------------------------------------------

def test_sanitize_candle_frame_drops_future_and_garbage():
    from dashboard.helpers import sanitize_candle_frame
    import pandas as pd

    df = pd.DataFrame({
        "ts": [1_700_000_000, 1_940_000_000, 1_000],  # valid 2023, garbage-2031, garbage-1970
        "close": [1.0, 2.0, 3.0],
    })
    out = sanitize_candle_frame(df)
    assert out["ts"].tolist() == [1_700_000_000]


def test_sanitize_candle_frame_milliseconds_table():
    from dashboard.helpers import sanitize_candle_frame
    import pandas as pd
    import time

    now_ms = int(time.time() * 1000)
    df = pd.DataFrame({"ts": [now_ms - 86400_000, now_ms - 2 * 86400_000], "close": [1.0, 2.0]})
    out = sanitize_candle_frame(df)
    assert len(out) == 2
    assert out["ts"].max() < 1e11  # converted to seconds
    assert out["ts"].is_monotonic_increasing


def test_sanitize_candle_frame_single_ms_row_in_seconds_table():
    from dashboard.helpers import sanitize_candle_frame
    import pandas as pd

    # seconds table + one corrupted millisecond row -> only the ms row is dropped
    df = pd.DataFrame({"ts": [1_780_000_000, 1_780_086_000, 1_786_000_000_000], "close": [1.0, 2.0, 3.0]})
    out = sanitize_candle_frame(df)
    assert 1_786_000_000_000 not in out["ts"].tolist()
    assert 1_780_000_000 in out["ts"].tolist()


def test_filter_sane_summary_rows_drops_ghost_pairs():
    """Ghost pairs: a table whose last row has a corrupted timestamp (future
    2031-style, pre-2013, ms-epoch or NaN) appeared in the pair list while its
    chart could never render a candle. The filter drops them at summary level."""
    from dashboard.helpers import filter_sane_summary_rows
    import time as _t

    now = int(_t.time())
    df = pd.DataFrame([
        {"ticker": "GOOD/USDT:USDT", "max_ts": now - 900},          # fresh
        {"ticker": "EDGY/USDT:USDT", "max_ts": now + 86400},        # +1d allowed (like sanitize)
        {"ticker": "GHOST/USDT:USDT", "max_ts": now + 10 * 86400},  # +10d future -> garbage
        {"ticker": "OLD/USDT:USDT", "max_ts": 1_234_567_890},       # 2009 -> garbage
        {"ticker": "MS/USDT:USDT", "max_ts": now * 1000},           # lone ms row in s frame -> garbage
        {"ticker": "NAN/USDT:USDT", "max_ts": None},
    ])
    out = filter_sane_summary_rows(df)
    assert out["ticker"].tolist() == ["GOOD/USDT:USDT", "EDGY/USDT:USDT"]


def test_filter_sane_summary_rows_ms_frame_converted_and_kept():
    from dashboard.helpers import filter_sane_summary_rows
    import time as _t

    now_ms = int(_t.time() * 1000)
    df = pd.DataFrame([
        {"ticker": "A/USDT:USDT", "max_ts": now_ms},
        {"ticker": "B/USDT:USDT", "max_ts": now_ms - 900_000},
    ])
    out = filter_sane_summary_rows(df)
    assert len(out) == 2  # whole ms frame converted, valid rows kept

    empty = filter_sane_summary_rows(pd.DataFrame())
    assert empty is not None and empty.empty
    assert filter_sane_summary_rows(None) is None


def test_demo_summary_passes_sanity_filter():
    """Demo mode must survive the ghost-pair filter (regression: demo used a
    fixed 1.8e9 max_ts, which drifts into the future as real time passes)."""
    from dashboard.helpers import generate_demo_summary, filter_sane_summary_rows

    for tf in ["15m", "1d"]:
        assert not filter_sane_summary_rows(generate_demo_summary(tf)).empty


def test_is_perp_and_find_perp_ticker():
    from dashboard.helpers import is_perp_symbol, find_perp_ticker

    assert is_perp_symbol("ADA/USDT:USDT")
    assert not is_perp_symbol("ADA/USDT")

    df = pd.DataFrame([
        {"ticker": "ADA/USDT", "exchange": "bybit"},
        {"ticker": "ADA/USDT:USDT", "exchange": "okx"},
    ])
    assert find_perp_ticker([df], "ADA", "okx") == "ADA/USDT:USDT"
    assert find_perp_ticker([df], "ADA", "bybit") is None
    assert find_perp_ticker([df], "BTC", "okx") is None


def test_build_pair_links_html_variants():
    from dashboard.helpers import build_pair_links_html

    perp = build_pair_links_html("ADA/USDT:USDT", "bybit")
    assert "trade/spot/ADA/USDT" in perp
    assert "trade/usdt/ADAUSDT" in perp
    assert "✅ Short: perp" in perp

    spot_with_perp = build_pair_links_html("ADA/USDT", "bybit", "ADA/USDT:USDT")
    assert "✅ Short: perp ADA/USDT:USDT" in spot_with_perp

    spot_no_perp = build_pair_links_html("XYZ/USDT", "bybit", None)
    assert "no perp" in spot_no_perp


# ---------------------------------------------------------------------------
# Live data: intraday->daily merge & poller JS
# ---------------------------------------------------------------------------

def _mk_df(ts_list, o=1.0, h=2.0, l=0.5, c=1.5, v=1.0):
    import pandas as pd
    n = len(ts_list)
    return pd.DataFrame({
        "ts": ts_list,
        "time": pd.to_datetime(ts_list, unit="s"),
        "open": [o] * n, "high": [h] * n, "low": [l] * n, "close": [c] * n, "volume": [v] * n,
    })


def test_merge_intraday_into_daily_appends_today():
    from dashboard.helpers import merge_intraday_into_daily
    import pandas as pd

    day = 1_786_000_000 - (1_786_000_000 % 86400)
    d1 = _mk_df([day - 86400])
    d15 = _mk_df([day, day + 900], o=1.5, h=2.5, l=1.4, c=1.8, v=3.0)
    out = merge_intraday_into_daily(d1, d15)

    assert len(out) == 2
    last = out.iloc[-1]
    assert int(last["ts"]) == day
    assert last["open"] == 1.5 and last["high"] == 2.5 and last["low"] == 1.4
    assert last["close"] == 1.8 and last["volume"] == 6.0


def test_merge_intraday_into_daily_replaces_stale_bar():
    from dashboard.helpers import merge_intraday_into_daily

    day = 1_786_000_000 - (1_786_000_000 % 86400)
    d1 = _mk_df([day])  # stale bar for today already present
    d15 = _mk_df([day + 900, day + 1800], o=1.1, h=2.2, l=1.0, c=2.0, v=2.0)
    out = merge_intraday_into_daily(d1, d15)

    assert len(out) == 1
    assert out.iloc[-1]["high"] == 2.2 and out.iloc[-1]["volume"] == 4.0


def test_merge_intraday_into_daily_older_15m_ignored():
    from dashboard.helpers import merge_intraday_into_daily

    day = 1_786_000_000 - (1_786_000_000 % 86400)
    d1 = _mk_df([day])
    d15 = _mk_df([day - 86400])  # 15m data older than daily chart
    out = merge_intraday_into_daily(d1, d15)
    assert len(out) == 1
    assert out.iloc[-1]["close"] == 1.5  # unchanged


def test_build_live_poller_js_variants():
    from dashboard.helpers import build_live_poller_js

    bybit = build_live_poller_js("bybit", "ADA/USDT:USDT", 900, 1000)
    assert "api.bybit.com" in bybit and "category=linear&symbol=ADAUSDT" in bybit
    assert "mainSeries.update" in bybit and "__STEP__" not in bybit

    okx_spot = build_live_poller_js("okx", "ADA/USDT", 86400, 2000)
    assert "instId=ADA-USDT" in okx_spot and "86400" in okx_spot and "2000" in okx_spot

    okx_perp = build_live_poller_js("okx", "ADA/USDT:USDT", 900, 1000)
    assert "instId=ADA-USDT-SWAP" in okx_perp

    gate = build_live_poller_js("gateio", "BTC/USDT:USDT", 900, 1000)
    assert "futures/usdt/tickers" in gate and "contract=BTC_USDT" in gate

    assert build_live_poller_js("unknown_ex", "ADA/USDT", 900, 1000) == ""  # unsupported
    assert build_live_poller_js("bybit", "ADA/USDT", 900, 0) == ""          # disabled


def test_build_live_poller_js_all_nine_exchanges_supported():
    """Every collector exchange must have a live chart poller — a missing one
    leaves the chart frozen while only the server-side chips keep updating."""
    from dashboard.helpers import build_live_poller_js

    cases = [
        ("bybit", "BTC/USDT:USDT", "api.bybit.com"),
        ("okx", "BTC/USDT:USDT", "okx.com"),
        ("gateio", "BTC/USDT:USDT", "gateio.ws"),
        ("kucoin", "BTC/USDT:USDT", "kucoin.com"),
        ("mexc", "BTC/USDT:USDT", "mexc.com"),
        ("bingx", "BTC/USDT:USDT", "bingx.com"),
        ("bitget", "BTC/USDT:USDT", "api.bitget.com/api/v2/mix"),
        ("htx", "BTC/USDT:USDT", "hbdm.com/linear-swap-ex"),
        ("coinex", "BTC/USDT:USDT", "coinex.com/v2/futures"),
    ]
    for exchange, symbol, marker in cases:
        js = build_live_poller_js(exchange, symbol, 900, 1000)
        assert js, f"no live poller for {exchange}"
        assert marker in js, f"unexpected endpoint for {exchange}"
        assert "__STEP__" not in js and "__PARSE__" not in js

    # spot variants resolve too
    assert "productType" not in build_live_poller_js("bitget", "BTC/USDT", 900, 1000)
    assert "symbol=btcusdt" in build_live_poller_js("htx", "BTC/USDT", 900, 1000)
    assert "spot/ticker" in build_live_poller_js("coinex", "BTC/USDT", 900, 1000)


def test_build_live_poller_js_never_gives_up():
    """The poller must retry forever: a previous version called clearInterval
    after 5 failed fetches, permanently freezing the chart on any hiccup."""
    from dashboard.helpers import _LIVE_POLLER_TEMPLATE, build_live_poller_js

    assert "clearInterval" not in _LIVE_POLLER_TEMPLATE
    js = build_live_poller_js("bybit", "BTC/USDT:USDT", 900, 1000)
    assert "clearInterval" not in js
    assert "inflight" in js  # overlapping-request guard


def test_build_live_poller_js_db_tick_mode():
    """DB mode: the poller reads the dashboard's own /tick endpoint (serving the
    TimescaleDB live row) first, with direct exchange REST only as fallback."""
    from dashboard.helpers import build_live_poller_js

    tick_path = "/tick?db=ohlcv_1d_high&ex=bybit&sym=BTC%2FUSDT%3AUSDT"
    js = build_live_poller_js(
        "bybit", "BTC/USDT:USDT", 900, 1000, tick_path=tick_path, tick_port=8511
    )
    assert tick_path in js and "8511" in js
    assert "document.referrer" in js  # host resolved client-side (srcdoc iframe)
    assert "+j.last" in js            # DB payload parse
    assert "api.bybit.com" in js      # direct REST stays as fallback

    # DB payload alone is enough even for an exchange without a REST mapping
    js2 = build_live_poller_js(
        "unknown_ex", "BTC/USDT:USDT", 900, 1000,
        tick_path=tick_path, tick_port=8511,
    )
    assert tick_path in js2
    assert "__TICK_PATH__" not in js2 and "__TICK_PORT__" not in js2


# ---------------------------------------------------------------------------
# In-memory gap stitching
# ---------------------------------------------------------------------------

def test_find_missing_bucket_ranges():
    from dashboard.helpers import find_missing_bucket_ranges
    assert find_missing_bucket_ranges([1, 2, 3], 900) == []
    assert find_missing_bucket_ranges([1, 2, 5, 6], 900) == [(3, 5)]
    assert find_missing_bucket_ranges([0, 10], 900) == [(1, 10)]
    assert find_missing_bucket_ranges([5], 900) == []
    assert find_missing_bucket_ranges([], 900) == []


def test_stitch_candle_gaps_fills_hole_with_fake_fetcher():
    from dashboard.helpers import stitch_candle_gaps
    import pandas as pd

    step = 86400
    base = 1_786_000_000 - (1_786_000_000 % step)
    df = _mk_df([base, base + step, base + 4 * step, base + 5 * step])  # hole of 2 days

    def fetcher(r0, r1):
        # returns ms candles for the requested bucket range
        return [[(base + b * step) * 1000, 1.0, 1.5, 0.9, 1.2, 5.0] for b in range(r0 - base // step, r1 - base // step)]

    out, added = stitch_candle_gaps(df, fetcher, step, include_tail=False)
    assert added == 2
    ts = out["ts"].tolist()
    assert ts == sorted(ts)
    assert base + 2 * step in ts and base + 3 * step in ts
    assert len(out) == 6


def test_stitch_candle_gaps_no_gap_noop_and_skips_huge_gaps():
    from dashboard.helpers import stitch_candle_gaps

    step = 900
    full = _mk_df(list(range(1_786_000_000, 1_786_000_000 + 5 * step, step)))
    out, added = stitch_candle_gaps(full, lambda r0, r1: (_ for _ in ()).throw(AssertionError("must not fetch")), step)
    assert added == 0 and len(out) == len(full)

    huge = _mk_df([1_786_000_000, 1_786_000_000 + 5000 * step])
    out2, added2 = stitch_candle_gaps(huge, lambda r0, r1: [], step, max_gap_buckets=2000)
    assert added2 == 0 and len(out2) == 2


def test_stitch_candle_gaps_fetches_closed_tail_only():
    from dashboard.helpers import stitch_candle_gaps

    step = 900
    now = 1_786_100_000
    # stored frame ends 3 closed buckets + 1 forming bucket behind `now`
    last_stored = now - 4 * step
    df = _mk_df([last_stored - step, last_stored])

    calls = []

    def fetcher(r0, r1):
        calls.append((r0, r1))
        return [[b * step * 1000, 1.0, 1.5, 0.9, 1.2, 5.0] for b in range(r0, r1)]

    out, added = stitch_candle_gaps(df, fetcher, step, include_tail=True, now_sec=now)
    assert added == 3  # three closed buckets, forming one excluded
    tail_call = calls[-1]
    assert tail_call[0] == last_stored // step + 1
    assert tail_call[1] == now // step  # half-open: forming bucket NOT fetched
    assert (out["ts"] == now // step * step).sum() == 0  # no forming bar added
    assert out["ts"].is_monotonic_increasing


def test_stitch_candle_gaps_tail_disabled():
    from dashboard.helpers import stitch_candle_gaps

    step = 900
    now = 1_786_100_000
    df = _mk_df([now - 3 * step, now - 2 * step])
    out, added = stitch_candle_gaps(df, lambda r0, r1: [], step, include_tail=False, now_sec=now)
    assert added == 0 and len(out) == len(df)


# ---------------------------------------------------------------------------
# Lightweight Charts page assembly (live-poller regression)
# ---------------------------------------------------------------------------


def _build_chart_html(with_volume=False, poller="POLLER_STUB();", history=""):
    import json as _json
    from dashboard.helpers import build_lightweight_chart_html

    # payloads are COMPACT ARRAYS: candles [ts,o,h,l,c], volume [ts,value,dir]
    candles = [[1000, 1.0, 1.2, 0.9, 1.1]]
    vol = [[1000, 10.0, 1]]
    return build_lightweight_chart_html(
        candles_json=_json.dumps(candles),
        volume_json=_json.dumps(vol) if with_volume else None,
        chart_height=470,
        chart_style="OHLCV Bars",
        live_poller_js=poller,
        history_loader_js=history,
    )


def test_chart_html_declares_candles_data_variable():
    """Regression: the lastBar initializer must reference a DECLARED JS
    variable. Previously it used a `candles` identifier that existed only in
    Python, so every chart died with `ReferenceError: candles is not
    defined`, the live poller was never installed and charts stayed frozen."""
    html = _build_chart_html()
    assert "const candlesData =" in html
    assert "mainSeries.setData(candlesData)" in html
    assert "let lastBar = candlesData.length ? Object.assign({}, candlesData[candlesData.length - 1]) : null;" in html
    # the bug pattern: bare `candles` identifier (never declared)
    assert "candles.length ? Object.assign({}, candles[" not in html
    assert "setData(candles);" not in html


def test_chart_html_declaration_precedes_all_uses():
    """`const candlesData` must appear before every identifier that reads it,
    otherwise the page again dies with a ReferenceError at init."""
    html = _build_chart_html()
    decl = html.index("const candlesData =")
    assert decl < html.index("mainSeries.setData(candlesData)")
    assert decl < html.index("let lastBar = candlesData.length")


def test_chart_html_injects_live_poller_without_visual_noise():
    html = _build_chart_html()
    assert "POLLER_STUB();" in html
    # no in-chart 'LIVE' decorations: neither the price-line axis title nor
    # the corner badge (redundant next to the always-on server LIVE chip)
    assert 'live-badge' not in html
    assert "title: 'LIVE'" not in html


def test_chart_html_volume_optional():
    off = _build_chart_html(with_volume=False)
    assert "volumeSeries" not in off
    assert "const volumeData =" not in off
    on = _build_chart_html(with_volume=True)
    assert "const volumeData =" in on
    assert "volumeSeries.setData(volumeData)" in on
    assert "allVolume = volumeData;" in on


def test_chart_html_maps_compact_arrays_and_tracks_all_candles():
    """Payloads are compact arrays remapped to objects in JS; `allCandles` /
    `allVolume` aliases exist for the infinite-history loader to prepend into."""
    html = _build_chart_html()
    assert "const candlesData = (" in html
    assert ".map(function(r){ return { time: r[0], open: r[1], high: r[2], low: r[3], close: r[4] }; });" in html
    assert "let allCandles = candlesData;" in html
    assert "let allVolume = null;" in html
    assert "UP_VOL_COLOR" in html and "DN_VOL_COLOR" in html


def test_build_series_arrays_shape_rounding_and_volume_dir():
    from dashboard.helpers import build_series_arrays

    df = pd.DataFrame({
        "ts": [1000, 1900],
        "open": [1.234567890123, 2.0],
        "high": [1.5, 2.5],
        "low": [1.0, 1.5],
        "close": [1.4, 1.8],
        "volume": [12345.678, float("nan")],
    })
    candles, vol = build_series_arrays(df, with_volume=True)
    assert candles == [[1000, 1.23456789, 1.5, 1.0, 1.4], [1900, 2.0, 2.5, 1.5, 1.8]]
    # volume: [ts, rounded value, dir (+1 up / -1 down)], NaN -> 0
    assert vol == [[1000, 12345.68, 1], [1900, 0.0, -1]]
    candles_nv, vol_nv = build_series_arrays(df, with_volume=False)
    assert vol_nv is None and candles_nv == candles


def test_rows_to_compact_candles_sanitizes_and_sorts():
    from dashboard.helpers import rows_to_compact_candles

    now = 1_786_100_000
    rows = [
        {"ts": now - 900, "open": 2, "high": 3, "low": 1, "close": 2.5, "volume": 10.005},
        {"ts": (now - 1800) * 1000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": None},  # ms epoch
        {"ts": 800_000_000, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},  # pre-2013 garbage
        {"ts": now + 10 * 86400, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},  # future garbage
    ]
    out = rows_to_compact_candles(rows, now_sec=now)
    assert [r[0] for r in out] == [now - 1800, now - 900]  # ascending, ms converted
    assert out[0][1:5] == [1.0, 2.0, 0.5, 1.5]
    assert out[0][5] == 0.0       # None volume -> 0
    assert out[1][5] == 10.01     # rounded to 2 decimals
    assert rows_to_compact_candles([], now_sec=now) == []
    assert rows_to_compact_candles(None, now_sec=now) == []


def test_rows_to_compact_candles_drops_nonfinite_values():
    """A single NaN/Inf must never reach json.dumps — the bare `NaN` literal
    is invalid JSON and would break response.json() in the chart's loader."""
    from dashboard.helpers import rows_to_compact_candles

    now = 1_786_100_000
    rows = [
        {"ts": now - 900, "open": 2, "high": 3, "low": 1, "close": 2.5, "volume": float("nan")},
        {"ts": now - 800, "open": float("nan"), "high": 3, "low": 1, "close": 2.5, "volume": 5},
        {"ts": now - 700, "open": 2, "high": 3, "low": 1, "close": float("inf"), "volume": 5},
    ]
    out = rows_to_compact_candles(rows, now_sec=now)
    assert len(out) == 1                      # both NaN-open and Inf-close rows skipped
    assert out[0][0] == now - 900
    assert out[0][5] == 0.0                   # NaN volume -> 0
    import json as _json
    s = _json.dumps({"c": out})
    assert "NaN" not in s and "Infinity" not in s


def test_build_history_loader_js_requires_endpoint_bits():
    from dashboard.helpers import build_history_loader_js

    assert build_history_loader_js(None, "t_on_x", 900, 8512) == ""
    assert build_history_loader_js("db", None, 900, 8512) == ""
    assert build_history_loader_js("db", "t_on_x", 900, None) == ""


def test_build_history_loader_js_content():
    from dashboard.helpers import build_history_loader_js

    js = build_history_loader_js("ohlcv15m", "ada_usdt_on_bybit", 900, 8512, chunk=1200)
    assert "subscribeVisibleLogicalRangeChange" in js
    assert "const PORT = 8512;" in js
    assert "/candles?db=ohlcv15m&table=ada_usdt_on_bybit" in js
    assert "const CHUNK = 1200;" in js
    assert "exhausted = true;" in js  # stops fetching at the start of history
    assert "olderC.concat(allCandles)" in js  # prepends older chunk
    # robust server discovery: probes /healthz on referrer-host, then localhost
    assert "/healthz" in js
    assert "http://localhost:" in js and "http://127.0.0.1:" in js
    assert "hist-status" in js  # drives the visible status badge


def test_chart_html_injects_history_loader():
    html = _build_chart_html(history="HISTORY_STUB();")
    assert "HISTORY_STUB();" in html
    off = _build_chart_html()
    assert "HISTORY_STUB" not in off


def test_chart_html_has_history_status_badge():
    """The loader's visible status badge (proves to the user which code the
    page runs and shows why history did not load)."""
    html = _build_chart_html()
    assert 'id="hist-status"' in html
