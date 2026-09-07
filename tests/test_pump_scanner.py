"""Unit tests for the standalone `pump_scanner.py` scanner (pure parts only).

The scanner reads the collector's stored OHLCV tables, detects cross-exchange
pumps (>= PUMP_MIN_PCT% over <= PUMP_WINDOW_DAYS days from a rolling low to a
peak), applies a "before the pump" (new-high) filter, and confirms coins across
exchanges. These tests pin the bits that do not need a live database: table
name -> ticker/exchange/kind inference, base and quote extraction, the junk-base
exclusions, the vectorised pump detection, the pre-pump filter, the cross-exchange
grouping, and the CLI override path (which recomputes the derived params).

Global config is snapshotted and restored so a test mutating it via `apply_args`
can never leak into the rest of the suite.
"""

import os

os.environ.setdefault("DASHBOARD_DEMO", "1")

import numpy as np

import pump_scanner as ps

import pytest


# ---------------------------------------------------------------------------
# Fixture: restore the module globals apply_args/_recompute_derived may reset.
# ---------------------------------------------------------------------------
_CONFIG_GLOBALS = [
    "PUMP_MIN_PCT", "PUMP_WINDOW_DAYS", "SCAN_TIMEFRAME", "PUMP_PRICE_SOURCE",
    "EXCHANGES_INCLUDE", "MIN_EXCHANGES", "REQUIRE_ALL_EXCHANGES", "PEAK_ALIGN_DAYS",
    "PRE_PUMP_DAYS", "PRE_PUMP_BELOW_PEAK_FACTOR", "RECENT_PUMPS_DAYS",
    "REPORT_TOP_N", "SAVE_TO_DB", "PRINT_RUN_STATISTICS", "INCLUDE_SPOT", "INCLUDE_SWAP",
]

_DERIVED_GLOBALS = [
    "BAR_MINUTES", "BAR_SEC", "BARS_IN_DAY", "WINDOW_BARS", "PRE_PUMP_BARS",
    "MERGE_GAP_BARS", "MIN_HISTORY_BARS", "SHORT_MIN_HISTORY_BARS", "THRESH_RATIO",
    "MIN_PLAUSIBLE_PEAK_TS", "DB_NAMES",
]


@pytest.fixture(autouse=True)
def _restore_config():
    saved = {g: getattr(ps, g) for g in _CONFIG_GLOBALS + _DERIVED_GLOBALS}
    yield
    for g, v in saved.items():
        setattr(ps, g, v)


# ---------------------------------------------------------------------------
# Table name -> ticker / exchange / kind
# ---------------------------------------------------------------------------


def test_parse_table_name_spot():
    assert ps.parse_table_name("btc_usdt_on_bybit") == ("BTC/USDT", "bybit", "spot")


def test_parse_table_name_perp():
    assert ps.parse_table_name("btc_usdt:usdt_on_bingx") == ("BTC/USDT:USDT", "bingx", "swap")


def test_parse_table_name_long_base():
    assert ps.parse_table_name("1000000babydoge_usdt_on_gateio") == (
        "1000000BABYDOGE/USDT", "gateio", "spot")


def test_parse_table_name_garbage():
    assert ps.parse_table_name("") == ("", "", "unknown")
    assert ps.parse_table_name("not_a_pair_table") == ("NOT_A_PAIR_TABLE", "", "unknown")
    # No `_on_`: the raw name is uppercased as-is, `_` is NOT turned into `/`.
    assert ps.parse_table_name("btc_usdt") == ("BTC_USDT", "", "unknown")


def test_base_of_ticker():
    assert ps.base_of_ticker("BTC/USDT") == "BTC"
    assert ps.base_of_ticker("BTC/USDT:USDT") == "BTC"
    assert ps.base_of_ticker("1000000BABYDOGE/USDT:USDT") == "1000000BABYDOGE"


def test_quote_of_ticker():
    assert ps.quote_of_ticker("BTC/USDT") == "USDT"
    assert ps.quote_of_ticker("BTC/USDT:USDT") == "USDT"


# ---------------------------------------------------------------------------
# Junk-base exclusions (test markets, bitget tokenized stocks)
# ---------------------------------------------------------------------------


def test_is_excluded_base_global_regex():
    assert ps.is_excluded_base("TEST251204") is True
    assert ps.is_excluded_base("BTC") is False


def test_is_excluded_base_exchange_regex():
    assert ps.is_excluded_base("RCOIN", "bitget") is True
    assert ps.is_excluded_base("RCOIN", "bybit") is False   # bitget-only rule
    assert ps.is_excluded_base("RNDR", "bitget") is False   # real R-* ticker safe


# ---------------------------------------------------------------------------
# Vectorised pump detection
# ---------------------------------------------------------------------------


def _series():
    """10 daily bars with a clear pump (min=1 -> peak close 5.5) near the end."""
    ts = np.arange(1750000000, 1750000000 + 10 * 86400, 86400, dtype=np.int64)
    low = np.array([1.0] * 10)
    high = np.array([1.1] * 5 + [1.1, 5.2, 5.5, 4.0, 3.0])
    close = np.array([1.0] * 5 + [1.0, 5.0, 5.5, 4.0, 3.0])
    return ts, low, high, close


def test_find_pumps_detects_above_threshold():
    ts, low, high, close = _series()
    now = int(ts[-1]) + 86400
    evs = ps.find_pumps(ts, low, high, now, close=close)
    assert len(evs) == 1
    e = evs[0]
    assert e["pump_pct"] > 300.0
    assert e["span_days"] <= ps.PUMP_WINDOW_DAYS


def test_find_pumps_skips_flat_series():
    ts = np.arange(1750000000, 1750000000 + 10 * 86400, 86400, dtype=np.int64)
    vals = np.array([1.0] * 10)
    evs = ps.find_pumps(ts, vals, vals, int(ts[-1]) + 86400, close=vals)
    assert evs == []


def test_find_pumps_high_low_catches_wick():
    """A wick that pierces high (but close stays flat) is caught in high_low."""
    ts, low, high, close = _series()
    # Make a table where only HIGH has the spike, close stays flat.
    flat_close = np.array([1.0] * 5 + [1.0, 1.0, 1.0, 1.0, 1.0])
    now = int(ts[-1]) + 86400
    evs_close = ps.find_pumps(ts, low, high, now, close=flat_close)          # close: none
    evs_hl = ps.find_pumps(ts, low, high, now, close=flat_close,
                           price_source="high_low")                          # high: yes
    assert evs_close == []
    assert len(evs_hl) >= 1


def test_find_pumps_respects_thresh_ratio():
    """A pump exactly at the threshold (ratio == thr) is NOT counted (strict >)."""
    ts = np.arange(1750000000, 1750000000 + 10 * 86400, 86400, dtype=np.int64)
    # Pump to exactly 4x (300%): ratio == THRESH_RATIO when pct=300.
    low = np.array([1.0] * 10)
    high = np.array([4.0] * 10)
    close = np.array([1.0] * 5 + [1.0, 4.0, 4.0, 4.0, 4.0])
    evs = ps.find_pumps(ts, low, high, int(ts[-1]) + 86400, close=close,
                        thresh_ratio=4.0)
    assert evs == []


# ---------------------------------------------------------------------------
# "Before the pump" filter (new-high, not a return to old highs)
# ---------------------------------------------------------------------------


def test_passes_prepump_when_prior_prices_below_peak():
    price = np.array([1.0] * 5 + [1.0, 5.0, 5.5, 4.0, 3.0])
    ok, pre = ps.passes_prepump_filter(price, start_idx=5, peak_price=4.0)
    assert ok is True
    assert pre == pytest.approx(1.0)


def test_passes_prepump_fails_when_prior_price_at_peak():
    price = np.array([4.0] * 10)
    ok, pre = ps.passes_prepump_filter(price, start_idx=5, peak_price=4.0)
    assert ok is False
    assert pre == pytest.approx(4.0)


def test_passes_prepump_trivially_passes_with_no_history_before():
    price = np.array([1.0, 5.0, 5.5])
    ok, pre = ps.passes_prepump_filter(price, start_idx=0, peak_price=5.5)
    assert ok is True
    assert pre is None


# ---------------------------------------------------------------------------
# Cross-exchange grouping
# ---------------------------------------------------------------------------


def _ev(base, exchange, pump_pct, peak_ts):
    return {"base": base, "exchange": exchange, "pump_pct": pump_pct,
            "min_pump_pct": pump_pct, "peak_ts": peak_ts, "min_price": 1.0,
            "peak_price": 1.0 + pump_pct / 100.0, "pre_max_high": None,
            "short_history": False}


def test_build_coin_results_requires_all_exchanges():
    catalog = {"BTC": {"bybit", "okx"}}
    # Only bybit has a qualifying event -> not all exchanges -> rejected.
    events = [dict(_ev("BTC", "bybit", 350.0, 1750000000))]
    coins, rej = ps.build_coin_results(catalog, events, 1750000000)
    assert coins == []
    assert rej["not_all"] == 1


def test_build_coin_results_min_exchanges():
    catalog = {"BTC": {"bybit", "okx"}}
    events = [
        dict(_ev("BTC", "bybit", 350.0, 1750000000)),
        dict(_ev("BTC", "okx", 320.0, 1750001000)),
    ]
    coins, rej = ps.build_coin_results(catalog, events, 1750000000)
    assert len(coins) == 1
    assert coins[0]["base"] == "BTC"
    assert coins[0]["min_pump_pct"] == 320.0   # weakest exchange decides


# ---------------------------------------------------------------------------
# CLI override path (recomputes derived params without touching a DB)
# ---------------------------------------------------------------------------


def test_apply_args_overrides_and_recomputes():
    from pump_scanner import build_arg_parser, apply_args
    args = build_arg_parser().parse_args(
        ["--pct", "200", "--days", "7", "--timeframe", "15m",
         "--exchanges", "bybit,okx", "--no-save-to-db", "--min-exchanges", "2"])
    apply_args(args)
    assert ps.PUMP_MIN_PCT == 200.0
    assert ps.PUMP_WINDOW_DAYS == 7.0
    assert ps.SCAN_TIMEFRAME == "15m"
    assert ps.BAR_MINUTES == 15
    assert ps.WINDOW_BARS == 672          # 7 d * 96 bars/day
    assert ps.THRESH_RATIO == 3.0         # 1 + 200/100
    assert ps.EXCHANGES_INCLUDE == {"bybit", "okx"}
    assert ps.SAVE_TO_DB is False
    assert ps.MIN_EXCHANGES == 2
    assert ps.DB_NAMES == [ps.settings.db_high_15m, ps.settings.db_low_15m]


def test_parse_exchange_list_empty_is_none():
    assert ps._parse_exchange_list(None) is None
    assert ps._parse_exchange_list("") is None
    assert ps._parse_exchange_list("  ") is None
    assert ps._parse_exchange_list("bybit,OKX") == {"bybit", "okx"}
