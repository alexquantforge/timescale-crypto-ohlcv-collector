import csv
from types import SimpleNamespace

import numpy as np

from short_setup_scanner import (
    _IDENT, _fmt_duration, _read_setups_csv, _screen_current_signal,
    _ticker_from_table, find_setups,
)


def test_formats_eta_duration():
    assert _fmt_duration(65) == "1м 05с"
    assert _fmt_duration(3661) == "1ч 01м"


def test_converts_database_table_name_to_exchange_symbol():
    assert _ticker_from_table("btc_usdt:usdt_on_bybit") == "BTC/USDT:USDT"


def test_reads_existing_csv_for_one_time_database_migration(tmp_path):
    fields = [
        "status", "invalidated", "base", "exchange", "market", "timeframe",
        "start_ts", "a_ts", "c_ts", "b_ts", "entry_ts", "pump_start", "a_price",
        "c_price", "b_price", "entry", "stop", "target", "pump_pct", "ac_drop_pct",
        "pump_retrace_pct", "cb_bounce_pct", "rr", "database", "table",
    ]
    row = {
        "status": "CURRENT", "invalidated": "False", "base": "BTC", "exchange": "bybit",
        "market": "swap", "timeframe": "15m", "start_ts": "1", "a_ts": "2", "c_ts": "3",
        "b_ts": "4", "entry_ts": "5", "pump_start": "100", "a_price": "150",
        "c_price": "135", "b_price": "145", "entry": "143", "stop": "146",
        "target": "135", "pump_pct": "50", "ac_drop_pct": "10",
        "pump_retrace_pct": "30", "cb_bounce_pct": "7", "rr": "2.6",
        "database": "ohlcv_15m", "table": "btc_usdt:usdt_on_bybit",
    }
    path = tmp_path / "short_setups.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    imported = _read_setups_csv(str(path))
    assert len(imported) == 1
    assert imported[0]["entry_ts"] == 5
    assert imported[0]["invalidated"] is False
    assert imported[0]["table"] == "btc_usdt:usdt_on_bybit"


def candles(c_low=136.0):
    # L=100 -> A=151 (>50%); C=136 retraces less than 30% of the impulse;
    # B=147 is a lower high. The close at index 11 confirms a drop.
    highs = np.array([110, 112, 120, 140, 151, 143, 140, 141, 142, 143, 147, 146, 145], dtype=float)
    lows = np.array([108, 100, 110, 130, 140, 137, c_low, 137, 139, 140, 143, 145.5, 144], dtype=float)
    closes = np.array([109, 110, 115, 135, 145, 140, 138, 139, 140, 142, 145, 145.8, 144.5], dtype=float)
    ts = np.arange(len(highs), dtype=np.int64) * 900
    return ts, lows, highs, closes


def test_finds_lower_high_setup_with_two_to_one_rr():
    ts, lows, highs, closes = candles()
    found = find_setups(ts, lows, highs, closes, bar_seconds=900, pivot_width=1)
    assert len(found) == 1
    event = found[0]
    assert event["pump_pct"] >= 50
    assert event["pump_retrace_pct"] <= 30
    assert event["b_price"] < event["a_price"]
    assert event["rr"] >= 2
    assert event["target"] == event["c_price"]
    assert event["stop"] > event["b_price"]


def test_pump_must_be_strictly_greater_than_50_percent():
    ts, lows, highs, closes = candles()
    highs[4] = 150.0  # Exactly +50% from L=100 is not a qualifying pump.
    found = find_setups(ts, lows, highs, closes, bar_seconds=900, pivot_width=1)
    assert found == []


def test_rejects_lower_local_a_if_prior_high_in_pump_leg_is_higher():
    ts, lows, highs, closes = candles()
    highs[3] = 155.0  # Above the proposed A at index 4 (150), between L and A.
    found = find_setups(ts, lows, highs, closes, bar_seconds=900, pivot_width=1)
    assert found == []


def test_rejects_setup_if_any_high_between_a_and_b_touches_or_crosses_a():
    ts, lows, highs, closes = candles()
    highs[8] = highs[4]  # B is still below A, but an intermediate wick retests A.
    found = find_setups(ts, lows, highs, closes, bar_seconds=900, pivot_width=1)
    assert found == []


def test_rejects_a_to_c_retracement_over_30_percent_of_impulse():
    ts, lows, highs, closes = candles(c_low=130.0)
    found = find_setups(ts, lows, highs, closes, bar_seconds=900, pivot_width=1)
    assert found == []


def test_rejects_setup_when_reward_risk_is_not_met():
    ts, lows, highs, closes = candles()
    found = find_setups(ts, lows, highs, closes, bar_seconds=900,
                        pivot_width=1, stop_buffer_pct=15.0)
    assert found == []


def test_perpetual_table_names_with_colon_are_scannable():
    assert _IDENT.fullmatch("btc_usdt:usdt_on_bybit")


def test_current_watch_keeps_live_setup_inside_c_b_with_good_health():
    args = SimpleNamespace(
        stop_buffer=0.5, rr=2.0, watch_min_tape=3.0,
        watch_min_depth_usd=1000.0, watch_max_spread_atr_pct=15.0,
        watch_min_7d_volume_usd=100_000.0,
    )
    event = {
        "invalidated": False, "a_price": 150.0, "c_price": 136.0,
        "b_price": 147.0, "entry": 143.0, "rr": 2.0,
    }
    snapshot = {
        "price": 137.0, "price_source": "LIVE", "trades_per_min": 4.0,
        "is_barcode": False, "depth_usd": 50_000.0,
        "spread_atr_pct": 8.0, "min_7d_volume_usd": 250_000.0,
    }
    signal, reason = _screen_current_signal(event, snapshot, args, now_ts=123)
    assert reason is None
    assert signal["status"] == "CURRENT"
    assert signal["entry"] == 137.0
    assert signal["entry_ts"] == 123
    assert signal["rr"] < 2.0  # Current RR is informational, not a watch filter.


def test_current_watch_keeps_inclusive_c_boundary_without_live_rr_filter():
    args = SimpleNamespace(
        stop_buffer=0.5, rr=2.0, watch_min_tape=3.0,
        watch_min_depth_usd=1000.0, watch_max_spread_atr_pct=15.0,
        watch_min_7d_volume_usd=100_000.0,
    )
    event = {"invalidated": False, "a_price": 150.0, "c_price": 136.0, "b_price": 147.0}
    snapshot = {
        "price": 136.0, "price_source": "LIVE", "trades_per_min": 4.0,
        "is_barcode": False, "depth_usd": 50_000.0,
        "spread_atr_pct": 8.0, "min_7d_volume_usd": 250_000.0,
    }
    signal, reason = _screen_current_signal(event, snapshot, args, now_ts=123)
    assert reason is None
    assert signal["entry"] == event["c_price"]
    assert signal["rr"] == 0.0


def test_current_watch_excludes_dead_tape_or_price_outside_c_b():
    args = SimpleNamespace(
        stop_buffer=0.5, rr=2.0, watch_min_tape=3.0,
        watch_min_depth_usd=1000.0, watch_max_spread_atr_pct=15.0,
        watch_min_7d_volume_usd=100_000.0,
    )
    event = {"invalidated": False, "a_price": 150.0, "c_price": 136.0, "b_price": 147.0}
    healthy = {
        "price": 146.0, "price_source": "LIVE", "trades_per_min": 4.0,
        "is_barcode": False, "depth_usd": 50_000.0,
        "spread_atr_pct": 8.0, "min_7d_volume_usd": 250_000.0,
    }
    dead = dict(healthy, trades_per_min=0.0)
    assert _screen_current_signal(event, dead, args, now_ts=123)[0] is None
    assert _screen_current_signal(event, dict(healthy, price=150.0), args, now_ts=123)[0] is None


def test_current_watch_applies_each_red_health_threshold():
    args = SimpleNamespace(
        stop_buffer=0.5, rr=2.0, watch_min_tape=3.0,
        watch_min_depth_usd=1000.0, watch_max_spread_atr_pct=15.0,
        watch_min_7d_volume_usd=100_000.0,
    )
    event = {"invalidated": False, "a_price": 150.0, "c_price": 136.0, "b_price": 147.0}
    healthy = {
        "price": 146.0, "price_source": "LIVE", "trades_per_min": 4.0,
        "is_barcode": False, "depth_usd": 50_000.0,
        "spread_atr_pct": 8.0, "min_7d_volume_usd": 250_000.0,
    }
    red_snapshots = (
        dict(healthy, is_barcode=True),
        dict(healthy, depth_usd=1000.0),
        dict(healthy, spread_atr_pct=15.0),
        dict(healthy, min_7d_volume_usd=100_000.0),
    )
    for snapshot in red_snapshots:
        assert _screen_current_signal(event, snapshot, args, now_ts=123)[0] is None
