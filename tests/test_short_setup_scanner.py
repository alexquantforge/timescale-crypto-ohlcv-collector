import numpy as np

from short_setup_scanner import find_setups


def candles(c_low=136.0):
    # L=100 -> A=150 (+50%); C=136 is a 28% retracement of the 50-point
    # impulse; B=147 is a lower high. The close at index 11 confirms a drop.
    highs = np.array([110, 112, 120, 140, 150, 143, 140, 141, 142, 143, 147, 146, 145], dtype=float)
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


def test_rejects_a_to_c_retracement_over_30_percent_of_impulse():
    ts, lows, highs, closes = candles(c_low=130.0)
    found = find_setups(ts, lows, highs, closes, bar_seconds=900, pivot_width=1)
    assert found == []


def test_rejects_setup_when_reward_risk_is_not_met():
    ts, lows, highs, closes = candles()
    found = find_setups(ts, lows, highs, closes, bar_seconds=900,
                        pivot_width=1, stop_buffer_pct=15.0)
    assert found == []
