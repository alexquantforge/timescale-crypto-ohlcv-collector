"""
Unit tests for ATR без паранормальных баров (Filtered Robust ATR).
"""
import pytest
from src.analytics.atr_filtered import compute_atr_no_paranormal_bars


def test_compute_atr_no_paranormal_bars_normal_series():
    # Synthetic normal bars with constant range ~10
    highs = [105, 106, 107, 108, 109, 110]
    lows = [95, 96, 97, 98, 99, 100]
    closes = [100, 101, 102, 103, 104, 105]

    atr = compute_atr_no_paranormal_bars(highs, lows, closes, period=5)
    assert atr > 0.0
    assert abs(atr - 10.0) < 1.0


def test_compute_atr_filters_paranormal_spike_bars():
    # Normal bars (range 10) + 1 giant paranormal spike bar at index 4 (High 200, Low 100 = range 100)
    highs = [105, 106, 107, 108, 200, 110]
    lows = [95, 96, 97, 98, 100, 100]
    closes = [100, 101, 102, 103, 104, 105]

    filtered_atr = compute_atr_no_paranormal_bars(
        highs, lows, closes, period=5, small_threshold=0.5, large_threshold=1.8
    )

    # Filtered ATR should ignore the spike bar (range 100) and stay around ~10, NOT ~28
    assert filtered_atr < 20.0
    assert filtered_atr > 5.0


def test_compute_atr_insufficient_data():
    assert compute_atr_no_paranormal_bars([100], [90], [95]) == 0.0
