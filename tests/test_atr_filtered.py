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


def test_rolling_atr_matches_prefix_recomputation():
    """
    compute_rolling_atr_no_paranormal_bars[i] must equal calling the point
    function on the prefix highs[:i+1] — the O(n) path must not change results.
    """
    import numpy as np
    from src.analytics.atr_filtered import compute_rolling_atr_no_paranormal_bars

    rng = np.random.default_rng(42)
    n = 60
    closes = 100 + np.cumsum(rng.normal(0, 0.8, n))
    highs = closes + np.abs(rng.normal(0, 1.2, n))
    lows = closes - np.abs(rng.normal(0, 1.2, n))

    rolling = compute_rolling_atr_no_paranormal_bars(highs, lows, closes, period=5)
    assert len(rolling) == n
    # Not enough history -> zeros
    assert rolling[0] == 0.0 and rolling[1] == 0.0

    for i in range(2, n):
        expected = compute_atr_no_paranormal_bars(highs[: i + 1], lows[: i + 1], closes[: i + 1], period=5)
        assert abs(rolling[i] - expected) < 1e-12


def test_rolling_atr_with_paranormal_spike_stays_robust():
    import numpy as np
    from src.analytics.atr_filtered import compute_rolling_atr_no_paranormal_bars

    n = 30
    closes = np.full(n, 100.0)
    highs = closes + 1.0
    lows = closes - 1.0
    # One giant paranormal spike bar
    highs[15] = 180.0
    lows[15] = 95.0

    rolling = compute_rolling_atr_no_paranormal_bars(highs, lows, closes, period=5)
    # Robust ATR on the window containing the spike must stay near ~2, not ~18
    assert rolling[-1] < 5.0
