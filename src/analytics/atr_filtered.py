"""
Module for calculating ATR without paranormal bars (Filtered Robust ATR).

Standard ATR (Average True Range) is significantly distorted by 'paranormal bars' —
abnormally large candles caused by news spikes, liquidity squeezes, or bad ticks,
as well as abnormally small candles (dojis/flats).

To determine the true average daily volatility of an asset, this algorithm filters
out paranormal bars falling outside the threshold window [small_threshold * ATR, large_threshold * ATR],
and iteratively recalculates the robust volatility value.
"""
from typing import Sequence, Union
import numpy as np


def _true_range(
    H: np.ndarray, L: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """Vectorized True Range for aligned High/Low/Close arrays."""
    prev_c = np.roll(C, 1)
    prev_c[0] = C[0]
    tr = np.maximum(H - L, np.maximum(np.abs(H - prev_c), np.abs(L - prev_c)))
    return np.where(np.isfinite(tr), tr, 0.0)


def _robust_atr_from_tr(
    window_tr: np.ndarray,
    small_threshold: float = 0.5,
    large_threshold: float = 1.8,
    max_iterations: int = 10,
) -> float:
    """
    Iteratively filters paranormal/tiny bars from a True-Range window
    and returns the converged robust ATR. Returns 0.0 for empty windows.
    """
    if len(window_tr) == 0:
        return 0.0

    # Initial volatility estimate using median (resistant to extreme outliers)
    current_atr = float(np.median(window_tr))
    if not np.isfinite(current_atr) or current_atr <= 0:
        current_atr = float(np.mean(window_tr))
    if not np.isfinite(current_atr) or current_atr <= 0:
        return 0.0

    # Iterative filtering of paranormal and tiny bars
    for _ in range(max_iterations):
        valid_bars = window_tr[
            (window_tr >= small_threshold * current_atr) &
            (window_tr <= large_threshold * current_atr)
        ]
        if len(valid_bars) == 0:
            break

        new_atr = float(np.mean(valid_bars))
        if not np.isfinite(new_atr) or new_atr <= 0:
            break

        # Convergence check: exit if ATR changed by less than 1%
        if abs(new_atr - current_atr) / max(abs(current_atr), 1e-12) < 0.01:
            current_atr = new_atr
            break

        current_atr = new_atr

    return max(float(current_atr), 0.0)


def compute_atr_no_paranormal_bars(
    highs: Sequence[Union[int, float]],
    lows: Sequence[Union[int, float]],
    closes: Sequence[Union[int, float]],
    period: int = 5,
    small_threshold: float = 0.5,
    large_threshold: float = 1.8,
    max_iterations: int = 10,
) -> float:
    """
    Calculates ATR (Average True Range) without paranormal bars.

    :param highs: Array of High prices
    :param lows: Array of Low prices
    :param closes: Array of Close prices
    :param period: Rolling window period for ATR calculation (default: 5)
    :param small_threshold: Cutoff factor for abnormally small bars (0.5 = less than 50% of ATR)
    :param large_threshold: Cutoff factor for paranormal bars (1.8 = greater than 180% of ATR)
    :param max_iterations: Maximum number of filtering iterations
    :return: Robust ATR without paranormal bars for the latest bar (float)
    """
    try:
        H = np.asarray(highs, dtype=float)
        L = np.asarray(lows, dtype=float)
        C = np.asarray(closes, dtype=float)
    except Exception:
        return 0.0

    n = len(C)
    if n < 3:
        return 0.0

    # Calculate True Range (vectorized, identical to the original definition)
    tr = _true_range(H, L, C)

    # Extract window for requested period
    window_tr = tr[max(0, n - period):n]
    if len(window_tr) == 0:
        return 0.0

    return _robust_atr_from_tr(
        window_tr,
        small_threshold=small_threshold,
        large_threshold=large_threshold,
        max_iterations=max_iterations,
    )


def compute_rolling_atr_no_paranormal_bars(
    highs: Sequence[Union[int, float]],
    lows: Sequence[Union[int, float]],
    closes: Sequence[Union[int, float]],
    period: int = 5,
    small_threshold: float = 0.5,
    large_threshold: float = 1.8,
    max_iterations: int = 10,
) -> np.ndarray:
    """
    Rolling per-bar ATR without paranormal bars.

    Returns an array aligned with the input bars where element ``i`` equals
    ``compute_atr_no_paranormal_bars(highs[:i+1], lows[:i+1], closes[:i+1], period)``
    — i.e. the robust ATR as it would have been known at bar ``i``.

    This is the vectorized-time equivalent used by the dashboard to draw
    historical ATR channels: True Range is computed once for the whole series
    (O(n)) and the iterative filter runs only on the trailing ``period`` bars
    of each position, instead of re-scanning the full prefix per bar (O(n^2)).

    :param highs: Array of High prices
    :param lows: Array of Low prices
    :param closes: Array of Close prices
    :param period: Rolling window period for ATR calculation (default: 5)
    :param small_threshold: Cutoff factor for abnormally small bars (0.5 = less than 50% of ATR)
    :param large_threshold: Cutoff factor for paranormal bars (1.8 = greater than 180% of ATR)
    :param max_iterations: Maximum number of filtering iterations
    :return: numpy array of robust ATR values (0.0 where not enough history)
    """
    try:
        H = np.asarray(highs, dtype=float)
        L = np.asarray(lows, dtype=float)
        C = np.asarray(closes, dtype=float)
    except Exception:
        return np.zeros(0, dtype=float)

    n = len(C)
    result = np.zeros(n, dtype=float)
    if n < 3:
        return result

    tr = _true_range(H, L, C)

    for i in range(2, n):
        window_tr = tr[max(0, i + 1 - period):i + 1]
        result[i] = _robust_atr_from_tr(
            window_tr,
            small_threshold=small_threshold,
            large_threshold=large_threshold,
            max_iterations=max_iterations,
        )

    return result
