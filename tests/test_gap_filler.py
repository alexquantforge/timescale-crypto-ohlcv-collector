"""
Unit tests for historical candle gap detection.
"""
import numpy as np


def test_contiguous_gap_range_grouping():
    existing_days = [100, 101, 102, 105, 106, 110]
    days_arr = np.array(existing_days, dtype=np.int64)
    full_range = np.arange(days_arr[0], days_arr[-1] + 1, dtype=np.int64)
    missing = np.setdiff1d(full_range, days_arr, assume_unique=True)

    # Missing days should be 103, 104, 107, 108, 109
    assert list(missing) == [103, 104, 107, 108, 109]
