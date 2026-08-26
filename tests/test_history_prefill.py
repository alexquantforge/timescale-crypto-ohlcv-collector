"""
Unit tests for the backward history-prefill helpers (src/core/history_prefill).

These encode the exact stop/pagination rules that repair truncated table
starts: progress is judged by timestamps, never by page size — the old
`len(page) == limit` check silently broke on exchanges with a smaller kline
cap and left perp tables with only the latest few days.
"""
from src.core.history_prefill import (
    extract_older_rows,
    prefill_needed,
    prefill_page_since_ms,
)

STEP = 900
LIMIT = 1000
NOW = 1_786_100_000  # arbitrary "now"


def test_prefill_needed_only_when_start_above_floor():
    floor = NOW - 180 * 86400
    assert prefill_needed(NOW - 10 * 86400, floor, slack_sec=1800) is True
    assert prefill_needed(floor + 100, floor, slack_sec=1800) is False  # within slack
    assert prefill_needed(floor, floor, slack_sec=1800) is False
    assert prefill_needed(None, floor, slack_sec=1800) is False
    assert prefill_needed(0, floor, slack_sec=1800) is False


def test_prefill_page_since_returns_full_page_below_oldest():
    oldest = NOW - 5 * 86400
    floor = NOW - 180 * 86400
    since = prefill_page_since_ms(oldest, STEP, LIMIT, None, floor)
    assert since == (oldest - LIMIT * STEP) * 1000


def test_prefill_page_since_clamps_at_target_floor():
    oldest = NOW - 5 * 86400
    floor = oldest - 100  # floor only 100s below the start
    since = prefill_page_since_ms(oldest, STEP, LIMIT, None, floor)
    assert since == floor * 1000
    # and when oldest == floor there is nothing left to fetch
    assert prefill_page_since_ms(floor, STEP, LIMIT, None, floor) is None


def test_prefill_page_since_none_when_exchange_window_reached():
    oldest = NOW - 5 * 86400
    # Gate.io-style window floor NEWER than the table start: older candles are
    # permanently out of reach -> stop (None), don't retry forever.
    floor_ms = oldest * 1000 + 1
    assert prefill_page_since_ms(oldest, STEP, LIMIT, floor_ms, 0) is None
    # window floor below the start: clamped to it
    win_ms = (oldest - 1000) * 1000
    assert prefill_page_since_ms(oldest, STEP, LIMIT, win_ms, 0) == win_ms


def test_extract_older_rows_filters_dedups_sorts():
    oldest = 10_000
    floor = 1_000
    r = lambda ts: [ts * 1000, 1, 2, 0.5, 1.5, 10]
    batch = [
        r(9_000), r(11_000),        # one older, one newer than table start
        r(9_000),                   # duplicate of the older row
        r(500),                     # below the retention floor -> dropped
        r(8_500),
    ]
    out = extract_older_rows(batch, oldest, floor)
    assert [int(c[0]) // 1000 for c in out] == [8_500, 9_000]  # ascending, deduped


def test_extract_older_rows_empty_means_no_progress():
    oldest = 10_000
    # exchange ignoring `since` returns its LATEST page -> nothing strictly older
    batch = [[t * 1000, 1, 1, 1, 1, 1] for t in range(20_000, 20_100)]
    assert extract_older_rows(batch, oldest, 0) == []
    assert extract_older_rows([], oldest, 0) == []
    assert extract_older_rows(None, oldest, 0) == []
