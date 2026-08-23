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


def _mk_repo(buckets):
    class FakeRepo:
        def __init__(self):
            self.buckets = buckets
            self.inserted_df = None
        async def get_stored_days(self, symbol, ccxt_id, timeframe="1d"):
            return list(self.buckets)
        async def upsert_ohlcv_batch(self, df, timeframe="1d"):
            self.inserted_df = df
            return len(df)
    return FakeRepo()


class FakeExchange:
    def __init__(self, candles):
        self.candles = candles  # [ts_ms, o, h, l, c, v]
    async def fetch_ohlcv(self, symbol, timeframe="1d", since=None, limit=None):
        out = [c for c in self.candles if c[0] >= since]
        return out[:limit] if limit else out


def test_fill_history_gaps_daily_fills_missing_days():
    import asyncio
    from src.exchanges.gap_filler import fill_history_gaps

    step_ms = 86400_000
    # stored days 0,1,2 and 5,6 -> missing 3,4
    candles = [[i * step_ms, 1.0, 2.0, 0.5, 1.5, 10.0] for i in range(7)]
    repo = _mk_repo([0, 1, 2, 5, 6])
    ex = FakeExchange(candles)

    inserted = asyncio.run(
        fill_history_gaps(ex, "BTC/USDT:USDT", "bybit", repo, bf_limit=100, max_pages=5, timeframe="1d")
    )
    assert inserted == 2
    df = repo.inserted_df
    assert len(df) == 2
    assert sorted(df["ts"] // step_ms) == [3, 4]
    assert "url_trading" in df.columns  # repository maps these to url_of_*


def test_fill_history_gaps_15m_uses_900s_buckets():
    import asyncio
    from src.exchanges.gap_filler import fill_history_gaps

    step_ms = 900_000
    # stored 15m buckets 100..103 and 106 -> missing 104,105
    candles = [[i * step_ms, 1.0, 2.0, 0.5, 1.5, 3.0] for i in range(100, 107)]
    repo = _mk_repo([100, 101, 102, 103, 106])
    ex = FakeExchange(candles)

    inserted = asyncio.run(
        fill_history_gaps(ex, "BTC/USDT:USDT", "bybit", repo, bf_limit=100, max_pages=5, timeframe="15m")
    )
    assert inserted == 2
    assert sorted(repo.inserted_df["ts"] // step_ms) == [104, 105]


def test_fill_history_gaps_no_gaps_noop():
    import asyncio
    from src.exchanges.gap_filler import fill_history_gaps

    repo = _mk_repo([1, 2, 3, 4])
    ex = FakeExchange([])
    inserted = asyncio.run(
        fill_history_gaps(ex, "BTC/USDT:USDT", "bybit", repo, timeframe="1d")
    )
    assert inserted == 0 and repo.inserted_df is None
