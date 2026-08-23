"""
Gate.io rejects kline queries whose `from` is older than ~10000 recent points
("Candlestick too long ago. Maximum 10000 points recently are allowed").
The 15m engine must clamp its fetch cursor into that window instead of
failing the pair on every cycle (the root cause of pairs stuck at the
"initial fetch" phase with zero candles ever stored).
"""
import time


def test_gate_since_clamped_into_window():
    from src.core.updater_15m import clamp_ohlcv_since_ms, EXCHANGE_MAX_LOOKBACK_CANDLES_15M

    now_ms = int(time.time() * 1000)
    far_back = now_ms - 200 * 86400 * 1000  # engine asks for 180 days of 15m history

    for name in ("gateio", "gate"):
        out = clamp_ohlcv_since_ms(name, far_back)
        floor = now_ms - EXCHANGE_MAX_LOOKBACK_CANDLES_15M[name] * 900 * 1000
        assert abs(out - floor) < 60_000, f"{name}: clamp {out} vs floor {floor}"

    # inside the window -> untouched
    recent = now_ms - 1000 * 900 * 1000
    assert clamp_ohlcv_since_ms("gateio", recent) == recent


def test_other_exchanges_untouched():
    from src.core.updater_15m import clamp_ohlcv_since_ms

    now_ms = int(time.time() * 1000)
    far_back = now_ms - 400 * 86400 * 1000
    for name in ("bybit", "okx", "mexc", "bingx", "bitget", "htx", "coinex"):
        assert clamp_ohlcv_since_ms(name, far_back) == far_back


def test_floor_helpers():
    from src.core.updater_15m import ohlcv_since_floor_ms

    assert ohlcv_since_floor_ms("bybit") is None
    floor = ohlcv_since_floor_ms("gateio")
    assert floor is not None
    now_ms = int(time.time() * 1000)
    # ~9900 15m candles ≈ 103 days back, 2-hour tolerance for test run time
    assert abs((now_ms - floor) - 9900 * 900 * 1000) < 2 * 3600 * 1000
