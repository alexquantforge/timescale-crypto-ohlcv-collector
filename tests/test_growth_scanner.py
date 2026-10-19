from __future__ import annotations

from growth_scanner import (
    BAR_1D_SEC,
    DEFAULT_MAX_SPREAD_ATR_PCT,
    DEFAULT_MIN_7D_VOLUME_USD,
    build_parser,
    calculate_daily_metrics,
    compute_daily_gerchik_atr,
    find_max_low_to_high_move,
    parse_pair_table,
)


def test_parse_pair_tables_for_spot_and_swap():
    spot = parse_pair_table("btc_usdt_on_bybit", "ohlcv_15m")
    swap = parse_pair_table("btc_usdt:usdt_on_gateio", "ohlcv_15m")

    assert spot is not None
    assert (spot.symbol, spot.exchange, spot.ccxt_id, spot.market_type) == (
        "BTC/USDT", "bybit", "bybit", "spot"
    )
    assert swap is not None
    assert (swap.symbol, swap.exchange, swap.ccxt_id, swap.market_type) == (
        "BTC/USDT:USDT", "gateio", "gate", "swap"
    )
    assert parse_pair_table("not_a_pair_table", "ohlcv_15m") is None


def test_growth_is_the_strongest_low_to_later_high_move():
    move = find_max_low_to_high_move([
        (1_000, 100.0, 101.0),
        (1_900, 110.0, 120.0),
        (2_800, 90.0, 95.0),
        (3_700, 93.0, 110.0),
    ])

    assert move is not None
    assert move.start_ts == 2_800
    assert move.peak_ts == 3_700
    assert move.low_price == 90.0
    assert move.high_price == 110.0
    assert round(move.pct, 6) == round((110.0 / 90.0 - 1.0) * 100.0, 6)


def test_growth_does_not_use_a_future_low_for_an_earlier_high():
    move = find_max_low_to_high_move([
        (1_000, 100.0, 120.0),
        (1_900, 80.0, 81.0),
    ])

    assert move is not None
    assert move.start_ts == 1_000
    assert move.peak_ts == 1_000
    assert round(move.pct, 6) == 20.0


def test_same_15m_bar_can_supply_its_low_to_high_range():
    move = find_max_low_to_high_move([(1_000, 100.0, 125.0)])

    assert move is not None
    assert move.start_ts == move.peak_ts == 1_000
    assert move.pct == 25.0


def _daily_row(day: int, *, volume: float | None = None):
    close = 100.0 + day
    return {
        "Timestamp": day * BAR_1D_SEC,
        "high": close + 5.0,
        "low": close - 5.0,
        "close": close,
        "volume": float(day + 1 if volume is None else volume),
    }


def test_daily_metrics_use_seven_closed_bars_and_exclude_current_day():
    now_ts = 9 * BAR_1D_SEC + 6 * 3600
    # Days 2..8 are the most recent seven closed candles; day 9 is forming.
    rows = [_daily_row(day) for day in range(2, 9)]
    rows.append(_daily_row(9, volume=1_000_000))

    metrics = calculate_daily_metrics(rows, now_ts=now_ts, atr_period=5)

    assert metrics is not None
    assert metrics["min_7d_volume_usd"] == 3.0 * 97.0
    assert metrics["daily_gerchik_atr"] > 0


def test_daily_metrics_require_seven_full_daily_candles():
    now_ts = 9 * BAR_1D_SEC + 6 * 3600
    rows = [_daily_row(day) for day in range(3, 9)]

    assert calculate_daily_metrics(rows, now_ts=now_ts, atr_period=5) is None


def test_daily_gerchik_atr_requires_previous_close_for_full_period():
    highs = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    lows = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

    assert compute_daily_gerchik_atr(highs, lows, closes, period=5) > 0
    assert compute_daily_gerchik_atr(highs[-5:], lows[-5:], closes[-5:], period=5) == 0


def test_cli_defaults_and_volume_threshold_override():
    parser = build_parser()
    defaults = parser.parse_args(["--hours", "6", "--pct", "25"])
    custom = parser.parse_args([
        "--hours", "12", "--pct", "40", "--min-7d-volume-usd", "250000"
    ])

    assert defaults.min_7d_volume_usd == DEFAULT_MIN_7D_VOLUME_USD
    assert defaults.max_spread_atr_pct == DEFAULT_MAX_SPREAD_ATR_PCT
    assert custom.min_7d_volume_usd == 250_000
