import csv

import numpy as np

from short_setup_scanner import _fmt_duration, _read_setups_csv, _ticker_from_table, find_setups


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
