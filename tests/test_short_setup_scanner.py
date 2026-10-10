import asyncio
import csv
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from short_setup_scanner import (
    _IDENT, _args, _classify_post_entry_resolution, _filter_reentries_after_resolution,
    _fmt_duration,
    _matches_requested_symbol, _print_symbol_scan_diagnostics,
    _print_watch_filter_diagnostics, _read_setups_csv,
    _screen_current_signal, _ticker_from_table, _watch, find_setups, main,
)


def test_formats_eta_duration():
    assert _fmt_duration(65) == "1м 05с"
    assert _fmt_duration(3661) == "1ч 01м"


def test_classifies_stop_as_first_resolution():
    result = _classify_post_entry_resolution(
        np.array([900, 1800, 2700]),
        np.array([111.0, 108.0, 109.0]),
        np.array([95.0, 89.0, 95.0]),
        stop=110.0, target=90.0,
    )
    assert result["invalidated"] is True
    assert result["invalidation_reason"] == "stop_first"
    assert result["invalidation_ts"] == 900
    assert result["stop_hit_ts"] == 900
    assert result["target_hit_ts"] == 1800


def test_classifies_target_as_first_resolution():
    result = _classify_post_entry_resolution(
        np.array([900, 1800]),
        np.array([108.0, 111.0]),
        np.array([89.0, 95.0]),
        stop=110.0, target=90.0,
    )
    assert result["invalidation_reason"] == "target_first"
    assert result["invalidation_ts"] == 900
    assert result["target_hit_ts"] == 900
    assert result["stop_hit_ts"] == 1800


def test_marks_same_candle_stop_and_target_touch_as_ambiguous():
    result = _classify_post_entry_resolution(
        np.array([900]), np.array([111.0]), np.array([89.0]),
        stop=110.0, target=90.0,
    )
    assert result["invalidation_reason"] == "both_same_candle"
    assert result["invalidation_ts"] == 900


def test_leaves_setup_unresolved_when_neither_level_was_touched():
    result = _classify_post_entry_resolution(
        np.array([900, 1800]), np.array([108.0, 109.0]), np.array([95.0, 94.0]),
        stop=110.0, target=90.0,
    )
    assert result["invalidated"] is False
    assert result["invalidation_reason"] is None


def test_converts_database_table_name_to_exchange_symbol():
    assert _ticker_from_table("btc_usdt:usdt_on_bybit") == "BTC/USDT:USDT"


def test_symbol_filter_matches_base_ccxt_and_compact_pair():
    table = "jct_usdt:usdt_on_bybit"
    assert _matches_requested_symbol(table, {"JCT"})
    assert _matches_requested_symbol(table, {"JCT/USDT:USDT"})
    assert _matches_requested_symbol(table, {"JCTUSDT"})
    assert _matches_requested_symbol(table, set())
    assert not _matches_requested_symbol(table, {"BTC"})


def test_scanner_cli_accepts_targeted_retrace_and_one_shot_watch(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["short_setup_scanner.py", "--watch", "--watch-once", "--symbols", "JCT",
         "--exchanges", "bybit", "--max-retrace", "40"],
    )
    args = _args()
    assert args.watch is True
    assert args.watch_once is True
    assert args.symbols == "JCT"
    assert args.max_retrace == 40.0
    assert args.rr is None  # RR has no default threshold.


def test_legacy_rr_flag_is_ignored_and_warns(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["short_setup_scanner.py", "--rr", "1.0", "--watch-once"])
    with pytest.raises(SystemExit, match="--watch-once"):
        main()
    warning = capsys.readouterr().err
    assert "--rr" in warning
    assert "игнорируется" in warning


def test_watch_once_runs_exactly_one_cycle(monkeypatch):
    scan_calls = 0

    async def fake_scan(_args):
        nonlocal scan_calls
        scan_calls += 1
        return []

    async def fake_snapshots(*_args):
        return {}

    async def fake_save(*_args, **_kwargs):
        return 1

    monkeypatch.setattr("short_setup_scanner._scan", fake_scan)
    monkeypatch.setattr("short_setup_scanner._collect_watch_snapshots", fake_snapshots)
    monkeypatch.setattr("short_setup_scanner._save_setups_to_db", fake_save)
    args = SimpleNamespace(
        no_save_to_db=False, import_csv="", interval_minutes=60.0,
        timeframe="15m", watch_atr_period=5, watch_once=True,
        watch_show_invalidated_details=False,
    )
    asyncio.run(_watch(args))
    assert scan_calls == 1


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
    # L=100 -> A=151 (>50%); C=136 retraces less than 30% of the impulse;
    # B=147 is a lower high. The close at index 11 confirms a drop.
    highs = np.array([110, 112, 120, 140, 151, 143, 140, 141, 142, 143, 147, 146, 145], dtype=float)
    lows = np.array([108, 100, 110, 130, 140, 137, c_low, 137, 139, 140, 143, 145.5, 144], dtype=float)
    closes = np.array([109, 110, 115, 135, 145, 140, 138, 139, 140, 142, 145, 145.8, 144.5], dtype=float)
    ts = np.arange(len(highs), dtype=np.int64) * 900
    return ts, lows, highs, closes


def test_watch_finds_fresh_b_after_previous_attempt_hits_stop():
    # One A/C formation produces B1=145. Its confirmed entry is stopped by
    # the next candle's 146 high. That same candle can become a fresh lower
    # high B2=146; after pivot confirmation, a new entry at index 8 is valid.
    ts = np.arange(11, dtype=np.int64) * 900
    highs = np.array([101, 160, 135, 145, 141, 142, 146, 140, 139, 138, 137], dtype=float)
    lows = np.array([100, 150, 120, 130, 132, 128, 125, 130, 128, 126, 125], dtype=float)
    closes = np.array([100, 155, 130, 143, 138, 140, 134, 138, 137, 135, 133], dtype=float)

    attempts = find_setups(
        ts, lows, highs, closes, bar_seconds=900, pump_pct=50,
        pump_days=1, max_retrace=70, pivot_width=1, setup_days=10,
        reentry_since_ts=int(ts[6]),
    )
    historical_attempts = find_setups(
        ts, lows, highs, closes, bar_seconds=900, pump_pct=50,
        pump_days=1, max_retrace=70, pivot_width=1, setup_days=10,
    )

    assert len(attempts) == 2
    assert len(historical_attempts) == 1  # ordinary scans retain their prior behavior
    assert [event["b_price"] for event in attempts] == [145, 146]
    assert attempts[0]["stop"] == pytest.approx(145 * 1.005)
    assert attempts[1]["stop"] == pytest.approx(146 * 1.005)
    for event in attempts:
        entry_i = int(np.searchsorted(ts, event["entry_ts"], side="left"))
        event.update(_classify_post_entry_resolution(
            ts[entry_i + 1:], highs[entry_i + 1:], lows[entry_i + 1:],
            stop=event["stop"], target=event["target"],
        ))
        event.update({"database": "test_db", "table": "lumia_usdt_on_bybit"})

    assert attempts[0]["invalidation_reason"] == "stop_first"
    assert attempts[0]["invalidation_ts"] == ts[6]
    assert attempts[1]["invalidated"] is False
    assert attempts[1]["b_ts"] == ts[6]
    assert attempts[1]["entry_ts"] == ts[8]

    retained = _filter_reentries_after_resolution(attempts)
    assert retained == attempts  # retain the stopped attempt and its fresh successor


def test_watch_does_not_accept_reentry_formed_before_prior_stop():
    attempts = [
        {"database": "db", "table": "lumia", "a_ts": 1, "b_ts": 2,
         "entry_ts": 3, "invalidated": True, "invalidation_reason": "stop_first",
         "invalidation_ts": 10},
        {"database": "db", "table": "lumia", "a_ts": 1, "b_ts": 9,
         "entry_ts": 11, "invalidated": False},
        {"database": "db", "table": "lumia", "a_ts": 1, "b_ts": 10,
         "entry_ts": 12, "invalidated": False},
        {"database": "db", "table": "lumia", "a_ts": 2, "b_ts": 2,
         "entry_ts": 20, "invalidated": True, "invalidation_reason": "target_first",
         "invalidation_ts": 10},
        {"database": "db", "table": "lumia", "a_ts": 2, "b_ts": 10,
         "entry_ts": 21, "invalidated": False},
    ]

    retained = _filter_reentries_after_resolution(attempts)

    assert [(event["a_ts"], event["b_ts"]) for event in retained] == [
        (1, 2), (1, 10), (2, 2),
    ]


def test_finds_lower_high_setup_and_reports_reward_risk():
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


def test_pump_must_be_strictly_greater_than_50_percent():
    ts, lows, highs, closes = candles()
    highs[4] = 150.0  # Exactly +50% from L=100 is not a qualifying pump.
    found = find_setups(ts, lows, highs, closes, bar_seconds=900, pivot_width=1)
    assert found == []


def test_rejects_lower_local_a_if_prior_high_in_pump_leg_is_higher():
    ts, lows, highs, closes = candles()
    highs[3] = 155.0  # Above the proposed A at index 4 (150), between L and A.
    found = find_setups(ts, lows, highs, closes, bar_seconds=900, pivot_width=1)
    assert found == []


def test_rejects_setup_if_any_high_between_a_and_b_touches_or_crosses_a():
    ts, lows, highs, closes = candles()
    highs[8] = highs[4]  # B is still below A, but an intermediate wick retests A.
    found = find_setups(ts, lows, highs, closes, bar_seconds=900, pivot_width=1)
    assert found == []


def test_rejects_a_to_c_retracement_over_30_percent_of_impulse():
    ts, lows, highs, closes = candles(c_low=130.0)
    found = find_setups(ts, lows, highs, closes, bar_seconds=900, pivot_width=1)
    assert found == []


def test_symbol_diagnostics_explain_retrace_rejection(capsys):
    ts, lows, highs, closes = candles(c_low=130.0)
    diagnostics = {}
    found = find_setups(
        ts, lows, highs, closes, bar_seconds=900, pump_pct=50,
        pump_days=1, max_retrace=40, pivot_width=1, diagnostics=diagnostics,
        diagnostic_since_ts=0,
    )
    assert found == []
    assert diagnostics["pump_passes_in_window"] == 1
    assert diagnostics["rejection_counts"]["a_to_c_filter_failed"] == 1
    record = diagnostics["records"][0]
    assert record["pump_retrace_pct"] > 40
    assert record["failed_checks"] == ["max_retrace"]

    _print_symbol_scan_diagnostics(
        "ohlcv_15m", "test_usdt_on_bybit", diagnostics,
        SimpleNamespace(
            max_retrace=40, pump_pct=50, days=1, min_ac_drop=5,
            lower_high_pct=0, min_cb_bounce=2, confirm_drop=1, stop_buffer=0.5,
            setup_days=10, pivot_bars=1,
        ),
        found,
    )
    output = capsys.readouterr().out
    assert "L→A = 51.00%" in output
    assert "A→C откат от L→A импульса = 41.18%" in output
    assert "Первый блокер" in output
    assert "max_retrace" in output


def test_keeps_setup_when_reward_risk_is_below_two_and_reports_it():
    ts, lows, highs, closes = candles()
    found = find_setups(
        ts, lows, highs, closes, bar_seconds=900,
        pivot_width=1, stop_buffer_pct=15.0,
    )
    assert len(found) == 1
    assert 0 < found[0]["rr"] < 2
    assert found[0]["stop"] > found[0]["b_price"]


def test_perpetual_table_names_with_colon_are_scannable():
    assert _IDENT.fullmatch("btc_usdt:usdt_on_bybit")


def test_targeted_watch_prints_live_filter_values_and_thresholds(capsys):
    args = SimpleNamespace(
        watch_min_tape=3.0, watch_min_depth_usd=1000.0,
        watch_max_spread_atr_pct=15.0, watch_min_7d_volume_usd=100_000.0,
    )
    event = {
        "base": "LUMIA", "exchange": "bybit", "market": "swap",
        "c_price": 0.104, "b_price": 0.116,
    }
    snapshot = {
        "price": 0.112, "price_source": "LIVE", "trades_per_min": 2.0,
        "is_barcode": False, "depth_usd": 500.0,
        "spread_atr_pct": 20.0, "min_7d_volume_usd": 50_000.0,
    }
    _print_watch_filter_diagnostics(event, snapshot, args, "dead/insufficient tape")
    output = capsys.readouterr().out
    assert "Tape=2/мин (нужно ≥3/мин)" in output
    assert "Depth ±1%=$500" in output
    assert "Spread/1D ATR=20%" in output
    assert "Минимальный 7d $volume=$50,000" in output
    assert output.count("НЕ ПРОЙДЕНО") == 4


def test_current_watch_reports_why_a_pattern_was_already_invalidated():
    args = SimpleNamespace(
        stop_buffer=0.5, watch_min_tape=3.0,
        watch_min_depth_usd=1000.0, watch_max_spread_atr_pct=15.0,
        watch_min_7d_volume_usd=100_000.0,
    )
    event = {"invalidated": True, "invalidation_reason": "stop_first"}
    signal, reason = _screen_current_signal(event, {}, args, now_ts=123)
    assert signal is None
    assert reason == "already invalidated: stop сработал раньше цели"


def test_current_watch_keeps_live_setup_inside_c_b_with_good_health():
    args = SimpleNamespace(
        stop_buffer=0.5, watch_min_tape=3.0,
        watch_min_depth_usd=1000.0, watch_max_spread_atr_pct=15.0,
        watch_min_7d_volume_usd=100_000.0,
    )
    event = {
        "invalidated": False, "a_price": 150.0, "c_price": 136.0,
        "b_price": 147.0, "entry": 143.0, "rr": 2.0,
    }
    snapshot = {
        "price": 137.0, "price_source": "LIVE", "trades_per_min": 4.0,
        "is_barcode": False, "depth_usd": 50_000.0,
        "spread_atr_pct": 8.0, "min_7d_volume_usd": 250_000.0,
    }
    signal, reason = _screen_current_signal(event, snapshot, args, now_ts=123)
    assert reason is None
    assert signal["status"] == "CURRENT"
    assert signal["entry"] == 137.0
    assert signal["entry_ts"] == 123
    assert signal["rr"] < 2.0  # Current RR is informational, not a watch filter.


def test_current_watch_keeps_inclusive_c_boundary_without_live_rr_filter():
    args = SimpleNamespace(
        stop_buffer=0.5, watch_min_tape=3.0,
        watch_min_depth_usd=1000.0, watch_max_spread_atr_pct=15.0,
        watch_min_7d_volume_usd=100_000.0,
    )
    event = {"invalidated": False, "a_price": 150.0, "c_price": 136.0, "b_price": 147.0}
    snapshot = {
        "price": 136.0, "price_source": "LIVE", "trades_per_min": 4.0,
        "is_barcode": False, "depth_usd": 50_000.0,
        "spread_atr_pct": 8.0, "min_7d_volume_usd": 250_000.0,
    }
    signal, reason = _screen_current_signal(event, snapshot, args, now_ts=123)
    assert reason is None
    assert signal["entry"] == event["c_price"]
    assert signal["rr"] == 0.0


def test_current_watch_excludes_dead_tape_or_price_outside_c_b():
    args = SimpleNamespace(
        stop_buffer=0.5, watch_min_tape=3.0,
        watch_min_depth_usd=1000.0, watch_max_spread_atr_pct=15.0,
        watch_min_7d_volume_usd=100_000.0,
    )
    event = {"invalidated": False, "a_price": 150.0, "c_price": 136.0, "b_price": 147.0}
    healthy = {
        "price": 146.0, "price_source": "LIVE", "trades_per_min": 4.0,
        "is_barcode": False, "depth_usd": 50_000.0,
        "spread_atr_pct": 8.0, "min_7d_volume_usd": 250_000.0,
    }
    dead = dict(healthy, trades_per_min=0.0)
    assert _screen_current_signal(event, dead, args, now_ts=123)[0] is None
    assert _screen_current_signal(event, dict(healthy, price=150.0), args, now_ts=123)[0] is None


def test_current_watch_applies_each_red_health_threshold():
    args = SimpleNamespace(
        stop_buffer=0.5, watch_min_tape=3.0,
        watch_min_depth_usd=1000.0, watch_max_spread_atr_pct=15.0,
        watch_min_7d_volume_usd=100_000.0,
    )
    event = {"invalidated": False, "a_price": 150.0, "c_price": 136.0, "b_price": 147.0}
    healthy = {
        "price": 146.0, "price_source": "LIVE", "trades_per_min": 4.0,
        "is_barcode": False, "depth_usd": 50_000.0,
        "spread_atr_pct": 8.0, "min_7d_volume_usd": 250_000.0,
    }
    red_snapshots = (
        dict(healthy, is_barcode=True),
        dict(healthy, depth_usd=1000.0),
        dict(healthy, spread_atr_pct=15.0),
        dict(healthy, min_7d_volume_usd=100_000.0),
    )
    for snapshot in red_snapshots:
        assert _screen_current_signal(event, snapshot, args, now_ts=123)[0] is None
