#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find historical and currently forming lower-high short setups in OHLCV tables.

Pattern: a >=N% pump within M days, A-to-C correction no deeper than a given
fraction of the preceding pump leg, a lower rebound high B, then a confirmed
move down from B. The proposed target is C and the stop is just above B; only
setups with the requested reward/risk ratio are reported.

This is a screening tool, not an execution engine or investment advice. It
uses stored candles only and does not place orders.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import math
import time
import datetime as dt
import json
import os
import re
import sys
import time
from typing import Any

import asyncpg
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import settings  # noqa: E402
from src.analytics.atr_filtered import compute_atr_no_paranormal_bars  # noqa: E402

_IDENT = re.compile(r"^[a-zA-Z0-9_:]+$")
RESULTS_DB = "pump_scanner_results"


def _db_names(timeframe: str) -> list[str]:
    if timeframe == "1d":
        return [settings.db_high_1d, settings.db_low_1d]
    return [settings.db_high_15m, settings.db_low_15m]


def _ticker_from_table(table: str) -> str:
    raw = str(table).lower().rsplit("_on_", 1)[0]
    return raw.replace("_", "/").upper()


def _parse_table(table: str) -> tuple[str, str, str]:
    name = table.lower()
    raw, _, exchange = name.rpartition("_on_")
    market_type = "swap" if ":" in raw else "spot"
    ticker = raw.replace("_", "/").upper()
    base = ticker.split("/", 1)[0].split(":", 1)[0]
    return base, exchange, market_type


def _matches_requested_symbol(table: str, requested_symbols: set[str]) -> bool:
    """Match a table by base (JCT), CCXT symbol, or compact pair (JCTUSDT)."""
    requested_symbols = {
        symbol.strip().upper() for symbol in requested_symbols if symbol.strip()
    }
    if not requested_symbols:
        return True
    base, _, _ = _parse_table(table)
    ticker = _ticker_from_table(table)
    spot_symbol = ticker.split(":", 1)[0]
    compact_symbols = {
        symbol.replace("/", "").replace(":", "")
        for symbol in requested_symbols
    }
    table_compact = {
        ticker.replace("/", "").replace(":", ""),
        spot_symbol.replace("/", "").replace(":", ""),
    }
    return (
        base.upper() in requested_symbols
        or ticker.upper() in requested_symbols
        or spot_symbol.upper() in requested_symbols
        or bool(table_compact & compact_symbols)
    )


def _pivots(values: np.ndarray, width: int, high: bool) -> np.ndarray:
    """Return strict local extrema, confirmed only after `width` later bars."""
    n = len(values)
    if n < 2 * width + 1:
        return np.array([], dtype=int)
    out = []
    for i in range(width, n - width):
        window = values[i - width:i + width + 1]
        # Choose the last equal extreme, avoiding a string of flat candles
        # becoming many pivots. Requiring the center to be the last extreme
        # also gives deterministic handling of equal highs/lows.
        extreme = np.max(window) if high else np.min(window)
        if values[i] == extreme and not np.any(window[width + 1:] == extreme):
            out.append(i)
    return np.asarray(out, dtype=int)


def find_setups(
    ts: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    closes: np.ndarray,
    *,
    bar_seconds: int,
    pump_pct: float = 50.0,
    pump_days: float = 3.0,
    max_retrace: float = 30.0,
    min_ac_drop_pct: float = 5.0,
    min_cb_bounce_pct: float = 2.0,
    lower_high_pct: float = 0.0,
    confirm_drop_pct: float = 1.0,
    stop_buffer_pct: float = 0.5,
    min_rr: float = 2.0,
    pivot_width: int = 2,
    setup_days: float = 10.0,
    diagnostics: dict[str, Any] | None = None,
    diagnostic_since_ts: int | None = None,
) -> list[dict[str, Any]]:
    """Detect setups and optionally explain recent candidate rejections.

    Timestamps are seconds. ``max_retrace`` is the fraction of the preceding
    L-to-A pump leg allowed to be given back at C: L=100, A=150, max=30 means
    C must be at least 135. Diagnostics do not change the detector's decisions.
    """
    n = min(len(ts), len(lows), len(highs), len(closes))
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "bar_count": n,
            "first_ts": int(ts[0]) if n else None,
            "last_ts": int(ts[n - 1]) if n else None,
            "window_start_ts": diagnostic_since_ts,
            "high_pivots_total": 0,
            "low_pivots_total": 0,
            "latest_high_pivot_ts": None,
            "latest_high_pivot_price": None,
            "a_pivots_in_window": 0,
            "pump_passes_in_window": 0,
            "rejection_counts": {},
            "records": [],
            "records_truncated": False,
            "early_reason": None,
            "setups_found": 0,
        })
    if n < max(2 * pivot_width + 3, 4):
        if diagnostics is not None:
            diagnostics["early_reason"] = (
                f"Недостаточно свечей: {n}, требуется не менее "
                f"{max(2 * pivot_width + 3, 4)} для pivot-bars={pivot_width}."
            )
        return []
    ts, lows, highs, closes = ts[:n], lows[:n], highs[:n], closes[:n]
    if not (np.all(np.isfinite(lows)) and np.all(np.isfinite(highs))
            and np.all(np.isfinite(closes))):
        valid = np.isfinite(lows) & np.isfinite(highs) & np.isfinite(closes)
        ts, lows, highs, closes = ts[valid], lows[valid], highs[valid], closes[valid]
        n = len(ts)
        if diagnostics is not None:
            diagnostics.update({
                "bar_count": n,
                "first_ts": int(ts[0]) if n else None,
                "last_ts": int(ts[n - 1]) if n else None,
            })

    high_pivots = _pivots(highs, pivot_width, True)
    low_pivots = _pivots(lows, pivot_width, False)
    if diagnostics is not None:
        diagnostics["high_pivots_total"] = len(high_pivots)
        diagnostics["low_pivots_total"] = len(low_pivots)
        if len(high_pivots):
            latest_high_i = int(high_pivots[-1])
            diagnostics["latest_high_pivot_ts"] = int(ts[latest_high_i])
            diagnostics["latest_high_pivot_price"] = float(highs[latest_high_i])
        diagnostics["a_pivots_in_window"] = sum(
            diagnostic_since_ts is None or ts[int(i)] >= diagnostic_since_ts
            for i in high_pivots
        )
    if not len(high_pivots) or not len(low_pivots):
        if diagnostics is not None:
            missing = []
            if not len(high_pivots):
                missing.append("локальных максимумов A")
            if not len(low_pivots):
                missing.append("локальных минимумов C")
            diagnostics["early_reason"] = (
                "Не найдено подтверждённых pivot-точек: " + " и ".join(missing)
                + f" (pivot-bars={pivot_width}; экстремуму нужны свечи с обеих сторон)."
            )
        return []

    def record_rejection(
        ai: int,
        reason: str,
        stage_rank: int,
        checks: list[str],
        *,
        li: int | None = None,
        ci: int | None = None,
        bi: int | None = None,
        ei: int | None = None,
        **extra: Any,
    ) -> None:
        if diagnostics is None:
            return
        if diagnostic_since_ts is not None and ts[int(ai)] < diagnostic_since_ts:
            return
        item: dict[str, Any] = {
            "reason": reason,
            "stage_rank": stage_rank,
            "a_ts": int(ts[int(ai)]),
            "a_price": float(highs[int(ai)]),
            "checks": list(checks),
        }
        if li is not None:
            item.update({"l_ts": int(ts[int(li)]), "l_price": float(lows[int(li)])})
        if ci is not None:
            item.update({"c_ts": int(ts[int(ci)]), "c_price": float(lows[int(ci)])})
        if bi is not None:
            item.update({"b_ts": int(ts[int(bi)]), "b_price": float(highs[int(bi)])})
        if ei is not None:
            item.update({"entry_ts": int(ts[int(ei)]), "entry": float(closes[int(ei)])})
        item.update(extra)
        counts = diagnostics["rejection_counts"]
        counts[reason] = counts.get(reason, 0) + 1
        records = diagnostics["records"]
        if len(records) == 500:
            records.pop(0)
            diagnostics["records_truncated"] = True
        records.append(item)

    pump_bars = max(1, int(pump_days * 86400 / bar_seconds))
    setup_bars = max(3, int(setup_days * 86400 / bar_seconds))
    result: list[dict[str, Any]] = []
    used_entry_indices: set[int] = set()
    low_pivot_set = set(map(int, low_pivots))

    for ai_value in high_pivots:
        ai = int(ai_value)

        # L is the lowest observed price in the allowed pump lookback.
        left = max(0, ai - pump_bars)
        if left >= ai:
            if diagnostics is not None:
                record_rejection(
                    ai, "insufficient_pre_a_history", 0,
                    [f"Для A={highs[ai]:.8g} недостаточно свечей слева, чтобы выбрать L."],
                )
            continue
        li = left + int(np.argmin(lows[left:ai]))
        leg = highs[ai] - lows[li]
        if np.max(highs[li:ai + 1]) > highs[ai]:
            if diagnostics is not None:
                prior_i = li + int(np.argmax(highs[li:ai + 1]))
                record_rejection(
                    ai, "higher_wick_before_a", 0,
                    [f"L={lows[li]:.8g} → A={highs[ai]:.8g}; до A был high "
                     f"{highs[prior_i]:.8g} в {_fmt_time(int(ts[prior_i]))}, выше выбранного A."],
                    li=li, prior_high=float(highs[prior_i]), prior_high_ts=int(ts[prior_i]),
                )
            continue
        if lows[li] <= 0 or leg <= 0:
            if diagnostics is not None:
                record_rejection(
                    ai, "non_positive_pump_leg", 0,
                    [f"Некорректная импульсная нога: L={lows[li]:.8g}, A={highs[ai]:.8g}."],
                    li=li,
                )
            continue
        actual_pump_pct = leg / lows[li] * 100.0
        if actual_pump_pct <= pump_pct:
            if diagnostics is not None:
                record_rejection(
                    ai, "pump_below_strict_threshold", 0,
                    [f"L→A = {actual_pump_pct:.2f}%: требуется строго больше {pump_pct:g}% — НЕ ПРОЙДЕНО."],
                    li=li, pump_pct=actual_pump_pct,
                )
            continue
        pump_elapsed = int(ts[ai] - ts[li])
        if pump_elapsed > pump_days * 86400:
            if diagnostics is not None:
                record_rejection(
                    ai, "pump_took_too_long", 0,
                    [f"L→A = {actual_pump_pct:.2f}% — проходит по росту, но занял "
                     f"{pump_elapsed / 3600:.2f} ч; лимит {pump_days:g} дн — НЕ ПРОЙДЕНО."],
                    li=li, pump_pct=actual_pump_pct, pump_elapsed_seconds=pump_elapsed,
                )
            continue
        if diagnostics is not None and (
            diagnostic_since_ts is None or ts[ai] >= diagnostic_since_ts
        ):
            diagnostics["pump_passes_in_window"] += 1
        pump_checks = []
        if diagnostics is not None:
            pump_checks = [
                f"L→A = {actual_pump_pct:.2f}% (нужно строго >{pump_pct:g}%) — ПРОЙДЕНО.",
                f"Длительность L→A = {pump_elapsed / 3600:.2f} ч "
                f"(лимит {pump_days:g} дн) — ПРОЙДЕНО.",
            ]

        # Walk B pivots chronologically, maintaining the lowest low since A.
        # That running minimum is C; it invalidates an earlier trough if price
        # makes a new low before B. This is linear in the number of bars, rather
        # than testing every possible C/B pair.
        b_candidates = high_pivots[(high_pivots > ai) & (high_pivots <= ai + setup_bars)]
        if not len(b_candidates):
            if diagnostics is not None:
                record_rejection(
                    ai, "no_confirmed_b_pivot", 1,
                    pump_checks + [
                        f"В следующие ≤{setup_days:g} дн нет подтверждённого локального high B "
                        f"после A (pivot-bars={pivot_width})."
                    ],
                    li=li, pump_pct=actual_pump_pct,
                )
            continue
        running_low_i = ai + 1
        ci = running_low_i
        high_crossed_a = False
        retest_i: int | None = None
        found_for_a = False
        for bi_value in b_candidates:
            bi = int(bi_value)
            while running_low_i <= bi:
                if highs[running_low_i] >= highs[ai]:
                    high_crossed_a = True
                    if retest_i is None:
                        retest_i = running_low_i
                if lows[running_low_i] < lows[ci]:
                    ci = running_low_i
                running_low_i += 1
            # A is invalidated by ANY later wick touching/crossing A before B,
            # even when the eventual rebound pivot B itself is a lower high.
            if high_crossed_a:
                if diagnostics is not None:
                    checks = pump_checks + [
                        f"После A high={highs[retest_i]:.8g} в {_fmt_time(int(ts[retest_i]))} "
                        f"достиг/пересёк A={highs[ai]:.8g}: нижний пик A нарушен — НЕ ПРОЙДЕНО."
                    ]
                    record_rejection(
                        ai, "a_retested_after_peak", 1, checks,
                        li=li, ci=ci, bi=bi, pump_pct=actual_pump_pct,
                        retest_ts=int(ts[retest_i]), retest_high=float(highs[retest_i]),
                    )
                break
            if ci not in low_pivot_set:
                if diagnostics is not None:
                    tentative = ""
                    if highs[ai] > lows[ci] and leg > 0:
                        tentative_retrace = (highs[ai] - lows[ci]) / leg * 100.0
                        tentative = (
                            f"; предварительный A→C откат={tentative_retrace:.2f}% импульса "
                            f"(C пока не подтверждён)"
                        )
                    record_rejection(
                        ai, "c_not_confirmed_local_low", 1,
                        pump_checks + [
                            f"Минимум после A до B: C?={lows[ci]:.8g} "
                            f"({_fmt_time(int(ts[ci]))}), но он не подтверждён как локальный low "
                            f"при pivot-bars={pivot_width}{tentative}."
                        ],
                        li=li, ci=ci, bi=bi, pump_pct=actual_pump_pct,
                    )
                continue
            if ts[bi] - ts[ai] > setup_days * 86400:
                if diagnostics is not None:
                    record_rejection(
                        ai, "a_to_b_window_exceeded", 1,
                        pump_checks + [
                            f"A→B занял {(ts[bi] - ts[ai]) / 86400:.2f} дн; "
                            f"лимит {setup_days:g} дн — НЕ ПРОЙДЕНО."
                        ],
                        li=li, ci=ci, bi=bi, pump_pct=actual_pump_pct,
                    )
                continue
            if highs[ai] <= lows[ci] or leg <= 0:
                if diagnostics is not None:
                    record_rejection(
                        ai, "no_a_to_c_drop", 1,
                        pump_checks + [
                            f"C={lows[ci]:.8g} не ниже A={highs[ai]:.8g}; "
                            "коррекция A→C отсутствует."
                        ],
                        li=li, ci=ci, bi=bi, pump_pct=actual_pump_pct,
                    )
                continue
            drop_pct = (highs[ai] - lows[ci]) / highs[ai] * 100.0
            retrace_pct = (highs[ai] - lows[ci]) / leg * 100.0
            ac_checks = []
            if diagnostics is not None:
                ac_checks = pump_checks + [
                    f"A→C падение от цены A = {drop_pct:.2f}% "
                    f"(минимум {min_ac_drop_pct:g}%) — "
                    f"{'ПРОЙДЕНО' if drop_pct >= min_ac_drop_pct else 'НЕ ПРОЙДЕНО'}.",
                    f"A→C откат от L→A импульса = {retrace_pct:.2f}% "
                    f"((A−C)/(A−L)); лимит ≤{max_retrace:g}% — "
                    f"{'ПРОЙДЕНО' if retrace_pct <= max_retrace else 'НЕ ПРОЙДЕНО'}.",
                ]
            ac_failed = []
            if drop_pct < min_ac_drop_pct:
                ac_failed.append("min_ac_drop")
            if retrace_pct > max_retrace:
                ac_failed.append("max_retrace")
            if ac_failed:
                if diagnostics is not None:
                    record_rejection(
                        ai, "a_to_c_filter_failed", 2, ac_checks,
                        li=li, ci=ci, bi=bi, pump_pct=actual_pump_pct,
                        ac_drop_pct=drop_pct, pump_retrace_pct=retrace_pct,
                        failed_checks=ac_failed,
                    )
                continue

            required_b_high = highs[ai] * (1.0 - lower_high_pct / 100.0)
            if highs[bi] >= required_b_high:
                if diagnostics is not None:
                    record_rejection(
                        ai, "b_not_lower_high", 3,
                        ac_checks + [
                            f"B={highs[bi]:.8g}; требуется B < {required_b_high:.8g} "
                            f"(A={highs[ai]:.8g}, lower-high ≥{lower_high_pct:g}%) — НЕ ПРОЙДЕНО."
                        ],
                        li=li, ci=ci, bi=bi, pump_pct=actual_pump_pct,
                        ac_drop_pct=drop_pct, pump_retrace_pct=retrace_pct,
                    )
                continue
            bounce_pct = (highs[bi] - lows[ci]) / lows[ci] * 100.0
            b_checks = []
            if diagnostics is not None:
                b_checks = ac_checks + [
                    f"B={highs[bi]:.8g} < A={highs[ai]:.8g} — ПРОЙДЕНО.",
                    f"C→B отскок = {bounce_pct:.2f}% (минимум {min_cb_bounce_pct:g}%) — "
                    f"{'ПРОЙДЕНО' if bounce_pct >= min_cb_bounce_pct else 'НЕ ПРОЙДЕНО'}.",
                ]
            if bounce_pct < min_cb_bounce_pct:
                if diagnostics is not None:
                    record_rejection(
                        ai, "c_to_b_bounce_too_small", 3, b_checks,
                        li=li, ci=ci, bi=bi, pump_pct=actual_pump_pct,
                        ac_drop_pct=drop_pct, pump_retrace_pct=retrace_pct,
                        cb_bounce_pct=bounce_pct,
                    )
                continue

            stop = highs[bi] * (1.0 + stop_buffer_pct / 100.0)
            # The first confirmed close-down after B is the reproducible
            # historical entry proxy. Ignore entries after target C traded.
            end = min(n, bi + setup_bars + 1)
            confirm_start = bi + pivot_width + 1
            confirm_threshold = highs[bi] * (1.0 - confirm_drop_pct / 100.0)
            confirmed_entry = False
            for ei in range(confirm_start, end):
                entry = closes[ei]
                if entry <= confirm_threshold:
                    confirmed_entry = True
                    entry_checks = []
                    if diagnostics is not None:
                        entry_checks = b_checks + [
                            f"Подтверждение: close={entry:.8g} ≤ {confirm_threshold:.8g} "
                            f"(B−{confirm_drop_pct:g}%) — ПРОЙДЕНО."
                        ]
                    target_touches = np.flatnonzero(lows[bi + 1:ei + 1] <= lows[ci])
                    if len(target_touches) or entry <= lows[ci]:
                        if diagnostics is not None:
                            target_i = bi + 1 + int(target_touches[0]) if len(target_touches) else ei
                            record_rejection(
                                ai, "target_touched_before_entry", 4,
                                entry_checks + [
                                    f"C={lows[ci]:.8g} был затронут до/на свече entry "
                                    f"({_fmt_time(int(ts[target_i]))}); сигнал после entry не подтверждается."
                                ],
                                li=li, ci=ci, bi=bi, ei=ei, pump_pct=actual_pump_pct,
                                ac_drop_pct=drop_pct, pump_retrace_pct=retrace_pct,
                                cb_bounce_pct=bounce_pct,
                                target_touch_ts=int(ts[target_i]),
                            )
                        break
                    risk = stop - entry
                    reward = entry - lows[ci]
                    if risk <= 0 or reward <= 0:
                        if diagnostics is not None:
                            record_rejection(
                                ai, "non_positive_risk_or_reward", 4,
                                entry_checks + [
                                    f"Risk=stop-entry={risk:.8g}, reward=entry-C={reward:.8g}; "
                                    "оба должны быть положительными — НЕ ПРОЙДЕНО."
                                ],
                                li=li, ci=ci, bi=bi, ei=ei, pump_pct=actual_pump_pct,
                                ac_drop_pct=drop_pct, pump_retrace_pct=retrace_pct,
                                cb_bounce_pct=bounce_pct,
                            )
                        break
                    rr = reward / risk
                    if rr < min_rr:
                        if diagnostics is not None:
                            record_rejection(
                                ai, "original_rr_below_minimum", 5,
                                entry_checks + [
                                    f"Исходный RR={rr:.2f}:1; требуется ≥{min_rr:g}:1 — НЕ ПРОЙДЕНО."
                                ],
                                li=li, ci=ci, bi=bi, ei=ei, pump_pct=actual_pump_pct,
                                ac_drop_pct=drop_pct, pump_retrace_pct=retrace_pct,
                                cb_bounce_pct=bounce_pct, rr=rr,
                            )
                        break
                    if ei in used_entry_indices:
                        if diagnostics is not None:
                            record_rejection(
                                ai, "entry_already_used", 5,
                                entry_checks + [
                                    f"Свеча entry {_fmt_time(int(ts[ei]))} уже использована другим A-кандидатом."
                                ],
                                li=li, ci=ci, bi=bi, ei=ei, pump_pct=actual_pump_pct,
                                ac_drop_pct=drop_pct, pump_retrace_pct=retrace_pct,
                                cb_bounce_pct=bounce_pct, rr=rr,
                            )
                        break
                    used_entry_indices.add(ei)
                    result.append({
                        "start_ts": int(ts[li]), "a_ts": int(ts[ai]),
                        "c_ts": int(ts[ci]), "b_ts": int(ts[bi]),
                        "entry_ts": int(ts[ei]), "pump_start": float(lows[li]),
                        "a_price": float(highs[ai]), "c_price": float(lows[ci]),
                        "b_price": float(highs[bi]), "entry": float(entry),
                        "stop": float(stop), "target": float(lows[ci]),
                        "pump_pct": (highs[ai] / lows[li] - 1) * 100,
                        "ac_drop_pct": drop_pct,
                        "pump_retrace_pct": retrace_pct,
                        "cb_bounce_pct": bounce_pct,
                        "rr": rr,
                    })
                    found_for_a = True
                    break
            if not confirmed_entry and diagnostics is not None:
                available_start = min(confirm_start, n)
                available_end = min(end, n)
                if available_end > available_start:
                    window_closes = closes[available_start:available_end]
                    min_close_offset = int(np.argmin(window_closes))
                    min_close_i = available_start + min_close_offset
                    close_detail = (
                        f"минимальный close после подтверждения pivot B="
                        f"{closes[min_close_i]:.8g} в {_fmt_time(int(ts[min_close_i]))}"
                    )
                else:
                    min_close_i = None
                    close_detail = "после B пока недостаточно свечей для проверки close"
                record_rejection(
                    ai, "no_confirmed_drop_close", 4,
                    b_checks + [
                        f"Нужен close ≤{confirm_threshold:.8g} (B−{confirm_drop_pct:g}%) "
                        f"после {pivot_width} свечей подтверждения B; {close_detail}."
                    ],
                    li=li, ci=ci, bi=bi, pump_pct=actual_pump_pct,
                    ac_drop_pct=drop_pct, pump_retrace_pct=retrace_pct,
                    cb_bounce_pct=bounce_pct,
                    lowest_post_b_close_i=min_close_i,
                )
            if found_for_a:
                break

    result.sort(key=lambda r: (r["entry_ts"], r["a_ts"]))
    if diagnostics is not None:
        diagnostics["setups_found"] = len(result)
    return result


_DIAGNOSTIC_REASON_LABELS = {
    "insufficient_pre_a_history": "недостаточно свечей до A для выбора L",
    "higher_wick_before_a": "между L и A был high выше выбранного A",
    "non_positive_pump_leg": "некорректная или неположительная нога L→A",
    "pump_below_strict_threshold": "рост L→A не превысил порог",
    "pump_took_too_long": "нога L→A длилась дольше заданного срока",
    "no_confirmed_b_pivot": "после A нет подтверждённого локального high B",
    "a_retested_after_peak": "после A был high, достигший или превысивший A",
    "c_not_confirmed_local_low": "минимум C пока не подтверждён как локальный low",
    "a_to_b_window_exceeded": "A→B вышло за лимит setup-days",
    "no_a_to_c_drop": "нет снижения от A к C",
    "a_to_c_filter_failed": "не пройден фильтр A→C (min-drop и/или max-retrace)",
    "b_not_lower_high": "B не является достаточно низким lower high",
    "c_to_b_bounce_too_small": "отскок C→B меньше заданного минимума",
    "target_touched_before_entry": "цель C достигнута до/на подтверждении entry",
    "non_positive_risk_or_reward": "риск или потенциальная награда неположительны",
    "original_rr_below_minimum": "исходный RR ниже порога --rr",
    "entry_already_used": "свеча entry уже использована другим A-кандидатом",
    "no_confirmed_drop_close": "нет подтверждённого close ниже B на заданный процент",
}


def _print_symbol_scan_diagnostics(
    database: str,
    table: str,
    diagnostics: dict[str, Any],
    args: argparse.Namespace,
    found: list[dict[str, Any]],
) -> None:
    """Print the closest recent pattern attempts and the first rule that rejected each."""
    base, exchange, market = _parse_table(table)
    print(
        f"\nДиагностика {base}/{exchange} {market} [{database}.{table}]:",
        flush=True,
    )
    first_ts, last_ts = diagnostics.get("first_ts"), diagnostics.get("last_ts")
    window_start = diagnostics.get("window_start_ts")
    if last_ts is not None:
        window_label = (
            f"с {_fmt_time(int(window_start))}"
            if window_start is not None else "за всю историю"
        )
        print(
            f"  OHLC-свечей: {diagnostics.get('bar_count', 0)}; данные "
            f"{_fmt_time(int(first_ts)) if first_ts is not None else '?'} — "
            f"{_fmt_time(int(last_ts))}; подробный разбор A-кандидатов {window_label}.",
            flush=True,
        )
    if diagnostics.get("early_reason"):
        print(f"  Разбор остановлен: {diagnostics['early_reason']}", flush=True)
        return

    print(
        f"  Подтверждённых локальных high/low: "
        f"{diagnostics.get('high_pivots_total', 0)}/"
        f"{diagnostics.get('low_pivots_total', 0)}; high-кандидатов A в окне: "
        f"{diagnostics.get('a_pivots_in_window', 0)}; L→A прошло все pump-проверки: "
        f"{diagnostics.get('pump_passes_in_window', 0)}; полных паттернов: {len(found)}.",
        flush=True,
    )
    print(
        f"  Настройки: pump >{args.pump_pct:g}% за ≤{args.days:g} дн; "
        f"A→C retrace ≤{args.max_retrace:g}% от ноги L→A; "
        f"падение A→C ≥{args.min_ac_drop:g}% от A; B ниже A минимум на "
        f"{args.lower_high_pct:g}%; отскок C→B ≥{args.min_cb_bounce:g}%; "
        f"подтверждение close ниже B на {args.confirm_drop:g}%; "
        f"stop=B+{args.stop_buffer:g}%; исходный RR ≥{args.rr:g}:1; "
        f"B ищется до {args.setup_days:g} дн от A, "
        f"entry-close — до {args.setup_days:g} дн после B; "
        f"pivot-bars={max(1, args.pivot_bars)}.",
        flush=True,
    )
    rejection_counts = diagnostics.get("rejection_counts", {})
    if rejection_counts:
        summary = "; ".join(
            f"{_DIAGNOSTIC_REASON_LABELS.get(reason, reason)}: {count}"
            for reason, count in sorted(
                rejection_counts.items(), key=lambda item: (-item[1], item[0])
            )
        )
        print(f"  Причины отсева в окне: {summary}.", flush=True)

    records = diagnostics.get("records", [])
    grouped: dict[tuple[int, float], dict[str, Any]] = {}
    for record in records:
        key = (int(record["a_ts"]), float(record["a_price"]))
        prior = grouped.get(key)
        record_rank = (
            int(record.get("stage_rank", 0)), int(record.get("b_ts", -1) or -1)
        )
        prior_rank = (
            int(prior.get("stage_rank", 0)), int(prior.get("b_ts", -1) or -1)
        ) if prior else (-1, -1)
        if prior is None or record_rank > prior_rank:
            grouped[key] = record

    attempts = list(grouped.values())
    pump_passed = [
        record for record in attempts
        if int(record.get("stage_rank", 0)) >= 1 and record.get("l_price") is not None
    ]
    selected = pump_passed or attempts
    selected.sort(
        key=lambda record: (
            int(record.get("a_ts", 0)),
            int(record.get("stage_rank", 0)),
        ),
        reverse=True,
    )
    if selected:
        print("  Ближайшие варианты (до 5 последних A, для каждого показан самый дальний этап проверки):", flush=True)
        for number, record in enumerate(selected[:5], start=1):
            a_stamp = _fmt_time(int(record["a_ts"]))
            print(
                f"    Вариант {number}: A={record['a_price']:.8g} ({a_stamp})",
                flush=True,
            )
            if record.get("l_price") is not None:
                print(
                    f"      L={record['l_price']:.8g} "
                    f"({_fmt_time(int(record['l_ts']))}); "
                    f"C={record['c_price']:.8g} "
                    f"({_fmt_time(int(record['c_ts']))})" if record.get("c_price") is not None
                    else f"      L={record['l_price']:.8g} ({_fmt_time(int(record['l_ts']))})",
                    flush=True,
                )
            if record.get("b_price") is not None:
                print(
                    f"      B={record['b_price']:.8g} "
                    f"({_fmt_time(int(record['b_ts']))})",
                    flush=True,
                )
            for check in record.get("checks", []):
                print(f"      {check}", flush=True)
            if record.get("reason"):
                label = _DIAGNOSTIC_REASON_LABELS.get(
                    record["reason"], record["reason"]
                )
                print(f"      Первый блокер по этой ветке: {label}.", flush=True)
                if int(record.get("stage_rank", 0)) < 4:
                    print(
                        "      Следующие этапы для этой ветки не проверялись, "
                        "так как она уже не прошла указанное условие.",
                        flush=True,
                    )
            if record.get("failed_checks"):
                print(
                    "      Непройденные проверки: "
                    + ", ".join(record["failed_checks"]) + ".",
                    flush=True,
                )
    elif not found:
        if diagnostics.get("a_pivots_in_window", 0) == 0:
            latest_a_ts = diagnostics.get("latest_high_pivot_ts")
            if latest_a_ts is not None:
                print(
                    f"  В окне нет подтверждённого A-пика; последний найденный "
                    f"high-pivot был {diagnostics['latest_high_pivot_price']:.8g} "
                    f"({_fmt_time(int(latest_a_ts))}), вне окна подробного разбора.",
                    flush=True,
                )
            else:
                print("  В данных нет подтверждённых локальных high-пивотов A.", flush=True)
        else:
            print(
                "  Для последних A-кандидатов дальнейшие ветки не прошли "
                "проверки, перечисленные выше.",
                flush=True,
            )

    for event in sorted(found, key=lambda item: item["entry_ts"], reverse=True)[:5]:
        print(
            f"  Паттерн ПРОШЁЛ структурные проверки: "
            f"L={event['pump_start']:.8g} → A={event['a_price']:.8g} → "
            f"C={event['c_price']:.8g} → B={event['b_price']:.8g}; "
            f"pump={event['pump_pct']:.2f}%, "
            f"A→C retrace={event['pump_retrace_pct']:.2f}% "
            f"(лимит {args.max_retrace:g}%), исходный RR={event['rr']:.2f}:1.",
            flush=True,
        )
    if diagnostics.get("records_truncated"):
        print(
            "  Примечание: список отсеянных вариантов ограничен 500 записями; "
            "сводные счётчики учитывают все варианты в окне.",
            flush=True,
        )


_INVALIDATION_REASON_LABELS = {
    "stop_first": "stop сработал раньше цели",
    "target_first": "цель C достигнута раньше stop",
    "both_same_candle": "stop и C задеты в одной свече; порядок внутри неё неизвестен",
}


def _classify_post_entry_resolution(
    timestamps: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    *,
    stop: float,
    target: float,
) -> dict[str, Any]:
    """Explain whether a setup's stop or target was first touched after entry.

    With OHLC candles, if both thresholds occur in the same candle their
    intrabar order cannot be recovered; that case is reported as ambiguous.
    """
    timestamps = np.asarray(timestamps)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    n = min(len(timestamps), len(highs), len(lows))
    timestamps, highs, lows = timestamps[:n], highs[:n], lows[:n]

    stop_hits = np.flatnonzero(np.isfinite(highs) & (highs >= stop))
    target_hits = np.flatnonzero(np.isfinite(lows) & (lows <= target))
    first_stop = int(stop_hits[0]) if len(stop_hits) else None
    first_target = int(target_hits[0]) if len(target_hits) else None

    if first_stop is None and first_target is None:
        reason = None
        resolution_index = None
    elif first_target is None or (first_stop is not None and first_stop < first_target):
        reason = "stop_first"
        resolution_index = first_stop
    elif first_stop is None or first_target < first_stop:
        reason = "target_first"
        resolution_index = first_target
    else:
        reason = "both_same_candle"
        resolution_index = first_stop

    return {
        "invalidated": reason is not None,
        "invalidation_reason": reason,
        "invalidation_ts": (
            int(timestamps[resolution_index]) if resolution_index is not None else None
        ),
        "invalidation_high": (
            float(highs[resolution_index]) if resolution_index is not None else None
        ),
        "invalidation_low": (
            float(lows[resolution_index]) if resolution_index is not None else None
        ),
        "stop_hit_ts": int(timestamps[first_stop]) if first_stop is not None else None,
        "target_hit_ts": int(timestamps[first_target]) if first_target is not None else None,
    }


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}ч {minutes:02d}м"
    return f"{minutes}м {secs:02d}с"


def _fmt_time(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ищет памп → ограниченный откат A-C → lower high B → подтверждённый шорт-сетап.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--timeframe", choices=("15m", "1d"), default="15m")
    p.add_argument("--pump-pct", type=float, default=50.0,
                   help="Требуемый строгий минимум роста L→A, %% (по умолчанию >50%%)")
    p.add_argument("--days", type=float, default=3.0, help="Максимальная длительность L→A, дни")
    p.add_argument("--max-retrace", type=float, default=30.0,
                   help="Максимальный откат A→C как %% от роста L→A")
    p.add_argument("--min-ac-drop", type=float, default=5.0,
                   help="Минимальное падение A→C от цены A, %% (отсечь мелкий шум)")
    p.add_argument("--min-cb-bounce", type=float, default=2.0,
                   help="Минимальный отскок C→B, %%")
    p.add_argument("--lower-high-pct", type=float, default=0.0,
                   help="На сколько %% B должен быть ниже A; 0 = любое lower high")
    p.add_argument("--confirm-drop", type=float, default=1.0,
                   help="Подтвердить разворот закрытием на N%% ниже B")
    p.add_argument("--stop-buffer", type=float, default=0.5,
                   help="Буфер стопа выше B, %%")
    p.add_argument("--rr", type=float, default=2.0, help="Минимум reward/risk до цели C")
    p.add_argument("--pivot-bars", type=int, default=2,
                   help="Свечей слева/справа для подтверждения локального пика/дна")
    p.add_argument("--setup-days", type=float, default=10.0,
                   help="Сколько максимум дней от A до сигнала")
    p.add_argument("--active-hours", type=float, default=6.0,
                   help="Считать сигнал текущим, если он моложе N часов")
    p.add_argument("--exchanges", default="", help="Список бирж через запятую, пусто = все")
    p.add_argument("--symbols", default="",
                   help="Фильтр тикеров; при указании печатать подробную диагностику отсева")
    p.add_argument("--no-spot", action="store_true", help="Не сканировать spot")
    p.add_argument("--no-swap", action="store_true", help="Не сканировать perpetual swaps")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit-tables", type=int, default=0, help="Для теста ограничить число таблиц; 0 = все")
    p.add_argument("--out", default="short_setups.csv", help="Дополнительный CSV-экспорт результатов")
    p.add_argument("--no-save-to-db", action="store_true",
                   help="Не сохранять запуск и сетапы в pump_scanner_results")
    p.add_argument("--import-csv", default="",
                   help="Импортировать уже готовый CSV в базу без повторного сканирования")
    p.add_argument("--watch", action="store_true",
                   help="Непрерывно искать только текущие сетапы и печатать их каждый цикл")
    p.add_argument("--watch-once", action="store_true",
                   help="С --watch выполнить один проход и завершить (для быстрой проверки)")
    p.add_argument("--interval-minutes", type=float, default=60.0,
                   help="Интервал между стартами watch-проходов, минут")
    p.add_argument("--watch-atr-period", type=int, default=5,
                   help="Период 1D ATR для фильтра спреда в watch-режиме")
    p.add_argument("--watch-min-tape", type=float, default=3.0,
                   help="Минимум сделок/мин за последние 5 минут")
    p.add_argument("--watch-min-depth-usd", type=float, default=1000.0,
                   help="Минимальная глубина стакана ±1%%, USD")
    p.add_argument("--watch-max-spread-atr-pct", type=float, default=15.0,
                   help="Максимальный спред как %% 1D ATR")
    p.add_argument("--watch-min-7d-volume-usd", type=float, default=100000.0,
                   help="Минимальный min(volume × low) за 7 закрытых дневных свечей")
    p.add_argument("--watch-show-invalidated-details", action="store_true",
                   help="В watch-режиме печатать пары и первую свечу касания stop/цели для уже завершённых сигналов")
    return p.parse_args()


async def _scan(args: argparse.Namespace) -> list[dict[str, Any]]:
    dbs = _db_names(args.timeframe)
    bar_sec = 900 if args.timeframe == "15m" else 86400
    pools: dict[str, asyncpg.Pool] = {}
    requested_symbols = {
        symbol.strip().upper()
        for symbol in args.symbols.split(",")
        if symbol.strip()
    }
    sem = asyncio.Semaphore(max(1, args.concurrency))
    try:
        for db in dbs:
            pools[db] = await asyncpg.create_pool(
                user=settings.db_user, password=settings.db_password,
                host=settings.db_host, port=settings.db_port, database=db,
                min_size=1, max_size=max(1, args.concurrency),
            )

        tasks: list[tuple[str, asyncpg.Pool, str]] = []
        for db, pool in pools.items():
            async with pool.acquire() as conn:
                names = await conn.fetch(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='public' AND table_name LIKE '%_on_%' ORDER BY table_name"
                )
            for row in names:
                table = row["table_name"]
                base, exchange, kind = _parse_table(table)
                if args.exchanges and exchange not in {x.strip().lower() for x in args.exchanges.split(",")}:
                    continue
                if not _matches_requested_symbol(table, requested_symbols):
                    continue
                if (kind == "spot" and args.no_spot) or (kind == "swap" and args.no_swap):
                    continue
                if not _IDENT.fullmatch(table):
                    continue
                tasks.append((db, pool, table))
        if args.limit_tables:
            tasks = tasks[:args.limit_tables]
        total_tables = len(tasks)
        table_diagnostics: dict[tuple[str, str], dict[str, Any]] = {}
        if requested_symbols and not tasks:
            exchange_scope = args.exchanges or "все биржи"
            market_scope = "без spot" if args.no_spot else "spot и swap"
            print(
                f"Подробная диагностика: не найдено OHLCV-таблиц для "
                f"{','.join(sorted(requested_symbols))} при фильтрах "
                f"биржи={exchange_scope}, рынок={market_scope}.",
                flush=True,
            )
        symbol_scope = (
            f"; тикеры: {','.join(sorted(requested_symbols))}"
            if requested_symbols else ""
        )
        print(f"Сканирую {total_tables} таблиц, timeframe={args.timeframe}{symbol_scope}; критерии: "
              f"памп >{args.pump_pct:g}% за ≤{args.days:g} дн, A→C ≤{args.max_retrace:g}% роста, "
              f"RR ≥{args.rr:g}:1")
        scan_started = time.monotonic()

        async def one(db: str, pool: asyncpg.Pool, table: str) -> list[dict[str, Any]]:
            async with sem:
                try:
                    async with pool.acquire() as conn:
                        rows = await conn.fetch(
                            f'SELECT "Timestamp", low, high, close FROM "{table}" ORDER BY "Timestamp" ASC'
                        )
                    if len(rows) < 8:
                        if requested_symbols:
                            table_diagnostics[(db, table)] = {
                                "bar_count": len(rows),
                                "early_reason": (
                                    f"В таблице только {len(rows)} свечей; для анализа pivot "
                                    "нужно минимум 8."
                                ),
                            }
                        return []
                    arr = np.asarray([tuple(r) for r in rows], dtype=np.float64)
                    pattern_diagnostics: dict[str, Any] | None = (
                        {} if requested_symbols else None
                    )
                    diagnostic_window_days = max(
                        30.0, args.setup_days + args.days + 1.0
                    )
                    found = find_setups(
                        arr[:, 0].astype(np.int64), arr[:, 1], arr[:, 2], arr[:, 3],
                        bar_seconds=bar_sec, pump_pct=args.pump_pct, pump_days=args.days,
                        max_retrace=args.max_retrace, min_ac_drop_pct=args.min_ac_drop,
                        min_cb_bounce_pct=args.min_cb_bounce, lower_high_pct=args.lower_high_pct,
                        confirm_drop_pct=args.confirm_drop, stop_buffer_pct=args.stop_buffer,
                        min_rr=args.rr, pivot_width=max(1, args.pivot_bars), setup_days=args.setup_days,
                        diagnostics=pattern_diagnostics,
                        diagnostic_since_ts=(
                            int(arr[-1, 0] - diagnostic_window_days * 86400)
                            if pattern_diagnostics is not None else None
                        ),
                    )
                    if pattern_diagnostics is not None:
                        table_diagnostics[(db, table)] = pattern_diagnostics
                    base, exchange, kind = _parse_table(table)
                    for event in found:
                        # Save the last database close for the live watcher's
                        # documented fallback when a venue ticker is unavailable.
                        event["last_close"] = float(arr[-1, 3])
                        event["last_bar_ts"] = int(arr[-1, 0])
                        entry_i = int(np.searchsorted(arr[:, 0], event["entry_ts"], side="left"))
                        later_bars = arr[entry_i + 1:]
                        event.update(_classify_post_entry_resolution(
                            later_bars[:, 0], later_bars[:, 2], later_bars[:, 1],
                            stop=float(event["stop"]), target=float(event["target"]),
                        ))
                        event.update({"database": db, "table": table, "base": base,
                                      "exchange": exchange, "market": kind})
                    return found
                except Exception as exc:
                    if requested_symbols:
                        table_diagnostics[(db, table)] = {
                            "early_reason": f"Ошибка чтения/анализа таблицы: {type(exc).__name__}: {exc}"
                        }
                    print(f"⚠️ [{db}.{table}] {exc}", file=sys.stderr)
                    return []

        completed = 0

        async def run_task(task: tuple[str, asyncpg.Pool, str]) -> list[dict[str, Any]]:
            nonlocal completed
            found = await one(*task)
            completed += 1
            if completed % 100 == 0 or completed == total_tables:
                elapsed = time.monotonic() - scan_started
                rate = elapsed / completed if completed else 0.0
                eta = rate * (total_tables - completed)
                print(f"Прогресс: {completed}/{total_tables} таблиц | "
                      f"прошло {_fmt_duration(elapsed)} | осталось ~{_fmt_duration(eta)}")
            return found

        batches = await asyncio.gather(*(run_task(task) for task in tasks))
        results = [event for group in batches for event in group]
        now = int(dt.datetime.now(dt.timezone.utc).timestamp())
        for event in results:
            recent = 0 <= now - event["entry_ts"] <= args.active_hours * 3600
            event["status"] = ("RESOLVED" if event.get("invalidated") else
                                "CURRENT" if recent else "HISTORY")
            for key in ("start", "a", "c", "b", "entry"):
                event[f"{key}_time"] = dt.datetime.fromtimestamp(
                    event[f"{key}_ts"], dt.timezone.utc
                ).isoformat(timespec="minutes")
        if requested_symbols and tasks:
            for db, _, table in sorted(tasks, key=lambda task: (task[0], task[2])):
                table_events = [
                    event for event in results
                    if event["database"] == db and event["table"] == table
                ]
                _print_symbol_scan_diagnostics(
                    db, table, table_diagnostics.get((db, table), {}),
                    args, table_events,
                )
        return results
    finally:
        for pool in pools.values():
            await pool.close()



def _daily_db_name(db_name: str, timeframe: str) -> str | None:
    if timeframe == "1d":
        return db_name
    return {
        settings.db_high_15m: settings.db_high_1d,
        settings.db_low_15m: settings.db_low_1d,
    }.get(db_name)


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


async def _daily_health_metrics(pool: asyncpg.Pool, table: str, now_ts: int,
                                atr_period: int) -> dict[str, float | None]:
    """Read the closed daily bars used by the dashboard's spread/volume chips."""
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f'SELECT "Timestamp" AS ts, low, high, close, volume '
                f'FROM "{table}" ORDER BY "Timestamp" DESC LIMIT $1',
                max(60, atr_period + 2),
            )
    except Exception:
        return {"min_7d_volume_usd": None, "atr_1d": None}
    if not rows:
        return {"min_7d_volume_usd": None, "atr_1d": None}

    data = np.asarray([tuple(row) for row in rows], dtype=np.float64)[::-1]
    timestamps = data[:, 0]
    if np.nanmedian(timestamps) > 1e11:
        timestamps = timestamps / 1000.0
    closed = data[timestamps <= now_ts - 86400]
    min_7d_volume = None
    if len(closed):
        tail = closed[-7:]
        volumes = tail[:, 4] * tail[:, 1]
        finite_volumes = volumes[np.isfinite(volumes) & (volumes >= 0)]
        if len(finite_volumes) == len(tail):
            min_7d_volume = float(np.min(finite_volumes))

    atr = None
    if len(data) >= 3:
        atr_value = compute_atr_no_paranormal_bars(
            highs=data[:, 2], lows=data[:, 1], closes=data[:, 3],
            period=max(1, int(atr_period)),
            small_threshold=settings.atr_small_threshold,
            large_threshold=settings.atr_large_threshold,
        )
        atr = _finite_positive(atr_value)
    return {"min_7d_volume_usd": min_7d_volume, "atr_1d": atr}


async def _fetch_watch_market_snapshot(client, symbol: str, last_close: float,
                                       daily_metrics: dict) -> dict:
    """Fetch the current quote, book and tape for one candidate pair."""
    ticker_result, book_result, trades_result = await asyncio.gather(
        client.fetch_ticker(symbol),
        client.fetch_order_book(symbol, limit=50),
        client.fetch_trades(symbol, limit=200),
        return_exceptions=True,
    )
    ticker = ticker_result if isinstance(ticker_result, dict) else {}
    book = book_result if isinstance(book_result, dict) else {}
    trades = trades_result if isinstance(trades_result, list) else []

    last = _finite_positive(ticker.get("last"))
    if last is None:
        last, price_source = _finite_positive(last_close), "DB"
    else:
        price_source = "LIVE"

    bids = book.get("bids") or []
    asks = book.get("asks") or []
    bid = ask = depth = spread_atr_pct = None
    try:
        bid, ask = _finite_positive(bids[0][0]), _finite_positive(asks[0][0])
        if bid is not None and ask is not None and ask >= bid:
            mid = (bid + ask) / 2.0
            depth = sum(
                float(price) * float(amount) for price, amount in bids
                if _finite_positive(price) and _finite_positive(amount)
                and float(price) >= mid * 0.99
            ) + sum(
                float(price) * float(amount) for price, amount in asks
                if _finite_positive(price) and _finite_positive(amount)
                and float(price) <= mid * 1.01
            )
            atr = daily_metrics.get("atr_1d")
            if atr and atr > 0:
                spread_atr_pct = (ask - bid) / atr * 100.0
    except (TypeError, ValueError, IndexError):
        bid = ask = depth = spread_atr_pct = None

    now_ms = time.time() * 1000.0
    recent = []
    try:
        recent = [
            trade for trade in trades
            if trade.get("timestamp")
            and float(trade["timestamp"]) >= now_ms - 300_000
        ]
    except (TypeError, ValueError):
        recent = []
    trades_per_min = len(recent) / 5.0 if isinstance(trades_result, list) else None
    trade_prices = []
    try:
        trade_prices = [float(t["price"]) for t in trades if t.get("price")]
    except (TypeError, ValueError):
        pass
    barcode = len(trades) >= 30 and len(set(trade_prices)) <= 4

    return {
        "price": last, "price_source": price_source,
        "bid": bid, "ask": ask, "depth_usd": depth,
        "trades_per_min": trades_per_min, "is_barcode": barcode,
        "spread_atr_pct": spread_atr_pct,
        "min_7d_volume_usd": daily_metrics.get("min_7d_volume_usd"),
    }


def _screen_current_signal(event: dict, snapshot: dict, args: argparse.Namespace,
                           now_ts: int) -> tuple[dict | None, str | None]:
    """Apply current-price, RR and the dashboard's red-chip exclusion rules."""
    if event.get("invalidated"):
        reason = event.get("invalidation_reason")
        label = _INVALIDATION_REASON_LABELS.get(
            reason, f"причина не записана ({reason or 'unknown'})"
        )
        return None, f"already invalidated: {label}"
    price = _finite_positive(snapshot.get("price"))
    c_price = _finite_positive(event.get("c_price"))
    b_price = _finite_positive(event.get("b_price"))
    a_price = _finite_positive(event.get("a_price"))
    if price is None or c_price is None or b_price is None or a_price is None:
        return None, "missing price or pattern level"
    if not c_price <= price <= b_price:
        return None, "price outside C-B"

    stop = b_price * (1.0 + args.stop_buffer / 100.0)
    risk, reward = stop - price, price - c_price
    if risk <= 0:
        return None, "non-positive risk"
    # Keep the current RR as an informational value on the saved snapshot, but
    # do not invalidate a live setup when price movement has reduced it below
    # the historical signal's --rr threshold.
    current_rr = reward / risk

    tape = snapshot.get("trades_per_min")
    if tape is None or tape < args.watch_min_tape or snapshot.get("is_barcode"):
        return None, "dead/insufficient tape"
    depth = snapshot.get("depth_usd")
    if depth is None or depth <= args.watch_min_depth_usd:
        return None, "thin/missing orderbook"
    spread = snapshot.get("spread_atr_pct")
    if spread is None or spread >= args.watch_max_spread_atr_pct:
        return None, "wide/missing spread or ATR"
    min_volume = snapshot.get("min_7d_volume_usd")
    if min_volume is None or min_volume <= args.watch_min_7d_volume_usd:
        return None, "low/missing 7d dollar volume"

    signal = dict(event)
    # In watch mode the persisted entry is the actionable current quote, while
    # A/C/B and L remain the historical pattern points.
    signal.update({
        "status": "CURRENT", "invalidated": False,
        "entry_ts": int(now_ts), "entry": price,
        "stop": stop, "target": c_price, "rr": current_rr,
        "current_price_source": snapshot.get("price_source", "DB"),
        "depth_usd": depth, "trades_per_min": tape,
        "spread_atr_pct": spread, "min_7d_volume_usd": min_volume,
    })
    return signal, None


def _print_watch_filter_diagnostics(
    event: dict[str, Any], snapshot: dict[str, Any], args: argparse.Namespace,
    reason: str,
) -> None:
    """Explain every live watch check when the user explicitly targeted symbols."""
    price = _finite_positive(snapshot.get("price"))
    c_price = _finite_positive(event.get("c_price"))
    b_price = _finite_positive(event.get("b_price"))
    tape = snapshot.get("trades_per_min")
    depth = snapshot.get("depth_usd")
    spread = snapshot.get("spread_atr_pct")
    min_volume = snapshot.get("min_7d_volume_usd")
    barcode = bool(snapshot.get("is_barcode"))

    def value(number: Any, suffix: str = "", number_format: str = ".4g") -> str:
        if number is None:
            return "н/д"
        try:
            number_value = float(number)
            if not math.isfinite(number_value):
                return "н/д"
            return f"{number_value:{number_format}}{suffix}"
        except (TypeError, ValueError):
            return "н/д"

    price_ok = (
        price is not None and c_price is not None and b_price is not None
        and c_price <= price <= b_price
    )
    tape_ok = tape is not None and tape >= args.watch_min_tape and not barcode
    depth_ok = depth is not None and depth > args.watch_min_depth_usd
    spread_ok = spread is not None and spread < args.watch_max_spread_atr_pct
    volume_ok = min_volume is not None and min_volume > args.watch_min_7d_volume_usd
    print(
        f"  Подробно по live-фильтрам {event['base']}/{event['exchange']} "
        f"{event['market']}: отказ='{reason}'.",
        flush=True,
    )
    print(
        f"    Цена {value(price, number_format='.8g')} ({snapshot.get('price_source', '?')}) в "
        f"[C={value(c_price)}, B={value(b_price)}]: "
        f"{'ПРОЙДЕНО' if price_ok else 'НЕ ПРОЙДЕНО'}.",
        flush=True,
    )
    print(
        f"    Tape={value(tape, '/мин')} (нужно ≥{args.watch_min_tape:g}/мин) "
        f"; barcode={'да' if barcode else 'нет'}: "
        f"{'ПРОЙДЕНО' if tape_ok else 'НЕ ПРОЙДЕНО'}.",
        flush=True,
    )
    print(
        f"    Depth ±1%=${value(depth, number_format=',.0f')} "
        f"(нужно >${args.watch_min_depth_usd:,.0f}): "
        f"{'ПРОЙДЕНО' if depth_ok else 'НЕ ПРОЙДЕНО'}.",
        flush=True,
    )
    print(
        f"    Spread/1D ATR={value(spread, '%')} "
        f"(нужно <{args.watch_max_spread_atr_pct:g}%): "
        f"{'ПРОЙДЕНО' if spread_ok else 'НЕ ПРОЙДЕНО'}.",
        flush=True,
    )
    print(
        f"    Минимальный 7d $volume=${value(min_volume, number_format=',.0f')} "
        f"(нужно >${args.watch_min_7d_volume_usd:,.0f}): "
        f"{'ПРОЙДЕНО' if volume_ok else 'НЕ ПРОЙДЕНО'}.",
        flush=True,
    )


async def _collect_watch_snapshots(events: list[dict], args: argparse.Namespace,
                                   now_ts: int) -> dict[tuple[str, str], dict]:
    """Fetch one live market snapshot per structurally uninvalidated pair."""
    from src.exchanges.client import close_exchange_safely, create_exchange

    pairs: dict[tuple[str, str], dict] = {}
    for event in events:
        if event.get("invalidated"):
            continue
        key = (str(event["database"]), str(event["table"]))
        pairs.setdefault(key, {
            "db": key[0], "table": key[1], "exchange": event["exchange"],
            "symbol": event.get("ticker") or _ticker_from_table(event["table"]),
            "last_close": event.get("last_close"),
        })
    if not pairs:
        return {}

    daily_pools: dict[str, asyncpg.Pool | None] = {}
    for candidate in pairs.values():
        daily_db = _daily_db_name(candidate["db"], args.timeframe)
        candidate["daily_db"] = daily_db
        if daily_db and daily_db not in daily_pools:
            try:
                daily_pools[daily_db] = await asyncpg.create_pool(
                    user=settings.db_user, password=settings.db_password,
                    host=settings.db_host, port=settings.db_port, database=daily_db,
                    min_size=1, max_size=max(2, min(8, args.concurrency * 2)),
                )
            except Exception as exc:
                print(f"⚠️ Не удалось открыть дневную БД {daily_db}: {exc}", file=sys.stderr)
                daily_pools[daily_db] = None

    clients: dict[str, Any] = {}
    try:
        venue_ids = {
            candidate["exchange"]: settings.exchange_map_1d.get(
                candidate["exchange"], candidate["exchange"]
            ) for candidate in pairs.values()
        }
        for exchange, ccxt_id in venue_ids.items():
            client = None
            try:
                client = create_exchange(ccxt_id)
                await client.load_markets()
                clients[exchange] = client
            except Exception as exc:
                print(f"⚠️ Не удалось получить markets для {exchange}: {exc}", file=sys.stderr)
                if client is not None:
                    await close_exchange_safely(client, exchange)

        semaphore = asyncio.Semaphore(max(1, min(args.concurrency, 8)))
        snapshots: dict[tuple[str, str], dict] = {}

        async def one(key, candidate):
            async with semaphore:
                daily_pool = daily_pools.get(candidate.get("daily_db"))
                daily_metrics = (
                    await _daily_health_metrics(
                        daily_pool, candidate["table"], now_ts, args.watch_atr_period
                    ) if daily_pool else {"min_7d_volume_usd": None, "atr_1d": None}
                )
                client = clients.get(candidate["exchange"])
                if client:
                    try:
                        snapshot = await _fetch_watch_market_snapshot(
                            client, candidate["symbol"], float(candidate.get("last_close") or 0),
                            daily_metrics,
                        )
                    except Exception as exc:
                        print(f"⚠️ {candidate['exchange']} {candidate['symbol']}: {exc}", file=sys.stderr)
                        snapshot = {
                            "price": _finite_positive(candidate.get("last_close")),
                            "price_source": "DB", "depth_usd": None,
                            "trades_per_min": None, "is_barcode": False,
                            "spread_atr_pct": None,
                            "min_7d_volume_usd": daily_metrics.get("min_7d_volume_usd"),
                        }
                else:
                    snapshot = {
                        "price": _finite_positive(candidate.get("last_close")),
                        "price_source": "DB", "depth_usd": None,
                        "trades_per_min": None, "is_barcode": False,
                        "spread_atr_pct": None,
                        "min_7d_volume_usd": daily_metrics.get("min_7d_volume_usd"),
                    }
                snapshots[key] = snapshot

        await asyncio.gather(*(one(key, candidate) for key, candidate in pairs.items()))
        return snapshots
    finally:
        for client in clients.values():
            await close_exchange_safely(client)
        for pool in daily_pools.values():
            if pool:
                await pool.close()


async def _watch(args: argparse.Namespace) -> None:
    if args.no_save_to_db:
        raise SystemExit("--watch требует сохранения каждого прохода в БД")
    if args.import_csv:
        raise SystemExit("--watch нельзя сочетать с --import-csv")
    if args.interval_minutes <= 0:
        raise SystemExit("--interval-minutes должен быть больше нуля")
    if args.timeframe != "15m":
        raise SystemExit("--watch сканирует только 15m-данные; уберите --timeframe 1d")
    if args.watch_atr_period < 1:
        raise SystemExit("--watch-atr-period должен быть не меньше 1")

    interval = args.interval_minutes * 60.0
    schedule = (
        "один проход"
        if args.watch_once else f"проход раз в {args.interval_minutes:g} мин"
    )
    print(
        f"Текущий сканер запущен: {schedule}; "
        f"цена в диапазоне C-B, исходный RR ≥{args.rr:g}:1, ликвидность по красным порогам Dashboard. "
        "Остановка: Ctrl+C.", flush=True,
    )
    while True:
        cycle_started = time.monotonic()
        now_ts = int(time.time())
        print(f"\n=== Watch-проход {_fmt_time(now_ts)} ===", flush=True)
        try:
            events = await _scan(args)
            resolution_counts: dict[str, int] = {}
            for event in events:
                if event.get("invalidated"):
                    reason = event.get("invalidation_reason") or "unknown"
                    resolution_counts[reason] = resolution_counts.get(reason, 0) + 1
            resolved_count = sum(resolution_counts.values())
            stop_touched_count = sum(event.get("stop_hit_ts") is not None for event in events)
            target_touched_count = sum(event.get("target_hit_ts") is not None for event in events)
            both_touched_count = sum(
                event.get("stop_hit_ts") is not None and event.get("target_hit_ts") is not None
                for event in events
            )
            resolution_summary = ", ".join(
                f"{_INVALIDATION_REASON_LABELS.get(reason, reason)}: {count}"
                for reason, count in sorted(resolution_counts.items())
            ) or "нет"
            print(
                f"Кандидатов по паттерну: {len(events)}; уже разрешены после entry: "
                f"{resolved_count}; ещё без касания уровней: {len(events) - resolved_count}.\n"
                f"  Касались после entry: stop (high ≥ stop) — {stop_touched_count}; "
                f"цель C (low ≤ C) — {target_touched_count}; оба уровня — {both_touched_count}.\n"
                f"  Первое событие: {resolution_summary}.",
                flush=True,
            )
            if args.watch_show_invalidated_details:
                for event in events:
                    if not event.get("invalidated"):
                        continue
                    reason = event.get("invalidation_reason") or "unknown"
                    label = _INVALIDATION_REASON_LABELS.get(reason, reason)
                    stop_ts = event.get("stop_hit_ts")
                    target_ts = event.get("target_hit_ts")
                    print(
                        f"  RESOLVED {event['base']}/{event['exchange']} {event['market']} "
                        f"{label}; first={_fmt_time(event['invalidation_ts'])}; "
                        f"stop={event['stop']:.8g} first_stop_touch="
                        f"{_fmt_time(stop_ts) if stop_ts is not None else 'нет'}; "
                        f"C={event['target']:.8g} first_C_touch="
                        f"{_fmt_time(target_ts) if target_ts is not None else 'нет'}; "
                        f"first_candle_H/L={event['invalidation_high']:.8g}/"
                        f"{event['invalidation_low']:.8g} "
                        f"[{event['database']}.{event['table']}]",
                        flush=True,
                    )

            snapshots = await _collect_watch_snapshots(events, args, now_ts)
            accepted: list[dict[str, Any]] = []
            rejected: dict[str, int] = {}
            for event in events:
                key = (str(event["database"]), str(event["table"]))
                signal, reason = _screen_current_signal(
                    event, snapshots.get(key, {}), args, now_ts
                )
                if signal:
                    accepted.append(signal)
                elif reason and not event.get("invalidated"):
                    rejected[reason] = rejected.get(reason, 0) + 1
                    if args.symbols.strip():
                        _print_watch_filter_diagnostics(
                            event, snapshots.get(key, {}), args, reason
                        )
            accepted.sort(key=lambda item: (item["exchange"], item["base"], item["entry_ts"]))
            diagnostics = {
                "candidate_count": len(events),
                "resolved_count": resolved_count,
                "unresolved_count": len(events) - resolved_count,
                "stop_touched_count": stop_touched_count,
                "target_c_touched_count": target_touched_count,
                "both_levels_touched_count": both_touched_count,
                "first_touch_reason_counts": resolution_counts,
            }
            run_id = await _save_setups_to_db(
                accepted, args, source="current_watch", diagnostics=diagnostics
            )
            print(
                f"Текущих сетапов: {len(accepted)}; прогон #{run_id} сохранён "
                f"в {RESULTS_DB}. Отсеяно live-фильтрами (цена/ликвидность): "
                + (", ".join(f"{reason}: {count}" for reason, count in sorted(rejected.items()))
                   or "нет"),
                flush=True,
            )
            for signal in accepted:
                print(
                    f"{signal['base']:<14} {signal['exchange']:<8} {signal['market']:<5} "
                    f"{signal['current_price_source']}={signal['entry']:.8g} "
                    f"L={signal['pump_start']:.8g} A={signal['a_price']:.8g} "
                    f"C={signal['c_price']:.8g} B={signal['b_price']:.8g} "
                    f"stop={signal['stop']:.8g} RR-now={signal['rr']:.2f}:1 "
                    f"Tape={signal['trades_per_min']:.1f}/min "
                    f"Depth=${signal['depth_usd']:,.0f} "
                    f"Spread={signal['spread_atr_pct']:.1f}% 7dMin=${signal['min_7d_volume_usd']:,.0f} "
                    f"[{signal['database']}.{signal['table']}]",
                    flush=True,
                )
        except Exception as exc:
            print(f"❌ Watch-проход завершился ошибкой: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

        if args.watch_once:
            break

        wait = max(0.0, interval - (time.monotonic() - cycle_started))
        if wait > 0:
            print(f"Следующий проход через {wait / 60:.1f} мин.\n", flush=True)
            await asyncio.sleep(wait)
        else:
            print("Проход занял не меньше интервала; следующий начинается сразу.\n", flush=True)


async def _init_results_db() -> asyncpg.Pool:
    """Create/reuse the shared scanner-results database and setup tables."""
    admin = await asyncpg.connect(
        user=settings.db_user, password=settings.db_password,
        host=settings.db_host, port=settings.db_port, database="postgres",
    )
    try:
        exists = await admin.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", RESULTS_DB
        )
        if not exists:
            await admin.execute(f'CREATE DATABASE "{RESULTS_DB}"')
    finally:
        await admin.close()

    pool = await asyncpg.create_pool(
        user=settings.db_user, password=settings.db_password,
        host=settings.db_host, port=settings.db_port,
        database=RESULTS_DB, min_size=1, max_size=4,
    )
    async with pool.acquire() as conn:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS "short_setup_runs" (
                "run_id" BIGSERIAL PRIMARY KEY,
                "started_at" TIMESTAMPTZ NOT NULL DEFAULT now(),
                "timeframe" TEXT NOT NULL,
                "setups_count" INTEGER NOT NULL,
                "config" JSONB NOT NULL
            )
        ''')
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS "short_setup_events" (
                "run_id" BIGINT NOT NULL REFERENCES "short_setup_runs"("run_id") ON DELETE CASCADE,
                "event_no" INTEGER NOT NULL,
                "status" TEXT NOT NULL,
                "invalidated" BOOLEAN NOT NULL,
                "base" TEXT NOT NULL,
                "exchange" TEXT NOT NULL,
                "market" TEXT NOT NULL,
                "ticker" TEXT NOT NULL,
                "source_db" TEXT NOT NULL,
                "source_table" TEXT NOT NULL,
                "timeframe" TEXT NOT NULL,
                "start_ts" BIGINT NOT NULL,
                "a_ts" BIGINT NOT NULL,
                "c_ts" BIGINT NOT NULL,
                "b_ts" BIGINT NOT NULL,
                "entry_ts" BIGINT NOT NULL,
                "pump_start" DOUBLE PRECISION NOT NULL,
                "a_price" DOUBLE PRECISION NOT NULL,
                "c_price" DOUBLE PRECISION NOT NULL,
                "b_price" DOUBLE PRECISION NOT NULL,
                "entry" DOUBLE PRECISION NOT NULL,
                "stop" DOUBLE PRECISION NOT NULL,
                "target" DOUBLE PRECISION NOT NULL,
                "pump_pct" DOUBLE PRECISION NOT NULL,
                "ac_drop_pct" DOUBLE PRECISION NOT NULL,
                "pump_retrace_pct" DOUBLE PRECISION NOT NULL,
                "cb_bounce_pct" DOUBLE PRECISION NOT NULL,
                "rr" DOUBLE PRECISION NOT NULL,
                PRIMARY KEY ("run_id", "event_no")
            )
        ''')
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS "short_setup_events_run_entry_idx"
            ON "short_setup_events" ("run_id", "entry_ts" DESC)
        ''')
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS "short_setup_events_run_filter_idx"
            ON "short_setup_events" ("run_id", "status", "exchange", "rr")
        ''')
    return pool


async def _save_setups_to_db(events: list[dict[str, Any]], args: argparse.Namespace,
                             source: str = "scan",
                             diagnostics: dict[str, Any] | None = None) -> int:
    """Atomically save run metadata and all setup rows in pump_scanner_results."""
    pool = await _init_results_db()
    config = {
        "source": source, "timeframe": args.timeframe,
        "symbols": sorted({
            symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()
        }),
        "pump_pct": args.pump_pct, "pump_days": args.days,
        "max_retrace_pct": args.max_retrace, "min_ac_drop_pct": args.min_ac_drop,
        "min_cb_bounce_pct": args.min_cb_bounce,
        "lower_high_pct": args.lower_high_pct,
        "confirm_drop_pct": args.confirm_drop,
        "stop_buffer_pct": args.stop_buffer, "min_rr": args.rr,
        "pivot_width": args.pivot_bars, "setup_days": args.setup_days,
        "active_hours": args.active_hours,
    }
    if diagnostics is not None:
        config["watch_resolution_diagnostics"] = diagnostics
    if args.watch:
        config.update({
            "watch_mode": True,
            "watch_once": args.watch_once,
            "watch_interval_minutes": args.interval_minutes,
            "watch_atr_period": args.watch_atr_period,
            "watch_min_tape": args.watch_min_tape,
            "watch_min_depth_usd": args.watch_min_depth_usd,
            "watch_max_spread_atr_pct": args.watch_max_spread_atr_pct,
            "watch_min_7d_volume_usd": args.watch_min_7d_volume_usd,
        })
    event_columns = [
        "run_id", "event_no", "status", "invalidated", "base", "exchange", "market",
        "ticker", "source_db", "source_table", "timeframe", "start_ts", "a_ts", "c_ts",
        "b_ts", "entry_ts", "pump_start", "a_price", "c_price", "b_price", "entry",
        "stop", "target", "pump_pct", "ac_drop_pct", "pump_retrace_pct", "cb_bounce_pct", "rr",
    ]
    records = []
    for event_no, event in enumerate(events):
        records.append((
            None, event_no, str(event.get("status") or "HISTORY"),
            bool(event.get("invalidated")), str(event["base"]), str(event["exchange"]),
            str(event["market"]), _ticker_from_table(event["table"]),
            str(event["database"]), str(event["table"]), str(args.timeframe),
            int(event["start_ts"]), int(event["a_ts"]), int(event["c_ts"]),
            int(event["b_ts"]), int(event["entry_ts"]), float(event["pump_start"]),
            float(event["a_price"]), float(event["c_price"]), float(event["b_price"]),
            float(event["entry"]), float(event["stop"]), float(event["target"]),
            float(event["pump_pct"]), float(event["ac_drop_pct"]),
            float(event["pump_retrace_pct"]), float(event["cb_bounce_pct"]), float(event["rr"]),
        ))
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                run_id = await conn.fetchval(
                    '''INSERT INTO "short_setup_runs" ("timeframe", "setups_count", "config")
                       VALUES ($1, $2, $3::jsonb) RETURNING "run_id"''',
                    args.timeframe, len(events), json.dumps(config),
                )
                if records:
                    records = [(run_id, *record[1:]) for record in records]
                    await conn.copy_records_to_table(
                        "short_setup_events", records=records, columns=event_columns
                    )
        return int(run_id)
    finally:
        await pool.close()


def _read_setups_csv(path: str) -> list[dict[str, Any]]:
    """One-time migration of an existing scanner CSV into the results database."""
    integer_fields = ("start_ts", "a_ts", "c_ts", "b_ts", "entry_ts")
    float_fields = (
        "pump_start", "a_price", "c_price", "b_price", "entry", "stop", "target",
        "pump_pct", "ac_drop_pct", "pump_retrace_pct", "cb_bounce_pct", "rr",
    )
    events = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for field in integer_fields:
                row[field] = int(float(row[field]))
            for field in float_fields:
                row[field] = float(row[field])
            row["invalidated"] = str(row.get("invalidated", "")).strip().lower() in {
                "1", "true", "yes", "y"
            }
            row["status"] = str(row.get("status") or "HISTORY").upper()
            row["database"] = row.get("database") or row.get("source_db") or ""
            row["table"] = row.get("table") or row.get("source_table") or ""
            row["base"] = str(row["base"]).upper()
            row["exchange"] = str(row["exchange"]).lower()
            row["market"] = str(row.get("market") or "spot")
            if not row["database"] or not row["table"]:
                raise ValueError("CSV row has no database/table source")
            events.append(row)
    return events



def main() -> int:
    args = _args()
    if args.watch_once and not args.watch:
        raise SystemExit("--watch-once можно использовать только вместе с --watch")
    if args.watch:
        try:
            asyncio.run(_watch(args))
        except KeyboardInterrupt:
            print("Watch остановлен пользователем.", flush=True)
        return 0

    imported = bool(args.import_csv)
    if imported:
        if args.no_save_to_db:
            raise SystemExit("--import-csv требует сохранения в БД; уберите --no-save-to-db")
        events = _read_setups_csv(args.import_csv)
        if events and events[0].get("timeframe") in ("15m", "1d"):
            args.timeframe = events[0]["timeframe"]
        print(f"Импортирую {len(events)} сетапов из {args.import_csv} в {RESULTS_DB}…")
    else:
        events = asyncio.run(_scan(args))

    if not args.no_save_to_db:
        run_id = asyncio.run(_save_setups_to_db(events, args, source="csv_import" if imported else "scan"))
        print(f"Сохранено в БД {RESULTS_DB}: прогон #{run_id}, сетапов {len(events)}")
    elif not imported:
        print("Сохранение в БД отключено (--no-save-to-db).")

    if imported:
        print(f"Импорт завершён: {len(events)} сетапов → {RESULTS_DB}, прогон #{run_id}")
        return 0

    fields = ["status", "invalidated", "base", "exchange", "market", "timeframe", "start_ts", "start_time",
              "a_ts", "a_time", "c_ts", "c_time", "b_ts", "b_time", "entry_ts", "entry_time",
              "pump_start", "a_price", "c_price", "b_price", "entry", "stop", "target",
              "pump_pct", "ac_drop_pct", "pump_retrace_pct", "cb_bounce_pct", "rr", "database", "table"]
    for event in events:
        event["timeframe"] = args.timeframe
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)

    if not events:
        print("Подходящих сетапов не найдено.")
    else:
        print(f"Найдено {len(events)} сетапов → {args.out}")
        for e in sorted(events, key=lambda x: x["entry_ts"], reverse=True)[:100]:
            print(f"{e['status']:7} {e['base']:<12} {e['exchange']:<8} {e['market']:<5} "
                  f"entry {_fmt_time(e['entry_ts'])}  A={e['a_price']:.8g} C={e['c_price']:.8g} "
                  f"B={e['b_price']:.8g} entry={e['entry']:.8g} stop={e['stop']:.8g} "
                  f"target={e['target']:.8g} RR={e['rr']:.2f}:1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
