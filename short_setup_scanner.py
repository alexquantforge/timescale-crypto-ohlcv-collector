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
import datetime as dt
import os
import re
import sys
from typing import Any

import asyncpg
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import settings  # noqa: E402

_IDENT = re.compile(r"^[a-zA-Z0-9_]+$")


def _db_names(timeframe: str) -> list[str]:
    if timeframe == "1d":
        return [settings.db_high_1d, settings.db_low_1d]
    return [settings.db_high_15m, settings.db_low_15m]


def _parse_table(table: str) -> tuple[str, str, str]:
    name = table.lower()
    raw, _, exchange = name.rpartition("_on_")
    market_type = "swap" if ":" in raw else "spot"
    ticker = raw.replace("_", "/").upper()
    base = ticker.split("/", 1)[0].split(":", 1)[0]
    return base, exchange, market_type


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
) -> list[dict[str, Any]]:
    """Pure pattern detector; values are OHLC candle prices, timestamps in seconds.

    max_retrace is the fraction of the *preceding L-to-A pump leg* allowed to
    be given back at C. Example: L=100, A=150, max_retrace=30 -> C >= 135.
    """
    n = min(len(ts), len(lows), len(highs), len(closes))
    if n < max(2 * pivot_width + 3, 4):
        return []
    ts, lows, highs, closes = ts[:n], lows[:n], highs[:n], closes[:n]
    if not (np.all(np.isfinite(lows)) and np.all(np.isfinite(highs))
            and np.all(np.isfinite(closes))):
        valid = np.isfinite(lows) & np.isfinite(highs) & np.isfinite(closes)
        ts, lows, highs, closes = ts[valid], lows[valid], highs[valid], closes[valid]
        n = len(ts)

    high_pivots = _pivots(highs, pivot_width, True)
    low_pivots = _pivots(lows, pivot_width, False)
    if not len(high_pivots) or not len(low_pivots):
        return []

    pump_bars = max(1, int(pump_days * 86400 / bar_seconds))
    setup_bars = max(3, int(setup_days * 86400 / bar_seconds))
    result: list[dict[str, Any]] = []
    used_entry_indices: set[int] = set()
    low_pivot_set = set(map(int, low_pivots))

    for ai in high_pivots:
        # L is the lowest observed price in the allowed pump lookback.
        left = max(0, int(ai) - pump_bars)
        if left >= ai:
            continue
        li = left + int(np.argmin(lows[left:ai]))
        leg = highs[ai] - lows[li]
        if lows[li] <= 0 or leg / lows[li] * 100.0 < pump_pct:
            continue
        if ts[ai] - ts[li] > pump_days * 86400:
            continue

        # Walk B pivots chronologically, maintaining the lowest low since A.
        # That running minimum is C; it invalidates an earlier trough if price
        # makes a new low before B. This is linear in the number of bars, rather
        # than testing every possible C/B pair.
        b_candidates = high_pivots[(high_pivots > ai) & (high_pivots <= ai + setup_bars)]
        running_low_i = int(ai) + 1
        ci = running_low_i
        found_for_a = False
        for bi in b_candidates:
            while running_low_i <= bi:
                if lows[running_low_i] < lows[ci]:
                    ci = running_low_i
                running_low_i += 1
            if ci not in low_pivot_set:
                continue
            if ts[bi] - ts[ai] > setup_days * 86400:
                continue
            if highs[ai] <= lows[ci] or leg <= 0:
                continue
            drop_pct = (highs[ai] - lows[ci]) / highs[ai] * 100.0
            retrace_pct = (highs[ai] - lows[ci]) / leg * 100.0
            if drop_pct < min_ac_drop_pct or retrace_pct > max_retrace:
                continue
            if highs[bi] >= highs[ai] * (1.0 - lower_high_pct / 100.0):
                continue
            bounce_pct = (highs[bi] - lows[ci]) / lows[ci] * 100.0
            if bounce_pct < min_cb_bounce_pct:
                continue

            stop = highs[bi] * (1.0 + stop_buffer_pct / 100.0)
            # The first confirmed close-down after B is the reproducible
            # historical entry proxy. Ignore entries after target C traded.
            end = min(n, int(bi) + setup_bars + 1)
            # Do not backdate the entry into the bars needed to confirm B.
            for ei in range(int(bi) + pivot_width + 1, end):
                entry = closes[ei]
                if entry <= highs[bi] * (1.0 - confirm_drop_pct / 100.0):
                    if np.min(lows[bi + 1:ei + 1]) <= lows[ci] or entry <= lows[ci]:
                        break
                    risk = stop - entry
                    reward = entry - lows[ci]
                    if risk <= 0 or reward <= 0:
                        break
                    rr = reward / risk
                    if rr >= min_rr and ei not in used_entry_indices:
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
            if found_for_a:
                break

    result.sort(key=lambda r: (r["entry_ts"], r["a_ts"]))
    return result


def _fmt_time(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ищет памп → ограниченный откат A-C → lower high B → подтверждённый шорт-сетап.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--timeframe", choices=("15m", "1d"), default="15m")
    p.add_argument("--pump-pct", type=float, default=50.0, help="Минимальный рост L→A, %")
    p.add_argument("--days", type=float, default=3.0, help="Максимальная длительность L→A, дни")
    p.add_argument("--max-retrace", type=float, default=30.0,
                   help="Максимальный откат A→C как %% от роста L→A")
    p.add_argument("--min-ac-drop", type=float, default=5.0,
                   help="Минимальное падение A→C от цены A, % (отсечь мелкий шум)")
    p.add_argument("--min-cb-bounce", type=float, default=2.0,
                   help="Минимальный отскок C→B, %")
    p.add_argument("--lower-high-pct", type=float, default=0.0,
                   help="На сколько %% B должен быть ниже A; 0 = любое lower high")
    p.add_argument("--confirm-drop", type=float, default=1.0,
                   help="Подтвердить разворот закрытием на N%% ниже B")
    p.add_argument("--stop-buffer", type=float, default=0.5,
                   help="Буфер стопа выше B, %")
    p.add_argument("--rr", type=float, default=2.0, help="Минимум reward/risk до цели C")
    p.add_argument("--pivot-bars", type=int, default=2,
                   help="Свечей слева/справа для подтверждения локального пика/дна")
    p.add_argument("--setup-days", type=float, default=10.0,
                   help="Сколько максимум дней от A до сигнала")
    p.add_argument("--active-hours", type=float, default=6.0,
                   help="Считать сигнал текущим, если он моложе N часов")
    p.add_argument("--exchanges", default="", help="Список бирж через запятую, пусто = все")
    p.add_argument("--no-spot", action="store_true", help="Не сканировать spot")
    p.add_argument("--no-swap", action="store_true", help="Не сканировать perpetual swaps")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit-tables", type=int, default=0, help="Для теста ограничить число таблиц; 0 = все")
    p.add_argument("--out", default="short_setups.csv", help="CSV-файл результатов")
    return p.parse_args()


async def _scan(args: argparse.Namespace) -> list[dict[str, Any]]:
    dbs = _db_names(args.timeframe)
    bar_sec = 900 if args.timeframe == "15m" else 86400
    pools: dict[str, asyncpg.Pool] = {}
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
                if (kind == "spot" and args.no_spot) or (kind == "swap" and args.no_swap):
                    continue
                if not _IDENT.fullmatch(table):
                    continue
                tasks.append((db, pool, table))
        if args.limit_tables:
            tasks = tasks[:args.limit_tables]
        print(f"Сканирую {len(tasks)} таблиц, timeframe={args.timeframe}; критерии: "
              f"памп ≥{args.pump_pct:g}% за ≤{args.days:g} дн, A→C ≤{args.max_retrace:g}% роста, "
              f"RR ≥{args.rr:g}:1")

        async def one(db: str, pool: asyncpg.Pool, table: str) -> list[dict[str, Any]]:
            async with sem:
                try:
                    async with pool.acquire() as conn:
                        rows = await conn.fetch(
                            f'SELECT "Timestamp", low, high, close FROM "{table}" ORDER BY "Timestamp" ASC'
                        )
                    if len(rows) < 8:
                        return []
                    arr = np.asarray([tuple(r) for r in rows], dtype=np.float64)
                    found = find_setups(
                        arr[:, 0].astype(np.int64), arr[:, 1], arr[:, 2], arr[:, 3],
                        bar_seconds=bar_sec, pump_pct=args.pump_pct, pump_days=args.days,
                        max_retrace=args.max_retrace, min_ac_drop_pct=args.min_ac_drop,
                        min_cb_bounce_pct=args.min_cb_bounce, lower_high_pct=args.lower_high_pct,
                        confirm_drop_pct=args.confirm_drop, stop_buffer_pct=args.stop_buffer,
                        min_rr=args.rr, pivot_width=max(1, args.pivot_bars), setup_days=args.setup_days,
                    )
                    base, exchange, kind = _parse_table(table)
                    for event in found:
                        entry_i = int(np.searchsorted(arr[:, 0], event["entry_ts"], side="left"))
                        later_highs = arr[entry_i + 1:, 2]
                        later_lows = arr[entry_i + 1:, 1]
                        event["invalidated"] = bool(
                            np.any(later_highs >= event["stop"])
                            or np.any(later_lows <= event["target"])
                        )
                        event.update({"database": db, "table": table, "base": base,
                                      "exchange": exchange, "market": kind})
                    return found
                except Exception as exc:
                    print(f"⚠️ [{db}.{table}] {exc}", file=sys.stderr)
                    return []

        completed = 0

        async def run_task(task: tuple[str, asyncpg.Pool, str]) -> list[dict[str, Any]]:
            nonlocal completed
            found = await one(*task)
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                print(f"Прогресс: {completed}/{len(tasks)} таблиц")
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
        return results
    finally:
        for pool in pools.values():
            await pool.close()


def main() -> int:
    args = _args()
    events = asyncio.run(_scan(args))
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
