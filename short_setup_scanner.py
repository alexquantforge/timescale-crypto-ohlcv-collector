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
import time
from typing import Any

import asyncpg
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import settings  # noqa: E402

_IDENT = re.compile(r"^[a-zA-Z0-9_]+$")
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
    p.add_argument("--out", default="short_setups.csv", help="Дополнительный CSV-экспорт результатов")
    p.add_argument("--no-save-to-db", action="store_true",
                   help="Не сохранять запуск и сетапы в pump_scanner_results")
    p.add_argument("--import-csv", default="",
                   help="Импортировать уже готовый CSV в базу без повторного сканирования")
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
        total_tables = len(tasks)
        print(f"Сканирую {total_tables} таблиц, timeframe={args.timeframe}; критерии: "
              f"памп ≥{args.pump_pct:g}% за ≤{args.days:g} дн, A→C ≤{args.max_retrace:g}% роста, "
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
        return results
    finally:
        for pool in pools.values():
            await pool.close()



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
                             source: str = "scan") -> int:
    """Atomically save run metadata and all setup rows in pump_scanner_results."""
    pool = await _init_results_db()
    config = {
        "source": source, "timeframe": args.timeframe,
        "pump_pct": args.pump_pct, "pump_days": args.days,
        "max_retrace_pct": args.max_retrace, "min_ac_drop_pct": args.min_ac_drop,
        "min_cb_bounce_pct": args.min_cb_bounce,
        "lower_high_pct": args.lower_high_pct,
        "confirm_drop_pct": args.confirm_drop,
        "stop_buffer_pct": args.stop_buffer, "min_rr": args.rr,
        "pivot_width": args.pivot_bars, "setup_days": args.setup_days,
        "active_hours": args.active_hours,
    }
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
