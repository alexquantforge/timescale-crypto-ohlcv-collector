"""
Priority-pair channel between the dashboard and the 15m engine.

The dashboard NEVER downloads or computes market data for the charts: it only
publishes WHICH pairs the user is looking at (the open pair plus its ±5
neighbours) into a small coordination table. The 15m engine runs a parallel
"priority lane" task that refreshes exactly those pairs in TimescaleDB every
second, so the dashboard can stay a pure renderer of stored rows.

    dashboard  --publish_priority_pairs()-->  dashboard_priority_pairs
    15m engine --read_priority_pairs()----->  (refresh loop, 1 s)

The table lives in ONE well-known database (the 15m HIGH db by default) so
both sides agree on the meeting point regardless of which tier a pair's candle
table currently sits in.

Everything that can be pure is pure (normalize/expire/select), so the lane is
unit-testable without a database.
"""
from __future__ import annotations

import time
from typing import Iterable, List, Optional, Sequence, Tuple

PRIORITY_TABLE = "dashboard_priority_pairs"

# A published set is honoured for this long; the dashboard re-publishes on
# every rerun, so a closed browser tab stops the lane instead of pinning the
# engine to a pair nobody watches.
DEFAULT_TTL_SEC = 90.0

# Hard cap on how many pairs the lane may serve at once (current pair ±5 = 11).
MAX_PRIORITY_PAIRS = 12

CREATE_SQL = (
    f'CREATE TABLE IF NOT EXISTS {PRIORITY_TABLE} ('
    " exchange text NOT NULL,"
    " symbol text NOT NULL,"
    " updated_at timestamptz NOT NULL DEFAULT now(),"
    " PRIMARY KEY (exchange, symbol))"
)

# The engine's side of the handshake: when it last SERVED a watched pair, and
# how many candles that service wrote. Without it the dashboard publishes into
# the void — "the dashboard wakes the updater" is unobservable, and a collector
# that is not running (or is stuck) looked exactly like one that refused to fix
# the gap. The ALTERs are idempotent so an existing table just grows columns.
SERVE_SQLS = (
    f"ALTER TABLE {PRIORITY_TABLE} ADD COLUMN IF NOT EXISTS served_at timestamptz",
    f"ALTER TABLE {PRIORITY_TABLE} ADD COLUMN IF NOT EXISTS served_bars int",
    # Per-timeframe, because there are TWO engines and one watch list: with a
    # single stamp, `main.py run` (1D only) made every 15m chart look "serviced"
    # while the 15m hole stayed exactly as it was. The note is only worth
    # printing if it can name the engine that is missing.
    f"ALTER TABLE {PRIORITY_TABLE} ADD COLUMN IF NOT EXISTS served_15m_at timestamptz",
    f"ALTER TABLE {PRIORITY_TABLE} ADD COLUMN IF NOT EXISTS served_1d_at timestamptz",
)


def serve_column(timeframe: str) -> str:
    """Which heartbeat column this engine stamps / this chart reads."""
    return "served_15m_at" if str(timeframe or "").lower().startswith("15") else "served_1d_at"

# DDL throttle. The lane reads this table once a second, and a
# `CREATE/ALTER ... IF NOT EXISTS` per second per engine is pure catalog noise, so
# the statements run once and then every DDL_MIN_INTERVAL_SEC.
#
# Deliberately NOT a per-connection memo: asyncpg hands out `PoolConnectionProxy`
# objects, and those cannot be weakly referenced — a `weakref.WeakSet` of
# connections raised `TypeError: cannot create weak reference to
# 'PoolConnectionProxy' object` inside `read_priority_pairs()`, which the lane
# loop swallowed as a cycle error EVERY TICK: the priority lane did not refresh a
# single pair, and the only trace was a repeating warning. Keyed process-wide
# instead (one coordination table per process), with `force=True` on the
# "schema says it is not there yet" paths so a fresh database is never left
# without its table.
_DDL_DONE_AT = [0.0]
DDL_MIN_INTERVAL_SEC = 300.0


def lane_since_sec(
    last_ts: int, now: int, step: int, catchup_max_bars: int = 2000, tail_bars: int = 3
) -> Tuple[int, int]:
    """Where the priority lane starts fetching for one pair, and how many bars to ask for.

    Returns `(since_sec, want_bars)`.

    Three cases, deliberately different:

    * The table is a bar or two behind (normal). Start ONE BAR BEFORE its last
      row: that row is a forming bar frozen mid-flight, and the writer replaces
      everything from the oldest fetched timestamp on, which only covers the
      stale bar if the fetch actually returns it.
    * The table is HOURS behind — the collector was off, the symbol was filtered
      out and came back, an engine restarted. Anchoring near `now` (what this
      used to do, with `limit=10`) refreshes the tail *around* the hole and the
      chart keeps the gap forever. So the fetch starts at the hole itself and the
      response length grows to cover it; one request per lane tick, so a big hole
      walks closed over a few ticks instead of turning the lane into a
      history-backfill job.
    * A hole deeper than `catchup_max_bars` is history, not a tail — the full
      sweep owns that, and the lane must not spend an hour paging the exchange
      for a pair somebody glanced at. Tail-only again.
    """
    tail = max(1, int(tail_bars))
    if last_ts and int(last_ts) > 1 and step > 0:
        hole = max(0, (int(now) - int(last_ts)) // step)
        cap = max(0, int(catchup_max_bars))
        if hole <= cap:
            return int(last_ts) - step, hole + tail + 1
    return int(now) - (tail + 1) * step, tail + 2


# Rate limiter for "the lane could not do its job" lines. Both engines tick every
# few seconds per watched pair, so an unbounded warning would bury the log (and a
# silent `return 0` was the other failure mode: a stale chart with no reason
# recorded). One line per pair per interval is enough to spot a pattern.
_LANE_WARN_AT: dict = {}


def lane_warn_due(key, min_interval_sec: float = 600.0) -> bool:
    """True when a warning for `key` has not been printed recently (and records it)."""
    now = time.time()
    last = _LANE_WARN_AT.get(key)
    if last is not None and now - last < min_interval_sec:
        return False
    _LANE_WARN_AT[key] = now
    if len(_LANE_WARN_AT) > 2000:
        for k, _ in sorted(_LANE_WARN_AT.items(), key=lambda kv: kv[1])[:1000]:
            _LANE_WARN_AT.pop(k, None)
    return True


def normalize_pairs(
    pairs: Iterable, limit: int = MAX_PRIORITY_PAIRS
) -> List[Tuple[str, str]]:
    """
    Cleans a published set into ordered unique (exchange, symbol) tuples.

    Accepts dicts ({"ex": ..., "sym": ...}) as produced by the dashboard's
    neighbour warmer, or plain 2-tuples. Order is preserved (the open pair is
    published first, so it survives the `limit` cut).
    """
    out: List[Tuple[str, str]] = []
    seen = set()
    for item in pairs or []:
        if isinstance(item, dict):
            exchange, symbol = item.get("ex"), item.get("sym")
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            exchange, symbol = item[0], item[1]
        else:
            continue
        exchange = str(exchange or "").strip()
        symbol = str(symbol or "").strip()
        if not exchange or not symbol or "/" not in symbol:
            continue
        key = (exchange, symbol)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= limit:
            break
    return out


def select_fresh(
    rows: Sequence, now: Optional[float] = None, ttl_sec: float = DEFAULT_TTL_SEC
) -> List[Tuple[str, str]]:
    """
    Keeps only pairs whose publication is younger than `ttl_sec`.

    `rows` are (exchange, symbol, age_sec) triples — the age is computed by the
    database (`EXTRACT(EPOCH FROM (now() - updated_at))`) so engine and
    dashboard clocks never have to agree.
    """
    _ = now  # kept for signature symmetry / future clock injection
    fresh: List[Tuple[str, str]] = []
    for row in rows or []:
        try:
            exchange, symbol, age = row[0], row[1], float(row[2])
        except (TypeError, ValueError, IndexError):
            continue
        if not exchange or not symbol:
            continue
        if age <= ttl_sec:
            fresh.append((str(exchange), str(symbol)))
    return fresh[:MAX_PRIORITY_PAIRS]


def due_pairs(
    pairs: Sequence[Tuple[str, str]],
    last_run: dict,
    interval_sec: float,
    now: Optional[float] = None,
) -> List[Tuple[str, str]]:
    """
    Pairs whose own refresh interval has elapsed.

    The lane ticks once per second for the whole set, but a pair that is still
    being fetched (or was just refreshed) must not be queued again — otherwise
    a slow exchange piles up overlapping requests and burns the rate limit.
    """
    now = time.time() if now is None else now
    return [
        p for p in pairs
        if now - float(last_run.get(p, 0.0)) >= interval_sec
    ]



def resolve_exchange_alias(published: str, name_to_id: dict) -> Tuple[str, str]:
    """
    Maps a published exchange label onto (engine_name, ccxt_id).

    The two engines disagree on table suffixes: 15m tables are
    `..._on_gateio` (engine name) while 1D tables are `..._on_gate` (ccxt id),
    and the dashboard publishes whatever its summary row carried. Accepting
    BOTH spellings keeps a published pair from being resolved to a table name
    that does not exist (which would otherwise create an empty junk table).
    """
    label = (published or "").strip().lower()
    if not label:
        return "", ""
    if label in name_to_id:                       # engine name, e.g. "gateio"
        return label, name_to_id[label]
    for name, ccxt_id in name_to_id.items():      # ccxt id, e.g. "gate"
        if label == str(ccxt_id).lower():
            return name, str(ccxt_id)
    return label, label                           # unknown: pass through


async def ensure_priority_table(conn, force: bool = False) -> None:
    """Creates the handshake table (and grows the heartbeat columns) if needed."""
    now = time.time()
    if not force and now - float(_DDL_DONE_AT[0]) < DDL_MIN_INTERVAL_SEC:
        return
    _DDL_DONE_AT[0] = now
    await conn.execute(CREATE_SQL)
    for ddl in SERVE_SQLS:
        await conn.execute(ddl)


def _missing_schema_error(e: Exception) -> bool:
    """'table/column does not exist' in asyncpg's (or a fake's) words."""
    msg = str(e).lower()
    return "does not exist" in msg or "undefined" in type(e).__name__.lower()


async def _fetch_healed(conn, kind: str, sql: str, *args):
    """`fetch`/`fetchrow` with the same one-shot re-ensure as `_execute_healed`.

    Matters for the ENGINE side most: a lane that cannot read the table logs a
    cycle error every second forever, and if the throttle above decided "schema is
    already ensured" after the table disappeared (a restore, a rename), that cycle
    error would be the last thing the lane ever did.
    """
    try:
        return await getattr(conn, kind)(sql, *args)
    except Exception as e:
        if not _missing_schema_error(e):
            raise
        await ensure_priority_table(conn, force=True)
        return await getattr(conn, kind)(sql, *args)


async def _execute_healed(conn, sql: str, *args) -> None:
    """One statement, re-ensuring the schema once if the DB does not know it.

    The DDL throttle above makes "the table exists but my memo says it was
    created" possible after a fresh restore; retrying once with `force=True`
    costs nothing and beats failing the lane tick.
    """
    try:
        await conn.execute(sql, *args)
    except Exception as e:
        if not _missing_schema_error(e):
            raise
        await ensure_priority_table(conn, force=True)
        await conn.execute(sql, *args)


async def publish_priority_pairs(conn, pairs: Iterable, ttl_sec: float = DEFAULT_TTL_SEC) -> int:
    """
    Dashboard side: replaces the published working set.

    Writes the current set and deletes everything else that has expired, so a
    pair the user navigated away from stops being refreshed after `ttl_sec`.
    Returns how many pairs were published.
    """
    normalized = normalize_pairs(pairs)
    await ensure_priority_table(conn)
    insert_sql = (
        f"INSERT INTO {PRIORITY_TABLE} (exchange, symbol, updated_at)"
        " VALUES ($1, $2, now())"
        " ON CONFLICT (exchange, symbol) DO UPDATE SET updated_at = now()"
    )
    delete_sql = (
        f"DELETE FROM {PRIORITY_TABLE}"
        f" WHERE updated_at < now() - ($1 || ' seconds')::interval"
    )
    try:
        if normalized:
            await conn.executemany(insert_sql, normalized)
        await conn.execute(delete_sql, str(int(ttl_sec * 4)))
    except Exception as e:
        if not _missing_schema_error(e):
            raise
        # table gone (restore / rename): create it and announce the set again
        await ensure_priority_table(conn, force=True)
        if normalized:
            await conn.executemany(insert_sql, normalized)
        await conn.execute(delete_sql, str(int(ttl_sec * 4)))
    return len(normalized)


async def mark_lane_service(
    conn, pairs: Sequence[Tuple[str, str]], candles: int = 0, timeframe: str = "15m"
) -> int:
    """Engine side of the heartbeat: stamp the pairs this loop just served.

    Small (the watch list is capped at `MAX_PRIORITY_PAIRS`) and called at most
    every few seconds, so the dashboard can always answer "is a live collector
    actually working on the pair the user is looking at, or is it just waiting
    for a process that is not running?"
    """
    col = serve_column(timeframe)
    stamped = 0
    for exchange, symbol in normalize_pairs(pairs):
        await _execute_healed(
            conn,
            f"UPDATE {PRIORITY_TABLE} SET served_at = now(), served_bars = $3,"
            f" {col} = now() WHERE exchange = $1 AND symbol = $2",
            exchange, symbol, int(candles),
        )
        stamped += 1
    return stamped


_PULSE_SQL = (
    f"SELECT count(*) AS watched,"
    f" count(*) FILTER (WHERE served_at > now() - ($1 || ' seconds')::interval) AS served,"
    f" EXTRACT(EPOCH FROM (now() - max(served_at))) AS idle_sec,"
    f" count(*) FILTER (WHERE served_15m_at > now() - ($1 || ' seconds')::interval)"
    f" AS served_15m,"
    f" count(*) FILTER (WHERE served_1d_at > now() - ($1 || ' seconds')::interval) AS served_1d,"
    f" EXTRACT(EPOCH FROM (now() - max(served_15m_at))) AS idle_15m,"
    f" EXTRACT(EPOCH FROM (now() - max(served_1d_at))) AS idle_1d"
    f" FROM {PRIORITY_TABLE}"
    f" WHERE updated_at > now() - ($2 || ' seconds')::interval"
)


async def lane_pulse(conn, stale_sec: float = 120.0) -> dict:
    """How alive is the lane? Dashboard-side, one cheap aggregate query.

    `watched` — pairs currently published; `served` — of those, how many an
    engine stamped within `stale_sec`; `idle_sec` — age of the newest stamp
    overall. `served == 0` with `watched > 0` means NO collector process is
    servicing the pair the user is looking at: nothing will be written to the
    database, and a chart hole will stay open no matter how often the page is
    reloaded. Saying that is far more useful than silently re-painting the hole.
    """
    await ensure_priority_table(conn)
    row = await _fetch_healed(conn, "fetchrow", _PULSE_SQL,
                              str(int(stale_sec)), str(int(DEFAULT_TTL_SEC)))
    def _num(key: str, default=0):
        try:
            v = row[key]
        except (TypeError, LookupError, KeyError):
            return default
        return default if v is None else v

    if not row:
        return {"watched": 0, "served": 0, "idle_sec": None,
                "served_15m": 0, "served_1d": 0, "idle_15m": None, "idle_1d": None}
    out = {
        "watched": int(_num("watched") or 0),
        "served": int(_num("served") or 0),
        "idle_sec": (float(_num("idle_sec")) if _num("idle_sec") is not None else None),
    }
    for tf in ("15m", "1d"):
        out[f"served_{tf}"] = int(_num(f"served_{tf}") or 0)
        idle = _num(f"idle_{tf}")
        out[f"idle_{tf}"] = float(idle) if idle is not None else None
    return out


async def read_priority_pairs(conn, ttl_sec: float = DEFAULT_TTL_SEC) -> List[Tuple[str, str]]:
    """Engine side: the pairs the dashboard is currently displaying."""
    await ensure_priority_table(conn)
    rows = await _fetch_healed(
        conn, "fetch",
        f"SELECT exchange, symbol,"
        f" EXTRACT(EPOCH FROM (now() - updated_at)) AS age"
        f" FROM {PRIORITY_TABLE} ORDER BY updated_at DESC"
    )
    return select_fresh(
        [(r["exchange"], r["symbol"], r["age"]) for r in rows], ttl_sec=ttl_sec
    )
