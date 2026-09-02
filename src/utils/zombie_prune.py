"""
Pruning of the ZOMBIE SPOT tables left behind by the perp-first switch.

Background: when a base asset gets a perpetual on an exchange, the collector
starts writing `BASE/USDT:USDT` and stops touching the old `BASE/USDT` spot
table — but nothing drops it. What is left is a table that is never written
again, usually holds FEWER candles than its perp (the perp is older/broader),
and is still counted by every scan, every summary and every "hide dead spot
duplicates" pass of the dashboard. The dashboard can hide it from the UI, but
the table keeps costing the database.

The rule implemented here is exactly the one the operator asked for:

    prune a spot pair table when the PERP of the same base on the same exchange,
    in the same database, has MORE bars than the spot table.

Three things it deliberately refuses to do:

* it never touches a perp table, a non-pair table, or a spot table that has no
  perp counterpart — a name only becomes a candidate through the perp's own
  name, never by guessing the other direction;
* it never prunes on a failed or missing count (a timeout during COUNT(*) is
  `unknown`, and `unknown` is kept — deleting is not retryable);
* it never prunes a spot table the collector is STILL writing (default: must be
  idle for `stale_hours`), because such a table is smaller but not dead, and
  deleting live data is the one mistake this file cannot unsay.

Dry-run by default; `mode="trash"` (the default) parks tables in a separate
schema instead of dropping them, so a wrong call is a rename away from being
undone.
"""
from __future__ import annotations

import asyncio
import math
import re
import time
from typing import Any, Iterable, Optional

# Quotes the collector uses for both sides of a pair. A perp table whose name is
# written without the colon (`btc_usdt_usdt_on_bybit`, older migrations) is only
# recognised when the doubled tail is one of these — anything else stays alone.
_QUOTES = ("usdt", "usdc", "usd", "dai", "tusd", "fdusd", "brl", "eur", "try")

# Tables the dashboard/engine own and that are NOT pair tables, even though the
# pair filter could otherwise match them.
_NEVER_PRUNE = {"dashboard_live_ticks"}

_ZOMBIE_SCHEMA = "zombie_pruned"


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------
def split_pair_table(name: str) -> Optional[tuple[str, str]]:
    """`1000rats_usdt:usdt_on_bybit` → ('1000rats_usdt:usdt', 'bybit'); None if
    the table is not a pair table.

    Splits on the LAST `_on_`: base symbols legitimately contain underscores
    (`1_000cats_usdt_on_bybit` from some sanitizers), and a name like
    `lion_on_me_usdt_on_bybit` must not be read as symbol `lion` on exchange
    `me_usdt_on_bybit`.
    """
    if not name or "_on_" not in name:
        return None
    sym, _, ex = name.rpartition("_on_")
    sym, ex = sym.strip(), ex.strip().lower()
    if not sym or not ex or name in _NEVER_PRUNE:
        return None
    return sym, ex


def is_perp_symbol(sym: str) -> bool:
    """`1000rats_usdt:usdt` (or the colonless doubled-quote form) is a perp."""
    if ":" in sym:
        return True
    return _doubled_quote_tail(sym) is not None


def _doubled_quote_tail(sym: str) -> Optional[str]:
    """The trailing `_usdt` of `btc_usdt_usdt` when the part before it already
    ends in the same quote. Returns the quote, or None if the name is not of
    that shape (so `btc_usdt` stays a spot and `doge_usdc_usdt` is not guessed).
    """
    parts = sym.rsplit("_", 1)
    if len(parts) != 2:
        return None
    head, tail = parts[0].lower(), parts[1].lower()
    if tail in _QUOTES and head.endswith("_" + tail):
        return tail
    return None


def perp_to_spot_name(name: str) -> Optional[str]:
    """The spot table name that this PERP table would have replaced.

    `1000rats_usdt:usdt_on_bybit` → `1000rats_usdt_on_bybit`
    `btc_usdt_usdt_on_bybit`      → `btc_usdt_on_bybit`
    Returns None for a spot table or anything that is not a pair table, so a
    candidate pair can only ever be produced from a name that is provably a
    perp. That asymmetry is the point: guessing `spot → perp` would need to
    know which quote each exchange marginals in, and a wrong guess deletes a
    table nobody asked about.
    """
    parts = split_pair_table(name)
    if parts is None:
        return None
    sym, ex = parts
    if ":" in sym:
        sym = sym.split(":", 1)[0]
    else:
        quote = _doubled_quote_tail(sym)
        if quote is None:
            return None
        sym = sym[: -(len(quote) + 1)]
    if not sym:
        return None
    return f"{sym}_on_{ex}"


# ---------------------------------------------------------------------------
# deciding
# ---------------------------------------------------------------------------
def zombie_verdict(
    spot_bars: Optional[int],
    perp_bars: Optional[int],
    spot_idle_sec: Optional[float] = None,
    stale_sec: float = 24 * 3600,
) -> tuple[str, str]:
    """('prune' | 'keep' | 'unknown', reason for the operator).

    `spot_bars`/`perp_bars` are None when the count could not be taken; the
    verdict is then `unknown`, which the caller must treat as untouchable.
    `spot_idle_sec` is None when the table's freshness is unknown — with a
    positive `stale_sec` that is also `unknown`, because "we could not check
    whether this is live data" is not a licence to delete it. A spot table with
    no rows at all is the one exception: nothing is at stake, and an empty table
    has no last write to check in the first place.
    """
    if spot_bars is None or perp_bars is None:
        return "unknown", "bar count unavailable (query failed or timed out)"
    if spot_bars < 0 or perp_bars < 0 or (spot_idle_sec is not None and math.isnan(spot_idle_sec)):
        return "unknown", "negative bar count or NaN idle time (corrupt row)"
    if spot_bars >= perp_bars:
        return "keep", f"spot has {spot_bars} bars, perp {perp_bars} — spot is not smaller"
    if stale_sec and stale_sec > 0:
        if spot_idle_sec is None:
            if spot_bars == 0:
                # Nothing is at stake in an empty table, and it has no last write
                # to inspect: the freshness rule cannot be evaluated, but the
                # scan cost it adds is real. `trash` mode (the default) makes this
                # decision one ALTER away from being undone, which is why the
                # unknown timestamp does not have to block it.
                return "prune", f"spot table is empty while its perp holds {perp_bars} bars"
            return "unknown", "last write of the spot table is unknown"
        if spot_idle_sec < stale_sec:
            return (
                "keep",
                f"spot is smaller but still collected (last write "
                f"{spot_idle_sec / 3600.0:.1f}h ago) — perp-first has not retired it",
            )
    if spot_bars == 0:
        return "prune", f"spot table is empty while its perp holds {perp_bars} bars"
    idle = "never written" if not spot_idle_sec else f"idle {spot_idle_sec / 3600.0:.1f}h"
    return (
        "prune",
        f"spot {spot_bars} bars < perp {perp_bars} bars, {idle}",
    )


def over_prune_guard(prune_count: int, pair_tables: int, max_fraction: float) -> Optional[str]:
    """A refusal message when the prune set is implausibly large.

    A bug in the name mapping would otherwise delete a third of the database in
    one run, so a mass prune needs an explicit second consent.
    """
    if max_fraction <= 0 or pair_tables <= 0:
        return None
    if prune_count <= max_fraction * pair_tables:
        return None
    return (
        f"refusing to prune {prune_count} of {pair_tables} pair tables "
        f"(> {max_fraction * 100:.0f}%): check the report, then re-run with a "
        f"larger --max-fraction if this really is the list you want"
    )


def quote_ident(name: str) -> str:
    """SQL identifier quoting (a `"` inside a table name is doubled, as in PG)."""
    return '"' + str(name).replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
def pair_tables_sql() -> str:
    """Every pair table of the database with an ESTIMATED row count.

    One catalog query for thousands of tables: pg_class + the sum over the
    inheritance children (Timescale chunks are ordinary child tables, so their
    reltuples is the cheapest per-table row estimate that exists). Estimates are
    only used to order the work and for `exact=False`; a prune decision needs the
    real COUNT(*).
    """
    return (
        "SELECT c.relname AS table_name, "
        "  GREATEST(COALESCE(c.reltuples, 0)::bigint, "
        "           COALESCE((SELECT SUM(cc.reltuples)::bigint "
        "                     FROM pg_catalog.pg_inherits i "
        "                     JOIN pg_catalog.pg_class cc ON cc.oid = i.inhrelid "
        "                     WHERE i.inhparent = c.oid), 0)) AS est_bars, "
        "  (SELECT COALESCE(MAX(a.attname), '') FROM pg_catalog.pg_attribute a "
        "    WHERE a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped "
        "      AND a.attname = 'Timestamp') AS has_time_col "
        "FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') "
        # '\\_' — a literal underscore. Unescaped, '_' is a single-character
        # wildcard and unrelated tables join the pair list.
        "  AND c.relname LIKE '%\\_on\\_%' "
        "  AND c.relname <> 'dashboard_live_ticks'"
    )


def count_and_last_sql(table_name: str) -> str:
    """Exact bar count + last written time of ONE table, in a single pass.

    `Timestamp` may hold seconds or milliseconds; the caller normalises through
    `normalize_last_ts`, which is also what the dashboard does, so a ms table is
    never mistaken for 'written in 2039' and kept forever.
    """
    return (
        f"SELECT COUNT(*)::bigint AS bars, MAX(\"Timestamp\") AS last_ts "
        f"FROM {quote_ident(table_name)}"
    )


def normalize_last_ts(last_ts: Any, now: Optional[float] = None) -> Optional[float]:
    """Epoch (s or ms) or datetime → seconds since epoch; None when unusable."""
    if last_ts is None:
        return None
    now = time.time() if now is None else float(now)
    try:
        if hasattr(last_ts, "timestamp"):
            v = float(last_ts.timestamp())
        else:
            v = float(last_ts)
            if v > 1e11:          # milliseconds
                v /= 1000.0
    except (TypeError, ValueError):
        return None
    if v <= 0 or v > now + 31536000:   # nonsense (garbage rows, far future)
        return None
    return now - v


def spot_perp_pairs(tables: Iterable[str]) -> dict[str, str]:
    """spot table name → perp table name, for pairs where BOTH exist here.

    The only source of candidates in this module: `measure` counts these tables,
    `plan_pruning` decides on them, and a name that is not in here can never be
    pruned. Derived from the perp side only (see perp_to_spot_name).
    """
    names = set(tables)
    pairs: dict[str, str] = {}
    for name in names:
        spot = perp_to_spot_name(name)
        if spot and spot in names:
            pairs[spot] = name
    return pairs


def plan_pruning(
    tables: dict[str, int],
    counts: dict[str, dict],
    stale_sec: float = 24 * 3600,
) -> list[dict]:
    """The full decision list from a catalog snapshot + measured counts.

    `tables` maps every pair table → estimated bars (used only to report what
    was not measured); `counts` maps table → {"bars": int|None, "last_ts": …}.
    Returns one record per CANDIDATE pair (a spot whose perp exists), each with
    a verdict; non-candidate tables are not listed — they are not up for
    discussion.
    """
    out: list[dict] = []
    for spot, perp in sorted(spot_perp_pairs(tables).items()):
        s_c, p_c = counts.get(spot) or {}, counts.get(perp) or {}
        spot_bars, perp_bars = s_c.get("bars"), p_c.get("bars")
        idle = normalize_last_ts(s_c.get("last_ts"))
        verdict, reason = zombie_verdict(spot_bars, perp_bars, idle, stale_sec)
        out.append({
            "db_table": spot,
            "spot": spot,
            "perp": perp,
            "spot_bars": spot_bars,
            "perp_bars": perp_bars,
            "spot_est_bars": tables.get(spot),
            "perp_est_bars": tables.get(perp),
            "spot_idle_sec": None if idle is None else round(idle, 1),
            "verdict": verdict,
            "reason": reason,
        })
    return out


# ---------------------------------------------------------------------------
# database work
# ---------------------------------------------------------------------------
async def read_pair_tables(conn) -> dict[str, int]:
    rows = await conn.fetch(pair_tables_sql())
    return {r["table_name"]: int(r["est_bars"] or 0) for r in rows}


async def measure(
    target, table_names: Iterable[str], concurrency: int = 8,
    statement_timeout_sec: float = 60.0, progress_every: int = 500,
) -> dict[str, dict]:
    """COUNT(*) + MAX(Timestamp) per table, bounded, failures as None.

    A failing count is recorded as `{"bars": None, "error": …}` rather than
    dropped: `plan_pruning` then calls that pair `unknown`, which keeps it.

    Pass a POOL, not a connection, when `concurrency > 1`. One connection cannot
    host several concurrent `transaction()` blocks — asyncpg answers
    "cannot use the implicit transaction of an explicit transaction block" — and
    the first end-to-end run of this file found exactly that: every count but the
    first "failed", and the tool reported every pair as `unknown` (correctly
    deleting nothing, but uselessly). With a bare connection the work is
    serialized and the per-query timeout is left to the caller's own setting.
    """
    names = [t for t in table_names]
    is_pool = hasattr(target, "acquire")
    sem = asyncio.Semaphore(max(1, int(concurrency)) if is_pool else 1)
    out: dict[str, dict] = {}
    done = 0
    lock = asyncio.Lock()

    async def _count_one(tbl: str):
        sql = count_and_last_sql(tbl)
        if is_pool:
            async with target.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        f"SET LOCAL statement_timeout = {int(statement_timeout_sec * 1000)}"
                    )
                    return await conn.fetchrow(sql)
        return await target.fetchrow(sql)

    async def one(tbl: str):
        nonlocal done
        async with sem:
            try:
                row = await _count_one(tbl)
                rec = {
                    "bars": int(row["bars"]) if row and row["bars"] is not None else None,
                    "last_ts": row["last_ts"] if row else None,
                }
            except Exception as e:
                # reported, never retried blindly, and never read as "empty"
                rec = {"bars": None, "last_ts": None, "error": f"{type(e).__name__}: {e}"}
        async with lock:
            out[tbl] = rec
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"[prune] counted {done}/{len(names)} tables", flush=True)

    await asyncio.gather(*(one(t) for t in names))
    return out


def candidate_tables(tables: dict[str, int]) -> tuple[list[str], int]:
    """Every table that must be COUNT(*)ed (both sides of every duplicate), and
    how many pair tables the database holds at all (the mass-prune guard's
    denominator)."""
    pairs = spot_perp_pairs(tables)
    return sorted(set(pairs) | set(pairs.values())), len(tables)


def statements_for(actions: list[dict], mode: str, schema: str = _ZOMBIE_SCHEMA) -> list[str]:
    """The DDL for a prune plan — kept separate from execution so it is testable.

    `trash` moves the table into another schema (reversible with one ALTER, and
    the dashboard stops seeing it because its catalog read is public-only);
    `drop` is the irreversible one.
    """
    stmts: list[str] = []
    if mode == "trash":
        stmts.append(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)}")
    for a in actions:
        tbl = a.get("spot") or a.get("db_table")
        if not tbl:
            continue
        if mode == "trash":
            stmts.append(
                f"ALTER TABLE {quote_ident('public')}.{quote_ident(tbl)} "
                f"SET SCHEMA {quote_ident(schema)}"
            )
        elif mode == "drop":
            stmts.append(f"DROP TABLE IF EXISTS {quote_ident('public')}.{quote_ident(tbl)} CASCADE")
        else:
            raise ValueError(f"unknown mode {mode!r} (expected 'trash' or 'drop')")
    return stmts


async def apply_pruning(conn, actions: list[dict], mode: str, schema: str = _ZOMBIE_SCHEMA) -> list[dict]:
    """Run the plan and return one record per statement.

    A record carries `spot` on the table statements and `spot=None` on the
    `CREATE SCHEMA` preamble: that statement can fail on its own (missing
    permission), and counting it as a moved table would report a deletion that
    did not happen. Callers report "N of M tables" over the `spot`-bearing
    records. A failed preamble makes every following ALTER fail too, and each
    gets its own error line — nothing is skipped quietly.
    """
    tables = [a.get("spot") or a.get("db_table") for a in actions]
    tables = [t for t in tables if t]
    results: list[dict] = []
    names = iter(tables)
    for sql in statements_for(actions, mode, schema):
        tbl = None if sql.startswith("CREATE SCHEMA") else next(names, None)
        try:
            await conn.execute(sql)
            results.append({"sql": sql, "spot": tbl, "ok": True})
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if mode == "trash" and "already exists" in err:
                # The usual reason: a previous run parked a table with the same
                # name there. Saying so beats making someone read the DDL.
                err += (
                    f' — "zombie_pruned"."{tbl}" is taken; undo the earlier move '
                    f'with ALTER TABLE zombie_pruned."{tbl}" SET SCHEMA public, or drop it'
                )
            results.append({"sql": sql, "spot": tbl, "ok": False, "error": err})
    return results


def summarize(plan: list[dict]) -> dict:
    s = {"prune": 0, "keep": 0, "unknown": 0}
    for row in plan:
        s[row["verdict"]] = s.get(row["verdict"], 0) + 1
    s["candidates"] = len(plan)
    return s
