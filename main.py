"""
CLI entry point for Timescale Crypto OHLCV Collector.
"""
import asyncio
import logging
import sys
import subprocess
import asyncpg
import typer
from rich.console import Console
from rich.table import Table

from config.settings import settings
from src.core.updater import MarketDataEngine
from src.core.updater_15m import main_15m_loop
from src.db.connection import get_db_pools, close_all_db_pools
from src.db.migrations import ensure_databases_exist

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")
console = Console()
app = typer.Typer(
    help="⚡ Timescale Crypto OHLCV Collector — Multi-Timeframe (1D/15M) Market Collector"
)


async def _run_all_timeframes():
    """Runs 1D and 15M collection loops concurrently in the same process."""
    engine_1d = MarketDataEngine(timeframe="1d")
    await asyncio.gather(
        engine_1d.start_loop(),
        main_15m_loop(),
    )


@app.command()
def run(
    timeframe: str = typer.Option("1d", help="Candle timeframe to collect: '1d', '15m', 'all'"),
):
    """
    Start the market data collection loop ('1d', '15m', or 'all' for both concurrently).
    """
    tf = timeframe.lower()
    if tf == "all":
        console.print("[bold green]Starting Market Data Engine for ALL timeframes (1D + 15M concurrently)...[/bold green]")
        try:
            asyncio.run(_run_all_timeframes())
        except KeyboardInterrupt:
            console.print("[bold yellow]Engine stopped by user.[/bold yellow]")
            sys.exit(0)
    elif tf == "15m":
        console.print("[bold green]Starting Dedicated 15M Market Data Updater (Bybit, Gate, MEXC, OKX, BingX)...[/bold green]")
        try:
            asyncio.run(main_15m_loop())
        except KeyboardInterrupt:
            console.print("[bold yellow]15M Engine stopped by user.[/bold yellow]")
            sys.exit(0)
    else:
        console.print(f"[bold green]Starting Market Data Engine for timeframe '{tf}'...[/bold green]")
        engine = MarketDataEngine(timeframe=tf)
        try:
            asyncio.run(engine.start_loop())
        except KeyboardInterrupt:
            console.print("[bold yellow]Engine stopped by user.[/bold yellow]")
            sys.exit(0)


@app.command()
def init_db():
    """
    Initialize TimescaleDB databases and extension checks.
    """
    async def _init():
        await ensure_databases_exist()

    console.print("[bold blue]Checking TimescaleDB Databases & Extensions...[/bold blue]")
    asyncio.run(_init())
    console.print("[bold green]✓ All 4 Databases & Extensions checked successfully![/bold green]")


@app.command()
def dashboard(
    host: str = typer.Option("0.0.0.0", help="Host address for Streamlit dashboard"),
    port: int = typer.Option(8501, help="Port for Streamlit dashboard"),
):
    """
    Launch the Streamlit Web Dashboard.
    """
    console.print(f"[bold cyan]Launching Streamlit Dashboard on http://{host}:{port}...[/bold cyan]")
    cmd = [
        "streamlit",
        "run",
        "dashboard/app.py",
        f"--server.address={host}",
        f"--server.port={port}",
    ]
    subprocess.run(cmd)


@app.command()
def summary(
    timeframe: str = typer.Option("1d", help="Timeframe mode to inspect: '1d' or '15m'"),
):
    """
    Display summary statistics for the historical databases.
    """
    async def _summary():
        if timeframe == "15m":
            high_db = settings.db_high_15m
            low_db = settings.db_low_15m
        else:
            high_db = settings.db_high_1d
            low_db = settings.db_low_1d

        total_tables = 0
        tier_counts = {"HIGH": 0, "LOW": 0}
        ex_counts = {}

        for tier, db_name in [("HIGH", high_db), ("LOW", low_db)]:
            try:
                conn = await asyncpg.connect(
                    host=settings.db_host,
                    port=settings.db_port,
                    user=settings.db_user,
                    password=settings.db_password,
                    database=db_name,
                )
                tables = await conn.fetch(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name LIKE '%_on_%'"
                )
                total_tables += len(tables)
                tier_counts[tier] = len(tables)

                for r in tables:
                    tbl = r["table_name"]
                    ex = tbl.rsplit("_on_", 1)[-1] if "_on_" in tbl else "unknown"
                    ex_counts[ex] = ex_counts.get(ex, 0) + 1

                await conn.close()
            except Exception as e:
                console.print(f"[yellow]Could not connect to {db_name}: {e}[/yellow]")

        table = Table(title=f"📊 Historical Databases Summary ({timeframe.upper()})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold green")

        table.add_row("HIGH Volume Database", high_db)
        table.add_row("LOW Volume Database", low_db)
        table.add_row("Total Symbol Tables", f"{total_tables:,}")
        table.add_row("HIGH Volume Tier Tables", f"{tier_counts['HIGH']:,}")
        table.add_row("LOW Volume Tier Tables", f"{tier_counts['LOW']:,}")

        console.print(table)

        if ex_counts:
            ex_table = Table(title="🏛️ Exchange Pair Breakdown")
            ex_table.add_column("Exchange", style="yellow")
            ex_table.add_column("Unique Symbols", style="magenta")
            for ex, cnt in sorted(ex_counts.items()):
                ex_table.add_row(ex, str(cnt))
            console.print(ex_table)

    asyncio.run(_summary())


@app.command(name="check-gaps")
def check_gaps(
    timeframe: str = typer.Option("1d", help="Timeframe to inspect: '1d' or '15m'"),
    table: str = typer.Option("", help="Specific table (e.g. btc_usdt_usdt_on_bybit). Empty = scan all tables"),
    top: int = typer.Option(15, help="Show top N tables by missing candles"),
):
    """
    Detect missing candles (gaps) in stored tables — read-only diagnostic.
    """
    import datetime as _dt

    async def _scan():
        if timeframe == "15m":
            dbs = [("HIGH", settings.db_high_15m), ("LOW", settings.db_low_15m)]
        else:
            dbs = [("HIGH", settings.db_high_1d), ("LOW", settings.db_low_1d)]
        step = 900 if timeframe == "15m" else 86400
        results = []

        for tier, db_name in dbs:
            try:
                conn = await asyncpg.connect(
                    host=settings.db_host, port=settings.db_port,
                    user=settings.db_user, password=settings.db_password,
                    database=db_name,
                )
            except Exception as e:
                console.print(f"[yellow]Could not connect to {db_name}: {e}[/yellow]")
                continue

            if table:
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=$1)", table
                )
                tables = [(table, tier)] if exists else []
                if not tables:
                    console.print(f"[yellow]Table '{table}' not found in {db_name}[/yellow]")
            else:
                rows = await conn.fetch(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema='public' AND table_name LIKE '%_on_%'"
                )
                tables = [(r["table_name"], tier) for r in rows]

            for tbl, tier_label in tables:
                try:
                    brows = await conn.fetch(
                        f'SELECT DISTINCT "Timestamp" / {step} AS b FROM "{tbl}" ORDER BY b ASC'
                    )
                except Exception:
                    continue
                buckets = [int(r["b"]) for r in brows]
                if len(buckets) < 2:
                    continue

                missing_total = (buckets[-1] - buckets[0] + 1) - len(buckets)
                biggest, biggest_at = 0, buckets[0]
                prev = buckets[0]
                for b in buckets[1:]:
                    if b - prev - 1 > biggest:
                        biggest = b - prev - 1
                        biggest_at = prev
                    prev = b

                if missing_total > 0:
                    results.append({
                        "table": tbl,
                        "tier": tier_label,
                        "first": _dt.datetime.fromtimestamp(buckets[0] * step, tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                        "last": _dt.datetime.fromtimestamp(buckets[-1] * step, tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                        "stored": len(buckets),
                        "missing": missing_total,
                        "biggest": biggest,
                        "biggest_at": _dt.datetime.fromtimestamp(biggest_at * step, tz=_dt.timezone.utc).strftime("%Y-%m-%d"),
                    })
            await conn.close()
        return results

    results = asyncio.run(_scan())

    if not results:
        console.print(f"[bold green]✓ No gaps found ({timeframe}) — every table is continuous.[/bold green]")
        return

    results.sort(key=lambda r: r["missing"], reverse=True)
    out = Table(title=f"🔍 Candle Gaps ({timeframe.upper()}) — {len(results)} tables with missing data")
    out.add_column("Table", style="cyan")
    out.add_column("Tier", style="yellow")
    out.add_column("From", style="white")
    out.add_column("To", style="white")
    out.add_column("Stored", justify="right", style="green")
    out.add_column("Missing", justify="right", style="bold red")
    out.add_column("Biggest gap", justify="right", style="magenta")
    out.add_column("After", style="dim")
    for r in results[: top]:
        unit = "days" if timeframe == "1d" else "bars"
        out.add_row(
            r["table"], r["tier"], r["first"], r["last"],
            f"{r['stored']:,}", f"{r['missing']:,}", f"{r['biggest']:,} {unit}", r["biggest_at"],
        )
    console.print(out)
    console.print(
        "[dim]Gaps are auto-repaired on the next collector cycle (CHECK_AND_FILL_GAPS=1). "
        "Run `python main.py run` to trigger repair.[/dim]"
    )


@app.command(name="diagnose-pair")
def diagnose_pair(
    symbol: str = typer.Argument(..., help="Pair as shown in the dashboard, e.g. '0G/USDT' or '0G/USDT:USDT'"),
    exchange: str = typer.Option("bybit", help="Exchange id as used by the collector (bybit, gateio, okx, ...)"),
):
    """
    Explain WHY a pair's chart is stale: which tables exist in the 4 databases,
    how far behind each one is, and whether the collector still selects this
    exact symbol (perp-first) or writes to the perp table instead.
    """
    import time
    import datetime as _dt
    from src.exchanges.symbol_selector import (
        should_skip_pair,
        select_symbols_perp_first,
    )

    base = (symbol.split("/")[0] or "").upper()
    spot_sym = f"{base}/USDT"
    perp_sym = f"{base}/USDT:USDT"

    def _tbl(sym: str) -> str:
        return f"{sym.replace('/', '_').replace('-', '_')}_on_{exchange}".lower()

    candidates = {spot_sym: _tbl(spot_sym), perp_sym: _tbl(perp_sym)}

    async def _scan():
        out = []
        dbs = [
            ("1D HIGH", settings.db_high_1d), ("1D LOW", settings.db_low_1d),
            ("15M HIGH", settings.db_high_15m), ("15M LOW", settings.db_low_15m),
        ]
        for label, db_name in dbs:
            try:
                conn = await asyncpg.connect(
                    host=settings.db_host, port=settings.db_port,
                    user=settings.db_user, password=settings.db_password,
                    database=db_name, timeout=15,
                )
            except Exception as e:
                console.print(f"[yellow]Could not connect to {db_name}: {e}[/yellow]")
                continue
            try:
                for sym, tbl in candidates.items():
                    exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=$1)",
                        tbl,
                    )
                    if not exists:
                        continue
                    row = await conn.fetchrow(
                        f'SELECT MIN("Timestamp") AS mn, MAX("Timestamp") AS mx, COUNT(*) AS n FROM "{tbl}"'
                    )
                    mx = int(row["mx"]) if row and row["mx"] else 0
                    mn = int(row["mn"]) if row and row["mn"] else 0
                    if mx > 1e11:  # legacy ms-epoch table
                        mx //= 1000
                        mn //= 1000
                    age_h = (time.time() - mx) / 3600.0 if mx else float("inf")
                    out.append({
                        "db": label, "table": tbl, "symbol": sym,
                        "rows": int(row["n"]) if row else 0,
                        "first": _dt.datetime.fromtimestamp(mn, tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M") if mn else "-",
                        "last": _dt.datetime.fromtimestamp(mx, tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M") if mx else "-",
                        "age_h": age_h,
                    })
            finally:
                await conn.close()
        return out

    found = asyncio.run(_scan())

    t = Table(title=f"🔎 Tables for {base} @ {exchange}")
    t.add_column("Database", style="cyan")
    t.add_column("Table", style="white")
    t.add_column("Rows", justify="right", style="green")
    t.add_column("First", style="dim")
    t.add_column("Last (UTC)", style="bold")
    t.add_column("Behind", justify="right", style="magenta")
    if not found:
        console.print(f"[yellow]No table for {base} @ {exchange} in any of the 4 databases.[/yellow]")
    else:
        for r in sorted(found, key=lambda r: (r["db"], r["table"])):
            behind = "—" if r["age_h"] == float("inf") else f"{r['age_h']:.1f}h"
            style = "red" if r["age_h"] > 24 else "green"
            t.add_row(r["db"], r["table"], f"{r['rows']:,}", r["first"],
                      f"[{style}]{r['last']}[/{style}]", behind)
        console.print(t)

    # --- Would the collector still pick THIS symbol? ------------------------
    console.print("\n[bold]Collector selection check (perp-first):[/bold]")
    if should_skip_pair(symbol, exchange):
        console.print(f"  [red]✗ {symbol} is filtered out by should_skip_pair() "
                      f"(leveraged/stock/blacklisted token) — never collected.[/red]")
    try:
        import ccxt as _ccxt
        ex = getattr(_ccxt, settings.exchange_map_1d.get(exchange, exchange))({"enableRateLimit": True})
        markets = ex.load_markets()
        syms = [s for s in markets if s in (spot_sym, perp_sym)]
        selected = select_symbols_perp_first(syms, markets, exchange)
        console.print(f"  markets present: {', '.join(syms) or 'none'}")
        console.print(f"  collector would collect: [bold]{', '.join(selected) or 'nothing'}[/bold]")
        if perp_sym in markets and symbol.split(":")[0] == symbol:
            console.print(
                f"  [yellow]⚠ {spot_sym} is a SPOT leftover: a perpetual ({perp_sym}) exists, "
                f"so perp-first makes the collector write only {_tbl(perp_sym)}. "
                f"The spot table stays frozen forever — open the perp pair in the dashboard "
                f"(or drop the stale spot table).[/yellow]"
            )
    except Exception as e:
        console.print(f"  [yellow]ccxt check skipped: {e}[/yellow]")

    console.print(
        "\n[dim]15m engine serves: bybit, gateio, mexc, okx, bingx "
        f"(ALLOWED_EXCHANGES={','.join(settings.allowed_exchanges) or 'all'}, "
        f"EXCLUDED_EXCHANGES={','.join(settings.excluded_exchanges) or 'none'}).[/dim]"
    )


@app.command(name="check-continuity")
def check_continuity(
    table: str = typer.Argument(..., help="Table, e.g. 'jusung_usdt:usdt_on_gateio'"),
    timeframe: str = typer.Option("15m", help="Timeframe of the table: '15m' or '1d'"),
    last: int = typer.Option(60, help="How many latest candles to inspect"),
    threshold_pct: float = typer.Option(0.05, help="Report jumps larger than this % of price"),
):
    """
    Report candles whose OPEN does not continue the previous CLOSE.

    Tells apart the two reasons a chart can look broken:
      * the stored candles really jump (collector/exchange data), or
      * only the on-screen live bar did (browser-side poller).
    """
    import datetime as _dt

    step = 900 if timeframe == "15m" else 86400

    async def _read():
        dbs = (
            [settings.db_high_15m, settings.db_low_15m]
            if timeframe == "15m"
            else [settings.db_high_1d, settings.db_low_1d]
        )
        for db_name in dbs:
            try:
                conn = await asyncpg.connect(
                    host=settings.db_host, port=settings.db_port,
                    user=settings.db_user, password=settings.db_password,
                    database=db_name, timeout=15,
                )
            except Exception:
                continue
            try:
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=$1)",
                    table,
                )
                if not exists:
                    continue
                rows = await conn.fetch(
                    f'SELECT "Timestamp" AS ts, open, high, low, close, volume '
                    f'FROM "{table}" ORDER BY "Timestamp" DESC LIMIT {int(last)}'
                )
                return db_name, [dict(r) for r in reversed(rows)]
            finally:
                await conn.close()
        return None, []

    db_name, rows = asyncio.run(_read())
    if not rows:
        console.print(f"[yellow]Table '{table}' not found (or empty) for {timeframe}.[/yellow]")
        return

    out = Table(title=f"🔗 Continuity of last {len(rows)} {timeframe} candles — {table} @ {db_name}")
    out.add_column("Time (UTC)", style="cyan")
    out.add_column("Prev close", justify="right")
    out.add_column("Open", justify="right")
    out.add_column("Jump", justify="right", style="bold")
    out.add_column("Note", style="dim")

    breaks = 0
    missing = 0
    prev = None
    for r in rows:
        ts = int(r["ts"])
        if ts > 1e11:
            ts //= 1000
        when = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
        if prev is not None:
            gap_bars = (ts - prev["ts"]) // step - 1
            if gap_bars > 0:
                missing += gap_bars
                out.add_row(when, "", "", "", f"⛔ {gap_bars} candle(s) missing before this one")
            pc, op = float(prev["close"] or 0), float(r["open"] or 0)
            if pc > 0 and op > 0:
                jump = (op - pc) / pc * 100.0
                if abs(jump) >= threshold_pct:
                    breaks += 1
                    out.add_row(when, f"{pc:.8g}", f"{op:.8g}", f"{jump:+.3f}%", "open ≠ prev close")
        prev = {"ts": ts, "close": r["close"]}

    if breaks or missing:
        console.print(out)
    console.print(
        f"[bold]{breaks}[/bold] open/close break(s) ≥ {threshold_pct}% and "
        f"[bold]{missing}[/bold] missing candle(s) in the last {len(rows)} bars."
    )
    if not breaks and not missing:
        console.print(
            "[green]✓ Stored candles are continuous — a gap seen in the dashboard came "
            "from the browser-side live bar, not from the database.[/green]"
        )


def _fmt_names(names: list[str], limit: int = 8) -> str:
    """First few table names, then how many more. Enough to undo from, short
    enough to read on a terminal that is already showing a 20-row table."""
    if not names:
        return ""
    return ", ".join(names[:limit]) + (f" … +{len(names) - limit} more" if len(names) > limit else "")


@app.command(name="prune-zombie-spots")
def prune_zombie_spots(
    timeframe: str = typer.Option("all", help="'1d', '15m' or 'all' — 'all' covers all 4 databases"),
    apply: bool = typer.Option(False, "--apply", help="Run the DDL. Without it nothing is touched (dry-run)"),
    mode: str = typer.Option("trash", help="'trash' = move into schema zombie_pruned (reversible), 'drop' = DROP TABLE"),
    stale_hours: float = typer.Option(24.0, help="Only prune a spot table idle this long (0 = ignore freshness)"),
    exact: bool = typer.Option(True, "--exact/--estimate-only", help="COUNT(*) the candidate pairs, or trust catalog estimates"),
    concurrency: int = typer.Option(8, help="Parallel COUNT(*) queries — the collector is writing, be polite"),
    limit: int = typer.Option(0, help="Prune at most N tables this run (0 = all). Use a small N as a canary"),
    max_fraction: float = typer.Option(0.35, help="Refuse to prune more than this share of a database's pair tables without --yes"),
    yes: bool = typer.Option(False, "--yes", help="Accept the mass-prune guard"),
    report: str = typer.Option("", help="Write every decision as JSON to this file"),
    purge_parked_tables: bool = typer.Option(
        False, "--purge-parked",
        help="Only report (and with --yes, DROP) what is already parked in the "
             "zombie_pruned schema — frees the disk a previous --apply kept",
    ),
):
    """
    Delete (or park) spot pair tables that their own perp outgrew.

    Rule: for a perp BASE/USDT:USDT on exchange E, if the spot BASE/USDT table
    on E in the same database holds FEWER bars, the spot table is a leftover of
    the perp-first switch — it is never written again, and it still costs every
    scan, summary and dashboard query its attention. Perp tables, non-pair
    tables and spots with no perp counterpart are never touched, and a table
    that is still being collected is kept even when it is smaller.

    Reads all 4 databases (HIGH/LOW × 15m/1D) by default. DRY-RUN unless
    --apply, and the default action is a reversible schema move.
    """
    import json as _json
    import time as _time

    from src.utils.zombie_prune import (
        ZOMBIE_SCHEMA,
        apply_pruning,
        candidate_tables,
        measure,
        over_prune_guard,
        plan_pruning,
        prune_order,
        purge_parked,
        read_pair_tables,
        summarize,
    )

    zombie_schema = ZOMBIE_SCHEMA

    if purge_parked_tables and not apply:
        console.print(
            "[red]--purge-parked deletes tables, so it needs --apply too[/red] "
            "(without it: nothing is dropped, and that is the point of asking)."
        )
        raise typer.Exit(2)
    if mode not in ("trash", "drop"):
        console.print("[red]--mode must be 'trash' or 'drop'[/red]")
        raise typer.Exit(2)
    if apply and not exact:
        console.print(
            "[red]--apply needs --exact counts.[/red] Catalog estimates (reltuples) are "
            "stale until ANALYZE and would decide deletions on a guess."
        )
        raise typer.Exit(2)

    if timeframe == "all":
        targets = [("15m/HIGH", settings.db_high_15m), ("15m/LOW", settings.db_low_15m),
                   ("1D/HIGH", settings.db_high_1d), ("1D/LOW", settings.db_low_1d)]
    elif timeframe == "15m":
        targets = [("15m/HIGH", settings.db_high_15m), ("15m/LOW", settings.db_low_15m)]
    elif timeframe == "1d":
        targets = [("1D/HIGH", settings.db_high_1d), ("1D/LOW", settings.db_low_1d)]
    else:
        console.print("[red]--timeframe must be '1d', '15m' or 'all'[/red]")
        raise typer.Exit(2)

    all_rows, total_prune = [], 0

    async def _scan_db(label, db_name):
        nonlocal total_prune
        try:
            # A pool, not one connection: COUNT(*) runs concurrently, and a
            # single connection cannot hold concurrent transaction blocks.
            pool = await asyncpg.create_pool(
                host=settings.db_host, port=settings.db_port,
                user=settings.db_user, password=settings.db_password, database=db_name,
                min_size=1, max_size=max(1, int(concurrency)),
            )
        except Exception as e:
            console.print(f"[yellow]{label} {db_name}: cannot connect ({e}) — skipped[/yellow]")
            return
        try:
            conn = await pool.acquire()
            if purge_parked_tables:
                # A different question than the rest of the command: not "what
                # should be moved" but "what did a previous run move, and is it
                # now safe to give the disk back".
                res = await purge_parked(conn, drop=yes)
                listed = [r["spot"] for r in res if r.get("listed")]
                gone = [r["spot"] for r in res if r["ok"] and not r.get("listed")]
                bad = [r for r in res if not r["ok"]]
                if not res:
                    console.print(f"[dim]{label} {db_name}: nothing parked in "
                                  f"{zombie_schema}, nothing to free[/dim]")
                elif listed:
                    console.print(
                        f"[yellow]{label} {db_name}: {len(listed)} table(s) parked in "
                        f"{zombie_schema}[/yellow] — they cost disk, not scan time; "
                        f"re-run with --yes to drop them for good"
                    )
                    console.print(f"[dim]{_fmt_names(listed)}[/dim]")
                else:
                    console.print(
                        f"[bold]dropped {len(gone)} parked table(s) of {db_name}"
                        + (f", [red]{len(bad)} failed[/red]" if bad else "") + "[/bold]"
                    )
                    console.print(f"[dim]{_fmt_names(gone)}[/dim]")
                for r in bad:
                    console.print(f"[red]{r['spot']}: {r['error'][:160]}[/red]")
                all_rows.extend({"db": db_name, "tier": label, **r} for r in res)
                return
            tables = await read_pair_tables(conn)
            if not tables:
                console.print(f"[dim]{label} {db_name}: no pair tables found[/dim]")
                return
            to_count, n_tables = candidate_tables(tables)
            if not to_count:
                console.print(f"[green]\u2713 {label} {db_name}: {n_tables} pair tables, "
                              f"no spot/perp duplicates[/green]")
                return
            if not exact:
                counts = {tb: {"bars": tables.get(tb), "last_ts": None} for tb in to_count}
            else:
                console.print(f"[dim]{label} {db_name}: counting {len(to_count)} tables "
                              f"(both sides of {len(to_count) // 2} duplicate(s))\u2026[/dim]")
                t0 = _time.time()
                counts = await measure(conn, to_count, concurrency=concurrency)
                console.print(f"[dim]{label}: counted in {_time.time() - t0:.1f}s[/dim]")
            plan = plan_pruning({t: tables.get(t, 0) for t in to_count}, counts,
                                stale_sec=max(0.0, float(stale_hours)) * 3600.0)
            summ = summarize(plan)
            total_prune += summ["prune"]
            for row in plan:
                row["db"] = db_name
                row["tier"] = label
            all_rows.extend(plan)
            console.print(
                f"\n[bold]{label}[/bold] {db_name}: {n_tables} pair tables, "
                f"{summ['candidates']} spot/perp duplicate(s) → "
                f"[red]{summ['prune']} to prune[/red], {summ['keep']} kept, "
                f"{summ['unknown']} unknown"
            )
            if plan:
                out = Table("spot table", "spot bars", "perp bars", "spot idle",
                            "perp idle", "verdict", "why",
                            title=f"{label}: decisions (first 20 of {len(plan)})")
                for row in plan[:20]:
                    idle = "—" if row["spot_idle_sec"] is None else f"{row['spot_idle_sec'] / 3600:.1f}h"
                    p_idle = ("—" if row.get("perp_idle_sec") is None
                              else f"{row['perp_idle_sec'] / 3600:.1f}h")
                    out.add_row(
                        row["spot"],
                        "?" if row["spot_bars"] is None else f"{row['spot_bars']}",
                        "?" if row["perp_bars"] is None else f"{row['perp_bars']}",
                        idle,
                        p_idle,
                        row["verdict"],
                        row["reason"][:70],
                    )
                console.print(out)
            if apply and summ["prune"]:
                actions = prune_order(plan)
                if limit and limit > 0:
                    actions = actions[:limit]
                refusal = None if yes else over_prune_guard(len(actions), n_tables, max_fraction)
                if refusal:
                    console.print(f"[red]{refusal}[/red]")
                    # Say so in the report too: the JSON is what an operator
                    # re-reads afterwards, and "prune" rows there must not look
                    # like work that was already done.
                    all_rows.append({"db": db_name, "tier": label, "tables": len(actions),
                                     "verdict": "refused-by-guard", "reason": refusal})
                    return
                res = await apply_pruning(conn, actions, mode)
                moved = [r for r in res if r.get("spot")]
                ok = sum(1 for r in moved if r["ok"])
                bad = [r for r in res if not r["ok"]]
                console.print(
                    f"[bold]{mode}[/bold]ed {ok}/{len(actions)} table(s) of {db_name}"
                    + (f", [red]{len(bad)} failed[/red]: {bad[0]['error'][:120]}" if bad else "")
                )
                # Which ones. "trashed 5/5" is not an audit trail: an operator who
                # wants to undo a run must not have to re-derive the list from the
                # report file they may not even have passed.
                names = [r["spot"] for r in moved if r["ok"]]
                if names:
                    console.print(f"[dim]{_fmt_names(names)}[/dim]")
                if limit and limit > 0 and summ["prune"] > len(actions):
                    console.print(
                        f"[dim]{summ['prune'] - len(actions)} further candidate(s) of "
                        f"{db_name} were left alone by --limit[/dim]"
                    )
                for r in bad:
                    r["db"] = db_name
                    r.setdefault("spot", None)
                all_rows.extend({"db": db_name, "sql": r["sql"], "ok": r["ok"],
                                 "error": r.get("error")} for r in bad)
        finally:
            await pool.release(conn)
            await pool.close()

    async def _run():
        for label, db in targets:
            await _scan_db(label, db)

    asyncio.run(_run())

    if report:
        with open(report, "w", encoding="utf-8") as fh:
            _json.dump(all_rows, fh, ensure_ascii=False, indent=1, default=str)
        console.print(f"[dim]report → {report}[/dim]")

    if not apply:
        console.print(
            f"\n[yellow]Dry-run: {total_prune} table(s) would be pruned. Nothing was changed.[/yellow] "
            f"Re-run with --apply (default mode=trash, reversible: "
            f"ALTER TABLE zombie_pruned.<table> SET SCHEMA public)."
        )
    else:
        console.print(
            "[green]Done.[/green] Press 🔄 Refresh data in the dashboard (or wait for "
            "DASH_SCAN_INVENTORY_TTL_SEC) for the pair list to shrink."
        )


if __name__ == "__main__":
    app()
