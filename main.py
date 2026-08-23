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


if __name__ == "__main__":
    app()
