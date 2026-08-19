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


if __name__ == "__main__":
    app()
