#!/usr/bin/env python3
"""
================================================================================
Growth Scan — pairs that rose more than X% from the lowest LOW of the last N
days to the current CLOSE, printed to the terminal with detailed market info.
================================================================================

Standalone, NOT connected to the dashboard. It reads the pair tables the
collector has already stored in TimescaleDB, computes, for every pair:

        growth % = (current_close - min_low_over_window) / min_low * 100

and prints only the pairs whose growth is >= --pct. For each match it then
watches the live exchange orderbook + trade tape (spread, depth, CVD, vitality)
and prints the trading links (spot / swap) on that exchange.

Defaults: last 5 days, rise >= 100%. Both are adjustable.

Usage (from the repo root, after `poetry install`):

    poetry run python scan_growth.py                 # 5 days, >=100%
    poetry run python scan_growth.py --days 7 --pct 150
    poetry run python scan_growth.py --timeframe 1d
    poetry run python scan_growth.py --timeframe 15m --days 3 --pct 80
    poetry run python scan_growth.py --exchanges bybit,gateio --top 20
    poetry run python scan_growth.py --no-orderbook --top 40
    poetry run python scan_growth.py --min-usd 250000 --sort volume

DB credentials come from the same `config/settings` (`.env` / `db_config.py`).
The exchange fetches use `SOCKS5_PROXY` like the engines do (set `--proxy ''`
to force direct), and markets are loaded per exchange once, with a hard timeout,
so a slow market list does not hang the scan.
================================================================================
"""

import argparse
import asyncio
import datetime as dt
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import asyncpg
from rich.console import Console
from rich.panel import Panel

# Add the repo root to sys.path so `config` and `src` import regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings  # noqa: E402
from src.analytics.atr_filtered import compute_atr_no_paranormal_bars  # noqa: E402
from src.analytics.orderbook import fetch_orderbook_snapshot  # noqa: E402
from src.exchanges.client import close_exchange_safely, create_exchange  # noqa: E402
from src.exchanges.symbol_selector import get_exchange_url, get_swap_url  # noqa: E402

console = Console()

# ---------------------------------------------------------------------------
# Settings we reuse
# ---------------------------------------------------------------------------
DBS = [
    ("1D/HIGH", settings.db_high_1d),
    ("1D/LOW", settings.db_low_1d),
    ("15M/HIGH", settings.db_high_15m),
    ("15M/LOW", settings.db_low_15m),
]

# engine-name -> ccxt id (1D), and the reverse (ccxt id -> engine name), so we
# can resolve the table suffix whether it is `..._on_gate` (1D) or
# `..._on_gateio` (15M) back to both a human label and a ccxt id.
_EX_MAP = dict(settings.exchange_map_1d)
_CCXT_TO_EX = {}
for _k, _v in _EX_MAP.items():
    _CCXT_TO_EX.setdefault(str(_v).lower(), _k)
_KNOWN_CCXT_IDS = set(str(v).lower() for v in _EX_MAP.values()) | set(
    str(v).lower() for v in settings.exchange_map_15m.values()
)

_PAIR_RE = re.compile(r"^[A-Za-z0-9_\-\$.]{1,64}:[A-Za-z0-9]{1,16}$|^[A-Za-z0-9_\-\$.]{1,64}$")
_EX_RE = re.compile(r"^[A-Za-z0-9]{1,16}$")
_UNITS_MS = 100000000000  # epoch values >= this are milliseconds


def resolve_exchange(suffix: str) -> Tuple[str, str]:
    """(engine-name, ccxt-id) for a table suffix like 'gateio' or 'gate'.

    `_on_gate` is what the 1D engine writes (ccxt id); `_on_gateio` is what the
    15M engine writes (engine name). Both must resolve to the SAME ccxt id and
    to the SAME human label used by `get_exchange_url` / `get_swap_url` (which
    expect the engine name, e.g. 'gateio', not 'gate').
    """
    s = str(suffix or "").lower()
    # engine name -> ccxt id (the common case: gateio, bybit, mexc, ...)
    if s in _EX_MAP:
        return s, str(_EX_MAP[s])
    if s in _CCXT_TO_EX:
        return _CCXT_TO_EX[s], s
    if s in _KNOWN_CCXT_IDS:
        return s, s
    return s, s


def ticker_from_table(table: str) -> Tuple[str, str, str]:
    """Infer (ticker, exchange-label, ccxt-id) from a stored table name.

    The collector writes `{symbol.replace('/', '_').replace('-', '_')}_on_{ex}`
    lowercased, so `btc_usdt:usdt_on_bybit` -> `BTC/USDT:USDT` @ bybit, and
    `1000000babydoge_usdt_on_gateio` -> `1000000BABYDOGE/USDT` @ gateio.

    The exchange id is the LAST `_on_`-separated component; the pair part is
    everything before it (spot `BASE_QUOTE`, perp `BASE_QUOTE:SETTLE`). The
    tables are split from the right so a pair name that itself contains `_on_`
    (a base with an underscore in its symbol, e.g. gate's internal ids) does not
    break the parse.
    """
    t = (table or "").strip().lower()
    if "_on_" not in t:
        return "", "", ""
    pair, ex_suffix = t.rsplit("_on_", 1)
    if not _EX_RE.match(ex_suffix) or not _PAIR_RE.match(pair):
        return "", "", ""
    ex_label, ccxt_id = resolve_exchange(ex_suffix)
    if ":" in pair:
        base_quote, settle = pair.split(":", 1)
        base, quote = base_quote.split("_", 1)
        ticker = f"{base.upper()}/{quote.upper()}:{settle.upper()}"
    else:
        base, quote = pair.split("_", 1)
        ticker = f"{base.upper()}/{quote.upper()}"
    return ticker, ex_label, ccxt_id


# ---------------------------------------------------------------------------
# Growth query (one aggregate row per table, cheap)
# ---------------------------------------------------------------------------
def _growth_sql(table: str, cutoff: int) -> str:
    """One row per table: current close/ts, and min low + its ts in the window.

    `Timestamp` is seconds in most tables but milliseconds in legacy ones, so
    the cutoff is matched in both units (seconds directly, ms via cutoff*1000
    only when the value is in the ms range).
    """
    _t = (table or "").strip().lower()
    if "_on_" not in _t:
        return ""
    _p, _ex = _t.rsplit("_on_", 1)
    if not _EX_RE.match(_ex) or not _PAIR_RE.match(_p):
        return ""
    win = (
        f'"Timestamp" >= {cutoff}'
        f' OR ("Timestamp" >= {_UNITS_MS} AND "Timestamp" >= {cutoff * 1000})'
    )
    return (
        f'SELECT '
        f'(SELECT "close" FROM "{table}" ORDER BY "Timestamp" DESC LIMIT 1) AS cur_close, '
        f'(SELECT "Timestamp" FROM "{table}" ORDER BY "Timestamp" DESC LIMIT 1) AS last_ts, '
        f'(SELECT MIN("low") FROM "{table}" WHERE {win}) AS min_low, '
        f'(SELECT "Timestamp" FROM "{table}" WHERE {win} '
        f'ORDER BY "low" ASC NULLS LAST LIMIT 1) AS min_ts, '
        f'(SELECT COALESCE(SUM("volume"), 0) FROM "{table}" WHERE {win}) AS vol'
    )


async def scan_tables(
    db_label: str, db_name: str, days: int, pool_size: int = 6
) -> List[dict]:
    """Scan one database and return candidate rows (growth >= 0) for `days`."""
    cutoff = int(time.time()) - int(days) * 86400
    out: List[dict] = []
    try:
        pool = await asyncpg.create_pool(
            host=settings.db_host, port=settings.db_port,
            user=settings.db_user, password=settings.db_password,
            database=db_name, min_size=1, max_size=pool_size,
            command_timeout=30, timeout=15,
        )
    except Exception as e:
        console.print(f"[yellow]scan_growth[/yellow] cannot connect to {db_name}: "
                      f"{type(e).__name__}: {e}")
        return out
    try:
        rows = await pool.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name LIKE '%_on_%'"
        )
        tables = [r["table_name"] for r in rows]
        if not tables:
            console.print(f"[dim]{db_label} {db_name}: no pair tables[/dim]")
            return out
        sem = asyncio.Semaphore(pool_size)

        async def _one(tbl):
            sql = _growth_sql(tbl, cutoff)
            if not sql:
                return
            async with sem:
                try:
                    async with pool.acquire() as conn:
                        r = await conn.fetchrow(sql)
                except Exception as e:
                    console.print(f"[dim]  {db_label} {tbl}: {type(e).__name__}: {e}[/dim]")
                    return
            cur, mn = r["cur_close"], r["min_low"]
            try:
                cur = float(cur)
                mn = float(mn)
            except (TypeError, ValueError):
                return
            if cur <= 0 or mn <= 0:
                return
            growth = (cur - mn) / mn * 100.0
            out.append({
                "db_label": db_label, "db_name": db_name, "table": tbl,
                "growth": growth, "cur_close": cur, "min_low": mn,
                "min_ts": r["min_ts"], "last_ts": r["last_ts"],
                "vol": float(r["vol"] or 0.0),
            })

        await asyncio.gather(*[_one(t) for t in tables])
    finally:
        await pool.close()
    return out


# ---------------------------------------------------------------------------
# Live per-pair detail (orderbook / tape / ATR)
# ---------------------------------------------------------------------------
async def _atr_from_table(db_name: str, table: str, period: int) -> float:
    """Filtered ATR from the stored candles of `table` (the engine's estimator)."""
    try:
        pool = await asyncpg.create_pool(
            host=settings.db_host, port=settings.db_port,
            user=settings.db_user, password=settings.db_password,
            database=db_name, min_size=1, max_size=2, command_timeout=15, timeout=10,
        )
    except Exception:
        return 0.0
    try:
        limit = max(int(period) * 3, 30)
        rows = await pool.fetch(
            f'SELECT "Timestamp" AS ts, high, low, close FROM "{table}" '
            f'ORDER BY "Timestamp" DESC LIMIT {limit}'
        )
    except Exception:
        return 0.0
    finally:
        await pool.close()
    if not rows:
        return 0.0
    rows = list(reversed(rows))
    return compute_atr_no_paranormal_bars(
        highs=[r["high"] for r in rows],
        lows=[r["low"] for r in rows],
        closes=[r["close"] for r in rows],
        period=int(period),
        small_threshold=settings.atr_small_threshold,
        large_threshold=settings.atr_large_threshold,
    )


async def _orderbook_detail(exchange, symbol: str, db_name: str, table: str,
                            period: int) -> dict:
    """Orderbook + tape snapshot, with an ATR baseline from the stored candles."""
    atr = await _atr_from_table(db_name, table, period)
    detail = {}
    try:
        detail = await fetch_orderbook_snapshot(
            exchange=exchange,
            symbol=symbol,
            atr_no_paranormal=atr,
            fetch_limit=settings.ob_fetch_limit,
            trades_limit=settings.ob_trades_limit,
            trades_window_sec=settings.ob_trades_window_sec,
            depth_pct=settings.ob_depth_pct,
            fallback_limits=settings.ob_fallback_limits,
            timeout_sec=8.0,
        )
    except Exception as e:
        detail = {"error": f"{type(e).__name__}: {e}"}
    if isinstance(detail, dict) and "error" not in detail:
        detail["ob_atr_no_paranormal"] = round(float(atr or 0.0), 10)
    return detail


async def _load_markets_quiet(exchange, timeout: float = 60.0) -> bool:
    """load_markets with a hard timeout; True/False = has markets."""
    from src.utils.timeouts import hard_wait_for
    try:
        await hard_wait_for(exchange.load_markets(), timeout, label="load_markets")
        return bool(getattr(exchange, "markets", None))
    except Exception as e:
        console.print(f"[dim]  market load failed ({type(e).__name__})[/dim]")
        return False


def _fmt_link(url: str) -> str:
    return url if url else "(no link)"


def _fmt_growth(g: float, pct: float) -> str:
    color = "green" if g >= pct else "yellow"
    return f"[{color}]+{g:,.1f}%[/{color}]"


def _fmt_ts(ts) -> str:
    if not ts:
        return "—"
    try:
        ts = int(ts)
        if ts >= _UNITS_MS:
            ts //= 1000
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return "—"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run(args) -> None:
    days = int(args.days)
    pct = float(args.pct)
    tf_filter = args.timeframe.lower()
    ex_filter = {e.strip().lower() for e in (args.exchanges or "").split(",") if e.strip()}

    db_targets = [
        (label, name) for label, name in DBS
        if tf_filter == "all"
        or (tf_filter == "1d" and label.startswith("1D"))
        or (tf_filter == "15m" and label.startswith("15M"))
    ]

    console.print(Panel(
        f"Scanning {db_targets and len(db_targets) or 0} database(s) · window "
        f"[bold]{days}d[/bold] · min rise [bold]{pct:g}%[/bold]"
        + (f" · exchanges: {args.exchanges}" if args.exchanges else ""),
        title="📈 Growth Scan", style="bold cyan",
    ))

    # Pass 1: cheap per-table growth.
    all_cands: List[dict] = []
    for label, name in db_targets:
        console.print(f"[cyan]Scanning {label} [bold]{name}[/bold]…[/cyan]")
        t0 = time.time()
        cands = await scan_tables(label, name, days)
        all_cands.extend(cands)
        console.print(f"  [dim]{time.time() - t0:.1f}s · {len(cands)} table(s)[/dim]")

    # Filter to candidates above the bar.
    hits = [c for c in all_cands if c["growth"] >= pct]
    if args.min_usd and args.min_usd > 0:
        hits = [c for c in hits if c["vol"] >= float(args.min_usd)]
    if not hits:
        console.print("[bold red]No pairs matched[/bold red] — lower --pct or raise --days.")
        return

    if ex_filter:
        keep = []
        for c in hits:
            _t, ex_label, _ccxt = ticker_from_table(c["table"])
            if ex_label.lower() in ex_filter or _ccxt.lower() in ex_filter:
                keep.append(c)
        hits = keep

    sort_key = {"growth": lambda c: c["growth"], "volume": lambda c: c["vol"], "close": lambda c: c["cur_close"]}[args.sort]
    hits.sort(key=sort_key, reverse=True)
    if args.top and args.top > 0:
        hits = hits[: int(args.top)]

    console.print(f"[bold green]✓ {len(hits)} pair(s) rose ≥ {pct:g}% in {days}d[/bold green]")

    # Pass 2: live detail (orderbook / tape / links) per match.
    # Group matches by exchange so we load markets once per exchange.
    by_ex: Dict[str, List[dict]] = {}
    for c in hits:
        _t, ex_label, ccxt_id = ticker_from_table(c["table"])
        by_ex.setdefault((ex_label, ccxt_id), []).append(c)

    exchanges: dict = {}
    if not args.no_orderbook:
        console.print("[dim]Loading exchange markets (one per exchange)…[/dim]")
        for (ex_label, ccxt_id) in by_ex:
            try:
                ex = create_exchange(ccxt_id, use_proxy=not args.noproxy)
            except Exception as e:
                console.print(f"[yellow]  {ex_label}: cannot build client ({e})[/yellow]")
                continue
            ok = await _load_markets_quiet(ex)
            if not ok:
                console.print(f"[yellow]  {ex_label}: no markets — skipping live detail[/yellow]")
                await close_exchange_safely(ex, ex_label)
                exchanges[(ex_label, ccxt_id)] = None
                continue
            exchanges[(ex_label, ccxt_id)] = ex

    rank = 0
    for c in hits:
        rank += 1
        ticker, ex_label, ccxt_id = ticker_from_table(c["table"])
        ex_label, ccxt_id = resolve_exchange(c["table"].rsplit("_on_", 1)[-1])
        spot_url = get_exchange_url(ex_label, ticker)
        swap_url = get_swap_url(ex_label, ticker)

        lines = []
        lines.append(f"📅 [{_fmt_ts(c['last_ts'])}] current close  [bold]{c['cur_close']:,.8g}[/bold]")
        lines.append(f"📉 lowest low ({days}d)  [bold]{c['min_low']:,.8g}[/bold] on {_fmt_ts(c['min_ts'])}")
        lines.append(f"📊 volume ({days}d)  [bold]${c['vol']:,.0f}[/bold]")
        lines.append(f"🔗 Spot  {_fmt_link(spot_url)}")
        lines.append(f"🔗 Swap  {_fmt_link(swap_url)}")

        ex = exchanges.get((ex_label, ccxt_id)) if not args.no_orderbook else None
        if ex is not None:
            detail = await _orderbook_detail(ex, ticker, c["db_name"], c["table"],
                                             settings.atr_period)
            if detail.get("error"):
                lines.append(f"[red]orderbook: {detail['error']}[/red]")
            elif detail:
                atr = detail.get("ob_atr_no_paranormal") or 0.0
                lines.append(f"📘 bid {detail['ob_best_bid']:,.8g} / ask {detail['ob_best_ask']:,.8g} "
                             f"· spread {detail['ob_spread_abs']:,.8g} "
                             f"({detail['ob_spread_pct']:.3f}%)"
                             + (f" · ATR {atr:.6g}" if atr else ""))
                lines.append(f"💧 Depth ±{settings.ob_depth_pct:g}%: "
                             f"${detail['ob_bid_depth_usd']:,.0f} / ${detail['ob_ask_depth_usd']:,.0f} "
                             f"= [bold]${detail['ob_total_depth_usd']:,.0f}[/bold]"
                             f" · imbalance {detail['ob_imbalance']:.2f}")
                lines.append(f"⚡ {detail['ob_trades_per_min']:.1f} trades/min · "
                             f"CVD 5m {detail['ob_cvd_5m']:,.0f} · "
                             f"buy {detail['ob_buy_pressure_pct']:.0f}%")
                lines.append(f"❤️ Vitality {detail['ob_vitality_score']:.1f}/10 "
                             f"[bold]{detail['ob_vitality_grade']}[/bold]"
                             + (" · barcode" if detail["ob_is_barcode"] else ""))
        elif ex is None and not args.no_orderbook:
            lines.append("[dim](no live data — exchange markets unavailable)")

        title = (f"[bold]#{rank}[/bold] {ticker} [bold cyan]@ {ex_label}[/bold cyan] "
                 f"· {_fmt_growth(c['growth'], pct)} · {c['db_label']}")
        console.print(Panel("\n".join(lines), title=title, border_style="blue"))

    # cleanup
    for ex in exchanges.values():
        if ex is not None:
            await close_exchange_safely(ex, "scan_growth")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Find pairs that rose > X% from the lowest low of the last N days.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--days", type=int, default=5, help="Look-back window in days")
    p.add_argument("--pct", type=float, default=100.0, help="Minimum rise from the low, %%")
    p.add_argument("--timeframe", choices=["all", "1d", "15m"], default="all",
                   help="Which group of databases to scan")
    p.add_argument("--exchanges", type=str, default="",
                   help="Comma-separated exchange filter (bybit,gateio,…)")
    p.add_argument("--top", type=int, default=0, help="Show at most N results (0 = all)")
    p.add_argument("--sort", choices=["growth", "volume", "close"], default="growth",
                   help="Sort key for the results")
    p.add_argument("--min-usd", type=float, default=0.0,
                   help="Only show pairs with window volume >= this many USD")
    p.add_argument("--no-orderbook", action="store_true",
                   help="Skip live orderbook/tape fetches (DB-only scan)")
    p.add_argument("--noproxy", action="store_true",
                   help="Force direct exchange connections (no SOCKS5_PROXY)")
    args = p.parse_args()

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
