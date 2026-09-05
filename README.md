# ⚡ Timescale Crypto OHLCV Collector

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TimescaleDB](https://img.shields.io/badge/Database-TimescaleDB-yellow.svg)](https://www.timescale.com/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-cyan.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Async Crypto Market Data Collector (**Perpetual Swaps & Spot Markets**) with Filtered ATR (w/o Paranormal Bars), L2 Orderbook Depth, CVD, and TimescaleDB Storage. Supports **9+ crypto exchanges** with automated time partitioning, **Columnar Compression**, and interactive **TradingView Lightweight Charts** dashboard.

---

## 🌟 Key Features

(Working on this repo with an AI assistant? [`AGENTS.md`](AGENTS.md) records the standing
contract for `dashboard/`, the invariants that each cost a debugging round, and the dead ends
that must not be re-opened.)

* 🔄 **Perp-First & Spot Fallback Architecture:** Automatically prefers linear perpetual swaps (`BTC/USDT:USDT`) for each base asset, seamlessly falling back to spot (`BTC/USDT`) when no perpetual contract exists.
* 📊 **ATR without Paranormal Bars (Filtered Robust ATR):** Robust volatility calculation that filters out news spikes, squeezes, and abnormal outlier candles outside $[0.5 \times \text{ATR}, 1.8 \times \text{ATR}]$. The dashboard labels every one of them, because three estimators live here: `1D_ATR(N)` (mean of True Range over N closed **daily** candles — the strip and the metric cards take N from the sidebar's *ATR Period (daily bars)*, the daily tables from `ATR_PERIOD`) and `15m_ATR(N)` (Gerchik-smoothed over N **15-minute** bars, written by the 15m engine into `ob_gerchik_atr`). Labels are built by `format_atr_label()` from the values actually used, so moving the slider renames the chip; a stored column never silently claims to follow it.
* ⚡ **TimescaleDB Hypertables:** Time-series storage with automated chunk partitioning and **Columnar Compression**, reducing DB disk footprint by up to 90%+.
* 🛡️ **Gap-Filling & Backfill:** Automatic detection and backfilling of missing daily candle ranges, combined with deep historical backfill from 2018.
* 📈 **L2/L3 Orderbook & Trade Tape Snapshots:** Computes spread tightness (% relative to ATR without paranormal bars), orderbook depth (Bid/Ask/Total USD), Cumulative Volume Delta (CVD & 5m CVD), and liquidity grade (**Vitality Score A–F**).
* 🔀 **Dynamic Liquidity Tiering:** Classifies coins into `HIGH` ($\ge \$500,000$ USD/day) and `LOW` volume tiers.
* 🌐 **Interactive Web Dashboard:** Streamlit dashboard featuring Plotly & **TradingView Lightweight Charts** (OHLCV Bars & Candlesticks) with ATR & liquidity metrics and multi-exchange filters.
* 🖱 **Instant Pair Switching:** the Charts tab paints the candles already stored in TimescaleDB first — gap stitching (missing candles fetched from the exchange, in-memory) and the live ticker / orderbook / trade-tape feeds run in background daemon threads and are swapped in when they land — and a catch-up range the exchange refused to answer is reported as a failure and retried after `DASH_STITCH_RETRY_SEC` rather than being cached for an hour as if it were the answer, because "I could not fetch it" and "there are no candles there" are different facts and only the second one may be drawn as a flat bar. A swap repaints only when it actually filled candles, so browsing healthy pairs never resets your zoom. Prefetching is deliberately small and lazy: warming a pair also primes the DB-only page, fetches at most 3 missing ranges per timeframe as a head start (the rest is fetched when the pair is actually opened), and skips exchange traffic entirely for pairs whose collector stopped writing more than `DASH_WARM_STALE_SKIP_SEC` ago — the dead spot tables left by a spot→perp migration, whose hundreds of missing candles used to make the prefetch slower than the click it was meant to speed up.
* 🩹 **The dashboard wakes the collector, and it closes the gap:** opening a pair registers it in a tiny `dashboard_priority_pairs` table, and the 15m **and** 1D engines each refresh that pair on their own timeframe in the database where its table actually lives. That refresh used to fetch `limit=10` bars anchored near `now` ("the lane is deliberately light — no gap scan"), which on a stale table refreshed the tail *around* the hole and left the chart with a date gap forever — on the daily chart even more visibly, since a day off is a whole missing candle. Now `lane_since_sec` anchors the fetch at the table's own last bar and grows `limit` until the hole is bridged, as long as the hole is at most `PRIORITY_LANE_CATCHUP_MAX_BARS` bars (2000: ~20 days of 15m, 5 years of 1D — deeper than that is history, and the sweep owns history; `0` restores the old tail-only behaviour). Still one request per tick, still one bar back so the forming candle is REWRITTEN (the writer's `DELETE >= min_ts` only covers it if the fetch returns it), still never creating a table. A bridge longer than a few bars logs `[LANE] ⚡ wrote N bar(s) across a Xh hole`, and a fetch that fails logs a rate-limited `[LANE] ⚠️ … fetch failed … — the gap stays open until this works` instead of returning zero into the void.
Each engine also stamps a heartbeat **per timeframe** (`served_15m_at` / `served_1d_at`, batched and rate-limited) on the pairs it actually serviced, and the dashboard reads it back, so a chart whose wake-up nobody answered says exactly that — "⛔ no 15m engine is refreshing this pair (lane last answered 2.5h ago) — the hole stays in the database until `python main.py run --timeframe 15m` is up" — instead of quietly re-painting the same gap and looking like a stitching bug. Per-timeframe matters: one shared stamp made `main.py run` (1D only) look like a healthy lane while every 15m chart stayed stale. The DDL that grows these columns is throttled and re-runs itself if the table turns out to be missing — and it is deliberately keyed process-wide, not per connection: a `weakref` memo of connections raised `cannot create weak reference to 'PoolConnectionProxy' object` inside the lane's own read and killed the lane on every tick.
* 🔌 **No silent market-load storm:** every dashboard feed (live chips, orderbook, trade tape, the in-chart poller's REST fallback, and the chart's gap stitching) shares ONE ccxt instance per exchange, and that instance's `load_markets()` — 2–4 requests, slower than a fetch timeout on a loaded line — is guarded by a per-exchange lock, given its own longer timeout, moved to a background thread whenever the caller is on the render path, and retried with an exponential backoff after a failure. Before that, each path did `if not ex.markets: ex.load_markets()` on its own cadence, so one slow exchange produced `RequestTimeout: gate GET …/spot/currencies` for every watched pair every few seconds, no candles were fetched at all, and the caption reported the resulting hole as "the exchange returned no catch-up candles". `[markets]` in the console now names the exchange that cannot load and when it will retry. A third one: `apply_market_type_trim` removes whole market CATEGORIES from that load (`CCXT_MARKET_TYPES_SKIP=option` by default), because ccxt fetches one request per category sequentially under one timeout and a single slow category therefore yields no markets at all — which is how gate's option-contract list, a market type this project never trades, could starve a *perpetual* chart's live line. A deny-list, not an allow-list: bybit names its linear perpetuals `linear`, and mexc expresses the categories as flags, so both are left with everything they need. Two things make that guarantee real: `create_exchange`/`_new_sync_exchange` disable ccxt's `fetchCurrencies` (`CCXT_FETCH_CURRENCIES=true` to restore it) because that extra round trip is what pushed gate's and okx's market load past the engine's 30 s hard wait — a failed load means the exchange collects nothing for the cycle while the log looks healthy — and every store that exists to AVOID work (backoffs, cooldowns, the stitch cache, the scan throttle and its resume cursor) lives in `st.cache_resource` via `_state()`, because Streamlit creates a new `__main__` module for each rerun, so a module-level `X: dict = {}` is quietly rebuilt several times a minute.
* 🧹 **Zombie spot pruning:** `python main.py prune-zombie-spots` finds the spot tables a spot→perp migration left behind — a spot holding FEWER bars than the same base's perp on the same exchange in the same database — across all four databases, and by default only reports them (`--report out.json`). `--apply` parks them in the `zombie_pruned` schema (one `ALTER TABLE … SET SCHEMA public` away from being undone) unless you pass `--mode drop`; it refuses to decide on catalog estimates instead of `COUNT(*)`, keeps any spot written within `--stale-hours`, keeps anything it could not measure, and will not touch more than `--max-fraction` of a database's pair tables without `--yes`. Perp tables, non-pair tables and spots without a perp counterpart are never candidates, and `--limit N` takes the N candidates whose perp has the largest lead, so a partial run frees the most expensive tables first. `--purge-parked` then answers the follow-up question — what is still sitting in `zombie_pruned` (with `--yes`: drops it, which is where the disk comes back).
* 🧵 **A one-second cache may not blank a page:** `st.cache_data(ttl=1)` raises `KeyError: <hash>` from `ttl_cache.py` when an entry is evicted between its lookup and its read — a race inside Streamlit that the `run_every` health-strip fragment walks into, and the traceback lands where the strip was. Those reads now answer `None` for one tick, print one `[live]` line per minute, and any OTHER exception still propagates (a dead feed is reported, not hidden). The same rule covers the gap stitch: `BadSymbol: gate does not have market symbol 1000000BABYDOGE/USDT:USDT` — a delisted pair the collector still holds a table for — is stored as the empty answer it is, instead of one `[stitch]` line and a fresh request per chart page every `DASH_STITCH_RETRY_SEC`.
* 🚀 **Instant Dashboard Startup:** the table/column inventory is read from `pg_catalog` (not the `information_schema` view, which costs 30–250 s on a 14k-table database) and cached for `DASH_SCAN_INVENTORY_TTL_SEC`; background revalidation is throttled to `DASH_SNAPSHOT_REFRESH_SEC`, so a rerun never launches a scan storm against the collector. The pair list itself is ONE type-stable `UNION ALL` query per 120 tables (not one query per pair): a legacy table whose `ob_*` columns are TEXT no longer kills its chunk, mixed columns are flattened to TEXT in SQL and converted back in Python, a chunk that still fails retries once as an all-TEXT query, and the scan is bounded by `DASH_SCAN_BUDGET_SEC` (it renders what it has and never caches a truncated list as the startup snapshot). **The scan is also a guest in the collector's database:** one sweep runs at a time process-wide and it walks the databases of a timeframe sequentially (`DASH_SCAN_MAX_PARALLEL_DBS=1`), a chunk that merely *times out* is skipped instead of being re-read table by table (`DASH_SCAN_RECOVERY_MAX_TABLES` bounds the schema recovery only), the last result — complete or truncated — keeps rendering from memory so a rerun never waits for the database, and a truncated pair list is retried with a doubling backoff (`DASH_SNAPSHOT_REFRESH_SEC` → `DASH_SCAN_RETRY_MAX_SEC`). A sweep the budget cut short **resumes** at the chunk it never reached instead of restarting, and the rows earlier sweeps answered stay in the list for `DASH_SCAN_CARRYOVER_TTL_SEC` — on an 8 000-table database that is the difference between the pair list converging in a few sweeps and re-reading the same ~1 300 tables forever. A timeframe whose sweep was pushed aside by another timeframe's sweep retries after `DASH_SCAN_DEFER_RETRY_SEC` (a skip costs the database nothing), bounded to 3 quick tries so a genuinely busy server still gets its backoff. **A pair list only ever grows while a sweep is unfinished:** a pass that read fewer tables than the pass before it is MERGED into what the dashboard already had, the rows carried mid-sweep never age out (`DASH_SCAN_CARRYOVER_TTL_SEC` applies only once a sweep has wrapped, where retiring unconfirmed rows is the point), a tier whose last scan came back complete is not re-scanned more often than `DASH_SCAN_RESCAN_COMPLETE_SEC`, and a table the database reports as gone (`relation "wbtc_usdt_on_bitget" does not exist`) leaves the cached catalog instead of eating a retry every pass. All four exist because one tier's success starved the other into `rendering 0/8235 tables`, and that empty answer became the selector — every 15m chart disappeared from the dashboard while its tables sat untouched in Postgres. While a sweep is still ADDING tables the retry is capped at 2 × `DASH_SNAPSHOT_REFRESH_SEC` — a pass that answered nothing at all keeps the doubling backoff, because that is a loaded database, not a list being built — and while it IS being built the pause is only `DASH_SCAN_DEFER_RETRY_SEC` (8 s), because re-pacing a sweep that makes progress is what turned a 5-minute job into 40 minutes of "15m не загружается": 69 chunks answered 6 at a time, with minutes of silence between the passes. The catalog is not re-read during that build either: an expired `pg_catalog` listing is reused until the sweep wraps (their log paid `+catalog 14.2s`–`17.0s` on EVERY pass, a third of each pass for a list of tables the sweep is still busy walking). `partial` itself is now measured on the LIST: a sweep that ran out of budget but can vouch for every table the catalog names — read now, or carried from an earlier pass inside `DASH_SCAN_CARRYOVER_TTL_SEC` — is complete, so the badge clears and the tier drops to the 5-minute cycle instead of re-reading 6 394 of 8 312 tables on every retry forever (their "6514/8312 tables have an answer … a full rescan is running now" sat there by design, not by load). Carried rows may hold the list UP, but only rows inside the TTL may call it complete, so a database busy for a day keeps saying "incomplete" rather than freezing quietly. Set `DASH_SCAN_RESCAN_COMPLETE_SEC=900` to quarter the scan load when pair tables are added rarely, which is the normal case. **A pass is now allowed to finish what it started:** `DASH_SCAN_BUDGET_SEC` bounds only the FIRST paint of a process, while a later pass of a sweep that is still building the list runs to `DASH_SCAN_BUDGET_IDLE_SEC` (120 s) as long as nobody has touched the page for 2 s. Their 15m/LOW tier answers 4-6 chunks of 69 per 25 s pass, so 15-20 passes of 'pair list incomplete' were ONE round of reading; at 120 s the same work is 1-2 passes, and it is not more load because no chunk is asked twice. The catalog read has its own bound (`DASH_SCAN_CATALOG_TIMEOUT_SEC`, 45 s) because it is the one read a sweep cannot work around: their first pass after a restart spent 27.3 s of its 56 s on `pg_catalog` and answered 0 of 69 chunks. A listing that fails or times out now keeps the tables of the previous listing, leaves the sweep cursor where it was, and prints `the catalog listing failed` instead of either raising through the cached load (which took the page with it) or answering 'this database has no pairs'. Errors from a busy database are tagged with the CHUNK position (`chunk[14]`) instead of its size: 69 chunks of 120 tables all printed as `chunk[120]`, so the skip count was a set of one tag and a pass that read nothing claimed `2 chunk(s) skipped` on one line and `69 chunk(s) skipped` on the next. The badge quotes the chunk position of the database that is *still being read* (`14/69`), not the sum of both tiers (`0/70`, for a HIGH tier that had already finished in one chunk), plus the budget the pass actually ran on. And it is no longer paid before the first chunk at all: the listing a real read produced is kept on disk beside the summary snapshot, and a cold process starts from it — one `DASH_SCAN_INVENTORY_TTL_SEC` of staleness is the price, `forget_missing_relations` still drops what the database no longer has, and a newly listed pair waits at most one TTL plus the current sweep. Their `+catalog 23.6s` in front of `480/8296 tables covered` was the whole mechanism of 'after a restart the 15m list is empty'. Wave admission is estimated from the UNION queries of earlier waves, never from a wave's wall time: a chunk that needed the all-TEXT retry plus per-table recovery costs 30s and used to convince the sweep that no further wave could fit, which starved exactly the databases that have a few broken chunks.

When a market list hangs, measure the load before blaming the API. `DASH_MARKET_LOAD_DEBUG=true`
prints `[markets] gate: probe, 3 categories, 90s per request — spot FAILED after 91.8s …; future ok, 30
markets, 2.8s`, and that shape (big bodies to the wall, a 3 KB list in 2.8s, same host and instance) is
throughput, not latency. The operator's own A/B of the hung URL settled it: **92,784 B gzipped in 8.2s vs
1,137,626 B uncompressed in 22.2s** — so gzip stays on, `DASH_MARKET_LOAD_ACCEPT_ENCODING` is only for an
endpoint that returns a *corrupted* body, and the probe knows the difference: a leg that died on the clock
is never re-asked uncompressed (that would ask for 12x the bytes of the request that just failed), while a
leg that died on a decode error is asked once bare and, if it answers, the load is retried that way in the
same cycle instead of after a 20–900 s backoff. What actually moves a narrow pipe: widen the FIRST load
(`DASH_MARKET_LOAD_TIMEOUT_SEC=180`, then back to 60), trim what is fetched at all
(`CCXT_MARKET_TYPES_SKIP`, `CCXT_FETCH_CURRENCIES=false`) and let `markets_<exchange>.pkl` carry every
later restart — with `DASH_MARKETS_REFRESH_SEC=21600` so a reload is not re-burning the link hourly.

**Proxy: which half of the app uses it.** `SOCKS5_PROXY` (default `socks5://127.0.0.1:10808`) is honoured by
the *async* ccxt clients — the engines, the collector, and the dashboard's order-book metrics card — and by
nothing else: the dashboard's sync clients (markets load, chips, tape, gap stitching) go direct, because
`requests` needs `pysocks` for a `socks5://` URL and this project does not depend on it. That asymmetry was
the last unexplained number in the gate saga (one panel timing out on the tunnel while another crawled on a
direct 11–50 KB/s link to the same host), so it is no longer inferred from source: `create_exchange` logs
`exchange client gate: route=socks5://127.0.0.1:10808` / `route=direct (…)`, every `[markets]` line carries
its own route, `DASH_LIVE_SNAPSHOT_VIA_PROXY=false` moves the card off the tunnel, and
`DASH_SYNC_PROXY=http://127.0.0.1:10809` moves the sync clients onto it — with the HTTP port, or an explicit
warning that a SOCKS URL is being ignored.

**The engines had the same bug with a different symptom.** `load_markets` was given a 30 s total wait around a
client whose own per-request timeout is 20 s (15m engine) or 40 s (1d), while gate's market lists — spot *and*
linear perps, ~1.4 MB compressed — need minutes on a 7.6 KB/s link. So gate never loaded in either engine,
which is not "the exchange is flaky" but arithmetic: at the measured speed no timeout under a few hundred
seconds could have worked, and no retry count would change that. `EXCHANGE_MARKETS_LOAD_SEC` (default 240) now
sizes the load, the client timeout is widened to it for the load and restored right after, and
`EXCHANGE_MARKETS_TTL_SEC` (default 1800, the previous constant) says how often a reload is worth those minutes
at all — raise it to 21600 to match `DASH_MARKETS_REFRESH_SEC` if the only thing you miss is a listing that
appeared hours ago.

A restart gets the same treatment from the other side: a `load_markets()` is the
slowest thing the dashboard does before it can draw a price (gate: a ~94 KB spot list and a
~1.3 MB swap list, in sequence, each under `DASH_MARKET_LOAD_TIMEOUT_SEC`), and one endpoint
that stalls blanks that exchange's charts, tape and gap stitches for the life of the process.
The catalog a real load produced is therefore kept in the cache directory
(`markets_<exchange>.pkl`) and applied on a cold start when it is younger than
`DASH_MARKETS_DISK_TTL_SEC`, with a real reload scheduled in the background past
`DASH_MARKETS_REFRESH_SEC`. Only a network load writes that file — a disk-seeded catalog never
re-saves itself, so a permanently broken endpoint cannot keep its own cache warm — and the age
is printed (`[markets] gate: 1423 markets taken from disk (412s old, accepted up to 86400s)`),
never hidden. It also states its own severity: a sweep that answered chunks and moved its cursor is `st.info` reading *progress, not a fault — this clears by itself in about N more pass(es)*, with **no knob mentioned**, because a line that ends with 'DASH_SCAN_BUDGET_SEC … is the lever' is read as an error to fix even when it is a progress bar nine chunks from the end (the question it provoked, verbatim: 'что значит эта ошибка в дашборде и как ее исправить?'). The warning and the tuning advice stay on the case they actually describe: a sweep that is NOT advancing, where the doubling backoff is deliberate and the budget is the lever. The message is built by `_pair_list_notice()` — outside the render path, so the wording is testable without a database, and its estimate is computed from the two numbers already on the line (`missing tables / rows added last pass`) instead of cursor arithmetic, which used to print '9 chunks of tables missing' and '7 chunks of cursor left' one line apart. A converging sweep now puts **one caption** on the page (`⏳ 15m pair list is still being built: 3796/8346 tables read — about 4 min (2 pass(es)) left, nothing to do.`) with the full arithmetic in a collapsed expander: a paragraph of counters on a page that is working is what makes the operator ask whether the dashboard is printing debug output, while the fact that the list is still partial must stay visible. The branch where the sweep is NOT advancing keeps every word in the open, because there the numbers are the point. A **healthy sweep now puts one caption on the page** (`⏳ 15m pair list is still being built: 3796/8346 tables read — about 4 min (2 pass(es)) left, nothing to do.`) with the arithmetic in a collapsed expander, and the chunks are admitted in **waves of one pool-full** instead of queueing all 69 behind 6 connections: their own measurement said `124.4s — 42 chunk(s) skipped (db busy)` for ~3 300 tables gained, because every late chunk was granted a connection with almost no time left and was cancelled *mid-query* — Postgres still did the UNION over 120 tables, and the answer was thrown away. A wave is now started only while the remaining budget covers the wave the sweep just measured (its own EWMA, no new setting), the chunks it did not ask about stay behind the cursor, and the console reports `N/M tables covered` rather than `rendering <rows>`, because re-reading a chunk of already-carried tables adds no row and made a growing list look stalled at `5296/8270`.
* ⚡ **Fast Dual-Timeframe Charts:** the Charts tab shows the **15m chart on top and the 1D chart below** with **⏪ Prev / Next ⏭** buttons flanking every chart. Table summaries are cached 10 min, candle frames 60 s, and each chart loads only the last N candles — pair switching is effectively instant. Volume bars are hidden by default and can be toggled with the **Show volume bars** switch. A compact **health strip** above the charts shows green→red chips for trade-tape activity (trades/min), orderbook depth, spread vs the ATR named in the chip (`↔ Spread % 1D_ATR(5)`), and min(vol×low) 7-day dollar volume. A chip is `n/a` only when NO source has its input: the writer keeps bid/ask from `fetch_ticker` for every pair but the orderbook only for the pair on screen, so 🌊 Depth ±1% can legitimately be empty while the LIVE line under it prints a spread — and the spread fields are now derived from bid/ask regardless of the depth (they used to be computed inside the branch that required it, which is how the strip refused to show a number the line below it had). Every field takes the best of three sources (live DB row → the background feed's last value → the pair table's own collector snapshot) with no request on the render path, the tooltip of an `n/a` chip names the input that is missing, and a feed the exchange will not answer prints one rate-limited `[live] ⚠️ …` line instead of dying quietly into NULL. Next to it: direct **Spot/Swap exchange links** and a **shortability badge** (shortable when a perp exists). Chart price axes use compact trimmed formatting (1.10 → 1.1, 4250000 → 4.25M), and candle timestamps are sanitized (garbage future rows dropped, ms-tables auto-converted). Charts are **live**: a background daemon writes the live price / orderbook top / spread / trade-tape stats of the current pair and its ±5 neighbours into a `dashboard_live_ticks` table every second, and every live widget — the server-side LIVE chips, the health strip, and even the in-chart poller (through the dashboard's own `/tick` JSON endpoint on port 8511+) — reads those rows. The chart poller never gives up on errors and falls back to the exchange public REST directly (supported on all 9 exchanges: Bybit/OKX/Gate/KuCoin/MEXC/BingX/Bitget/HTX/CoinEx), the daily chart aggregates fresher 15m candles of the running day, DB data auto-reloads every 60 s, and the ±2 neighbouring pairs (`DASH_WARM_NEIGHBORS`) are prefetched — at most once per chart-page lifetime, starting `DASH_WARM_DELAY_SEC` after your click so the click is served first — which makes Prev/Next flipping a memory lookup. By default the 15m and 1D charts render **side by side** (smaller); a **⬓ Large stacked** toggle switches to 15m-top / 1D-bottom full-width charts.

---

## 🏛️ Supported Exchanges

Operates concurrently via `ccxt.async_support` with independent rate-limiting:
* **Bybit**, **Bitget**, **MEXC**, **KuCoin**, **Gate.io**, **BingX**, **HTX (Huobi)**, **CoinEx**, **OKX**.

---

## 🏗️ Clean Architecture

```text
timescale-crypto-ohlcv-collector/
├── config/                     # Configuration management (Pydantic Settings)
│   └── settings.py
├── src/                        # Application source code
│   ├── analytics/              # Mathematical analytics & indicators
│   │   ├── atr_filtered.py     # 🎯 ATR without paranormal bars
│   │   ├── orderbook.py        # Orderbook depth, spread & CVD analysis
│   │   └── vitality.py         # Market vitality scoring (Grade A-F)
│   ├── db/                     # TimescaleDB & asyncpg layer
│   │   ├── connection.py       # Async pool management
│   │   ├── migrations.py       # Hypertables & compression schema
│   │   └── repository.py       # SQL Repository
│   ├── exchanges/              # CCXT integration layer
│   │   ├── client.py           # CCXT Async factory & SOCKS5 lifecycle
│   │   ├── gap_filler.py       # Candle gap detection & backfill
│   │   └── symbol_selector.py  # Perp-First symbol selection
│   └── core/                   # Orchestration engine
│       ├── progress.py         # Unified progress & ETA calculation
│       └── updater.py          # Main market data engine loop
│   └── utils/                  # Dependency-free shared utilities
│       └── timeouts.py         # hard_wait_for: strictly bounded network timeouts
├── dashboard/                  # Web Dashboard
│   └── app.py                  # Streamlit dashboard app
├── tests/                      # Unit test suite (Pytest)
│   ├── test_atr_filtered.py
│   ├── test_gap_filler.py
│   └── test_symbol_selector.py
├── docker-compose.yml          # One-click launch (TimescaleDB + Engine + Dashboard)
├── Dockerfile                  # Container definition
├── pyproject.toml              # Poetry project configuration
├── requirements.txt            # Python dependencies
├── main.py                     # CLI entry point (Typer)
└── .env.example                # Environment template
```

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

1. Clone repository:
   ```bash
   git clone https://github.com/your-username/timescale-crypto-ohlcv-collector.git
   cd timescale-crypto-ohlcv-collector
   ```

2. Copy environment file:
   ```bash
   cp .env.example .env
   ```

3. Launch infrastructure in 1 command:
   ```bash
   docker compose up -d
   ```

4. Open **Streamlit Dashboard** in browser:
   `http://localhost:8501`

---

### Option 2: Local Execution with Poetry (Python 3.11+)

1. Install dependencies via Poetry:
   ```bash
   poetry install
   ```

2. Initialize TimescaleDB schema and hypertables:
   ```bash
   poetry run python main.py init-db
   ```

3. Start main data collection loop:
   ```bash
   poetry run python main.py run                    # 1D only — the default
   poetry run python main.py run --timeframe 15m    # the 15m engine, separate process
   poetry run python main.py run --timeframe all    # both, concurrently
   ```
   `run` without `--timeframe` collects **1D only**, so a 15m chart stays stale and
   the dashboard's wake-up for the pair you opened goes unanswered on that
   timeframe — the chart says which engine is missing (`⛔ no 15m engine is
   refreshing this pair …`) instead of pretending nobody is listening.

4. Launch Streamlit Web Dashboard:
   ```bash
   poetry run python main.py dashboard
   ```

5. View database summary stats:
   ```bash
   poetry run python main.py summary
   ```

---

## 🧪 Running Tests

Execute test suite for ATR without paranormal bars, symbol selection, and gap detection:

```bash
poetry run pytest -v
```

---

## 📐 Algorithm: ATR without Paranormal Bars

Which one, over which bars — the names the UI prints:

| label | timeframe | N from | stored column | written by |
|---|---|---|---|---|
| `1D_ATR(N)` | closed daily candles | sidebar *ATR Period (daily bars)* (live strip, metric cards) / `ATR_PERIOD` (tables) | `ob_atr_no_paranormal`, `ob_spread_atr_pct` | 1D engine |
| `15m_ATR(N)` | closed 15-minute candles | `ATR_PERIOD` | `ob_gerchik_atr`, `ob_spread_atr_pct` | 15m engine |

So `Spread % of ATR` in the liquidity table is a different measurement in a 15m row than in a daily row, and neither is the chip on the pair page (which divides by `1D_ATR(sidebar N)`). Every label says which.

Standard **ATR (Average True Range)** inflates volatility during single abnormal candles — *paranormal bars* (news spikes, squeezes, false breakouts).

The `compute_atr_no_paranormal_bars` algorithm:
1. Calculates True Range for every candle.
2. Initializes baseline volatility using median True Range (resistant to extreme outliers).
3. Filters out bars falling outside threshold window $[0.5 \times \text{ATR}, 1.8 \times \text{ATR}]$.
4. Iteratively recalculates robust volatility until convergence over the user-selected period $N$.

---

## 📜 License

Distributed under the **MIT License**.
