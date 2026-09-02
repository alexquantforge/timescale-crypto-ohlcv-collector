# ⚡ Timescale Crypto OHLCV Collector

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TimescaleDB](https://img.shields.io/badge/Database-TimescaleDB-yellow.svg)](https://www.timescale.com/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-cyan.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Async Crypto Market Data Collector (**Perpetual Swaps & Spot Markets**) with Filtered ATR (w/o Paranormal Bars), L2 Orderbook Depth, CVD, and TimescaleDB Storage. Supports **9+ crypto exchanges** with automated time partitioning, **Columnar Compression**, and interactive **TradingView Lightweight Charts** dashboard.

---

## 🌟 Key Features

* 🔄 **Perp-First & Spot Fallback Architecture:** Automatically prefers linear perpetual swaps (`BTC/USDT:USDT`) for each base asset, seamlessly falling back to spot (`BTC/USDT`) when no perpetual contract exists.
* 📊 **ATR without Paranormal Bars (Filtered Robust ATR):** Robust volatility calculation that filters out news spikes, squeezes, and abnormal outlier candles outside $[0.5 \times \text{ATR}, 1.8 \times \text{ATR}]$.
* ⚡ **TimescaleDB Hypertables:** Time-series storage with automated chunk partitioning and **Columnar Compression**, reducing DB disk footprint by up to 90%+.
* 🛡️ **Gap-Filling & Backfill:** Automatic detection and backfilling of missing daily candle ranges, combined with deep historical backfill from 2018.
* 📈 **L2/L3 Orderbook & Trade Tape Snapshots:** Computes spread tightness (% relative to ATR without paranormal bars), orderbook depth (Bid/Ask/Total USD), Cumulative Volume Delta (CVD & 5m CVD), and liquidity grade (**Vitality Score A–F**).
* 🔀 **Dynamic Liquidity Tiering:** Classifies coins into `HIGH` ($\ge \$500,000$ USD/day) and `LOW` volume tiers.
* 🌐 **Interactive Web Dashboard:** Streamlit dashboard featuring Plotly & **TradingView Lightweight Charts** (OHLCV Bars & Candlesticks) with ATR & liquidity metrics and multi-exchange filters.
* 🖱 **Instant Pair Switching:** the Charts tab paints the candles already stored in TimescaleDB first — gap stitching and the live ticker / orderbook / trade-tape feeds run in background daemon threads and are swapped in when they land. A swap repaints only when it actually filled candles, so browsing healthy pairs never resets your zoom. Prefetching is deliberately small and lazy: warming a pair also primes the DB-only page, fetches at most 3 missing ranges per timeframe as a head start (the rest is fetched when the pair is actually opened), and skips exchange traffic entirely for pairs whose collector stopped writing more than `DASH_WARM_STALE_SKIP_SEC` ago — the dead spot tables left by a spot→perp migration, whose hundreds of missing candles used to make the prefetch slower than the click it was meant to speed up.
* 🚀 **Instant Dashboard Startup:** the table/column inventory is read from `pg_catalog` (not the `information_schema` view, which costs 30–250 s on a 14k-table database) and cached for `DASH_SCAN_INVENTORY_TTL_SEC`; background revalidation is throttled to `DASH_SNAPSHOT_REFRESH_SEC`, so a rerun never launches a scan storm against the collector. The pair list itself is ONE type-stable `UNION ALL` query per 120 tables (not one query per pair): a legacy table whose `ob_*` columns are TEXT no longer kills its chunk, mixed columns are flattened to TEXT in SQL and converted back in Python, a chunk that still fails retries once as an all-TEXT query, and the scan is bounded by `DASH_SCAN_BUDGET_SEC` (it renders what it has and never caches a truncated list as the startup snapshot). **The scan is also a guest in the collector's database:** one sweep runs at a time process-wide and it walks the databases of a timeframe sequentially (`DASH_SCAN_MAX_PARALLEL_DBS=1`), a chunk that merely *times out* is skipped instead of being re-read table by table (`DASH_SCAN_RECOVERY_MAX_TABLES` bounds the schema recovery only), the last result — complete or truncated — keeps rendering from memory so a rerun never waits for the database, and a truncated pair list is retried with a doubling backoff (`DASH_SNAPSHOT_REFRESH_SEC` → `DASH_SCAN_RETRY_MAX_SEC`).
* ⚡ **Fast Dual-Timeframe Charts:** the Charts tab shows the **15m chart on top and the 1D chart below** with **⏪ Prev / Next ⏭** buttons flanking every chart. Table summaries are cached 10 min, candle frames 60 s, and each chart loads only the last N candles — pair switching is effectively instant. Volume bars are hidden by default and can be toggled with the **Show volume bars** switch. A compact **health strip** above the charts shows green→red chips for trade-tape activity (trades/min), orderbook depth, spread vs 5% of daily ATR, and min(vol×low) 7-day dollar volume. Next to it: direct **Spot/Swap exchange links** and a **shortability badge** (shortable when a perp exists). Chart price axes use compact trimmed formatting (1.10 → 1.1, 4250000 → 4.25M), and candle timestamps are sanitized (garbage future rows dropped, ms-tables auto-converted). Charts are **live**: a background daemon writes the live price / orderbook top / spread / trade-tape stats of the current pair and its ±5 neighbours into a `dashboard_live_ticks` table every second, and every live widget — the server-side LIVE chips, the health strip, and even the in-chart poller (through the dashboard's own `/tick` JSON endpoint on port 8511+) — reads those rows. The chart poller never gives up on errors and falls back to the exchange public REST directly (supported on all 9 exchanges: Bybit/OKX/Gate/KuCoin/MEXC/BingX/Bitget/HTX/CoinEx), the daily chart aggregates fresher 15m candles of the running day, DB data auto-reloads every 60 s, and the ±2 neighbouring pairs (`DASH_WARM_NEIGHBORS`) are prefetched — at most once per chart-page lifetime, starting `DASH_WARM_DELAY_SEC` after your click so the click is served first — which makes Prev/Next flipping a memory lookup. By default the 15m and 1D charts render **side by side** (smaller); a **⬓ Large stacked** toggle switches to 15m-top / 1D-bottom full-width charts.

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
   poetry run python main.py run
   ```

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

Standard **ATR (Average True Range)** inflates volatility during single abnormal candles — *paranormal bars* (news spikes, squeezes, false breakouts).

The `compute_atr_no_paranormal_bars` algorithm:
1. Calculates True Range for every candle.
2. Initializes baseline volatility using median True Range (resistant to extreme outliers).
3. Filters out bars falling outside threshold window $[0.5 \times \text{ATR}, 1.8 \times \text{ATR}]$.
4. Iteratively recalculates robust volatility until convergence over the user-selected period $N$.

---

## 📜 License

Distributed under the **MIT License**.
