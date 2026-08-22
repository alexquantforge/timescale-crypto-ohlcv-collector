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
* ⚡ **Fast Dual-Timeframe Charts:** the Charts tab shows the **15m chart on top and the 1D chart below** with **⏪ Prev / Next ⏭** buttons flanking every chart. Table summaries are cached 10 min, candle frames 60 s, and each chart loads only the last N candles — pair switching is effectively instant.

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
