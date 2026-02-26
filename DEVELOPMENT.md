# Quantitative Backtesting & Portfolio Optimization Engine

## Project Overview
Multi-asset backtesting engine with portfolio optimization (mean-variance, Kelly, risk parity),
Monte Carlo simulation, and risk analytics. Targets 12 assets across equities, bonds, commodities,
crypto, and real estate.

## Architecture
- `config.py` — Central configuration. All tunable parameters live here, no magic numbers in library code.
- `src/data/` — Data ingestion (yfinance) and return computation.
- `src/engine/` — Event-driven backtester and transaction cost model.
- `src/optimization/` — Portfolio optimizers (mean-variance, Kelly, risk parity).
- `src/risk/` — Risk metrics (Sharpe, VaR, drawdown) and regime detection.
- `src/monte_carlo/` — Forward simulation engine.
- `src/visualization/` — Chart generation (matplotlib).
- `data/cache/` — Parquet-cached price data (git-ignored).
- `output/` — Generated charts and reports (git-ignored).
- `tests/` — pytest test suite.

## Conventions
- Python 3.11+. Use type hints on all public functions.
- Use `config.py` for all parameters — never hardcode numbers in modules.
- DataFrames use DatetimeIndex; columns are ticker symbols.
- Log returns (not simple returns) for all statistical computations.
- Parquet for data caching; no CSV.
- numpy/pandas for numerics; scipy for optimization; sklearn for covariance estimation.
- pytest for testing. Run: `source venv/bin/activate && python -m pytest tests/ -v`

## Commands
- **Run tests:** `source venv/bin/activate && python -m pytest tests/ -v`
- **Run full pipeline:** `source venv/bin/activate && python main.py`
- **Run backtest only:** `source venv/bin/activate && python run_backtest.py`
- **Fetch data:** `source venv/bin/activate && python -m src.data.fetcher`
