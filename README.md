# Quantitative Backtesting & Portfolio Optimization Engine

> A production-quality multi-asset backtesting framework with Kelly Criterion portfolio optimization, Ledoit-Wolf covariance estimation, Monte Carlo simulation, and full risk analytics — backtested on 12 assets from 2014 to 2025.

---

## Results

| Strategy | CAGR | Sharpe | Sortino | Max Drawdown | Calmar |
|---|---|---|---|---|---|
| Equal Weight (1/N) | 13.2% | 0.478 | 0.458 | -27.8% | 0.321 |
| **Half-Kelly** | **33.0%** | **0.921** | **0.977** | **-29.1%** | **0.747** |
| Full Kelly | 33.1% | 0.916 | 0.970 | -31.9% | 0.683 |

*Universe: SPY, QQQ, IWM, EFA, TLT, IEF, GLD, SLV, USO, BTC-USD, DBA, VNQ — monthly rebalance — 10 bps commission + 5 bps slippage.*

---

## Charts

### Equity Curves
![Equity Curves](output/charts/equity_curves.png)

### Monthly Returns Heatmap (Half-Kelly)
![Monthly Returns Heatmap](output/charts/monthly_heatmap.png)

### Monte Carlo Fan Chart (1-Year Forward, 50,000 Paths)
![Monte Carlo](output/charts/monte_carlo_fan.png)

### Rolling 252-Day Sharpe Ratio
![Rolling Sharpe](output/charts/rolling_sharpe.png)

### Drawdowns
![Drawdowns](output/charts/drawdowns.png)

### Portfolio Weight Allocation Over Time
![Weight Evolution](output/charts/weight_evolution.png)

---

## Architecture

```
quant-backtest-engine/
│
├── config.py                     # Central configuration (all tunable params)
├── main.py                       # Single entry point — full pipeline
│
├── src/
│   ├── data/
│   │   ├── fetcher.py            # Yahoo Finance ingestion + parquet cache
│   │   └── returns.py            # Log-return computation and alignment
│   │
│   ├── engine/
│   │   ├── backtest.py           # Event-driven backtester core loop
│   │   └── costs.py              # Proportional transaction cost model
│   │
│   ├── optimization/
│   │   ├── covariance.py         # Sample / Ledoit-Wolf / EWMA estimators
│   │   ├── kelly.py              # Multi-asset Kelly criterion optimizer
│   │   ├── equal_weight.py       # 1/N benchmark strategy
│   │   └── signals.py            # Signal factories (wraps optimizers)
│   │
│   ├── risk/
│   │   ├── metrics.py            # Sharpe, Sortino, VaR, CVaR, Calmar, ...
│   │   └── regime.py             # Rolling vol regime classifier
│   │
│   ├── monte_carlo/
│   │   └── simulation.py         # Vectorized 50k-path forward simulator
│   │
│   └── visualization/
│       ├── charts.py             # 6 publication-quality matplotlib charts
│       └── report.py             # Text report generator
│
├── tests/                        # 141 pytest tests (100% passing)
├── data/cache/                   # Parquet price cache (git-ignored)
└── output/                       # Charts + report (git-ignored)
```

**Data flow:**

```
Yahoo Finance → fetcher.py → parquet cache
                    ↓
             returns.py (log returns)
                    ↓
        ┌───────────┴───────────┐
   covariance.py           equal_weight.py
   kelly.py                    │
        └───────────┬───────────┘
                signals.py
                    ↓
             backtest.py  ←  costs.py
                    ↓
        ┌───────────┴───────────┐
   metrics.py              simulation.py
   regime.py               (Monte Carlo)
        └───────────┬───────────┘
                    ↓
          charts.py + report.py
```

---

## Installation

**Requirements:** Python 3.11+

```bash
git clone https://github.com/schoudhary90210/quant-backtest-engine.git
cd quant-backtest-engine

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

**Run the full pipeline** (fetches data, runs backtests, generates all charts and report):

```bash
python main.py
```

**Run backtests only:**

```bash
python run_backtest.py
```

**Run the test suite:**

```bash
python -m pytest tests/ -v
```

**Fetch and cache market data only:**

```bash
python -m src.data.fetcher
```

All outputs are written to:
- `output/charts/` — 6 PNG charts at 300 DPI
- `output/report.txt` — full risk report

---

## Configuration

All parameters are in `config.py` — no magic numbers anywhere in the library code:

```python
ASSETS = ["SPY", "QQQ", "IWM", ...]   # 12-asset universe
START_DATE = "2010-01-01"
INITIAL_CAPITAL = 1_000_000
REBALANCE_FREQ = "monthly"             # 'daily' or 'monthly'
TRANSACTION_COST_BPS = 10             # round-trip commission
SLIPPAGE_BPS = 5
KELLY_FRACTION = 0.5                   # half-Kelly
MAX_POSITION_WEIGHT = 0.40            # 40% single-asset cap
MAX_LEVERAGE = 1.5                     # gross exposure cap
COV_METHOD = "ledoit_wolf"            # 'sample', 'ledoit_wolf', 'ewma'
MC_NUM_PATHS = 50_000
```

---

## Technical Deep Dive

### Kelly Criterion

The Kelly Criterion provides the theoretically optimal bet size that maximises the long-run geometric growth rate of a portfolio. For a single asset it simplifies to *f\* = (μ − r) / σ²*, but the multi-asset generalisation used here is:

**f\* = Σ⁻¹ (μ − r)**

where **Σ** is the N×N covariance matrix, **μ** is the vector of expected excess returns, and **r** is the risk-free rate. The system is solved via `numpy.linalg.solve` rather than explicit matrix inversion, which is numerically stable even for near-singular covariances (falling back to least-squares via `numpy.linalg.lstsq` when needed). In practice, full Kelly produces volatile portfolios that are deeply uncomfortable to hold through drawdowns. This engine defaults to **Half-Kelly (fraction = 0.5)**, which cuts position sizes in half while retaining most of the geometric-growth advantage — the empirical results above show essentially identical CAGR with materially better drawdown and Calmar ratio compared to Full Kelly.

Constraints are applied in sequence after solving: (1) assets with negative expected excess return over the risk-free rate are zeroed out entirely; (2) each position is clipped to `MAX_POSITION_WEIGHT` (40%); (3) if total gross exposure exceeds `MAX_LEVERAGE` (1.5×), all weights are scaled proportionally. Expected returns are estimated from a rolling `LOOKBACK_WINDOW` (252 days) of historical log returns, annualised by multiplying by 252.

### Ledoit-Wolf Covariance Shrinkage

The core challenge in multi-asset portfolio optimisation is estimating a reliable covariance matrix from finite historical data. The sample covariance matrix has estimation error that grows with N (number of assets) relative to T (number of observations), producing extreme, unstable portfolio weights. The Ledoit-Wolf (2004) estimator addresses this by shrinking the sample covariance **S** toward a structured target **μI** (scaled identity):

**Σ̂ = (1 − α) S + α μI**

where the optimal shrinkage intensity **α** is derived analytically, requiring no cross-validation. This engine exposes three swappable estimators — `SampleCovariance`, `LedoitWolfCovariance`, and `EWMACovariance` — all behind a common `estimate(returns) → pd.DataFrame` protocol, making them drop-in replaceable via the `COV_METHOD` config key. Ledoit-Wolf is the default because it consistently produces better-conditioned matrices (lower condition number) even when the number of observations is only a few multiples of the number of assets, which is the typical regime for monthly-rebalanced strategies with a lookback of 252 days and 12 assets.

### Monte Carlo Simulation

The Monte Carlo engine generates 50,000 forward paths of 252 trading days each. It operates in two modes: (1) univariate, fitting a normal distribution to historical portfolio returns and drawing i.i.d. daily returns; and (2) multi-asset correlated, drawing from a multivariate normal distribution parameterised by the per-asset historical means and the Ledoit-Wolf covariance matrix, then projecting through the fixed weight vector. The simulation is fully vectorised using NumPy — all 50,000 × 252 draws happen in a single `numpy.random.default_rng.multivariate_normal` call, making it fast enough for interactive use. Results include the terminal wealth distribution, per-path maximum drawdown (computed via vectorised cumulative-maximum), and underwater period lengths. The random seed is fixed via `config.MC_SEED` ensuring full reproducibility.

---

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
