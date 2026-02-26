"""
Run Kelly (Half-Kelly) vs Equal Weight backtest on the 12-asset universe.

Usage:
    source venv/bin/activate && python run_backtest.py
"""

import logging
import numpy as np

from src.data.fetcher import fetch_prices
from src.engine.backtest import run_backtest, BacktestResult
from src.optimization.equal_weight import equal_weight_signal
from src.optimization.signals import make_kelly_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def sharpe_ratio(result: BacktestResult, risk_free_rate: float = 0.04) -> float:
    """Annualized Sharpe ratio from daily returns."""
    daily = result.daily_returns
    if len(daily) < 2:
        return 0.0
    excess = daily - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def main():
    # ── Fetch price data ──
    logger.info("Fetching price data...")
    prices = fetch_prices()
    logger.info("Price matrix: %s", prices.shape)

    # ── Build signals ──
    kelly_signal = make_kelly_signal(fraction=0.5, cov_method="ledoit_wolf")

    # ── Run backtests ──
    logger.info("Running Equal Weight backtest...")
    ew_result = run_backtest(prices, equal_weight_signal, rebalance_freq="monthly")

    logger.info("Running Half-Kelly backtest...")
    kelly_result = run_backtest(prices, kelly_signal, rebalance_freq="monthly")

    # ── Report ──
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS: 12-Asset Universe (2010–2025)")
    print("=" * 60)

    for name, result in [("Equal Weight (1/N)", ew_result), ("Half-Kelly (LW)", kelly_result)]:
        sr = sharpe_ratio(result)
        print(f"\n  {name}:")
        print(f"    Total Return:  {result.total_return * 100:>8.2f}%")
        print(f"    CAGR:          {result.cagr * 100:>8.2f}%")
        print(f"    Max Drawdown:  {result.max_drawdown * 100:>8.2f}%")
        print(f"    Sharpe Ratio:  {sr:>8.3f}")
        print(f"    Total Costs:  ${result.total_costs.sum():>10,.0f}")
        print(f"    Avg Turnover:  {result.turnover.mean() * 100:>8.2f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
