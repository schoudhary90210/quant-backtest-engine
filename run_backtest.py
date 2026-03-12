"""
Full pipeline: backtest → risk metrics → regime analysis → Monte Carlo.

Usage:
    source venv/bin/activate && python run_backtest.py
"""

import logging
import numpy as np

from src.data.fetcher import fetch_prices
from src.engine.backtest import run_backtest, BacktestResult
from src.monte_carlo.simulation import run_monte_carlo
from src.optimization.equal_weight import equal_weight_signal
from src.optimization.signals import make_kelly_signal
from src.risk.metrics import compute_risk_report
from src.risk.regime import regime_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _section(title: str) -> None:
    logger.info("\n%s\n  %s\n%s", "=" * 62, title, "=" * 62)


def main() -> None:
    # ── 1. Fetch data ──
    logger.info("Fetching price data...")
    prices = fetch_prices()
    logger.info("Price matrix: %s", prices.shape)

    # ── 2. Build signals ──
    kelly_signal = make_kelly_signal(fraction=0.5, cov_method="ledoit_wolf")

    # ── 3. Run backtests ──
    logger.info("Running Equal Weight backtest...")
    ew_result = run_backtest(prices, equal_weight_signal, rebalance_freq="monthly")

    logger.info("Running Half-Kelly backtest...")
    kelly_result = run_backtest(prices, kelly_signal, rebalance_freq="monthly")

    # ── 4. Risk metrics ──
    _section("BACKTEST RESULTS: 12-Asset Universe")

    for name, result in [("Equal Weight (1/N)", ew_result), ("Half-Kelly (LW)", kelly_result)]:
        report = compute_risk_report(result.daily_returns)
        logger.info("\n  ─── %s ───", name)
        logger.info("\n%s", report)
        logger.info("  Total Costs:         $%s", f"{result.total_costs.sum():,.0f}")
        logger.info("  Avg Daily Turnover:   %7.2f%%", result.turnover.mean() * 100)

    # ── 5. Regime analysis (Kelly) ──
    _section("REGIME ANALYSIS: Half-Kelly")
    stats = regime_stats(kelly_result.daily_returns)
    for regime_name in ("bull", "sideways", "bear"):
        if regime_name in stats:
            logger.info("\n%s", stats[regime_name])

    # ── 6. Monte Carlo (Kelly) ──
    _section("MONTE CARLO SIMULATION: Half-Kelly (50k paths, 1yr)")
    logger.info("Running Monte Carlo simulation (50,000 paths × 252 days)...")
    mc = run_monte_carlo(
        daily_returns=kelly_result.daily_returns,
        n_paths=50_000,
        n_days=252,
        initial_capital=1_000_000,
        seed=42,
    )
    logger.info("\n%s", mc)

    _section("DONE")


if __name__ == "__main__":
    main()
