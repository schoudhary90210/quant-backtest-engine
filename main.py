"""
Quantitative Backtesting & Portfolio Optimization Engine
=========================================================
Single entry point. Runs the full pipeline end-to-end:

    1. Fetch & cache price data
    2. Run backtests (Equal Weight, Half-Kelly, Full Kelly, BL + Kelly)
    3. Walk-forward out-of-sample validation  [if WALK_FORWARD=True]
    4. Run Monte Carlo simulation
    5. Generate all charts
    6. Generate and save text report

Flags
-----
WALK_FORWARD : bool
    Run anchored walk-forward OOS validation (adds ~2 min for 15-year data).
RETURN_ESTIMATOR : str
    "black_litterman" (default) or "historical_mean".  Controls which return
    estimator is used for the *primary* Kelly backtest and walk-forward loop.
    Both estimators are always run and compared in the return-estimator chart.

Usage:
    source venv/bin/activate && python main.py
"""

import logging
from pathlib import Path

import config
from src.data.fetcher import fetch_prices
from src.engine.backtest import run_backtest
from src.monte_carlo.simulation import run_monte_carlo
from src.optimization.equal_weight import equal_weight_signal
from src.optimization.signals import make_bl_kelly_signal, make_kelly_signal
from src.risk.metrics import compute_risk_report
from src.validation.oos_metrics import compare_is_oos
from src.validation.walk_forward import WalkForwardConfig, run_walk_forward
from src.visualization.charts import (
    plot_drawdowns,
    plot_equity_curves,
    plot_is_vs_oos_equity,
    plot_monthly_heatmap,
    plot_monte_carlo_fan,
    plot_return_estimator_comparison,
    plot_rolling_sharpe,
    plot_weight_evolution,
)
from src.visualization.report import generate_report

# ── Pipeline flags ────────────────────────────────────────────────────────
# Set WALK_FORWARD=False for fast iteration (skips the ~2-min WF loop).
WALK_FORWARD = True
# "black_litterman" uses BL posterior as expected-return input to Kelly.
# "historical_mean" uses raw annualised log-return mean (original approach).
RETURN_ESTIMATOR = "black_litterman"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 62)
    logger.info("  Quant Backtest Engine — Full Pipeline")
    logger.info("=" * 62)

    # ── 1. Data ──────────────────────────────────────────────────
    logger.info("[1/6] Fetching price data...")
    prices = fetch_prices()
    logger.info("      %d assets × %d trading days", len(prices.columns), len(prices))

    # ── 2. Backtests ─────────────────────────────────────────────
    logger.info("[2/6] Running backtests...")

    # Historical Mean + Kelly (original approach)
    hk_signal = make_kelly_signal(fraction=0.5, cov_method="ledoit_wolf")
    fk_signal = make_kelly_signal(fraction=1.0, cov_method="ledoit_wolf")

    # Black-Litterman + Kelly (new approach)
    bl_signal = make_bl_kelly_signal(fraction=0.5)

    ew_result = run_backtest(prices, equal_weight_signal, rebalance_freq="monthly")
    hk_result = run_backtest(prices, hk_signal,          rebalance_freq="monthly")
    fk_result = run_backtest(prices, fk_signal,          rebalance_freq="monthly")
    bl_result = run_backtest(prices, bl_signal,          rebalance_freq="monthly")

    backtest_results = {
        "Equal Weight": ew_result,
        "Half-Kelly":   hk_result,
        "Full Kelly":   fk_result,
        "BL + Kelly":   bl_result,
    }

    hk_metrics = compute_risk_report(hk_result.daily_returns)
    bl_metrics = compute_risk_report(bl_result.daily_returns)
    ew_metrics = compute_risk_report(ew_result.daily_returns)

    logger.info("      Equal Weight  — CAGR %.1f%%  Sharpe %.3f",
                ew_result.cagr * 100, ew_metrics.sharpe_ratio)
    logger.info("      Half-Kelly    — CAGR %.1f%%  Sharpe %.3f",
                hk_result.cagr * 100, hk_metrics.sharpe_ratio)
    logger.info("      Full Kelly    — CAGR %.1f%%", fk_result.cagr * 100)
    logger.info("      BL + Kelly    — CAGR %.1f%%  Sharpe %.3f",
                bl_result.cagr * 100, bl_metrics.sharpe_ratio)

    # Choose which IS result drives the walk-forward comparison
    primary_result = bl_result if RETURN_ESTIMATOR == "black_litterman" else hk_result

    # ── 3. Walk-Forward Validation ───────────────────────────────
    wf_result = None
    if WALK_FORWARD:
        logger.info("[3/6] Walk-forward OOS validation (estimator=%s)...",
                    RETURN_ESTIMATOR)
        wf_result = run_walk_forward(
            prices,
            wf_config=WalkForwardConfig(),
            in_sample_result=primary_result,
            return_estimator=RETURN_ESTIMATOR,
        )
        logger.info(
            "      OOS start: %s | IS Sharpe %.3f | OOS Sharpe %.3f",
            wf_result.oos_start_date.date(),
            wf_result.in_sample_metrics.sharpe_ratio,
            wf_result.oos_metrics.sharpe_ratio,
        )
        compare_is_oos(wf_result.in_sample_metrics, wf_result.oos_metrics)

    # ── 4. Monte Carlo ───────────────────────────────────────────
    logger.info("[4/6] Running Monte Carlo (%d paths × %d days)...",
                config.MC_NUM_PATHS, config.MC_HORIZON_DAYS)
    mc_result = run_monte_carlo(
        daily_returns=primary_result.daily_returns,
        n_paths=config.MC_NUM_PATHS,
        n_days=config.MC_HORIZON_DAYS,
        initial_capital=config.INITIAL_CAPITAL,
        seed=config.MC_SEED,
        store_paths=True,
    )
    logger.info("      P(profit)=%.1f%%  Median terminal $%s",
                mc_result.prob_profit * 100,
                f"{mc_result.median_terminal_wealth:,.0f}")

    # ── 5. Charts ────────────────────────────────────────────────
    logger.info("[5/6] Generating charts → %s", config.CHART_DIR)
    Path(config.CHART_DIR).mkdir(parents=True, exist_ok=True)

    # Core strategy equity curves (EW / HK / FK for the main comparison)
    core_equity_map = {
        "Equal Weight": ew_result.equity_curve,
        "Half-Kelly":   hk_result.equity_curve,
        "Full Kelly":   fk_result.equity_curve,
    }
    returns_map = {
        "Equal Weight": ew_result.daily_returns,
        "Half-Kelly":   hk_result.daily_returns,
        "Full Kelly":   fk_result.daily_returns,
    }

    saved = []

    p = plot_equity_curves(core_equity_map)
    saved.append(p)
    logger.info("      + %s", p.name)

    p = plot_monte_carlo_fan(mc_result.equity_paths, config.INITIAL_CAPITAL)
    saved.append(p)
    logger.info("      + %s", p.name)

    p = plot_rolling_sharpe(returns_map)
    saved.append(p)
    logger.info("      + %s", p.name)

    p = plot_drawdowns(core_equity_map)
    saved.append(p)
    logger.info("      + %s", p.name)

    p = plot_monthly_heatmap(hk_result.daily_returns, strategy_name="Half-Kelly")
    saved.append(p)
    logger.info("      + %s", p.name)

    p = plot_weight_evolution(hk_result.positions, strategy_name="Half-Kelly")
    saved.append(p)
    logger.info("      + %s", p.name)

    if wf_result is not None:
        p = plot_is_vs_oos_equity(
            wf_result.in_sample_result.equity_curve,
            wf_result.oos_result.equity_curve,
            oos_start_date=wf_result.oos_start_date,
        )
        saved.append(p)
        logger.info("      + %s", p.name)

    # Return estimator comparison: EW vs Historical Mean + Kelly vs BL + Kelly
    estimator_equity = {
        "Equal Weight":            ew_result.equity_curve,
        "Historical Mean + Kelly": hk_result.equity_curve,
        "BL + Kelly":              bl_result.equity_curve,
    }
    estimator_metrics = {
        "Equal Weight":            (ew_result.cagr, ew_metrics.sharpe_ratio),
        "Historical Mean + Kelly": (hk_result.cagr, hk_metrics.sharpe_ratio),
        "BL + Kelly":              (bl_result.cagr, bl_metrics.sharpe_ratio),
    }
    p = plot_return_estimator_comparison(estimator_equity, estimator_metrics)
    saved.append(p)
    logger.info("      + %s", p.name)

    logger.info("      %d charts saved.", len(saved))

    # ── 6. Report ────────────────────────────────────────────────
    logger.info("[6/6] Generating report → %s", config.REPORT_PATH)
    generate_report(backtest_results, kelly_result=primary_result)

    # ── Done ─────────────────────────────────────────────────────
    logger.info("Pipeline complete.")
    logger.info("      Charts: %s", config.CHART_DIR)
    logger.info("      Report: %s", config.REPORT_PATH)


if __name__ == "__main__":
    main()
