"""
Covariance estimator comparison helper.

Provides:
1. ``compare_covariance_estimators`` — diagnostic metrics (eigenvalues, condition number, etc.)
2. ``run_covariance_backtest_comparison`` — full backtest per covariance method with Sharpe,
   CAGR, max drawdown, turnover, and condition number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from src.data.returns import compute_log_returns
from src.engine.backtest import BacktestResult, run_backtest
from src.optimization.covariance import get_covariance_estimator
from src.optimization.signals import make_bl_kelly_signal, make_kelly_signal
from src.risk.metrics import compute_risk_report, turnover_analysis

logger = logging.getLogger(__name__)

# Default methods to compare (order matters for chart legends)
DEFAULT_METHODS = [
    "sample",
    "ledoit_wolf",
    "ledoit_wolf_nonlinear",
    "ewma",
    "rmt_denoised",
    "dcc_garch",
]


def compare_covariance_estimators(
    returns: pd.DataFrame,
    methods: list[str] | None = None,
) -> dict[str, dict]:
    """
    Run multiple covariance estimators on *returns* and collect diagnostics.

    Parameters
    ----------
    returns : pd.DataFrame
        T x N log-return matrix.
    methods : list[str] or None
        Estimator names to compare.  Defaults to ``DEFAULT_METHODS``.

    Returns
    -------
    dict[str, dict]
        Mapping of method name to a dict with keys:
        - ``cov``              : pd.DataFrame — the N x N covariance matrix
        - ``eigenvalues``      : np.ndarray — sorted descending
        - ``condition_number`` : float
        - ``trace``            : float
        - ``frobenius_norm``   : float
        - ``max_eigenvalue``   : float
        - ``min_eigenvalue``   : float
    """
    if methods is None:
        methods = list(DEFAULT_METHODS)

    results: dict[str, dict] = {}
    for method in methods:
        estimator = get_covariance_estimator(method)
        cov = estimator.estimate(returns)
        eigenvals = np.linalg.eigvalsh(cov.values)
        eigenvals_sorted = np.sort(eigenvals)[::-1]

        results[method] = {
            "cov": cov,
            "eigenvalues": eigenvals_sorted,
            "condition_number": float(np.linalg.cond(cov.values)),
            "trace": float(np.trace(cov.values)),
            "frobenius_norm": float(np.linalg.norm(cov.values, "fro")),
            "max_eigenvalue": float(eigenvals.max()),
            "min_eigenvalue": float(eigenvals.min()),
        }

    return results


# ---------------------------------------------------------------------------
# Backtest-based comparison
# ---------------------------------------------------------------------------


@dataclass
class CovarianceComparisonResult:
    """Per-method comparison result combining backtest performance and diagnostics."""

    method: str
    sharpe: float
    cagr: float
    max_drawdown: float
    sortino: float
    annualized_turnover: float
    total_cost_drag_bps: float
    condition_number: float
    backtest_result: BacktestResult


def run_covariance_backtest_comparison(
    prices: pd.DataFrame,
    methods: list[str] | None = None,
    return_estimator: str = "black_litterman",
    kelly_fraction: float = config.KELLY_FRACTION,
    rebalance_freq: str = config.REBALANCE_FREQ,
    cost_model=None,
    max_turnover: float | None = config.MAX_TURNOVER,
) -> dict[str, CovarianceComparisonResult]:
    """
    Run a full backtest per covariance method and collect performance metrics.

    Uses the active return estimator (BL or historical mean) so the comparison
    reflects the deployed strategy, not a different one.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices (rows = dates, columns = tickers).
    methods : list[str] or None
        Covariance estimator names.  Defaults to ``config.COV_COMPARISON_METHODS``.
    return_estimator : str
        ``"black_litterman"`` or ``"historical_mean"``.
    kelly_fraction : float
        Kelly fraction (0.5 = half-Kelly).
    rebalance_freq : str
        ``"daily"`` or ``"monthly"``.
    cost_model : TransactionCostModel or None
        Transaction cost model to use.
    max_turnover : float or None
        Per-rebalance turnover cap.

    Returns
    -------
    dict[str, CovarianceComparisonResult]
    """
    if methods is None:
        methods = list(config.COV_COMPARISON_METHODS)

    results: dict[str, CovarianceComparisonResult] = {}

    for method in methods:
        logger.info("  Comparing covariance method: %s", method)

        # Build signal using the active return estimator
        if return_estimator == "black_litterman":
            signal_fn = make_bl_kelly_signal(
                cov_method=method,
                fraction=kelly_fraction,
                max_turnover=max_turnover,
            )
        else:
            signal_fn = make_kelly_signal(
                cov_method=method,
                fraction=kelly_fraction,
                max_turnover=max_turnover,
            )

        # Run backtest
        bt = run_backtest(
            prices,
            signal_fn,
            rebalance_freq=rebalance_freq,
            cost_model=cost_model,
        )

        # Compute metrics
        risk = compute_risk_report(bt.daily_returns)
        turn = turnover_analysis(bt)

        # Condition number from covariance snapshot on last lookback window
        log_rets = compute_log_returns(prices)
        lookback = min(config.LOOKBACK_WINDOW, len(log_rets))
        window_rets = log_rets.iloc[-lookback:]
        cov_est = get_covariance_estimator(method)
        cov_mat = cov_est.estimate(window_rets)
        cond_num = float(np.linalg.cond(cov_mat.values))

        results[method] = CovarianceComparisonResult(
            method=method,
            sharpe=risk.sharpe_ratio,
            cagr=bt.cagr,
            max_drawdown=bt.max_drawdown,
            sortino=risk.sortino_ratio,
            annualized_turnover=turn["annualized_turnover"],
            total_cost_drag_bps=turn["total_cost_drag_bps"],
            condition_number=cond_num,
            backtest_result=bt,
        )

    return results
