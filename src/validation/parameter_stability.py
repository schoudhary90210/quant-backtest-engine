"""
Parameter stability analysis.

Rolls a training window across historical prices, evaluates every
parameter combination on each window, and records which parameters
win.  Stable strategies show low coefficient of variation (CV) in
their optimal parameters over time.
"""

from __future__ import annotations

import itertools
import logging
from typing import Callable

import numpy as np
import pandas as pd

import config
from src.engine.backtest import run_backtest
from src.risk.metrics import sharpe_ratio

logger = logging.getLogger(__name__)

CV_THRESHOLD = 0.50  # CV below this → "stable"


def parameter_stability_analysis(
    prices: pd.DataFrame,
    strategy_factory: Callable[..., Callable],
    param_grid: dict[str, list],
    window_years: int = 3,
    step_months: int = 3,
    rebalance_freq: str = config.REBALANCE_FREQ,
    cost_model=None,
) -> dict:
    """
    Rolling-window parameter stability analysis.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price matrix (DatetimeIndex, columns = tickers).
    strategy_factory : callable
        ``strategy_factory(**params)`` returns a signal function
        compatible with ``run_backtest``.
    param_grid : dict[str, list]
        Parameter name → list of candidate values.
    window_years : int
        Training window length in years.
    step_months : int
        Step size between windows in months.
    rebalance_freq : str
        Rebalance frequency passed to ``run_backtest``.
    cost_model
        Transaction cost model (or None for default proportional).

    Returns
    -------
    dict
        Keys:
        - parameter_timeseries : pd.DataFrame
            Index = window end date, columns = param names, values = best params.
        - parameter_cv : dict[str, float]
            Coefficient of variation per numeric parameter.
        - sharpe_timeseries : pd.Series
            Index = window end date, values = best Sharpe for that window.
        - is_stable : bool
            True if all numeric parameter CVs are below CV_THRESHOLD.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combos = list(itertools.product(*param_values))

    # Build rolling windows
    window_delta = pd.DateOffset(years=window_years)
    step_delta = pd.DateOffset(months=step_months)

    start_date = prices.index[0]
    end_date = prices.index[-1]

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start = start_date
    while True:
        window_end = current_start + window_delta
        if window_end > end_date:
            break
        windows.append((current_start, window_end))
        current_start = current_start + step_delta

    if not windows:
        raise ValueError(
            f"Not enough data for a {window_years}-year window "
            f"(data spans {start_date.date()} to {end_date.date()})"
        )

    logger.info(
        "Parameter stability: %d windows × %d combos = %d backtests",
        len(windows), len(all_combos), len(windows) * len(all_combos),
    )

    best_params_list: list[dict] = []
    best_sharpe_list: list[float] = []
    window_end_dates: list[pd.Timestamp] = []

    for i, (w_start, w_end) in enumerate(windows):
        window_prices = prices.loc[w_start:w_end]
        if len(window_prices) < 126:  # at least ~6 months of data
            continue

        best_sharpe = -np.inf
        best_combo = all_combos[0]

        for combo in all_combos:
            params = dict(zip(param_names, combo))
            signal_fn = strategy_factory(**params)
            result = run_backtest(
                window_prices, signal_fn,
                rebalance_freq=rebalance_freq,
                cost_model=cost_model,
            )
            sr = sharpe_ratio(result.daily_returns)
            if sr > best_sharpe:
                best_sharpe = sr
                best_combo = combo

        best_params = dict(zip(param_names, best_combo))
        best_params_list.append(best_params)
        best_sharpe_list.append(best_sharpe)
        window_end_dates.append(w_end)

        if (i + 1) % 5 == 0 or i == len(windows) - 1:
            logger.info(
                "      Window %d/%d (→%s): best Sharpe %.3f",
                i + 1, len(windows), w_end.date(), best_sharpe,
            )

    # Build output structures
    param_ts = pd.DataFrame(best_params_list, index=window_end_dates)
    sharpe_ts = pd.Series(best_sharpe_list, index=window_end_dates, name="best_sharpe")

    # Coefficient of variation for numeric parameters
    parameter_cv: dict[str, float] = {}
    for col in param_ts.columns:
        vals = pd.to_numeric(param_ts[col], errors="coerce")
        if vals.notna().sum() < 2:
            continue
        mean_val = vals.mean()
        if abs(mean_val) < 1e-10:
            parameter_cv[col] = 0.0
        else:
            parameter_cv[col] = float(vals.std() / abs(mean_val))

    is_stable = all(cv < CV_THRESHOLD for cv in parameter_cv.values())

    return {
        "parameter_timeseries": param_ts,
        "parameter_cv": parameter_cv,
        "sharpe_timeseries": sharpe_ts,
        "is_stable": is_stable,
    }
