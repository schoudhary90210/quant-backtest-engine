"""
Signal factories that wire optimizers into the backtester.

Each factory returns a callable matching the SignalProvider protocol:
    signal_fn(date, prices, current_weights) → target_weights
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

import config
from src.data.returns import compute_log_returns
from src.optimization.covariance import CovarianceEstimator, get_covariance_estimator
from src.optimization.kelly import kelly_weights

logger = logging.getLogger(__name__)


def make_kelly_signal(
    lookback: int = config.LOOKBACK_WINDOW,
    cov_method: str = config.COV_METHOD,
    fraction: float = config.KELLY_FRACTION,
    risk_free_rate: float = config.RISK_FREE_RATE,
    max_weight: float = config.MAX_POSITION_WEIGHT,
    max_leverage: float = config.MAX_LEVERAGE,
    trading_days: int = 252,
) -> Callable[..., pd.Series]:
    """
    Create a Kelly criterion signal function.

    The returned callable estimates expected returns and covariance from
    a rolling lookback window of prices, then computes Kelly weights.

    Parameters
    ----------
    lookback : int
        Number of trading days for return/covariance estimation.
    cov_method : str
        Covariance estimator name ('sample', 'ledoit_wolf', 'ewma').
    fraction : float
        Kelly fraction (0.5 = half-Kelly).
    risk_free_rate : float
        Annualized risk-free rate.
    max_weight : float
        Maximum per-asset weight.
    max_leverage : float
        Maximum gross leverage.
    trading_days : int
        Trading days per year for annualization.

    Returns
    -------
    callable
        SignalProvider-compatible function.
    """
    cov_estimator = get_covariance_estimator(cov_method)

    def signal_fn(
        date: pd.Timestamp,
        prices: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        n_assets = len(current_weights)

        # Need enough history for the lookback window
        if len(prices) < lookback + 1:
            # Not enough data yet — hold equal weight as warm-up
            return pd.Series(1.0 / n_assets, index=current_weights.index)

        # Use the last `lookback` days of prices
        window_prices = prices.iloc[-(lookback + 1):]
        log_returns = compute_log_returns(window_prices)

        # Annualized expected returns (mean daily log return × 252)
        mu = log_returns.mean() * trading_days

        # Annualized covariance (daily cov × 252)
        cov = cov_estimator.estimate(log_returns) * trading_days

        weights = kelly_weights(
            expected_returns=mu,
            cov_matrix=cov,
            risk_free_rate=risk_free_rate,
            fraction=fraction,
            max_weight=max_weight,
            max_leverage=max_leverage,
        )

        return weights

    return signal_fn
