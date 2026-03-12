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
from src.optimization.black_litterman import (
    BlackLittermanConfig,
    BlackLittermanEstimator,
)
from src.optimization.covariance import CovarianceEstimator, get_covariance_estimator
from src.optimization.hrp import HierarchicalRiskParity
from src.optimization.kelly import kelly_weights
from src.optimization.risk_budget import RiskBudget

logger = logging.getLogger(__name__)


def make_kelly_signal(
    lookback: int = config.LOOKBACK_WINDOW,
    cov_method: str = config.COV_METHOD,
    fraction: float = config.KELLY_FRACTION,
    risk_free_rate: float = config.RISK_FREE_RATE,
    max_weight: float = config.MAX_POSITION_WEIGHT,
    max_leverage: float = config.MAX_LEVERAGE,
    trading_days: int = 252,
    max_turnover: float | None = None,
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
        window_prices = prices.iloc[-(lookback + 1) :]
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
            current_weights=current_weights,
            max_turnover=max_turnover,
        )

        return weights

    return signal_fn


def make_bl_kelly_signal(
    lookback: int = config.LOOKBACK_WINDOW,
    cov_method: str = config.COV_METHOD,
    fraction: float = config.KELLY_FRACTION,
    risk_free_rate: float = config.RISK_FREE_RATE,
    max_weight: float = config.MAX_POSITION_WEIGHT,
    max_leverage: float = config.MAX_LEVERAGE,
    bl_config: BlackLittermanConfig | None = None,
    trading_days: int = 252,
    max_turnover: float | None = None,
) -> Callable[..., pd.Series]:
    """
    Create a Black-Litterman + Kelly criterion signal function.

    Identical pipeline to ``make_kelly_signal`` except that expected returns
    are estimated via the BL posterior (equilibrium prior + momentum views)
    rather than from raw historical means.

    Parameters
    ----------
    lookback : int
        Rolling lookback window in trading days.
    cov_method : str
        Covariance estimator name (default from ``config.COV_METHOD``).
    fraction : float
        Kelly fraction (0.5 = half-Kelly).
    risk_free_rate : float
        Annualised risk-free rate.
    max_weight : float
        Per-asset weight cap.
    max_leverage : float
        Gross leverage cap.
    bl_config : BlackLittermanConfig or None
        BL hyperparameters.  Defaults to BlackLittermanConfig().
    trading_days : int
        Trading days per year for annualisation.

    Returns
    -------
    callable
        SignalProvider-compatible function.
    """
    cov_estimator = get_covariance_estimator(cov_method)
    bl_estimator = BlackLittermanEstimator(bl_config)

    def signal_fn(
        date: pd.Timestamp,
        prices: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        n_assets = len(current_weights)

        if len(prices) < lookback + 1:
            return pd.Series(1.0 / n_assets, index=current_weights.index)

        window_prices = prices.iloc[-(lookback + 1) :]
        log_returns = compute_log_returns(window_prices)

        # Annualised covariance (estimator selected by cov_method)
        cov = cov_estimator.estimate(log_returns) * trading_days

        # BL posterior returns (replaces raw historical mean)
        mu = bl_estimator.estimate(log_returns, cov)

        weights = kelly_weights(
            expected_returns=mu,
            cov_matrix=cov,
            risk_free_rate=risk_free_rate,
            fraction=fraction,
            max_weight=max_weight,
            max_leverage=max_leverage,
            current_weights=current_weights,
            max_turnover=max_turnover,
        )

        return weights

    return signal_fn


# ---------------------------------------------------------------------------
# Turnover limiting helper (shared by HRP / risk parity)
# ---------------------------------------------------------------------------


def _apply_turnover_limit(
    target: pd.Series,
    current: pd.Series,
    max_turnover: float | None,
) -> pd.Series:
    """Proportional scaling of trade to respect turnover limit."""
    if max_turnover is None:
        return target
    trade = target - current.reindex(target.index, fill_value=0.0)
    one_way = float(np.sum(np.abs(trade.values))) / 2.0
    if one_way > max_turnover and one_way > 1e-10:
        scale = max_turnover / one_way
        result = current.reindex(target.index, fill_value=0.0) + trade * scale
        return result.clip(lower=0.0)
    return target


# ---------------------------------------------------------------------------
# Volatility targeting wrapper
# ---------------------------------------------------------------------------


def wrap_vol_target(
    base_signal: Callable[..., pd.Series],
    target_vol: float = config.TARGET_VOL,
    max_leverage: float = config.VOL_TARGET_MAX_LEVERAGE,
    lookback: int = 63,
    cov_method: str = config.COV_METHOD,
    max_turnover: float | None = None,
    trading_days: int = 252,
) -> Callable[..., pd.Series]:
    """
    Wrap any signal function with volatility targeting.

    Pipeline:
    1. Call base_signal to get raw weights
    2. Estimate annualized covariance from recent prices (rolling window)
    3. Scale weights via VolatilityTargeter.apply(raw, cov)
    4. Re-apply turnover limit (scaling may have violated the base cap)

    Parameters
    ----------
    base_signal : callable
        Base signal function matching SignalProvider protocol.
    target_vol : float
        Annualized target volatility.
    max_leverage : float
        Max leverage multiplier.
    lookback : int
        Rolling lookback for covariance estimation.
    cov_method : str
        Covariance estimator name.
    max_turnover : float or None
        Per-rebalance one-way turnover cap (re-applied after scaling).
    trading_days : int
        Trading days per year for annualization.

    Returns
    -------
    callable
        Wrapped signal function (same signature as base_signal).
    """
    from src.optimization.vol_targeting import VolatilityTargeter

    targeter = VolatilityTargeter(target_vol=target_vol, max_leverage=max_leverage)
    cov_estimator = get_covariance_estimator(cov_method)
    leverage_history: list[tuple[pd.Timestamp, float]] = []

    # DCC-GARCH needs a longer window; use at least config.DCC_GARCH_WINDOW
    effective_lookback = (
        max(lookback, config.DCC_GARCH_WINDOW)
        if cov_method == "dcc_garch"
        else lookback
    )

    def signal_fn(
        date: pd.Timestamp,
        prices: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        raw = base_signal(date, prices, current_weights)

        # Need enough history for covariance estimation
        if len(prices) < effective_lookback + 1:
            leverage_history.append((date, 1.0))
            return raw

        window_prices = prices.iloc[-(effective_lookback + 1) :]
        log_returns = compute_log_returns(window_prices)
        cov = cov_estimator.estimate(log_returns) * trading_days  # annualize

        leverage = targeter.get_leverage(raw, cov)
        leverage_history.append((date, leverage))
        scaled = raw * leverage

        # Re-apply turnover limit after scaling
        scaled = _apply_turnover_limit(scaled, current_weights, max_turnover)

        return scaled

    # Attach for diagnostics
    signal_fn.targeter = targeter  # type: ignore[attr-defined]
    signal_fn.leverage_history = leverage_history  # type: ignore[attr-defined]
    return signal_fn


def make_vol_targeted_bl_signal(
    lookback: int = config.LOOKBACK_WINDOW,
    cov_method: str = config.COV_METHOD,
    fraction: float = config.KELLY_FRACTION,
    risk_free_rate: float = config.RISK_FREE_RATE,
    max_weight: float = config.MAX_POSITION_WEIGHT,
    max_leverage: float = config.MAX_LEVERAGE,
    bl_config: BlackLittermanConfig | None = None,
    trading_days: int = 252,
    max_turnover: float | None = None,
    target_vol: float = config.TARGET_VOL,
    vol_max_leverage: float = config.VOL_TARGET_MAX_LEVERAGE,
    vol_lookback: int = 63,
) -> Callable[..., pd.Series]:
    """
    BL + Kelly signal with volatility targeting overlay.

    Composes make_bl_kelly_signal() with wrap_vol_target().
    """
    # DCC-GARCH needs a longer window for the base signal too
    base_lookback = (
        max(lookback, config.DCC_GARCH_WINDOW)
        if cov_method == "dcc_garch"
        else lookback
    )
    base = make_bl_kelly_signal(
        lookback=base_lookback,
        cov_method=cov_method,
        fraction=fraction,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        max_leverage=max_leverage,
        bl_config=bl_config,
        trading_days=trading_days,
        max_turnover=max_turnover,
    )
    return wrap_vol_target(
        base,
        target_vol=target_vol,
        max_leverage=vol_max_leverage,
        lookback=vol_lookback,
        cov_method=cov_method,
        max_turnover=max_turnover,
        trading_days=trading_days,
    )


# ---------------------------------------------------------------------------
# HRP signal factory
# ---------------------------------------------------------------------------


def make_hrp_signal(
    lookback: int = config.LOOKBACK_WINDOW,
    cov_method: str = config.COV_METHOD,
    linkage_method: str = "ward",
    trading_days: int = 252,
    max_turnover: float | None = None,
) -> Callable[..., pd.Series]:
    """
    Create an HRP signal function.

    Parameters
    ----------
    lookback : int
        Rolling lookback window in trading days.
    cov_method : str
        Covariance estimator name.
    linkage_method : str
        Linkage method for hierarchical clustering.
    trading_days : int
        Trading days per year for annualisation.
    max_turnover : float or None
        Per-rebalance one-way turnover cap.

    Returns
    -------
    callable
        SignalProvider-compatible function.
    """
    cov_estimator = get_covariance_estimator(cov_method)
    hrp = HierarchicalRiskParity(linkage_method=linkage_method)

    def signal_fn(
        date: pd.Timestamp,
        prices: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        n_assets = len(current_weights)

        if len(prices) < lookback + 1:
            return pd.Series(1.0 / n_assets, index=current_weights.index)

        window_prices = prices.iloc[-(lookback + 1) :]
        log_returns = compute_log_returns(window_prices)

        cov = cov_estimator.estimate(log_returns) * trading_days
        weights = hrp.optimize(log_returns, cov)

        return _apply_turnover_limit(weights, current_weights, max_turnover)

    signal_fn.hrp_optimizer = hrp  # type: ignore[attr-defined]
    return signal_fn


# ---------------------------------------------------------------------------
# Risk parity signal factory
# ---------------------------------------------------------------------------


def make_risk_parity_signal(
    lookback: int = config.LOOKBACK_WINDOW,
    cov_method: str = config.COV_METHOD,
    budgets: dict[str, float] | None = None,
    trading_days: int = 252,
    max_turnover: float | None = None,
) -> Callable[..., pd.Series]:
    """
    Create a risk-parity (equal risk contribution) signal function.

    Parameters
    ----------
    lookback : int
        Rolling lookback window in trading days.
    cov_method : str
        Covariance estimator name.
    budgets : dict[str, float] or None
        Target risk budgets per asset.  None = equal risk contribution.
    trading_days : int
        Trading days per year for annualisation.
    max_turnover : float or None
        Per-rebalance one-way turnover cap.

    Returns
    -------
    callable
        SignalProvider-compatible function.
    """
    cov_estimator = get_covariance_estimator(cov_method)
    rb = RiskBudget(budgets=budgets)

    def signal_fn(
        date: pd.Timestamp,
        prices: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        n_assets = len(current_weights)

        if len(prices) < lookback + 1:
            return pd.Series(1.0 / n_assets, index=current_weights.index)

        window_prices = prices.iloc[-(lookback + 1) :]
        log_returns = compute_log_returns(window_prices)

        cov = cov_estimator.estimate(log_returns) * trading_days
        weights = rb.optimize(cov)

        return _apply_turnover_limit(weights, current_weights, max_turnover)

    return signal_fn


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------


def get_strategy_registry(
    cov_method: str = config.COV_METHOD,
    max_turnover: float | None = config.MAX_TURNOVER,
) -> dict[str, tuple[str, Callable[..., pd.Series]]]:
    """
    Return strategy ID -> (display_label, signal_fn) mapping.

    Reuses existing signal factories. All seven supported strategies
    are always included; callers filter by config.STRATEGIES_TO_RUN.
    """
    from src.optimization.equal_weight import equal_weight_signal

    return {
        "equal_weight": ("Equal Weight", equal_weight_signal),
        "half_kelly": ("Half-Kelly", make_kelly_signal(
            fraction=0.5, cov_method=cov_method, max_turnover=max_turnover,
        )),
        "full_kelly": ("Full Kelly", make_kelly_signal(
            fraction=1.0, cov_method=cov_method, max_turnover=max_turnover,
        )),
        "bl_kelly": ("BL + Kelly", make_bl_kelly_signal(
            fraction=0.5, cov_method=cov_method, max_turnover=max_turnover,
        )),
        "hrp": ("HRP", make_hrp_signal(
            cov_method=cov_method, max_turnover=max_turnover,
        )),
        "risk_parity": ("Risk Parity", make_risk_parity_signal(
            cov_method=cov_method, max_turnover=max_turnover,
        )),
        "vol_targeted_bl": ("Vol-Targeted BL", make_vol_targeted_bl_signal(
            cov_method=cov_method, max_turnover=max_turnover,
        )),
    }
