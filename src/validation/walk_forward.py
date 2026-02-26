"""
Anchored walk-forward out-of-sample validation.

Re-estimates Kelly weights once per calendar month using an expanding training
window anchored at the first available price date.  No future information is
ever used when computing weights for any rebalance period — the window only
ever grows backward.

The existing run_backtest() engine is used without modification; the
walk-forward signal is a thin wrapper that looks up pre-computed weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

import config as cfg
from src.data.returns import compute_log_returns
from src.engine.backtest import BacktestResult, run_backtest
from src.optimization.black_litterman import BlackLittermanConfig, BlackLittermanEstimator
from src.optimization.covariance import LedoitWolfCovariance
from src.optimization.kelly import kelly_weights
from src.optimization.signals import make_kelly_signal
from src.risk.metrics import RiskReport, compute_risk_report

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    """Configuration for anchored walk-forward validation.

    Attributes
    ----------
    min_train_days : int
        Minimum number of trading-day returns required before a rebalance
        date is considered valid for weight estimation.  Defaults to 756
        (approximately 3 calendar years).
    rebalance_freq : str
        Cadence label for the walk-forward schedule.  ``"M"`` = monthly,
        matching the standard backtest rebalance frequency.
    start_date : str
        Inclusive start of the full price data window.
    end_date : str
        Inclusive end of the full price data window.
    """

    min_train_days: int = 756
    rebalance_freq: str = "M"   # "M" = monthly cadence
    start_date: str = cfg.START_DATE
    end_date: str = cfg.END_DATE


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Outputs from a walk-forward validation run.

    Attributes
    ----------
    in_sample_result : BacktestResult
        Full-period backtest using the standard rolling-lookback Kelly signal.
    oos_result : BacktestResult
        Full-period backtest using pre-computed expanding-window Kelly weights
        (equal-weight warm-up before ``oos_start_date``).
    weight_history : pd.DataFrame
        Monthly rebalance dates (DatetimeIndex) × tickers — Kelly weights
        computed with training data ending on each rebalance date.
    oos_start_date : pd.Timestamp
        First rebalance date with enough training data (>= ``min_train_days``).
    in_sample_metrics : RiskReport
        Risk report for the in-sample backtest.
    oos_metrics : RiskReport
        Risk report for the OOS backtest.
    """

    in_sample_result: BacktestResult
    oos_result: BacktestResult
    weight_history: pd.DataFrame
    oos_start_date: pd.Timestamp
    in_sample_metrics: RiskReport
    oos_metrics: RiskReport


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_walk_forward(
    prices: pd.DataFrame,
    wf_config: WalkForwardConfig | None = None,
    in_sample_result: BacktestResult | None = None,
    kelly_fraction: float = cfg.KELLY_FRACTION,
    cov_method: str = cfg.COV_METHOD,
    risk_free_rate: float = cfg.RISK_FREE_RATE,
    max_weight: float = cfg.MAX_POSITION_WEIGHT,
    max_leverage: float = cfg.MAX_LEVERAGE,
    initial_capital: float = cfg.INITIAL_CAPITAL,
    return_estimator: str = "historical_mean",
    bl_config: BlackLittermanConfig | None = None,
) -> WalkForwardResult:
    """
    Run anchored walk-forward out-of-sample validation.

    For each calendar month in the price data the function:

    1. Slices all available log returns from the first price date up to and
       including that rebalance date (expanding window — no lookahead).
    2. Fits a Ledoit-Wolf covariance matrix on this slice.
    3. Computes annualised mean returns on this slice.
    4. Solves for fractional Kelly weights.
    5. Stores the resulting weights indexed by the rebalance date.

    Rebalance dates where the training slice is shorter than
    ``wf_config.min_train_days`` are skipped; the backtest engine uses
    equal-weight fallback during the warm-up period.

    The walk-forward result is then compared against a standard in-sample
    Kelly backtest (rolling ``LOOKBACK_WINDOW``-day window).

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.  Columns = tickers, index = DatetimeIndex.
    wf_config : WalkForwardConfig or None
        Walk-forward settings.  Uses defaults if None.
    in_sample_result : BacktestResult or None
        Pre-computed in-sample result to avoid redundant computation.
        If None, a standard Kelly backtest is run internally.
    kelly_fraction : float
        Fractional Kelly multiplier (0.5 = half-Kelly).
    cov_method : str
        Covariance estimator for the *in-sample* signal only.  Walk-forward
        always uses Ledoit-Wolf for numerical stability on short windows.
    risk_free_rate : float
        Annualised risk-free rate.
    max_weight : float
        Per-asset weight cap.
    max_leverage : float
        Gross leverage cap.
    initial_capital : float
        Starting portfolio value for both backtests.
    return_estimator : str
        ``"historical_mean"`` (default) uses raw annualised mean log returns.
        ``"black_litterman"`` replaces the historical mean with the BL
        posterior (equilibrium prior + expanding-window momentum views).
    bl_config : BlackLittermanConfig or None
        BL hyperparameters.  Only used when ``return_estimator="black_litterman"``.
        Defaults to ``BlackLittermanConfig()``.

    Returns
    -------
    WalkForwardResult
    """
    if wf_config is None:
        wf_config = WalkForwardConfig()

    if prices.empty:
        raise ValueError("prices DataFrame is empty")

    tickers = prices.columns.tolist()

    # ── 1. Log returns for the full price history ──────────────────────────
    log_returns = compute_log_returns(prices)

    # ── 2. Monthly rebalance dates (first trading day of each calendar month) ─
    rebalance_dates = _monthly_rebalance_dates(prices.index)

    # ── 3. Expanding-window weight estimation ──────────────────────────────
    cov_estimator = LedoitWolfCovariance()
    bl_estimator: BlackLittermanEstimator | None = (
        BlackLittermanEstimator(bl_config)
        if return_estimator == "black_litterman"
        else None
    )
    weight_records: list[dict] = []
    oos_start_date: pd.Timestamp | None = None

    for date in rebalance_dates:
        # Training slice: all returns up to and including this rebalance date.
        # log_returns.loc[:date] never includes any return after `date`.
        train_returns = log_returns.loc[:date]

        if len(train_returns) < wf_config.min_train_days:
            logger.debug(
                "Walk-forward skip %s: %d training days < min %d",
                date.date(),
                len(train_returns),
                wf_config.min_train_days,
            )
            continue

        if oos_start_date is None:
            oos_start_date = date

        # Annualised Ledoit-Wolf covariance from this expanding window
        cov: pd.DataFrame = cov_estimator.estimate(train_returns) * TRADING_DAYS

        # Expected returns — either raw historical mean or BL posterior
        if bl_estimator is not None:
            mu = bl_estimator.estimate(train_returns, cov)
        else:
            mu = train_returns.mean() * TRADING_DAYS

        w = kelly_weights(
            expected_returns=mu,
            cov_matrix=cov,
            risk_free_rate=risk_free_rate,
            fraction=kelly_fraction,
            max_weight=max_weight,
            max_leverage=max_leverage,
        )

        weight_records.append({"date": date, **w.to_dict()})

    if not weight_records:
        raise ValueError(
            f"No rebalance dates have >= {wf_config.min_train_days} training days. "
            "Extend the price history or reduce min_train_days."
        )

    weight_history = (
        pd.DataFrame(weight_records)
        .set_index("date")
        .reindex(columns=tickers, fill_value=0.0)
    )

    assert oos_start_date is not None  # guaranteed by the check above
    logger.info(
        "Walk-forward (%s): %d monthly windows | OOS start: %s",
        return_estimator,
        len(weight_history),
        oos_start_date.date(),
    )

    # ── 4. Walk-forward signal (wraps pre-computed weights) ────────────────
    wf_signal = _walk_forward_signal(weight_history, tickers)

    # ── 5. In-sample backtest (standard rolling-lookback Kelly) ────────────
    if in_sample_result is None:
        is_signal = make_kelly_signal(
            fraction=kelly_fraction,
            cov_method=cov_method,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
            max_leverage=max_leverage,
        )
        in_sample_result = run_backtest(
            prices,
            is_signal,
            initial_capital=initial_capital,
            rebalance_freq="monthly",
        )

    # ── 6. OOS backtest (walk-forward weights) ─────────────────────────────
    oos_result = run_backtest(
        prices,
        wf_signal,
        initial_capital=initial_capital,
        rebalance_freq="monthly",
    )

    # ── 7. Risk reports ────────────────────────────────────────────────────
    is_metrics = compute_risk_report(in_sample_result.daily_returns)
    oos_metrics = compute_risk_report(oos_result.daily_returns)

    return WalkForwardResult(
        in_sample_result=in_sample_result,
        oos_result=oos_result,
        weight_history=weight_history,
        oos_start_date=oos_start_date,
        in_sample_metrics=is_metrics,
        oos_metrics=oos_metrics,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _monthly_rebalance_dates(price_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the first trading date of each calendar month in *price_index*.

    Uses vectorised groupby — no Python loop over individual trading days.

    Parameters
    ----------
    price_index : pd.DatetimeIndex
        All trading dates in the price DataFrame.

    Returns
    -------
    pd.DatetimeIndex
        One date per calendar month: the first trading day of that month.
    """
    s = pd.Series(price_index, index=price_index, dtype="datetime64[ns]")
    first_per_month = s.groupby(price_index.to_period("M")).first()
    return pd.DatetimeIndex(first_per_month.values)


def _walk_forward_signal(
    weight_history: pd.DataFrame,
    tickers: list[str],
) -> Callable[..., pd.Series]:
    """
    Build a SignalProvider that returns pre-computed walk-forward weights.

    At each rebalance date the signal looks up the most recent entry in
    ``weight_history`` whose index date is <= the current rebalance date.
    During the warm-up period (before any entry in weight_history) the signal
    returns equal weights.

    The returned callable ignores the ``prices`` argument entirely, so it
    cannot inadvertently observe future price data — lookahead is structurally
    impossible.

    Parameters
    ----------
    weight_history : pd.DataFrame
        Pre-computed weights indexed by rebalance date.
    tickers : list[str]
        Full list of asset tickers (column order for alignment).

    Returns
    -------
    callable
        SignalProvider-compatible function.
    """
    n = len(tickers)
    equal_w = pd.Series(1.0 / n, index=tickers)

    def signal_fn(
        date: pd.Timestamp,
        prices: pd.DataFrame,          # not used — prevents any future lookahead
        current_weights: pd.Series,
    ) -> pd.Series:
        available = weight_history.loc[weight_history.index <= date]
        if available.empty:
            return equal_w.copy()
        return available.iloc[-1].reindex(tickers, fill_value=0.0)

    return signal_fn
