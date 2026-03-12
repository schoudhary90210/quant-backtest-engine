"""
Risk and performance metrics.

Sharpe ratio, Sortino ratio, maximum drawdown, Value-at-Risk,
Conditional VaR, Calmar ratio, and rolling statistics.

All functions accept a pd.Series of daily returns (simple or log).
Annualization assumes 252 trading days per year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

TRADING_DAYS = 252


@dataclass
class RiskReport:
    """Container for all computed risk metrics."""

    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    win_rate: float
    profit_factor: float

    def __str__(self) -> str:
        lines = [
            f"  Annualized Return:    {self.annualized_return * 100:>8.2f}%",
            f"  Annualized Volatility:{self.annualized_volatility * 100:>8.2f}%",
            f"  Sharpe Ratio:         {self.sharpe_ratio:>8.3f}",
            f"  Sortino Ratio:        {self.sortino_ratio:>8.3f}",
            f"  Max Drawdown:         {self.max_drawdown * 100:>8.2f}%",
            f"  Calmar Ratio:         {self.calmar_ratio:>8.3f}",
            f"  VaR (95%):            {self.var_95 * 100:>8.2f}%",
            f"  CVaR (95%):           {self.cvar_95 * 100:>8.2f}%",
            f"  Win Rate:             {self.win_rate * 100:>8.2f}%",
            f"  Profit Factor:        {self.profit_factor:>8.3f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def annualized_return(daily_returns: pd.Series) -> float:
    """
    Compound annualized return from a series of daily returns.

    Uses geometric compounding: (prod(1 + r_i))^(252/T) - 1
    """
    if len(daily_returns) == 0:
        return 0.0
    compound = (1.0 + daily_returns).prod()
    years = len(daily_returns) / TRADING_DAYS
    if years <= 0 or compound <= 0:
        return 0.0
    return float(compound ** (1.0 / years) - 1.0)


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns."""
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> float:
    """
    Annualized Sharpe ratio.

    (mean_daily_excess_return / std_daily_return) * sqrt(252)
    """
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - risk_free_rate / TRADING_DAYS
    vol = excess.std()
    if vol == 0.0:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(TRADING_DAYS))


def sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> float:
    """
    Annualized Sortino ratio.

    Uses downside deviation (only negative excess returns) in denominator.
    """
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - risk_free_rate / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.inf
    downside_std = float(np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS))
    if downside_std == 0.0:
        return 0.0
    ann_excess = float(excess.mean() * TRADING_DAYS)
    return ann_excess / downside_std


def max_drawdown(daily_returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown (negative number).

    Computed from the cumulative equity curve.
    """
    if len(daily_returns) == 0:
        return 0.0
    equity = (1.0 + daily_returns).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def calmar_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> float:
    """
    Calmar ratio: CAGR / |max drawdown|.

    Returns 0 if max drawdown is zero.
    """
    ann_ret = annualized_return(daily_returns)
    mdd = max_drawdown(daily_returns)
    if mdd == 0.0:
        return 0.0
    return float(ann_ret / abs(mdd))


def value_at_risk(
    daily_returns: pd.Series,
    confidence: float = config.VAR_CONFIDENCE,
) -> float:
    """
    Historical Value-at-Risk at given confidence level.

    Returns the loss threshold (negative number) not exceeded with
    `confidence` probability. E.g. VaR 95% = 5th percentile of returns.
    """
    if len(daily_returns) == 0:
        return 0.0
    return float(np.percentile(daily_returns, (1 - confidence) * 100))


def conditional_var(
    daily_returns: pd.Series,
    confidence: float = config.VAR_CONFIDENCE,
) -> float:
    """
    Conditional VaR (Expected Shortfall) at given confidence level.

    Mean of returns below the VaR threshold.
    """
    if len(daily_returns) == 0:
        return 0.0
    var = value_at_risk(daily_returns, confidence)
    tail = daily_returns[daily_returns <= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def win_rate(daily_returns: pd.Series) -> float:
    """Fraction of days with positive return."""
    if len(daily_returns) == 0:
        return 0.0
    return float((daily_returns > 0).mean())


def profit_factor(daily_returns: pd.Series) -> float:
    """
    Gross profit / gross loss.

    Returns inf if there are no losing days, 0 if no winning days.
    """
    if len(daily_returns) == 0:
        return 0.0
    gains = daily_returns[daily_returns > 0].sum()
    losses = daily_returns[daily_returns < 0].abs().sum()
    if losses == 0.0:
        return np.inf
    if gains == 0.0:
        return 0.0
    return float(gains / losses)


# ---------------------------------------------------------------------------
# Composite report
# ---------------------------------------------------------------------------


def compute_risk_report(
    daily_returns: pd.Series,
    risk_free_rate: float = config.RISK_FREE_RATE,
    confidence: float = config.VAR_CONFIDENCE,
) -> RiskReport:
    """Compute all risk metrics and return a RiskReport."""
    return RiskReport(
        annualized_return=annualized_return(daily_returns),
        annualized_volatility=annualized_volatility(daily_returns),
        sharpe_ratio=sharpe_ratio(daily_returns, risk_free_rate),
        sortino_ratio=sortino_ratio(daily_returns, risk_free_rate),
        max_drawdown=max_drawdown(daily_returns),
        calmar_ratio=calmar_ratio(daily_returns, risk_free_rate),
        var_95=value_at_risk(daily_returns, confidence),
        cvar_95=conditional_var(daily_returns, confidence),
        win_rate=win_rate(daily_returns),
        profit_factor=profit_factor(daily_returns),
    )


# ---------------------------------------------------------------------------
# Turnover & cost analysis
# ---------------------------------------------------------------------------


def turnover_analysis(
    backtest_result: object,
    cost_multipliers: list[float] | None = None,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> dict:
    """
    Analyse turnover and cost drag for a backtest result.

    Uses duck typing — accepts any object with `daily_returns`, `turnover`,
    `total_costs`, and `equity_curve` attributes (avoids circular import).

    Parameters
    ----------
    backtest_result : BacktestResult-like
        Must have .daily_returns, .turnover, .total_costs, .equity_curve.
    cost_multipliers : list[float] or None
        Multipliers for cost sensitivity analysis. Defaults to [0, 0.5, 1, 2, 3, 5].
    risk_free_rate : float
        Annualized risk-free rate for Sharpe computation.

    Returns
    -------
    dict
        Keys: annualized_turnover, avg_holding_period_days, total_cost_drag_bps,
              cost_sensitivity (list of {multiplier, sharpe} dicts).
    """
    if cost_multipliers is None:
        cost_multipliers = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

    daily_returns = backtest_result.daily_returns
    turnover_series = backtest_result.turnover
    cost_series = backtest_result.total_costs
    equity_curve = backtest_result.equity_curve

    n_days = len(daily_returns)
    years = n_days / TRADING_DAYS if n_days > 0 else 1.0

    # Total one-way turnover over the backtest
    total_turnover = float(turnover_series.sum())
    ann_turnover = total_turnover / years if years > 0 else 0.0

    # Average holding period
    avg_holding = TRADING_DAYS / ann_turnover if ann_turnover > 0 else float("inf")

    # Cost drag in bps/year
    avg_equity = float(equity_curve.mean()) if len(equity_curve) > 0 else 1.0
    total_costs = float(cost_series.sum())
    cost_drag_bps = (total_costs / avg_equity / years * 10_000) if years > 0 else 0.0

    # Cost sensitivity: adjust returns by varying cost multiplier
    cost_sensitivity: list[dict[str, float]] = []

    # Daily cost drag ratio
    cost_drag_daily = cost_series / equity_curve
    cost_drag_daily = cost_drag_daily.reindex(daily_returns.index, fill_value=0.0)

    for mult in cost_multipliers:
        # At mult=1 we get original returns; at mult=0 we add back all costs
        adjusted = daily_returns + cost_drag_daily * (1.0 - mult)
        sr = sharpe_ratio(adjusted, risk_free_rate)
        cost_sensitivity.append({"multiplier": mult, "sharpe": sr})

    return {
        "annualized_turnover": ann_turnover,
        "avg_holding_period_days": avg_holding,
        "total_cost_drag_bps": cost_drag_bps,
        "cost_sensitivity": cost_sensitivity,
    }
