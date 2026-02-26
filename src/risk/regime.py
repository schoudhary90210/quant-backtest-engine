"""
Market regime detection.

Classifies market regimes (bull/sideways/bear) based on
rolling volatility thresholds. Computes performance metrics
segmented by regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from src.risk.metrics import compute_risk_report, RiskReport

TRADING_DAYS = 252
REGIME_WINDOW = 60  # rolling days for vol estimation


@dataclass
class RegimeStats:
    """Performance summary for a single regime."""

    name: str
    n_days: int
    fraction: float
    report: RiskReport

    def __str__(self) -> str:
        return (
            f"  Regime: {self.name.upper()}  "
            f"({self.n_days} days, {self.fraction * 100:.1f}% of period)\n"
            + str(self.report)
        )


def classify_regimes(
    daily_returns: pd.Series,
    window: int = REGIME_WINDOW,
    thresholds: dict[str, float] | None = None,
) -> pd.Series:
    """
    Classify each trading day into a market regime.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio or benchmark returns.
    window : int
        Rolling window in trading days for volatility estimation.
    thresholds : dict or None
        Volatility thresholds (annualized). Defaults to config values.
        Keys: 'bull' (upper bound for bull), 'sideways' (upper bound for sideways).

    Returns
    -------
    pd.Series
        String labels ('bull', 'sideways', 'bear') indexed like daily_returns.
        First `window` days are NaN (insufficient history).
    """
    if thresholds is None:
        thresholds = config.REGIME_VOL_THRESHOLDS

    bull_threshold = thresholds["bull"]
    sideways_threshold = thresholds["sideways"]

    rolling_vol = daily_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)

    regimes = pd.Series(index=daily_returns.index, dtype=object)
    regimes[rolling_vol < bull_threshold] = "bull"
    regimes[(rolling_vol >= bull_threshold) & (rolling_vol < sideways_threshold)] = "sideways"
    regimes[rolling_vol >= sideways_threshold] = "bear"

    return regimes


def regime_stats(
    daily_returns: pd.Series,
    window: int = REGIME_WINDOW,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> dict[str, RegimeStats]:
    """
    Compute risk metrics segmented by market regime.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns.
    window : int
        Rolling window for regime classification.
    risk_free_rate : float
        Annualized risk-free rate.

    Returns
    -------
    dict[str, RegimeStats]
        Keys: 'bull', 'sideways', 'bear'. Only includes regimes with data.
    """
    regimes = classify_regimes(daily_returns, window)
    valid = regimes.dropna()
    total_days = len(valid)

    results: dict[str, RegimeStats] = {}
    for name in ("bull", "sideways", "bear"):
        mask = valid == name
        regime_returns = daily_returns.loc[valid.index[mask]]

        if len(regime_returns) < 2:
            continue

        report = compute_risk_report(regime_returns, risk_free_rate=risk_free_rate)
        results[name] = RegimeStats(
            name=name,
            n_days=len(regime_returns),
            fraction=len(regime_returns) / total_days if total_days > 0 else 0.0,
            report=report,
        )

    return results
