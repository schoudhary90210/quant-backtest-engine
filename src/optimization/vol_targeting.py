"""
Volatility-targeting layer for portfolio weights.

Scales weights so that *ex-ante* portfolio volatility (sqrt(w^T Σ w))
matches a target, capped at max_leverage.  Assumes the covariance
matrix passed in is already annualized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class VolatilityTargeter:
    """
    Scale portfolio weights to target a given annualized volatility.

    Parameters
    ----------
    target_vol : float
        Annualized target volatility (e.g. 0.10 = 10%).
    max_leverage : float
        Maximum leverage multiplier (caps the scaling factor).
    """

    def __init__(
        self,
        target_vol: float = 0.10,
        max_leverage: float = 2.0,
    ) -> None:
        self._target_vol = target_vol
        self._max_leverage = max_leverage

    def get_leverage(
        self,
        weights: pd.Series,
        covariance: pd.DataFrame,
    ) -> float:
        """
        Compute the leverage multiplier.

        leverage = min(target_vol / portfolio_vol, max_leverage)

        Parameters
        ----------
        weights : pd.Series
            Raw portfolio weights (indexed by ticker).
        covariance : pd.DataFrame
            Annualized N×N covariance matrix.

        Returns
        -------
        float
            Leverage multiplier (≥ 0, ≤ max_leverage).
            Returns 1.0 if portfolio vol is near zero.
        """
        w = weights.reindex(covariance.columns, fill_value=0.0).values
        cov = covariance.values
        port_var = float(w @ cov @ w)

        if port_var < 1e-20:
            return 1.0

        port_vol = np.sqrt(port_var)
        return min(self._target_vol / port_vol, self._max_leverage)

    def apply(
        self,
        weights: pd.Series,
        covariance: pd.DataFrame,
    ) -> pd.Series:
        """
        Scale weights so that ex-ante portfolio vol ≈ target_vol.

        Parameters
        ----------
        weights : pd.Series
            Raw target weights from the base signal.
        covariance : pd.DataFrame
            Annualized N×N covariance matrix.

        Returns
        -------
        pd.Series
            Scaled weights = weights × leverage.
        """
        leverage = self.get_leverage(weights, covariance)
        return weights * leverage
