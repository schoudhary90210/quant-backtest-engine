"""
Covariance matrix estimation.

Three swappable estimators sharing a common interface:
  - SampleCovariance:    unbiased sample covariance (baseline)
  - LedoitWolfCovariance: Ledoit-Wolf shrinkage estimator (sklearn)
  - EWMACovariance:       exponentially-weighted moving average

All estimators implement ``estimate(returns) → pd.DataFrame`` returning
an N×N covariance matrix indexed by ticker symbols.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import config


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class CovarianceEstimator(Protocol):
    """Common interface for covariance estimators."""

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate the covariance matrix from a returns DataFrame.

        Parameters
        ----------
        returns : pd.DataFrame
            T×N log-return matrix (rows = dates, columns = tickers).

        Returns
        -------
        pd.DataFrame
            N×N covariance matrix with ticker-labeled rows and columns.
        """
        ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

class SampleCovariance:
    """Unbiased sample covariance matrix (ddof=1)."""

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")
        return returns.cov()  # pandas default ddof=1


class LedoitWolfCovariance:
    """
    Ledoit-Wolf shrinkage covariance estimator.

    Shrinks the sample covariance toward a structured target (scaled identity)
    to reduce estimation error, especially when T ≈ N.
    """

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")

        tickers = returns.columns
        lw = LedoitWolf().fit(returns.values)
        cov_matrix = lw.covariance_

        return pd.DataFrame(cov_matrix, index=tickers, columns=tickers)


class EWMACovariance:
    """
    Exponentially-weighted moving average covariance.

    Recent observations get exponentially higher weight, controlled by
    ``decay`` (lambda). Higher lambda → slower decay → longer memory.

    Parameters
    ----------
    decay : float
        Decay factor (lambda), typically 0.94 for daily data (RiskMetrics).
    """

    def __init__(self, decay: float = config.EWMA_LAMBDA):
        if not 0 < decay < 1:
            raise ValueError(f"Decay must be in (0, 1), got {decay}")
        self.decay = decay

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")

        tickers = returns.columns
        # pandas ewm computes the EWMA covariance directly
        # span = 2/(1-lambda) - 1 converts decay to span for ewm
        # But more directly: use com = decay / (1 - decay)
        com = self.decay / (1.0 - self.decay)
        cov_matrix = returns.ewm(com=com).cov().iloc[-len(tickers):]
        # The last N rows of the multi-indexed result are the final estimate
        cov_matrix = cov_matrix.droplevel(0)
        cov_matrix.index = tickers
        cov_matrix.columns = tickers

        return cov_matrix


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_covariance_estimator(method: str = config.COV_METHOD) -> CovarianceEstimator:
    """
    Factory function to get a covariance estimator by name.

    Parameters
    ----------
    method : str
        'sample', 'ledoit_wolf', or 'ewma'.

    Returns
    -------
    CovarianceEstimator
    """
    estimators = {
        "sample": SampleCovariance,
        "ledoit_wolf": LedoitWolfCovariance,
        "ewma": EWMACovariance,
    }
    if method not in estimators:
        raise ValueError(
            f"Unknown covariance method '{method}'. "
            f"Choose from: {list(estimators.keys())}"
        )
    return estimators[method]()
