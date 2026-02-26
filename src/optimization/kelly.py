"""
Multi-asset Kelly Criterion optimizer.

Computes optimal portfolio weights via f* = C^{-1}(μ − r), then applies:
  - Fractional Kelly scaling (e.g. 0.5 for Half-Kelly)
  - Position cap per asset (default 40%)
  - Total gross leverage cap (default 1.5×)
  - Zero-weight for assets with negative expected excess return

Uses np.linalg.solve instead of matrix inversion for numerical stability.
Falls back to least-squares (np.linalg.lstsq) if the covariance is singular.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def kelly_weights(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = config.RISK_FREE_RATE,
    fraction: float = config.KELLY_FRACTION,
    max_weight: float = config.MAX_POSITION_WEIGHT,
    max_leverage: float = config.MAX_LEVERAGE,
) -> pd.Series:
    """
    Compute constrained fractional Kelly weights.

    Parameters
    ----------
    expected_returns : pd.Series
        Annualized expected return per asset (indexed by ticker).
    cov_matrix : pd.DataFrame
        N×N annualized covariance matrix.
    risk_free_rate : float
        Annualized risk-free rate.
    fraction : float
        Kelly fraction (1.0 = full Kelly, 0.5 = half-Kelly).
    max_weight : float
        Maximum absolute weight per asset.
    max_leverage : float
        Maximum gross exposure (sum of |weights|).

    Returns
    -------
    pd.Series
        Optimal portfolio weights indexed by ticker.
    """
    tickers = expected_returns.index
    n = len(tickers)

    if n == 0:
        return pd.Series(dtype=float)

    # ── Excess returns ──
    excess = expected_returns.values - risk_free_rate

    # ── Zero out assets with negative expected excess return ──
    mask = excess > 0
    if not mask.any():
        logger.warning(
            "All assets have negative excess returns; returning zero weights"
        )
        return pd.Series(0.0, index=tickers)

    # Work only with positive-excess-return assets
    active_idx = np.where(mask)[0]
    active_excess = excess[active_idx]
    active_cov = cov_matrix.values[np.ix_(active_idx, active_idx)]

    # ── Solve C @ f = (μ − r) for raw Kelly weights ──
    raw_weights = _solve_kelly(active_cov, active_excess)

    # ── Apply fractional Kelly ──
    raw_weights = raw_weights * fraction

    # ── Clip negative weights to zero (long-only after excess return filter) ──
    raw_weights = np.maximum(raw_weights, 0.0)

    # ── Apply per-position cap ──
    raw_weights = np.minimum(raw_weights, max_weight)

    # ── Apply leverage cap ──
    gross = np.sum(np.abs(raw_weights))
    if gross > max_leverage:
        raw_weights = raw_weights * (max_leverage / gross)

    # ── Build full weight vector ──
    full_weights = np.zeros(n)
    full_weights[active_idx] = raw_weights

    return pd.Series(full_weights, index=tickers)


def _solve_kelly(cov: np.ndarray, excess_returns: np.ndarray) -> np.ndarray:
    """
    Solve C @ f = excess_returns using np.linalg.solve.

    Falls back to least-squares if the covariance matrix is singular.
    """
    try:
        weights = np.linalg.solve(cov, excess_returns)
    except np.linalg.LinAlgError:
        logger.warning("Singular covariance matrix; falling back to lstsq")
        weights, _, _, _ = np.linalg.lstsq(cov, excess_returns, rcond=None)
    return weights
