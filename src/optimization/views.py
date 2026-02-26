"""
View generation for the Black-Litterman model.

Generates an absolute momentum view for every asset in the universe:

    "I believe asset i will deliver Q_i in annual return terms."

P is the identity matrix (one absolute view per asset).
Q_i is the trailing lookback-period compounded log return, annualised.
Omega is a diagonal uncertainty matrix; its diagonal is set proportional to
the annualised asset variance scaled by tau, making the views weakly held
relative to the equilibrium prior.

Reference: He & Litterman (1999), Idzorek (2005).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def generate_momentum_views(
    returns: pd.DataFrame,
    cov_matrix: np.ndarray,
    lookback: int = config.BL_VIEW_LOOKBACK,
    tau: float = config.BL_TAU,
    view_confidence: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate absolute momentum views for the Black-Litterman model.

    Each view says: "Asset i will return Q_i per year."  The pick matrix P
    is the N×N identity (absolute, not relative, views).

    View uncertainty Omega is set so that views are held with roughly
    ``view_confidence`` weight relative to the equilibrium prior:

        Omega_ii = (1 / view_confidence - 1) * tau * Sigma_ii

    With ``view_confidence=0.25`` the prior receives 75 % of the weight and
    the momentum signal 25 %, preventing momentum from overwhelming the prior
    on short estimation windows.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical log returns, shape T × N.  Used for the trailing Q signal.
    cov_matrix : np.ndarray
        Annualised N × N covariance matrix.  Used to scale view uncertainty.
    lookback : int
        Number of trading days over which to measure trailing momentum.
        Defaults to 252 (one year).
    tau : float
        Prior uncertainty scaling factor from the BL model.
    view_confidence : float
        In (0, 1).  Fraction of posterior weight allocated to views.
        0 = no confidence in views (posterior equals prior), 1 = full
        confidence (posterior equals views).

    Returns
    -------
    P : np.ndarray, shape (N, N)
        Pick matrix — identity for absolute views.
    Q : np.ndarray, shape (N,)
        View return vector, annualised log return for each asset.
    omega : np.ndarray, shape (N, N)
        Diagonal view-uncertainty matrix.
    """
    if not 0.0 < view_confidence < 1.0:
        raise ValueError(f"view_confidence must be in (0, 1), got {view_confidence}")

    N = returns.shape[1]
    P = np.eye(N)

    # Trailing annualised return: sum of log returns over min(lookback, T) days,
    # then scale up to a full year if the window is shorter than lookback.
    window = min(lookback, len(returns))
    raw_total = returns.iloc[-window:].sum().values  # total log ret
    Q = raw_total * (float(lookback) / float(window))  # annualise

    # Omega_ii = [(1 / confidence) - 1] * tau * Sigma_ii
    # This makes the prior contribution = (1 - view_confidence) of the total.
    scale = (1.0 / view_confidence - 1.0) * tau
    omega = np.diag(np.diag(cov_matrix) * scale)

    return P, Q, omega
