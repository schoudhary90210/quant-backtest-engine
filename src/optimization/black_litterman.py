"""
Black-Litterman return estimator.

Implements the He–Litterman / Idzorek formulation of the Black-Litterman
model in three composable steps:

  1. Equilibrium prior:   pi   = delta * Sigma * w_mkt
  2. Momentum views:      (P, Q, Omega)  via src.optimization.views
  3. Bayesian posterior:  mu_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
                                  [(tau*Sigma)^-1 * pi + P'*Omega^-1 * Q]

All matrix systems are solved with np.linalg.solve (never np.linalg.inv)
for numerical stability.

References
----------
Black, F. & Litterman, R. (1992).  "Global Portfolio Optimization."
    Financial Analysts Journal, 48(5), 28–43.
He, G. & Litterman, R. (1999).  "The intuition behind Black-Litterman
    model portfolios."  Goldman Sachs Investment Management Research.
Idzorek, T. (2005).  "A step-by-step guide to the Black-Litterman model."
    Zephyr Associates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from src.optimization.views import generate_momentum_views

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BlackLittermanConfig:
    """Hyperparameters for the Black-Litterman model.

    Attributes
    ----------
    risk_aversion : float
        Market risk-aversion coefficient (delta).  Calibrated so that
        equilibrium returns roughly equal observed historical excess returns
        for a well-diversified portfolio.  Standard value: 2.5.
    tau : float
        Uncertainty of the prior relative to the covariance matrix.
        Scales the prior covariance as tau * Sigma.  Typical range: 0.01–0.10.
        Smaller tau gives the prior more weight.
    use_views : bool
        When False the estimator returns pure equilibrium returns (no
        momentum signal is applied).  Useful as a baseline comparison.
    view_confidence : float
        Weight allocated to the momentum views vs the equilibrium prior.
        See views.generate_momentum_views for details.
    view_lookback : int
        Trading-day window for the trailing momentum signal.
    """

    risk_aversion: float = config.BL_RISK_AVERSION
    tau: float = config.BL_TAU
    use_views: bool = True
    view_confidence: float = 0.25
    view_lookback: int = config.BL_VIEW_LOOKBACK


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_equilibrium_returns(
    cov_matrix: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float = config.BL_RISK_AVERSION,
) -> np.ndarray:
    """
    Compute equilibrium (implied) excess returns.

        pi = delta * Sigma * w_mkt

    Parameters
    ----------
    cov_matrix : np.ndarray
        Annualised N × N covariance matrix.
    market_weights : np.ndarray
        Market-cap weights, shape (N,).  Must sum to 1 or will be normalised.
    risk_aversion : float
        Market risk-aversion coefficient (delta).

    Returns
    -------
    np.ndarray
        Equilibrium expected excess return vector, shape (N,).
    """
    w = np.asarray(market_weights, dtype=float)
    total = w.sum()
    if total > 0:
        w = w / total

    return risk_aversion * cov_matrix @ w


def compute_bl_posterior(
    pi: np.ndarray,
    cov_matrix: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    tau: float = config.BL_TAU,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Black-Litterman posterior mean and covariance.

    Precision-weighted Bayesian combination of the equilibrium prior and
    investor views (He–Litterman, 1999; Idzorek, 2005):

        mu_BL  = A^{-1} b
        Cov_BL = A^{-1}

    where

        A = (tau * Sigma)^{-1}  +  P' Omega^{-1} P
        b = (tau * Sigma)^{-1} pi  +  P' Omega^{-1} Q

    All systems are solved with np.linalg.solve — no explicit matrix
    inversion is performed.

    Parameters
    ----------
    pi : np.ndarray
        Equilibrium expected return vector, shape (N,).
    cov_matrix : np.ndarray
        Annualised covariance matrix, shape (N, N).
    P : np.ndarray
        Pick matrix, shape (K, N).
    Q : np.ndarray
        View return vector, shape (K,).
    omega : np.ndarray
        Diagonal view-uncertainty matrix, shape (K, K).
    tau : float
        Prior uncertainty scaling factor.

    Returns
    -------
    mu_bl : np.ndarray
        Posterior expected returns, shape (N,).
    cov_bl : np.ndarray
        Posterior covariance matrix, shape (N, N).
    """
    N = len(pi)
    K = len(Q)
    tau_cov = tau * cov_matrix                          # (N, N)

    # (tau * Sigma)^{-1} — solve tau_cov @ X = I for X
    # Using solve instead of inv: solve the N systems tau_cov @ x_i = e_i
    tau_cov_inv = np.linalg.solve(tau_cov, np.eye(N))  # (N, N)

    # Omega is diagonal; use element-wise operations to avoid full K×K inversion
    omega_diag = np.diag(omega)                         # (K,)
    if np.any(omega_diag <= 0):
        raise ValueError("omega diagonal must be strictly positive")

    # P' Omega^{-1} P  — shape (N, N)
    # Row k of P scaled by 1/omega_k, then P.T @ result
    P_scaled = P / omega_diag[:, np.newaxis]            # (K, N)
    PtOP = P.T @ P_scaled                               # (N, N)

    # P' Omega^{-1} Q  — shape (N,)
    PtOQ = P.T @ (Q / omega_diag)                       # (N,)

    # Precision matrix A and right-hand side b
    A = tau_cov_inv + PtOP                              # (N, N)
    b = tau_cov_inv @ pi + PtOQ                         # (N,)

    # mu_BL = A^{-1} b  (solve, not inv)
    mu_bl = np.linalg.solve(A, b)                       # (N,)

    # Cov_BL = A^{-1}  (solve against identity)
    cov_bl = np.linalg.solve(A, np.eye(N))              # (N, N)

    return mu_bl, cov_bl


# ---------------------------------------------------------------------------
# Estimator class
# ---------------------------------------------------------------------------


class BlackLittermanEstimator:
    """
    Black-Litterman expected return estimator.

    Composes equilibrium returns, momentum views, and Bayesian updating
    into a single ``estimate()`` call compatible with the existing
    walk-forward and signal-factory pipelines.

    Parameters
    ----------
    bl_config : BlackLittermanConfig or None
        Model hyperparameters.  Defaults to BlackLittermanConfig().
    """

    def __init__(self, bl_config: BlackLittermanConfig | None = None) -> None:
        self.cfg = bl_config if bl_config is not None else BlackLittermanConfig()

    def estimate(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        market_weights: pd.Series | None = None,
    ) -> pd.Series:
        """
        Compute BL posterior expected returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical log returns, shape (T, N).  Used for the momentum
            views (trailing window).  Must be in daily (not annualised) units.
        cov_matrix : pd.DataFrame
            Annualised N × N covariance matrix (already multiplied by 252).
        market_weights : pd.Series or None
            Market-cap proxy weights indexed by ticker.  If None, uses
            config.MARKET_CAP_WEIGHTS, normalised to the available tickers.

        Returns
        -------
        pd.Series
            Annualised BL posterior expected returns indexed by ticker.
        """
        tickers = returns.columns.tolist()
        N = len(tickers)
        cov = cov_matrix.values                         # (N, N)

        # ── Market weights ────────────────────────────────────────────────
        if market_weights is not None:
            mkt_w = market_weights.reindex(tickers, fill_value=0.0).values.astype(float)
        else:
            mkt_w = np.array(
                [config.MARKET_CAP_WEIGHTS.get(t, 0.0) for t in tickers],
                dtype=float,
            )
        mkt_sum = mkt_w.sum()
        mkt_w = mkt_w / mkt_sum if mkt_sum > 0 else np.ones(N) / N

        # ── Step 1: equilibrium returns ───────────────────────────────────
        pi = compute_equilibrium_returns(cov, mkt_w, self.cfg.risk_aversion)

        if not self.cfg.use_views:
            logger.debug("BL use_views=False: returning pure equilibrium prior")
            return pd.Series(pi, index=tickers)

        # ── Step 2: momentum views ────────────────────────────────────────
        P, Q, omega = generate_momentum_views(
            returns=returns,
            cov_matrix=cov,
            lookback=self.cfg.view_lookback,
            tau=self.cfg.tau,
            view_confidence=self.cfg.view_confidence,
        )

        # ── Step 3: BL posterior ──────────────────────────────────────────
        try:
            mu_bl, _ = compute_bl_posterior(
                pi=pi,
                cov_matrix=cov,
                P=P,
                Q=Q,
                omega=omega,
                tau=self.cfg.tau,
            )
        except np.linalg.LinAlgError:
            logger.warning("BL posterior solve failed; falling back to equilibrium")
            mu_bl = pi

        return pd.Series(mu_bl, index=tickers)
