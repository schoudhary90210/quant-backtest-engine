"""
Risk budgeting / risk parity portfolio optimizer.

Finds weights such that each asset's marginal risk contribution matches
a target budget.  When all budgets are equal (``budgets=None``), this
produces an equal-risk-contribution (ERC / risk parity) portfolio.

Uses the Roncalli (2013) formulation:
  minimize  sum_i [w_i * (Sigma @ w)_i  -  b_i * (w^T Sigma w)]^2
  s.t.      sum(w) = 1,  w >= 0

This avoids division by portfolio volatility and is numerically robust.

Uses ``scipy.optimize.minimize`` with the SLSQP solver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class RiskBudget:
    """
    Risk-budget portfolio optimizer.

    Parameters
    ----------
    budgets : dict[str, float] or None
        Target risk-contribution fractions per asset (must sum to 1, all >= 0).
        If ``None``, defaults to equal risk contribution (1/N per asset).
    """

    def __init__(self, budgets: dict[str, float] | None = None) -> None:
        self._budgets = budgets

    def optimize(self, covariance: pd.DataFrame) -> pd.Series:
        """
        Compute risk-budget-optimal weights.

        Parameters
        ----------
        covariance : pd.DataFrame
            N x N covariance matrix indexed by ticker.

        Returns
        -------
        pd.Series
            Long-only weights summing to 1, indexed by ticker.

        Raises
        ------
        ValueError
            If budgets are invalid or optimisation fails.
        """
        tickers = covariance.columns.tolist()
        n = len(tickers)
        cov = covariance.values.copy()

        if n == 0:
            return pd.Series(dtype=float)
        if n == 1:
            return pd.Series(1.0, index=tickers)

        # Resolve target budgets
        if self._budgets is None:
            target = np.ones(n) / n
        else:
            self._validate_budgets(self._budgets, tickers)
            target = np.array([self._budgets[t] for t in tickers])

        # Initial guess: inverse-volatility heuristic
        vols = np.sqrt(np.diag(cov))
        vols_safe = np.where(vols > 1e-15, vols, 1e-15)
        w0 = (1.0 / vols_safe) / np.sum(1.0 / vols_safe)

        # Roncalli (2013) objective:
        # minimize sum_i [w_i * (Sigma @ w)_i - b_i * (w^T Sigma w)]^2
        def objective(w: np.ndarray) -> float:
            sigma_w = cov @ w
            rc = w * sigma_w                    # risk contributions
            port_var = float(w @ sigma_w)       # w^T Sigma w
            deviations = rc - target * port_var
            return float(np.sum(deviations ** 2))

        # Analytical gradient for faster convergence
        def gradient(w: np.ndarray) -> np.ndarray:
            sigma_w = cov @ w
            rc = w * sigma_w
            port_var = float(w @ sigma_w)
            diff = rc - target * port_var  # n-vector

            grad = np.zeros(n)
            for i in range(n):
                # d(rc_i)/dw_j = delta_ij * sigma_w_i + w_i * cov_ij
                # d(port_var)/dw_j = 2 * sigma_w_j
                # d(objective)/dw_j = 2 * sum_i diff_i * (d(rc_i)/dw_j - b_i * d(port_var)/dw_j)
                pass

            # Vectorised gradient:
            # d_obj/d_w = 2 * J^T @ diff where J_ij = d(rc_i - b_i*pv) / d_w_j
            # d(rc_i)/d_w_j = delta_ij * (Sigma w)_i + w_i * Sigma_ij
            # d(b_i*pv)/d_w_j = 2 * b_i * (Sigma w)_j
            # So J_ij = delta_ij * (Sigma w)_i + w_i * Sigma_ij - 2*b_i*(Sigma w)_j

            J = np.diag(sigma_w) + np.outer(w, np.ones(n)) * cov - 2.0 * np.outer(target, sigma_w)
            return 2.0 * J.T @ diff

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(1e-8, 1.0) for _ in range(n)]

        result = minimize(
            objective,
            w0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-14},
        )

        if not result.success:
            raise ValueError(
                f"Risk-budget optimisation failed: {result.message}"
            )

        weights = result.x / result.x.sum()  # normalise for numerical safety
        return pd.Series(weights, index=tickers)

    def get_risk_contributions(
        self,
        weights: pd.Series,
        covariance: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute realised risk contributions for given weights.

        The risk contribution of asset *i* is ``w_i * (Cov @ w)_i``.
        Contributions sum to portfolio variance ``w^T Cov w``.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by ticker.
        covariance : pd.DataFrame
            N x N covariance matrix.

        Returns
        -------
        pd.Series
            Risk contribution per asset (sums to portfolio variance).
        """
        tickers = weights.index.tolist()
        w = weights.values
        cov = covariance.reindex(index=tickers, columns=tickers).values
        rc = self._risk_contributions_array(w, cov)
        return pd.Series(rc, index=tickers)

    @staticmethod
    def _risk_contributions_array(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Raw risk contributions: w_i * (Cov @ w)_i."""
        marginal = cov @ w
        return w * marginal

    @staticmethod
    def _validate_budgets(budgets: dict[str, float], tickers: list[str]) -> None:
        """Validate budget dict."""
        missing = set(tickers) - set(budgets.keys())
        if missing:
            raise ValueError(f"Budgets missing for tickers: {missing}")

        vals = np.array([budgets[t] for t in tickers])
        if (vals < 0).any():
            raise ValueError("All budgets must be non-negative")
        if abs(vals.sum() - 1.0) > 1e-6:
            raise ValueError(
                f"Budgets must sum to 1.0, got {vals.sum():.6f}"
            )
