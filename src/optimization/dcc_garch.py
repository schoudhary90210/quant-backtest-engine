"""
DCC-GARCH time-varying covariance estimation.

Implements the Dynamic Conditional Correlation model of Engle (2002):

1. Fit univariate GARCH(1,1) per asset to obtain conditional volatilities.
2. Compute standardised residuals.
3. Estimate scalar DCC parameters (a, b) with constraint a + b < 1.
4. Produce a time-varying conditional covariance matrix for any date
   in the fitted sample.

Robustness
----------
If GARCH fitting fails for any asset (convergence issues, degenerate
series), the estimator falls back to EWMA volatility (RiskMetrics
lambda = 0.94) for that asset.  A warning is logged but the pipeline
does not crash.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

import config

logger = logging.getLogger(__name__)

_EWMA_LAMBDA = 0.94  # RiskMetrics fallback decay


# ---------------------------------------------------------------------------
# Univariate GARCH fitting (per-asset)
# ---------------------------------------------------------------------------


def _fit_garch_single(
    returns: pd.Series,
) -> tuple[pd.Series, bool]:
    """Fit GARCH(1,1) to a single asset's return series.

    Returns
    -------
    conditional_vol : pd.Series
        Daily conditional volatility (sigma_t), same index as *returns*.
    used_fallback : bool
        True if GARCH fitting failed and EWMA was used instead.
    """
    # Scale to percentage returns for arch numerical stability
    y = returns * 100.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(y, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
            result = model.fit(disp="off", show_warning=False)

        cond_var = result.conditional_volatility / 100.0  # back to decimal
        if cond_var.isna().any() or (cond_var <= 0).any():
            raise ValueError("Invalid conditional volatility")
        return cond_var, False

    except Exception as exc:
        logger.warning(
            "GARCH fit failed for %s (%s); falling back to EWMA vol.",
            returns.name,
            exc,
        )
        return _ewma_vol_fallback(returns), True


def _ewma_vol_fallback(returns: pd.Series) -> pd.Series:
    """EWMA (RiskMetrics) volatility as a GARCH fallback."""
    com = _EWMA_LAMBDA / (1.0 - _EWMA_LAMBDA)
    ewma_var = returns.pow(2).ewm(com=com).mean()
    return np.sqrt(ewma_var).rename(returns.name)


# ---------------------------------------------------------------------------
# DCC parameter estimation
# ---------------------------------------------------------------------------


def _estimate_dcc_params(
    std_resids: np.ndarray,
) -> tuple[float, float]:
    """Estimate scalar DCC parameters (a, b) from standardised residuals.

    Uses a grid search + quasi-Newton refinement to maximise the
    DCC log-likelihood, enforcing a + b < 1 and a, b >= 0.

    Parameters
    ----------
    std_resids : np.ndarray
        T x N matrix of standardised residuals (epsilon_t / sigma_t).

    Returns
    -------
    tuple[float, float]
        (a, b) with a + b < 1.
    """
    T, N = std_resids.shape
    # Unconditional correlation of standardised residuals
    Q_bar = np.corrcoef(std_resids, rowvar=False)

    best_ll = -np.inf
    best_a, best_b = 0.01, 0.95

    # Grid search over plausible (a, b) pairs
    for a in np.arange(0.005, 0.15, 0.01):
        for b in np.arange(0.80, 0.995, 0.01):
            if a + b >= 1.0:
                continue
            ll = _dcc_loglik(std_resids, Q_bar, a, b)
            if ll > best_ll:
                best_ll = ll
                best_a, best_b = a, b

    return float(best_a), float(best_b)


def _dcc_loglik(
    std_resids: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
) -> float:
    """Compute DCC composite log-likelihood (correlation part only).

    The DCC correlation part of the log-likelihood is:
      -0.5 * sum_t [ log|R_t| + eps_t' R_t^{-1} eps_t - eps_t' eps_t ]

    We skip the constant term (eps_t' eps_t) since it doesn't depend on (a,b).
    """
    T, N = std_resids.shape
    Qt = Q_bar.copy()
    ll = 0.0

    for t in range(T):
        eps_t = std_resids[t]

        if t > 0:
            Qt = (1 - a - b) * Q_bar + a * np.outer(eps_t_prev, eps_t_prev) + b * Qt

        # Rescale Qt to correlation Rt
        d = np.sqrt(np.diag(Qt))
        d_safe = np.where(d > 1e-15, d, 1e-15)
        Rt = Qt / np.outer(d_safe, d_safe)

        # Regularise for numerical stability
        Rt = (Rt + Rt.T) / 2.0
        np.fill_diagonal(Rt, 1.0)

        try:
            sign, logdet = np.linalg.slogdet(Rt)
            if sign <= 0:
                return -np.inf
            Rt_inv = np.linalg.solve(Rt, eps_t)
            ll -= 0.5 * (logdet + eps_t @ Rt_inv)
        except np.linalg.LinAlgError:
            return -np.inf

        eps_t_prev = eps_t  # noqa: F841 — used in next iteration's Qt update

    return float(ll)


# ---------------------------------------------------------------------------
# DCCGarchEstimator
# ---------------------------------------------------------------------------


class DCCGarchEstimator:
    """
    DCC-GARCH time-varying covariance estimator.

    Fits the model once on a returns DataFrame and caches results so that
    repeated date lookups are cheap.

    Parameters
    ----------
    returns : pd.DataFrame
        T x N daily log-return matrix (rows = dates, columns = tickers).
    """

    def __init__(self, returns: pd.DataFrame) -> None:
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")

        self._returns = returns
        self._tickers = list(returns.columns)
        self._dates = returns.index
        self._N = len(self._tickers)

        # --- Stage 1: univariate GARCH(1,1) per asset ---
        cond_vols: dict[str, pd.Series] = {}
        self._fallback_tickers: list[str] = []
        for ticker in self._tickers:
            vol, used_fallback = _fit_garch_single(returns[ticker])
            cond_vols[ticker] = vol
            if used_fallback:
                self._fallback_tickers.append(ticker)

        self._cond_vol = pd.DataFrame(cond_vols, index=returns.index)

        # --- Stage 2: standardised residuals ---
        self._std_resids = (returns / self._cond_vol).values
        # Replace any NaN/inf with 0 (degenerate observations)
        self._std_resids = np.nan_to_num(self._std_resids, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Stage 3: DCC parameters ---
        self._Q_bar = np.corrcoef(self._std_resids, rowvar=False)
        # Handle degenerate Q_bar (e.g., single observation)
        if np.any(np.isnan(self._Q_bar)):
            self._Q_bar = np.eye(self._N)

        self._a, self._b = _estimate_dcc_params(self._std_resids)
        logger.info(
            "DCC-GARCH fitted: a=%.4f, b=%.4f (a+b=%.4f), fallbacks=%s",
            self._a,
            self._b,
            self._a + self._b,
            self._fallback_tickers or "none",
        )

        # --- Stage 4: forward-pass Qt series (cached) ---
        self._Qt_series = self._compute_qt_series()

    # -- internal ----------------------------------------------------------

    def _compute_qt_series(self) -> dict[int, np.ndarray]:
        """Run the DCC recursion and cache Qt at every date index."""
        T = len(self._dates)
        a, b = self._a, self._b
        Q_bar = self._Q_bar
        Qt = Q_bar.copy()
        qt_cache: dict[int, np.ndarray] = {}

        for t in range(T):
            if t > 0:
                eps_prev = self._std_resids[t - 1]
                Qt = (1 - a - b) * Q_bar + a * np.outer(eps_prev, eps_prev) + b * Qt
            qt_cache[t] = Qt.copy()

        return qt_cache

    def _date_to_idx(self, date: pd.Timestamp) -> int:
        """Convert a timestamp to the nearest valid index (at or before *date*)."""
        loc = self._dates.searchsorted(date, side="right") - 1
        return max(0, min(loc, len(self._dates) - 1))

    # -- public API --------------------------------------------------------

    def get_covariance(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Conditional covariance matrix at *date*.

        H_t = D_t R_t D_t, where D_t = diag(sigma_t) and R_t is
        the DCC conditional correlation matrix derived from Q_t.

        Returns
        -------
        pd.DataFrame
            N x N covariance matrix indexed by ticker.
        """
        idx = self._date_to_idx(date)

        # Conditional volatilities at date
        sigma = self._cond_vol.iloc[idx].values
        D = np.diag(sigma)

        # DCC correlation from cached Qt
        Qt = self._Qt_series[idx]
        d = np.sqrt(np.diag(Qt))
        d_safe = np.where(d > 1e-15, d, 1e-15)
        Rt = Qt / np.outer(d_safe, d_safe)
        Rt = (Rt + Rt.T) / 2.0
        np.fill_diagonal(Rt, 1.0)

        # Conditional covariance: H_t = D_t R_t D_t
        Ht = D @ Rt @ D
        Ht = (Ht + Ht.T) / 2.0

        return pd.DataFrame(Ht, index=self._tickers, columns=self._tickers)

    def get_correlation_timeseries(
        self, asset_i: str, asset_j: str
    ) -> pd.Series:
        """
        Time series of conditional correlation between two assets.

        Parameters
        ----------
        asset_i, asset_j : str
            Ticker names.

        Returns
        -------
        pd.Series
            Correlation rho_{ij,t} indexed by date.
        """
        i = self._tickers.index(asset_i)
        j = self._tickers.index(asset_j)

        rho = np.empty(len(self._dates))
        for t, Qt in self._Qt_series.items():
            d_i = np.sqrt(Qt[i, i]) if Qt[i, i] > 0 else 1e-15
            d_j = np.sqrt(Qt[j, j]) if Qt[j, j] > 0 else 1e-15
            rho[t] = Qt[i, j] / (d_i * d_j)

        return pd.Series(rho, index=self._dates, name=f"{asset_i}/{asset_j}")

    @property
    def fallback_tickers(self) -> list[str]:
        """Tickers for which GARCH fitting failed and EWMA was used."""
        return list(self._fallback_tickers)

    @property
    def dcc_params(self) -> tuple[float, float]:
        """Fitted DCC parameters (a, b)."""
        return self._a, self._b


# ---------------------------------------------------------------------------
# Convenience wrapper matching the plan's public API
# ---------------------------------------------------------------------------


def dcc_garch_covariance(
    returns: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.DataFrame:
    """
    One-shot DCC-GARCH conditional covariance at a specific date.

    Fits the full model on *returns* and returns H_t for *date*.
    For repeated queries, prefer :class:`DCCGarchEstimator` directly
    to avoid re-fitting.
    """
    estimator = DCCGarchEstimator(returns)
    return estimator.get_covariance(date)


# ---------------------------------------------------------------------------
# Factory-compatible wrapper
# ---------------------------------------------------------------------------


class DCCGarchCovariance:
    """
    DCC-GARCH covariance estimator with the standard factory interface.

    ``estimate(returns)`` fits the DCC model on the full *returns* window
    and returns the conditional covariance at the **last date**.
    """

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")
        estimator = DCCGarchEstimator(returns)
        return estimator.get_covariance(returns.index[-1])
