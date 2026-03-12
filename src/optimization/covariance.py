"""
Covariance matrix estimation.

Swappable estimators sharing a common interface:
  - SampleCovariance:              unbiased sample covariance (baseline)
  - LedoitWolfCovariance:          Ledoit-Wolf linear shrinkage (sklearn)
  - LedoitWolfNonlinearCovariance: kernel-based nonlinear shrinkage
  - EWMACovariance:                exponentially-weighted moving average
  - RMTDenoisedCovariance:         Random Matrix Theory denoised covariance

All estimators implement ``estimate(returns) -> pd.DataFrame`` returning
an N x N covariance matrix indexed by ticker symbols.
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
            T x N log-return matrix (rows = dates, columns = tickers).

        Returns
        -------
        pd.DataFrame
            N x N covariance matrix with ticker-labeled rows and columns.
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
    to reduce estimation error, especially when T ~ N.
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
    ``decay`` (lambda). Higher lambda -> slower decay -> longer memory.

    Parameters
    ----------
    decay : float
        Decay factor (lambda), typically 0.94 for daily data (RiskMetrics).
    """

    def __init__(self, decay: float = config.EWMA_LAMBDA) -> None:
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
        cov_matrix = returns.ewm(com=com).cov().iloc[-len(tickers) :]
        # The last N rows of the multi-indexed result are the final estimate
        cov_matrix = cov_matrix.droplevel(0)
        cov_matrix.index = tickers
        cov_matrix.columns = tickers

        return cov_matrix


# ---------------------------------------------------------------------------
# Random Matrix Theory (RMT) denoising
# ---------------------------------------------------------------------------


def marchenko_pastur_bounds(
    n_assets: int,
    n_observations: int,
    sigma2: float = 1.0,
) -> tuple[float, float]:
    """
    Compute the Marchenko-Pastur distribution bounds.

    For a random T x N matrix with i.i.d. entries of variance sigma2,
    the eigenvalues of the sample correlation matrix concentrate in
    [lambda_minus, lambda_plus] as T, N -> infinity with q = N/T fixed.

    Parameters
    ----------
    n_assets : int
        Number of assets (N).
    n_observations : int
        Number of observations (T).
    sigma2 : float
        Noise variance (1.0 for standardised correlation matrices).

    Returns
    -------
    tuple[float, float]
        (lambda_minus, lambda_plus).
    """
    q = n_assets / n_observations
    sqrt_q = np.sqrt(q)
    lambda_plus = sigma2 * (1 + sqrt_q) ** 2
    lambda_minus = sigma2 * (1 - sqrt_q) ** 2
    return float(lambda_minus), float(lambda_plus)


def marchenko_pastur_pdf(
    x: np.ndarray,
    n_assets: int,
    n_observations: int,
    sigma2: float = 1.0,
) -> np.ndarray:
    """
    Evaluate the Marchenko-Pastur probability density function.

    Parameters
    ----------
    x : np.ndarray
        Points at which to evaluate the PDF.
    n_assets : int
        Number of assets (N).
    n_observations : int
        Number of observations (T).
    sigma2 : float
        Noise variance.

    Returns
    -------
    np.ndarray
        PDF values (same shape as *x*).
    """
    q = n_assets / n_observations
    lambda_minus, lambda_plus = marchenko_pastur_bounds(n_assets, n_observations, sigma2)
    pdf = np.zeros_like(x, dtype=float)
    mask = (x >= lambda_minus) & (x <= lambda_plus) & (x > 0)
    pdf[mask] = (
        (1.0 / (2.0 * np.pi * sigma2 * q))
        * np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus))
        / x[mask]
    )
    return pdf


def rmt_denoise_covariance(
    corr: np.ndarray,
    n_observations: int,
    method: str = "constant_residual",
) -> np.ndarray:
    """
    Denoise a correlation matrix using Random Matrix Theory.

    Eigenvalues below the Marchenko-Pastur upper bound (lambda+) are
    treated as noise and replaced according to the chosen method.  The
    result is a symmetric, PSD matrix with unit diagonal.

    Parameters
    ----------
    corr : np.ndarray
        N x N sample correlation matrix.
    n_observations : int
        Number of observations (T) used to compute *corr*.
    method : str
        ``"constant_residual"`` — replace noise eigenvalues with their mean
        (preserves eigenvalue sum).
        ``"targeted_shrinkage"`` — zero noise eigenvalues and redistribute
        their mass uniformly across all eigenvalues, equivalent to shrinking
        the noise subspace toward the identity.

    Returns
    -------
    np.ndarray
        Denoised N x N correlation matrix.
    """
    N = corr.shape[0]

    # Eigendecompose (eigh returns ascending order)
    eigenvals, eigenvecs = np.linalg.eigh(corr)
    # Sort descending
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    _, lambda_plus = marchenko_pastur_bounds(N, n_observations)

    if method not in ("constant_residual", "targeted_shrinkage"):
        raise ValueError(
            f"Unknown RMT denoise method '{method}'. "
            f"Choose from: 'constant_residual', 'targeted_shrinkage'"
        )

    is_signal = eigenvals > lambda_plus

    # Edge cases: all signal or all noise — return as-is
    if is_signal.all() or not is_signal.any():
        return corr.copy()

    denoised = eigenvals.copy()

    if method == "constant_residual":
        # Replace noise eigenvalues with their mean (preserves trace)
        noise_mean = float(eigenvals[~is_signal].sum() / (~is_signal).sum())
        denoised[~is_signal] = noise_mean
    elif method == "targeted_shrinkage":
        # Zero noise eigenvalues, redistribute mass uniformly (shrink toward I)
        noise_sum = float(eigenvals[~is_signal].sum())
        denoised[~is_signal] = 0.0
        denoised += noise_sum / N

    # Reconstruct: V @ diag(lambda) @ V^T
    corr_denoised = (eigenvecs * denoised) @ eigenvecs.T

    # Force exact symmetry (numerical hygiene)
    corr_denoised = (corr_denoised + corr_denoised.T) / 2.0

    # Rescale to unit diagonal (proper correlation matrix)
    d = np.sqrt(np.diag(corr_denoised))
    corr_denoised = corr_denoised / np.outer(d, d)

    return corr_denoised


class RMTDenoisedCovariance:
    """
    Random Matrix Theory denoised covariance estimator.

    Computes the sample covariance, converts to correlation, denoises
    eigenvalues below the Marchenko-Pastur bound, then converts back
    to covariance using the original per-asset volatilities.

    Parameters
    ----------
    method : str
        ``"constant_residual"`` or ``"targeted_shrinkage"``.
    """

    def __init__(self, method: str = config.RMT_DENOISE_METHOD) -> None:
        self.method = method

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")

        tickers = returns.columns
        T, N = returns.shape

        # Sample covariance (ddof=1)
        sample_cov = returns.cov().values

        # Per-asset volatilities
        vols = np.sqrt(np.diag(sample_cov))

        # Convert to correlation
        D_inv = np.diag(1.0 / vols)
        corr = D_inv @ sample_cov @ D_inv

        # Denoise the correlation matrix
        corr_denoised = rmt_denoise_covariance(corr, T, self.method)

        # Convert back to covariance
        D = np.diag(vols)
        cov_denoised = D @ corr_denoised @ D

        return pd.DataFrame(cov_denoised, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# Nonlinear shrinkage (Ledoit-Wolf kernel-based)
# ---------------------------------------------------------------------------


def _epanechnikov_hilbert_transform(a: np.ndarray) -> np.ndarray:
    """Analytical Hilbert transform of the Epanechnikov kernel.

    For K(u) = (3/4)(1 - u^2), |u| <= 1, the principal-value integral

        H[K](a) = P.V. int_{-1}^{1} K(u)/(u + a) du

    has the closed form

        H[K](a) = (3/4) [2a + (1 - a^2) ln|(1 + a)/(a - 1)|]

    with limits H[K](0) = 0 and H[K](+-1) = +-3/2.
    """
    result = np.zeros_like(a, dtype=float)

    near_zero = np.abs(a) < 1e-10
    near_pos1 = np.abs(a - 1.0) < 1e-10
    near_neg1 = np.abs(a + 1.0) < 1e-10
    regular = ~(near_zero | near_pos1 | near_neg1)

    result[near_pos1] = 1.5
    result[near_neg1] = -1.5
    # near_zero stays 0

    ar = a[regular]
    result[regular] = 0.75 * (
        2.0 * ar + (1.0 - ar**2) * np.log(np.abs((1.0 + ar) / (ar - 1.0)))
    )
    return result


def ledoit_wolf_nonlinear(
    sample_cov: np.ndarray,
    n_observations: int,
) -> np.ndarray:
    """
    Nonlinear shrinkage of a sample covariance matrix.

    Applies a distinct optimal shrinkage intensity to each eigenvalue
    using kernel density estimation of the sample spectral distribution
    and its Hilbert transform, following the framework of Ledoit & Wolf
    (2017, 2018).

    .. note::

       This is a **kernel-based numerical approximation**, not the exact
       analytical formula from Ledoit & Wolf (2020, *Annals of Statistics*,
       "Analytical nonlinear shrinkage of large-dimensional covariance
       matrices").  The analytical formula requires solving the
       Marchenko-Pastur equation for the population spectral distribution.
       This implementation instead estimates the spectral density with an
       Epanechnikov kernel and computes the Hilbert transform analytically
       from that kernel.

    Parameters
    ----------
    sample_cov : np.ndarray
        N x N sample covariance matrix.
    n_observations : int
        Number of observations (T) used to compute *sample_cov*.

    Returns
    -------
    np.ndarray
        N x N nonlinearly-shrunk covariance matrix (symmetric, PSD).
    """
    N = sample_cov.shape[0]
    T = n_observations
    c = N / T  # concentration ratio

    # Eigendecompose
    eigenvals, eigenvecs = np.linalg.eigh(sample_cov)
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    eigenvals = np.maximum(eigenvals, 0.0)

    # Number of effective (non-zero) eigenvalues
    p = int(min(N, T))
    if p <= 1:
        return sample_cov.copy()

    lam = eigenvals[:p].copy()

    # Bandwidth: Silverman's rule adapted for Epanechnikov kernel
    std_lam = float(np.std(lam, ddof=1)) if p > 1 else float(lam[0])
    iqr = float(np.percentile(lam, 75) - np.percentile(lam, 25))
    spread = min(std_lam, iqr / 1.34) if iqr > 0 else std_lam
    h = 0.9 * spread * p ** (-0.2)
    h = max(h, 1e-6 * max(float(lam.max()), 1e-10))  # numerical floor

    shrunk = np.empty(p)

    for i in range(p):
        x = lam[i]
        a = (lam - x) / h  # standardised distances

        # Epanechnikov kernel density at lambda_i
        inside = np.abs(a) <= 1.0
        f_i = float(np.sum(0.75 * (1.0 - a[inside] ** 2))) / (p * h)

        # Hilbert transform of the KDE at lambda_i
        ht_vals = _epanechnikov_hilbert_transform(a)
        Hf_i = float(np.sum(ht_vals)) / (p * h * np.pi)

        # Optimal nonlinear shrinkage (Ledoit-Wolf formula):
        # delta_i = lambda_i / [(1 - c - c*lambda_i*H)^2 + (c*lambda_i*pi*f)^2]
        denom = (1.0 - c - c * x * Hf_i) ** 2 + (c * x * np.pi * f_i) ** 2
        shrunk[i] = x / max(denom, 1e-15)

    # Ensure non-negative eigenvalues
    shrunk = np.maximum(shrunk, 0.0)

    # Reconstruct full N x N matrix
    full_eigenvals = np.zeros(N)
    full_eigenvals[:p] = shrunk

    result = (eigenvecs * full_eigenvals) @ eigenvecs.T
    result = (result + result.T) / 2.0  # enforce exact symmetry
    return result


class LedoitWolfNonlinearCovariance:
    """
    Ledoit-Wolf nonlinear shrinkage covariance estimator.

    Unlike the linear estimator (``LedoitWolfCovariance``) which applies a
    single shrinkage intensity uniformly across all eigenvalues, this
    estimator computes a distinct optimal intensity for each eigenvalue.

    The implementation uses kernel density estimation of the spectral
    distribution.  See :func:`ledoit_wolf_nonlinear` for accuracy notes.
    """

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        if returns.empty:
            raise ValueError("Returns DataFrame is empty")

        tickers = returns.columns
        T = len(returns)
        sample_cov = returns.cov().values

        cov_matrix = ledoit_wolf_nonlinear(sample_cov, T)

        return pd.DataFrame(cov_matrix, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_covariance_estimator(method: str = config.COV_METHOD) -> CovarianceEstimator:
    """
    Factory function to get a covariance estimator by name.

    Parameters
    ----------
    method : str
        'sample', 'ledoit_wolf', 'ledoit_wolf_nonlinear', 'ewma',
        'rmt_denoised', or 'dcc_garch'.

    Returns
    -------
    CovarianceEstimator
    """
    from src.optimization.dcc_garch import DCCGarchCovariance

    estimators = {
        "sample": SampleCovariance,
        "ledoit_wolf": LedoitWolfCovariance,
        "ledoit_wolf_nonlinear": LedoitWolfNonlinearCovariance,
        "ewma": EWMACovariance,
        "rmt_denoised": RMTDenoisedCovariance,
        "dcc_garch": DCCGarchCovariance,
    }
    if method not in estimators:
        raise ValueError(
            f"Unknown covariance method '{method}'. "
            f"Choose from: {list(estimators.keys())}"
        )
    return estimators[method]()
