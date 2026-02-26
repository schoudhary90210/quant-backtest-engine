"""
Tests for the Black-Litterman return estimator.

Covers:
  - Equilibrium returns: positive weights → positive pi
  - No-view degeneracy: use_views=False → mu_BL equals pi
  - Confident-views convergence: very small Omega → mu_BL approaches Q
  - Tau sensitivity: larger tau → views get more relative weight
  - Omega sensitivity: larger Omega → posterior moves toward prior
  - View dimensions: P (N×N), Q (N,), Omega (N×N)
  - Integration with Kelly optimizer: valid weights produced
  - Market-weight normalisation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.black_litterman import (
    BlackLittermanConfig,
    BlackLittermanEstimator,
    compute_bl_posterior,
    compute_equilibrium_returns,
)
from src.optimization.views import generate_momentum_views
from src.optimization.kelly import kelly_weights

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_cov() -> np.ndarray:
    """Simple 3×3 diagonal covariance (annualised)."""
    return np.diag([0.04, 0.09, 0.16])  # σ² = 20%, 30%, 40% annual


@pytest.fixture
def uniform_weights() -> np.ndarray:
    """Equal market-cap weights for 3 assets."""
    return np.array([1 / 3, 1 / 3, 1 / 3])


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """252 days × 3 assets, stationary daily log returns."""
    np.random.seed(0)
    dates = pd.bdate_range("2022-01-01", periods=252)
    tickers = ["A", "B", "C"]
    data = np.random.normal(0.0004, 0.01, (252, 3))
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def sample_cov_df(simple_cov: np.ndarray, sample_returns: pd.DataFrame) -> pd.DataFrame:
    tickers = sample_returns.columns.tolist()
    return pd.DataFrame(simple_cov, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# compute_equilibrium_returns
# ---------------------------------------------------------------------------


def test_equilibrium_returns_positive_with_positive_weights(
    simple_cov: np.ndarray, uniform_weights: np.ndarray
) -> None:
    """All-positive market weights → positive equilibrium returns."""
    pi = compute_equilibrium_returns(simple_cov, uniform_weights)
    assert pi.shape == (3,)
    assert np.all(pi > 0), f"Expected all positive, got {pi}"


def test_equilibrium_returns_scales_with_risk_aversion(
    simple_cov: np.ndarray, uniform_weights: np.ndarray
) -> None:
    """Doubling risk_aversion doubles equilibrium returns."""
    pi1 = compute_equilibrium_returns(simple_cov, uniform_weights, risk_aversion=1.0)
    pi2 = compute_equilibrium_returns(simple_cov, uniform_weights, risk_aversion=2.0)
    np.testing.assert_allclose(pi2, 2.0 * pi1)


def test_equilibrium_returns_normalises_weights(simple_cov: np.ndarray) -> None:
    """Market weights are normalised internally — raw and scaled give same pi."""
    w = np.array([1.0, 1.0, 1.0])
    w_norm = w / w.sum()
    pi_raw = compute_equilibrium_returns(simple_cov, w)
    pi_norm = compute_equilibrium_returns(simple_cov, w_norm)
    np.testing.assert_allclose(pi_raw, pi_norm)


# ---------------------------------------------------------------------------
# compute_bl_posterior — no-view degeneracy
# ---------------------------------------------------------------------------


def test_no_views_returns_prior(
    simple_cov: np.ndarray, uniform_weights: np.ndarray
) -> None:
    """
    When P = 0 (no views contribute), the posterior should equal the prior.

    We achieve this by setting omega to infinity (very large values) so that
    Omega^{-1} → 0 and P'*Omega^{-1}*P → 0, reducing A to (tau*Sigma)^{-1}
    and b to (tau*Sigma)^{-1}*pi, giving mu_BL = pi.
    """
    N = 3
    pi = compute_equilibrium_returns(simple_cov, uniform_weights)
    P = np.eye(N)
    Q = pi.copy()  # views exactly match prior
    omega = np.diag(np.ones(N) * 1e12)  # infinite uncertainty → no views

    mu_bl, _ = compute_bl_posterior(pi, simple_cov, P, Q, omega, tau=0.05)
    np.testing.assert_allclose(mu_bl, pi, rtol=1e-4)


def test_confident_views_converge_to_Q(
    simple_cov: np.ndarray, uniform_weights: np.ndarray
) -> None:
    """Very confident views (tiny Omega) → posterior converges to Q."""
    N = 3
    pi = compute_equilibrium_returns(simple_cov, uniform_weights)
    P = np.eye(N)
    Q = np.array([0.30, 0.20, 0.10])  # views differ from prior
    omega = np.diag(np.ones(N) * 1e-10)  # near-zero uncertainty → pure view

    mu_bl, _ = compute_bl_posterior(pi, simple_cov, P, Q, omega, tau=0.05)
    np.testing.assert_allclose(
        mu_bl, Q, rtol=1e-3, err_msg="mu_BL should converge to Q when Omega→0"
    )


def test_posterior_between_prior_and_views(
    simple_cov: np.ndarray, uniform_weights: np.ndarray
) -> None:
    """Posterior mean is a compromise between prior and views."""
    N = 3
    pi = compute_equilibrium_returns(simple_cov, uniform_weights)
    P = np.eye(N)
    Q = pi * 2.0  # views are 2× prior
    omega = np.diag(np.diag(simple_cov) * 0.05)

    mu_bl, _ = compute_bl_posterior(pi, simple_cov, P, Q, omega, tau=0.05)

    # Each mu_bl element should be strictly between pi and Q
    for i in range(N):
        assert (
            pi[i] < mu_bl[i] < Q[i]
        ), f"Asset {i}: pi={pi[i]:.4f}, mu_bl={mu_bl[i]:.4f}, Q={Q[i]:.4f}"


# ---------------------------------------------------------------------------
# Tau and Omega sensitivity
# ---------------------------------------------------------------------------


def test_larger_tau_gives_more_weight_to_views(
    simple_cov: np.ndarray, uniform_weights: np.ndarray
) -> None:
    """
    Larger tau means the prior is less precise ((tau*Sigma)^{-1} is smaller),
    so views carry relatively more weight → posterior closer to Q.
    """
    N = 3
    pi = compute_equilibrium_returns(simple_cov, uniform_weights)
    P = np.eye(N)
    Q = pi * 3.0
    omega = np.diag(np.diag(simple_cov) * 0.05)

    mu_low_tau, _ = compute_bl_posterior(pi, simple_cov, P, Q, omega, tau=0.01)
    mu_high_tau, _ = compute_bl_posterior(pi, simple_cov, P, Q, omega, tau=0.20)

    # High tau → posterior closer to Q (larger deviation from pi)
    dist_low = np.linalg.norm(mu_low_tau - pi)
    dist_high = np.linalg.norm(mu_high_tau - pi)
    assert dist_high > dist_low, (
        f"Higher tau should move posterior toward views: "
        f"dist_low={dist_low:.4f}, dist_high={dist_high:.4f}"
    )


def test_larger_omega_gives_more_weight_to_prior(
    simple_cov: np.ndarray, uniform_weights: np.ndarray
) -> None:
    """
    Larger Omega (more view uncertainty) → posterior stays closer to prior.
    """
    N = 3
    pi = compute_equilibrium_returns(simple_cov, uniform_weights)
    P = np.eye(N)
    Q = pi * 3.0

    omega_small = np.diag(np.diag(simple_cov) * 0.01)
    omega_large = np.diag(np.diag(simple_cov) * 100.0)

    mu_small, _ = compute_bl_posterior(pi, simple_cov, P, Q, omega_small, tau=0.05)
    mu_large, _ = compute_bl_posterior(pi, simple_cov, P, Q, omega_large, tau=0.05)

    # Larger Omega → posterior closer to prior (smaller distance from pi)
    dist_small = np.linalg.norm(mu_small - pi)
    dist_large = np.linalg.norm(mu_large - pi)
    assert dist_large < dist_small, (
        f"Larger Omega should keep posterior closer to prior: "
        f"dist_small={dist_small:.4f}, dist_large={dist_large:.4f}"
    )


# ---------------------------------------------------------------------------
# generate_momentum_views — dimensions
# ---------------------------------------------------------------------------


def test_view_dimensions(sample_returns: pd.DataFrame, simple_cov: np.ndarray) -> None:
    """P is (N×N), Q is (N,), Omega is (N×N) diagonal."""
    N = sample_returns.shape[1]
    P, Q, omega = generate_momentum_views(sample_returns, simple_cov)
    assert P.shape == (N, N), f"P shape: {P.shape}"
    assert Q.shape == (N,), f"Q shape: {Q.shape}"
    assert omega.shape == (N, N), f"omega shape: {omega.shape}"


def test_pick_matrix_is_identity(
    sample_returns: pd.DataFrame, simple_cov: np.ndarray
) -> None:
    """Absolute views → P equals the identity matrix."""
    P, _, _ = generate_momentum_views(sample_returns, simple_cov)
    np.testing.assert_array_equal(P, np.eye(P.shape[0]))


def test_omega_is_diagonal(
    sample_returns: pd.DataFrame, simple_cov: np.ndarray
) -> None:
    """Omega must be a diagonal matrix (off-diagonal = 0)."""
    _, _, omega = generate_momentum_views(sample_returns, simple_cov)
    off_diag = omega - np.diag(np.diag(omega))
    np.testing.assert_array_equal(off_diag, 0.0)


def test_omega_diagonal_positive(
    sample_returns: pd.DataFrame, simple_cov: np.ndarray
) -> None:
    """All diagonal entries of Omega must be strictly positive."""
    _, _, omega = generate_momentum_views(sample_returns, simple_cov)
    assert np.all(np.diag(omega) > 0)


# ---------------------------------------------------------------------------
# BlackLittermanEstimator
# ---------------------------------------------------------------------------


def test_use_views_false_returns_equilibrium(
    sample_returns: pd.DataFrame, sample_cov_df: pd.DataFrame
) -> None:
    """use_views=False → estimate() returns pure equilibrium returns."""
    tickers = sample_returns.columns.tolist()
    cov = sample_cov_df.values

    mkt_w = np.array([1 / 3, 1 / 3, 1 / 3])
    pi_expected = compute_equilibrium_returns(cov, mkt_w, risk_aversion=2.5)
    pi_series = pd.Series(pi_expected, index=tickers)

    cfg = BlackLittermanConfig(use_views=False, risk_aversion=2.5)
    estimator = BlackLittermanEstimator(cfg)

    # Provide explicit uniform market weights via a mock config with known tickers
    mkt_series = pd.Series(mkt_w, index=tickers)
    mu = estimator.estimate(sample_returns, sample_cov_df, market_weights=mkt_series)

    np.testing.assert_allclose(mu.values, pi_series.values, rtol=1e-10)


def test_estimator_returns_series_indexed_by_ticker(
    sample_returns: pd.DataFrame, sample_cov_df: pd.DataFrame
) -> None:
    """estimate() returns a pd.Series with ticker index."""
    estimator = BlackLittermanEstimator()
    mu = estimator.estimate(sample_returns, sample_cov_df)
    assert isinstance(mu, pd.Series)
    assert list(mu.index) == list(sample_returns.columns)


def test_estimator_returns_finite_values(
    sample_returns: pd.DataFrame, sample_cov_df: pd.DataFrame
) -> None:
    """estimate() must return finite values (no NaN, no Inf)."""
    estimator = BlackLittermanEstimator()
    mu = estimator.estimate(sample_returns, sample_cov_df)
    assert np.all(np.isfinite(mu.values)), f"Non-finite values: {mu}"


# ---------------------------------------------------------------------------
# Integration: BL + Kelly → valid portfolio weights
# ---------------------------------------------------------------------------


def test_bl_kelly_integration_produces_valid_weights(
    sample_returns: pd.DataFrame, sample_cov_df: pd.DataFrame
) -> None:
    """BL expected returns feed into Kelly → weights satisfy constraints."""
    estimator = BlackLittermanEstimator()
    mu = estimator.estimate(sample_returns, sample_cov_df)

    max_weight = 0.40
    max_leverage = 1.5

    weights = kelly_weights(
        expected_returns=mu,
        cov_matrix=sample_cov_df,
        risk_free_rate=0.04,
        fraction=0.5,
        max_weight=max_weight,
        max_leverage=max_leverage,
    )

    assert isinstance(weights, pd.Series)
    assert np.all(weights >= 0), "Kelly weights must be non-negative"
    assert np.all(weights <= max_weight + 1e-8), "Per-asset cap violated"
    assert weights.sum() <= max_leverage + 1e-8, "Leverage cap violated"
