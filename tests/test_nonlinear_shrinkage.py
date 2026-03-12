"""Tests for Ledoit-Wolf nonlinear shrinkage (Phase 3B)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.covariance import (
    LedoitWolfNonlinearCovariance,
    SampleCovariance,
    get_covariance_estimator,
    ledoit_wolf_nonlinear,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(n_days: int = 500, n_assets: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    data = rng.normal(0, 0.01, (n_days, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


# ---------------------------------------------------------------------------
# ledoit_wolf_nonlinear function
# ---------------------------------------------------------------------------


class TestLedoitWolfNonlinearFunction:

    def test_symmetric(self):
        returns = _make_returns()
        sample_cov = returns.cov().values
        result = ledoit_wolf_nonlinear(sample_cov, len(returns))
        np.testing.assert_allclose(result, result.T, atol=1e-12)

    def test_psd(self):
        returns = _make_returns()
        sample_cov = returns.cov().values
        result = ledoit_wolf_nonlinear(sample_cov, len(returns))
        eigenvals = np.linalg.eigvalsh(result)
        assert eigenvals.min() >= -1e-10, f"Negative eigenvalue: {eigenvals.min()}"

    def test_no_nan(self):
        returns = _make_returns()
        sample_cov = returns.cov().values
        result = ledoit_wolf_nonlinear(sample_cov, len(returns))
        assert not np.any(np.isnan(result))

    def test_no_inf(self):
        returns = _make_returns()
        sample_cov = returns.cov().values
        result = ledoit_wolf_nonlinear(sample_cov, len(returns))
        assert not np.any(np.isinf(result))

    def test_condition_number_improvement_noisy(self):
        """On noisy data (high N/T), nonlinear shrinkage should reduce
        the condition number versus raw sample covariance."""
        returns = _make_returns(n_days=100, n_assets=30, seed=7)
        sample_cov = returns.cov().values
        shrunk = ledoit_wolf_nonlinear(sample_cov, len(returns))

        cond_sample = np.linalg.cond(sample_cov)
        cond_shrunk = np.linalg.cond(shrunk)
        assert cond_shrunk < cond_sample, (
            f"Shrunk cond ({cond_shrunk:.1f}) should be < "
            f"sample cond ({cond_sample:.1f})"
        )

    def test_preserves_shape(self):
        returns = _make_returns(n_assets=8)
        sample_cov = returns.cov().values
        result = ledoit_wolf_nonlinear(sample_cov, len(returns))
        assert result.shape == (8, 8)

    def test_single_eigenvalue_returns_copy(self):
        """p=1 edge case should return the original matrix."""
        cov = np.array([[0.01]])
        result = ledoit_wolf_nonlinear(cov, 100)
        np.testing.assert_allclose(result, cov)

    def test_identity_stable(self):
        """Applying nonlinear shrinkage to the identity should not blow up."""
        cov = np.eye(5) * 1e-4
        result = ledoit_wolf_nonlinear(cov, 500)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert result.shape == (5, 5)
        eigenvals = np.linalg.eigvalsh(result)
        assert eigenvals.min() >= -1e-10

    def test_various_n_t_ratios(self):
        """Stable across different concentration ratios."""
        for n_days, n_assets in [(500, 5), (200, 20), (100, 50)]:
            returns = _make_returns(n_days=n_days, n_assets=n_assets, seed=42)
            sample_cov = returns.cov().values
            result = ledoit_wolf_nonlinear(sample_cov, n_days)
            assert not np.any(np.isnan(result)), f"NaN for N={n_assets}, T={n_days}"
            assert not np.any(np.isinf(result)), f"Inf for N={n_assets}, T={n_days}"
            eigenvals = np.linalg.eigvalsh(result)
            assert eigenvals.min() >= -1e-10, (
                f"Negative eigenvalue for N={n_assets}, T={n_days}: {eigenvals.min()}"
            )


# ---------------------------------------------------------------------------
# LedoitWolfNonlinearCovariance estimator class
# ---------------------------------------------------------------------------


class TestLedoitWolfNonlinearCovariance:

    def test_output_shape(self):
        returns = _make_returns(n_assets=10)
        cov = LedoitWolfNonlinearCovariance().estimate(returns)
        assert cov.shape == (10, 10)
        assert list(cov.index) == list(cov.columns)

    def test_symmetric(self):
        returns = _make_returns()
        cov = LedoitWolfNonlinearCovariance().estimate(returns)
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-12)

    def test_psd(self):
        returns = _make_returns()
        cov = LedoitWolfNonlinearCovariance().estimate(returns)
        eigenvals = np.linalg.eigvalsh(cov.values)
        assert eigenvals.min() >= -1e-10

    def test_positive_diagonal(self):
        returns = _make_returns()
        cov = LedoitWolfNonlinearCovariance().estimate(returns)
        assert (np.diag(cov.values) > 0).all()

    def test_no_nan_no_inf(self):
        returns = _make_returns()
        cov = LedoitWolfNonlinearCovariance().estimate(returns)
        assert not np.any(np.isnan(cov.values))
        assert not np.any(np.isinf(cov.values))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            LedoitWolfNonlinearCovariance().estimate(pd.DataFrame())

    def test_ticker_labels_preserved(self):
        returns = _make_returns(n_assets=4)
        returns.columns = ["SPY", "QQQ", "TLT", "GLD"]
        cov = LedoitWolfNonlinearCovariance().estimate(returns)
        assert list(cov.index) == ["SPY", "QQQ", "TLT", "GLD"]
        assert list(cov.columns) == ["SPY", "QQQ", "TLT", "GLD"]

    def test_condition_improvement_vs_sample(self):
        """Nonlinear shrinkage should improve conditioning on noisy data."""
        returns = _make_returns(n_days=100, n_assets=30, seed=7)
        sample = SampleCovariance().estimate(returns)
        nonlinear = LedoitWolfNonlinearCovariance().estimate(returns)
        assert np.linalg.cond(nonlinear.values) < np.linalg.cond(sample.values)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:

    def test_get_ledoit_wolf_nonlinear(self):
        est = get_covariance_estimator("ledoit_wolf_nonlinear")
        assert isinstance(est, LedoitWolfNonlinearCovariance)

    def test_ledoit_wolf_still_works(self):
        """Original ledoit_wolf factory name must remain supported."""
        from src.optimization.covariance import LedoitWolfCovariance
        est = get_covariance_estimator("ledoit_wolf")
        assert isinstance(est, LedoitWolfCovariance)
