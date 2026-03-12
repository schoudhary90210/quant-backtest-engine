"""Tests for RMT denoising and covariance comparison (Phase 3A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.covariance import (
    RMTDenoisedCovariance,
    get_covariance_estimator,
    marchenko_pastur_bounds,
    marchenko_pastur_pdf,
    rmt_denoise_covariance,
)
from src.optimization.covariance_comparison import compare_covariance_estimators

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(n_days: int = 500, n_assets: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    data = rng.normal(0, 0.01, (n_days, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def _sample_correlation(returns: pd.DataFrame) -> np.ndarray:
    cov = returns.cov().values
    vols = np.sqrt(np.diag(cov))
    return cov / np.outer(vols, vols)


# ---------------------------------------------------------------------------
# Marchenko-Pastur bounds
# ---------------------------------------------------------------------------


class TestMarchenkoPasturBounds:

    def test_basic_sanity_n10_t1000(self):
        """MP bounds for N=10, T=1000 with sigma2=1."""
        lam_min, lam_max = marchenko_pastur_bounds(10, 1000)
        q = 10 / 1000
        expected_max = (1 + np.sqrt(q)) ** 2
        expected_min = (1 - np.sqrt(q)) ** 2
        assert pytest.approx(lam_max) == expected_max
        assert pytest.approx(lam_min) == expected_min
        # lambda_max should be slightly above 1 for small q
        assert lam_max > 1.0
        assert lam_min < 1.0
        assert lam_min > 0.0

    def test_sigma2_scaling(self):
        """Bounds should scale linearly with sigma2."""
        lam_min1, lam_max1 = marchenko_pastur_bounds(10, 100, sigma2=1.0)
        lam_min2, lam_max2 = marchenko_pastur_bounds(10, 100, sigma2=2.0)
        assert pytest.approx(lam_max2) == 2.0 * lam_max1
        assert pytest.approx(lam_min2) == 2.0 * lam_min1

    def test_q_one_gives_zero_lower_bound(self):
        """When N=T, lambda_minus = 0."""
        lam_min, _ = marchenko_pastur_bounds(100, 100)
        assert pytest.approx(lam_min) == 0.0


# ---------------------------------------------------------------------------
# Marchenko-Pastur PDF
# ---------------------------------------------------------------------------


class TestMarchenkoPasturPDF:

    def test_zero_outside_bounds(self):
        """PDF should be zero outside [lambda_minus, lambda_plus]."""
        lam_min, lam_max = marchenko_pastur_bounds(10, 1000)
        x_below = np.array([lam_min * 0.5])
        x_above = np.array([lam_max * 1.5])
        assert marchenko_pastur_pdf(x_below, 10, 1000)[0] == 0.0
        assert marchenko_pastur_pdf(x_above, 10, 1000)[0] == 0.0

    def test_positive_inside_bounds(self):
        """PDF should be positive inside the bounds."""
        lam_min, lam_max = marchenko_pastur_bounds(10, 1000)
        x = np.linspace(lam_min + 1e-6, lam_max - 1e-6, 50)
        pdf = marchenko_pastur_pdf(x, 10, 1000)
        assert (pdf > 0).all()

    def test_integrates_close_to_one(self):
        """Numerical integral of the PDF should be close to 1 (for q < 1)."""
        N, T = 10, 1000
        lam_min, lam_max = marchenko_pastur_bounds(N, T)
        x = np.linspace(lam_min, lam_max, 5000)
        pdf = marchenko_pastur_pdf(x, N, T)
        integral = np.trapezoid(pdf, x)
        assert pytest.approx(integral, abs=0.02) == 1.0


# ---------------------------------------------------------------------------
# RMT denoising
# ---------------------------------------------------------------------------


class TestRMTDenoiseCovariance:

    def test_trace_preservation_constant_residual(self):
        """Constant-residual denoising should preserve the trace."""
        returns = _make_returns(n_days=500, n_assets=10)
        corr = _sample_correlation(returns)
        denoised = rmt_denoise_covariance(corr, 500, method="constant_residual")
        # Trace of a correlation matrix = N
        assert pytest.approx(np.trace(denoised), abs=1e-8) == corr.shape[0]

    def test_trace_preservation_targeted_shrinkage(self):
        """Targeted-shrinkage denoising should preserve the trace."""
        returns = _make_returns(n_days=500, n_assets=10)
        corr = _sample_correlation(returns)
        denoised = rmt_denoise_covariance(corr, 500, method="targeted_shrinkage")
        assert pytest.approx(np.trace(denoised), abs=1e-8) == corr.shape[0]

    def test_symmetry(self):
        """Denoised matrix should be symmetric."""
        returns = _make_returns(n_days=500, n_assets=10)
        corr = _sample_correlation(returns)
        for method in ("constant_residual", "targeted_shrinkage"):
            denoised = rmt_denoise_covariance(corr, 500, method=method)
            np.testing.assert_allclose(denoised, denoised.T, atol=1e-12)

    def test_psd(self):
        """Denoised matrix should be positive semi-definite."""
        returns = _make_returns(n_days=500, n_assets=10)
        corr = _sample_correlation(returns)
        for method in ("constant_residual", "targeted_shrinkage"):
            denoised = rmt_denoise_covariance(corr, 500, method=method)
            eigenvals = np.linalg.eigvalsh(denoised)
            assert eigenvals.min() >= -1e-10, f"Negative eigenvalue: {eigenvals.min()}"

    def test_unit_diagonal(self):
        """Denoised correlation matrix should have unit diagonal."""
        returns = _make_returns(n_days=500, n_assets=10)
        corr = _sample_correlation(returns)
        for method in ("constant_residual", "targeted_shrinkage"):
            denoised = rmt_denoise_covariance(corr, 500, method=method)
            np.testing.assert_allclose(np.diag(denoised), 1.0, atol=1e-10)

    def test_all_signal_returns_copy(self):
        """When all eigenvalues are above lambda+, return unchanged."""
        # Use very few assets relative to observations so all are signal
        returns = _make_returns(n_days=5000, n_assets=3, seed=99)
        corr = _sample_correlation(returns)
        denoised = rmt_denoise_covariance(corr, 5000, method="constant_residual")
        np.testing.assert_allclose(denoised, corr, atol=1e-12)

    def test_invalid_method_raises(self):
        corr = np.eye(3)
        with pytest.raises(ValueError, match="Unknown RMT denoise method"):
            rmt_denoise_covariance(corr, 100, method="bogus")

    def test_factor_model_recovery(self):
        """RMT denoising should produce a better-conditioned matrix that
        better separates signal from noise in a synthetic factor model.

        We check that:
        1. The noise eigenvalue spread is reduced (std of noise eigs decreases).
        2. The condition number is lower.
        """
        rng = np.random.default_rng(42)
        N, T, K = 50, 200, 3

        # True factor structure: returns = F @ beta + noise
        F = rng.normal(0, 1, (T, K))
        beta = rng.normal(0, 0.3, (K, N))
        noise = rng.normal(0, 0.05, (T, N))
        returns_arr = F @ beta + noise

        # Sample correlation
        returns_df = pd.DataFrame(
            returns_arr, columns=[f"A{i}" for i in range(N)]
        )
        sample_cov = returns_df.cov().values
        sample_vols = np.sqrt(np.diag(sample_cov))
        sample_corr = sample_cov / np.outer(sample_vols, sample_vols)

        # Denoised
        denoised_corr = rmt_denoise_covariance(
            sample_corr, T, method="constant_residual"
        )

        _, lambda_plus = marchenko_pastur_bounds(N, T)

        # Sample noise eigenvalues have high variance
        sample_eigs = np.sort(np.linalg.eigvalsh(sample_corr))[::-1]
        denoised_eigs = np.sort(np.linalg.eigvalsh(denoised_corr))[::-1]

        sample_noise_std = np.std(sample_eigs[sample_eigs <= lambda_plus])
        denoised_noise_std = np.std(denoised_eigs[denoised_eigs <= lambda_plus])

        assert denoised_noise_std < sample_noise_std, (
            f"Denoised noise eigenvalue std ({denoised_noise_std:.4f}) should be "
            f"less than sample ({sample_noise_std:.4f})"
        )

        # Condition number should decrease
        cond_sample = np.linalg.cond(sample_corr)
        cond_denoised = np.linalg.cond(denoised_corr)
        assert cond_denoised < cond_sample


# ---------------------------------------------------------------------------
# RMTDenoisedCovariance estimator class
# ---------------------------------------------------------------------------


class TestRMTDenoisedCovariance:

    def test_output_shape(self):
        returns = _make_returns(n_assets=10)
        cov = RMTDenoisedCovariance().estimate(returns)
        assert cov.shape == (10, 10)
        assert list(cov.index) == list(cov.columns)

    def test_symmetric(self):
        returns = _make_returns()
        cov = RMTDenoisedCovariance().estimate(returns)
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-12)

    def test_positive_diagonal(self):
        returns = _make_returns()
        cov = RMTDenoisedCovariance().estimate(returns)
        assert (np.diag(cov.values) > 0).all()

    def test_psd(self):
        returns = _make_returns()
        cov = RMTDenoisedCovariance().estimate(returns)
        eigenvals = np.linalg.eigvalsh(cov.values)
        assert eigenvals.min() >= -1e-10

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            RMTDenoisedCovariance().estimate(pd.DataFrame())

    def test_ticker_labels_preserved(self):
        returns = _make_returns(n_assets=4)
        returns.columns = ["SPY", "QQQ", "TLT", "GLD"]
        cov = RMTDenoisedCovariance().estimate(returns)
        assert list(cov.index) == ["SPY", "QQQ", "TLT", "GLD"]
        assert list(cov.columns) == ["SPY", "QQQ", "TLT", "GLD"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:

    def test_get_rmt_denoised(self):
        est = get_covariance_estimator("rmt_denoised")
        assert isinstance(est, RMTDenoisedCovariance)

    def test_existing_methods_still_work(self):
        """Existing factory names must remain supported."""
        from src.optimization.covariance import (
            EWMACovariance,
            LedoitWolfCovariance,
            SampleCovariance,
        )

        assert isinstance(get_covariance_estimator("sample"), SampleCovariance)
        assert isinstance(get_covariance_estimator("ledoit_wolf"), LedoitWolfCovariance)
        assert isinstance(get_covariance_estimator("ewma"), EWMACovariance)

    def test_unknown_still_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_covariance_estimator("magic")


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


class TestCovarianceComparison:

    def test_smoke(self):
        returns = _make_returns(n_days=500, n_assets=10)
        result = compare_covariance_estimators(returns)
        assert set(result.keys()) == {
            "sample", "ledoit_wolf", "ledoit_wolf_nonlinear", "ewma",
            "rmt_denoised", "dcc_garch",
        }
        for name, data in result.items():
            assert "cov" in data
            assert "eigenvalues" in data
            assert "condition_number" in data
            assert "trace" in data
            assert "frobenius_norm" in data
            assert data["cov"].shape == (10, 10)
            assert len(data["eigenvalues"]) == 10
            # Eigenvalues should be sorted descending
            assert (np.diff(data["eigenvalues"]) <= 1e-12).all()

    def test_subset_methods(self):
        returns = _make_returns(n_days=500, n_assets=5)
        result = compare_covariance_estimators(returns, methods=["sample", "rmt_denoised"])
        assert set(result.keys()) == {"sample", "rmt_denoised"}

    def test_rmt_lower_condition_than_sample(self):
        """RMT denoising should reduce condition number vs raw sample."""
        returns = _make_returns(n_days=200, n_assets=20, seed=7)
        result = compare_covariance_estimators(returns, methods=["sample", "rmt_denoised"])
        assert result["rmt_denoised"]["condition_number"] <= result["sample"]["condition_number"]
