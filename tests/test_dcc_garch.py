"""Tests for DCC-GARCH covariance estimation (Phase 3C)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.covariance import get_covariance_estimator
from src.optimization.dcc_garch import (
    DCCGarchCovariance,
    DCCGarchEstimator,
    _ewma_vol_fallback,
    _fit_garch_single,
    dcc_garch_covariance,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(
    n_days: int = 500,
    n_assets: int = 3,
    seed: int = 42,
    vol: float = 0.01,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    data = rng.normal(0, vol, (n_days, n_assets))
    tickers = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=dates, columns=tickers)


def _make_regime_shift_returns(
    n_days: int = 600,
    seed: int = 42,
) -> pd.DataFrame:
    """Create returns with a clear regime shift at the midpoint.

    First half: low vol, low correlation between A0 and A1.
    Second half: high vol, high correlation between A0 and A1.
    A2 is independent noise throughout.
    """
    rng = np.random.default_rng(seed)
    half = n_days // 2
    dates = pd.bdate_range("2018-01-01", periods=n_days)

    # Calm regime: independent low-vol assets
    calm_factor = rng.normal(0, 0.005, half)
    calm_noise = rng.normal(0, 0.008, (half, 3))
    calm = calm_noise.copy()
    calm[:, 0] += calm_factor * 0.2  # weak factor loading
    calm[:, 1] += calm_factor * 0.1

    # Crisis regime: correlated high-vol assets
    crisis_factor = rng.normal(0, 0.03, n_days - half)
    crisis_noise = rng.normal(0, 0.005, (n_days - half, 3))
    crisis = crisis_noise.copy()
    crisis[:, 0] += crisis_factor * 1.0  # strong factor loading
    crisis[:, 1] += crisis_factor * 0.8

    data = np.vstack([calm, crisis])
    return pd.DataFrame(data, index=dates, columns=["A0", "A1", "A2"])


# ---------------------------------------------------------------------------
# Univariate GARCH fitting
# ---------------------------------------------------------------------------


class TestGarchFitting:

    def test_fit_returns_series(self):
        returns = _make_returns(n_days=500, n_assets=1)
        vol, used_fallback = _fit_garch_single(returns.iloc[:, 0])
        assert isinstance(vol, pd.Series)
        assert len(vol) == 500
        assert not used_fallback

    def test_fit_no_nan(self):
        returns = _make_returns(n_days=500, n_assets=1)
        vol, _ = _fit_garch_single(returns.iloc[:, 0])
        assert not vol.isna().any()

    def test_fallback_on_constant_series(self):
        """A constant series should trigger the EWMA fallback."""
        dates = pd.bdate_range("2020-01-01", periods=200)
        const = pd.Series(0.0, index=dates, name="CONST")
        vol, used_fallback = _fit_garch_single(const)
        assert used_fallback
        assert len(vol) == 200

    def test_ewma_fallback_positive(self):
        returns = _make_returns(n_days=300, n_assets=1)
        vol = _ewma_vol_fallback(returns.iloc[:, 0])
        assert (vol >= 0).all()
        assert len(vol) == 300


# ---------------------------------------------------------------------------
# DCCGarchEstimator
# ---------------------------------------------------------------------------


class TestDCCGarchEstimator:

    def test_covariance_shape(self):
        returns = _make_returns(n_days=500, n_assets=4)
        est = DCCGarchEstimator(returns)
        cov = est.get_covariance(returns.index[-1])
        assert cov.shape == (4, 4)
        assert list(cov.index) == list(cov.columns)

    def test_covariance_symmetric(self):
        returns = _make_returns(n_days=500, n_assets=3)
        est = DCCGarchEstimator(returns)
        cov = est.get_covariance(returns.index[-1])
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-12)

    def test_covariance_psd(self):
        returns = _make_returns(n_days=500, n_assets=3)
        est = DCCGarchEstimator(returns)
        cov = est.get_covariance(returns.index[-1])
        eigenvals = np.linalg.eigvalsh(cov.values)
        assert eigenvals.min() >= -1e-10, f"Negative eigenvalue: {eigenvals.min()}"

    def test_covariance_no_nan_no_inf(self):
        returns = _make_returns(n_days=500, n_assets=3)
        est = DCCGarchEstimator(returns)
        cov = est.get_covariance(returns.index[-1])
        assert not np.any(np.isnan(cov.values))
        assert not np.any(np.isinf(cov.values))

    def test_date_lookup_first_and_last(self):
        returns = _make_returns(n_days=500, n_assets=3)
        est = DCCGarchEstimator(returns)
        cov_first = est.get_covariance(returns.index[0])
        cov_last = est.get_covariance(returns.index[-1])
        assert cov_first.shape == (3, 3)
        assert cov_last.shape == (3, 3)
        # Should generally differ (time-varying)
        assert not np.allclose(cov_first.values, cov_last.values)

    def test_date_lookup_between_dates(self):
        """A date between two trading days should return the nearest prior."""
        returns = _make_returns(n_days=500, n_assets=3)
        est = DCCGarchEstimator(returns)
        # Pick a weekend date between two trading days
        mid_date = returns.index[250] + pd.Timedelta(days=1)
        cov = est.get_covariance(mid_date)
        assert cov.shape == (3, 3)

    def test_dcc_params_valid(self):
        returns = _make_returns(n_days=500, n_assets=3)
        est = DCCGarchEstimator(returns)
        a, b = est.dcc_params
        assert a >= 0
        assert b >= 0
        assert a + b < 1.0

    def test_correlation_timeseries_shape(self):
        returns = _make_returns(n_days=500, n_assets=3)
        est = DCCGarchEstimator(returns)
        rho = est.get_correlation_timeseries("A0", "A1")
        assert isinstance(rho, pd.Series)
        assert len(rho) == 500

    def test_correlation_timeseries_bounded(self):
        returns = _make_returns(n_days=500, n_assets=3)
        est = DCCGarchEstimator(returns)
        rho = est.get_correlation_timeseries("A0", "A1")
        assert rho.min() >= -1.0 - 1e-10
        assert rho.max() <= 1.0 + 1e-10

    def test_self_correlation_is_one(self):
        returns = _make_returns(n_days=500, n_assets=3)
        est = DCCGarchEstimator(returns)
        rho = est.get_correlation_timeseries("A0", "A0")
        np.testing.assert_allclose(rho.values, 1.0, atol=1e-10)

    def test_positive_diagonal(self):
        returns = _make_returns(n_days=500, n_assets=3)
        est = DCCGarchEstimator(returns)
        cov = est.get_covariance(returns.index[-1])
        assert (np.diag(cov.values) > 0).all()

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            DCCGarchEstimator(pd.DataFrame())

    def test_ticker_labels_preserved(self):
        returns = _make_returns(n_days=500, n_assets=3)
        returns.columns = ["SPY", "TLT", "GLD"]
        est = DCCGarchEstimator(returns)
        cov = est.get_covariance(returns.index[-1])
        assert list(cov.index) == ["SPY", "TLT", "GLD"]
        assert list(cov.columns) == ["SPY", "TLT", "GLD"]


# ---------------------------------------------------------------------------
# Regime shift test
# ---------------------------------------------------------------------------


class TestRegimeShift:

    def test_correlation_rises_during_crisis(self):
        """DCC should show higher A0/A1 correlation in the crisis period."""
        returns = _make_regime_shift_returns(n_days=600, seed=42)
        est = DCCGarchEstimator(returns)
        rho = est.get_correlation_timeseries("A0", "A1")

        half = len(returns) // 2
        calm_mean = float(rho.iloc[:half].mean())
        crisis_mean = float(rho.iloc[half:].mean())
        assert crisis_mean > calm_mean, (
            f"Crisis correlation ({crisis_mean:.3f}) should exceed "
            f"calm correlation ({calm_mean:.3f})"
        )

    def test_independent_asset_stays_low(self):
        """A2 (independent noise) should have low correlation with A0 throughout."""
        returns = _make_regime_shift_returns(n_days=600, seed=42)
        est = DCCGarchEstimator(returns)
        rho = est.get_correlation_timeseries("A0", "A2")
        assert abs(float(rho.mean())) < 0.4


# ---------------------------------------------------------------------------
# Fallback coverage
# ---------------------------------------------------------------------------


class TestFallback:

    def test_fallback_tickers_reported(self):
        """Estimator with a degenerate asset should report fallback."""
        returns = _make_returns(n_days=500, n_assets=3)
        # Make one asset constant → GARCH will fail
        returns["A2"] = 0.0
        est = DCCGarchEstimator(returns)
        assert "A2" in est.fallback_tickers

    def test_fallback_still_produces_valid_cov(self):
        """Even with fallback assets, the output should be valid."""
        returns = _make_returns(n_days=500, n_assets=3)
        returns["A2"] = 0.0
        est = DCCGarchEstimator(returns)
        cov = est.get_covariance(returns.index[-1])
        assert cov.shape == (3, 3)
        assert not np.any(np.isnan(cov.values))
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-12)

    def test_all_assets_fallback(self):
        """All-constant assets should still produce a result (identity-like)."""
        dates = pd.bdate_range("2020-01-01", periods=200)
        returns = pd.DataFrame(
            0.0, index=dates, columns=["X", "Y", "Z"]
        )
        est = DCCGarchEstimator(returns)
        cov = est.get_covariance(returns.index[-1])
        assert cov.shape == (3, 3)
        assert not np.any(np.isnan(cov.values))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestConvenienceFunction:

    def test_dcc_garch_covariance(self):
        returns = _make_returns(n_days=500, n_assets=3)
        cov = dcc_garch_covariance(returns, returns.index[-1])
        assert cov.shape == (3, 3)
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Factory-compatible wrapper
# ---------------------------------------------------------------------------


class TestDCCGarchCovariance:

    def test_estimate_interface(self):
        returns = _make_returns(n_days=500, n_assets=3)
        est = DCCGarchCovariance()
        cov = est.estimate(returns)
        assert cov.shape == (3, 3)
        assert list(cov.index) == list(cov.columns)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            DCCGarchCovariance().estimate(pd.DataFrame())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:

    def test_get_dcc_garch(self):
        est = get_covariance_estimator("dcc_garch")
        assert isinstance(est, DCCGarchCovariance)

    def test_existing_methods_still_work(self):
        from src.optimization.covariance import (
            EWMACovariance,
            LedoitWolfCovariance,
            LedoitWolfNonlinearCovariance,
            RMTDenoisedCovariance,
            SampleCovariance,
        )

        assert isinstance(get_covariance_estimator("sample"), SampleCovariance)
        assert isinstance(get_covariance_estimator("ledoit_wolf"), LedoitWolfCovariance)
        assert isinstance(
            get_covariance_estimator("ledoit_wolf_nonlinear"),
            LedoitWolfNonlinearCovariance,
        )
        assert isinstance(get_covariance_estimator("ewma"), EWMACovariance)
        assert isinstance(get_covariance_estimator("rmt_denoised"), RMTDenoisedCovariance)
