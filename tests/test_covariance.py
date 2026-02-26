"""Tests for src/optimization/covariance.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.covariance import (
    EWMACovariance,
    LedoitWolfCovariance,
    SampleCovariance,
    get_covariance_estimator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(n_days: int = 500, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    data = np.random.normal(0, 0.01, (n_days, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


# ---------------------------------------------------------------------------
# Shared interface tests
# ---------------------------------------------------------------------------


class TestSharedInterface:
    """All three estimators should produce valid N×N PSD matrices."""

    @pytest.mark.parametrize(
        "EstimatorClass",
        [
            SampleCovariance,
            LedoitWolfCovariance,
            EWMACovariance,
        ],
    )
    def test_output_shape(self, EstimatorClass):
        returns = _make_returns(n_assets=4)
        estimator = (
            EstimatorClass()
            if EstimatorClass != EWMACovariance
            else EstimatorClass(decay=0.94)
        )
        cov = estimator.estimate(returns)
        assert cov.shape == (4, 4)
        assert list(cov.index) == list(cov.columns)

    @pytest.mark.parametrize(
        "EstimatorClass",
        [
            SampleCovariance,
            LedoitWolfCovariance,
            EWMACovariance,
        ],
    )
    def test_symmetric(self, EstimatorClass):
        returns = _make_returns()
        estimator = (
            EstimatorClass()
            if EstimatorClass != EWMACovariance
            else EstimatorClass(decay=0.94)
        )
        cov = estimator.estimate(returns)
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-12)

    @pytest.mark.parametrize(
        "EstimatorClass",
        [
            SampleCovariance,
            LedoitWolfCovariance,
            EWMACovariance,
        ],
    )
    def test_positive_diagonal(self, EstimatorClass):
        returns = _make_returns()
        estimator = (
            EstimatorClass()
            if EstimatorClass != EWMACovariance
            else EstimatorClass(decay=0.94)
        )
        cov = estimator.estimate(returns)
        assert (np.diag(cov.values) > 0).all()

    @pytest.mark.parametrize(
        "EstimatorClass",
        [
            SampleCovariance,
            LedoitWolfCovariance,
            EWMACovariance,
        ],
    )
    def test_empty_raises(self, EstimatorClass):
        estimator = (
            EstimatorClass()
            if EstimatorClass != EWMACovariance
            else EstimatorClass(decay=0.94)
        )
        with pytest.raises(ValueError, match="empty"):
            estimator.estimate(pd.DataFrame())


class TestSampleCovariance:

    def test_matches_pandas(self):
        returns = _make_returns()
        cov = SampleCovariance().estimate(returns)
        pd.testing.assert_frame_equal(cov, returns.cov())


class TestLedoitWolfCovariance:

    def test_shrinkage_reduces_condition_number(self):
        """Shrunk covariance should have a lower condition number than sample."""
        returns = _make_returns(n_days=50, n_assets=30, seed=99)
        sample_cov = SampleCovariance().estimate(returns).values
        lw_cov = LedoitWolfCovariance().estimate(returns).values

        cond_sample = np.linalg.cond(sample_cov)
        cond_lw = np.linalg.cond(lw_cov)
        assert cond_lw < cond_sample


class TestEWMACovariance:

    def test_invalid_decay_raises(self):
        with pytest.raises(ValueError, match="Decay"):
            EWMACovariance(decay=0.0)
        with pytest.raises(ValueError, match="Decay"):
            EWMACovariance(decay=1.0)
        with pytest.raises(ValueError, match="Decay"):
            EWMACovariance(decay=-0.5)

    def test_higher_decay_smoother(self):
        """Higher decay (more memory) → more similar to sample covariance."""
        returns = _make_returns(n_days=500)
        sample = SampleCovariance().estimate(returns).values
        ewma_high = EWMACovariance(decay=0.99).estimate(returns).values
        ewma_low = EWMACovariance(decay=0.80).estimate(returns).values

        diff_high = np.linalg.norm(ewma_high - sample)
        diff_low = np.linalg.norm(ewma_low - sample)
        assert diff_high < diff_low


class TestFactory:

    def test_get_sample(self):
        est = get_covariance_estimator("sample")
        assert isinstance(est, SampleCovariance)

    def test_get_ledoit_wolf(self):
        est = get_covariance_estimator("ledoit_wolf")
        assert isinstance(est, LedoitWolfCovariance)

    def test_get_ewma(self):
        est = get_covariance_estimator("ewma")
        assert isinstance(est, EWMACovariance)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_covariance_estimator("magic")
