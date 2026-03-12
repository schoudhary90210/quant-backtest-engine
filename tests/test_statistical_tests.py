"""Tests for src/validation/statistical_tests.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.validation.statistical_tests import (
    cpcv_backtest,
    deflated_sharpe_ratio,
    minimum_backtest_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_returns(
    n_assets: int = 3,
    n_days: int = 1000,
    drift: float = 0.0005,
    vol: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily log returns with given drift and vol."""
    np.random.seed(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    data = np.random.normal(drift, vol, size=(n_days, n_assets))
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=dates, columns=tickers)


def _equal_weight_strategy(train_returns: pd.DataFrame) -> pd.Series:
    """Simple equal-weight strategy for CPCV tests."""
    n = len(train_returns.columns)
    return pd.Series(1.0 / n, index=train_returns.columns)


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


class TestDeflatedSharpeRatio:
    def test_single_trial_equals_psr(self) -> None:
        """With n_trials=1, DSR should equal PSR(benchmark=0)."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=1.0,
            n_trials=1,
            n_observations=500,
            skewness=0.0,
            kurtosis=0.0,
        )
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=1.0,
            benchmark_sharpe=0.0,
            n_observations=500,
            skewness=0.0,
            kurtosis=0.0,
        )
        assert dsr["dsr"] == pytest.approx(psr["psr"], abs=1e-10)

    def test_many_trials_deflates(self) -> None:
        """More trials should lower the DSR."""
        dsr_1 = deflated_sharpe_ratio(
            observed_sharpe=1.0,
            n_trials=1,
            n_observations=500,
            skewness=0.0,
            kurtosis=0.0,
        )
        dsr_100 = deflated_sharpe_ratio(
            observed_sharpe=1.0,
            n_trials=100,
            n_observations=500,
            skewness=0.0,
            kurtosis=0.0,
        )
        assert dsr_100["dsr"] < dsr_1["dsr"]

    def test_high_sharpe_significant(self) -> None:
        dsr = deflated_sharpe_ratio(
            observed_sharpe=2.0,
            n_trials=1,
            n_observations=1000,
            skewness=0.0,
            kurtosis=0.0,
        )
        assert dsr["is_significant"] is True
        assert dsr["dsr"] > 0.99

    def test_zero_sharpe(self) -> None:
        dsr = deflated_sharpe_ratio(
            observed_sharpe=0.0,
            n_trials=1,
            n_observations=500,
            skewness=0.0,
            kurtosis=0.0,
        )
        assert dsr["dsr"] == pytest.approx(0.5, abs=0.01)

    def test_negative_sharpe(self) -> None:
        dsr = deflated_sharpe_ratio(
            observed_sharpe=-0.5,
            n_trials=1,
            n_observations=500,
            skewness=0.0,
            kurtosis=0.0,
        )
        assert dsr["dsr"] < 0.5

    def test_return_keys(self) -> None:
        result = deflated_sharpe_ratio(
            observed_sharpe=1.0,
            n_trials=5,
            n_observations=252,
            skewness=0.1,
            kurtosis=1.0,
        )
        assert set(result.keys()) == {"dsr", "expected_max_sr", "sr_std", "is_significant"}

    def test_expected_max_sr_increases_with_trials(self) -> None:
        r5 = deflated_sharpe_ratio(1.0, 5, 500, 0.0, 0.0)
        r50 = deflated_sharpe_ratio(1.0, 50, 500, 0.0, 0.0)
        assert r50["expected_max_sr"] > r5["expected_max_sr"]


# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------


class TestProbabilisticSharpeRatio:
    def test_high_sr_large_t_approx_one(self) -> None:
        result = probabilistic_sharpe_ratio(
            observed_sharpe=2.0,
            benchmark_sharpe=0.0,
            n_observations=2000,
            skewness=0.0,
            kurtosis=0.0,
        )
        assert result["psr"] > 0.999

    def test_sr_equals_benchmark_is_half(self) -> None:
        result = probabilistic_sharpe_ratio(
            observed_sharpe=0.5,
            benchmark_sharpe=0.5,
            n_observations=500,
            skewness=0.0,
            kurtosis=0.0,
        )
        assert result["psr"] == pytest.approx(0.5, abs=1e-10)

    def test_sr_below_benchmark(self) -> None:
        result = probabilistic_sharpe_ratio(
            observed_sharpe=0.3,
            benchmark_sharpe=1.0,
            n_observations=500,
            skewness=0.0,
            kurtosis=0.0,
        )
        assert result["psr"] < 0.5

    def test_return_keys(self) -> None:
        result = probabilistic_sharpe_ratio(
            observed_sharpe=1.0,
            benchmark_sharpe=0.5,
            n_observations=252,
            skewness=-0.2,
            kurtosis=3.0,
        )
        assert set(result.keys()) == {"psr", "sr_std", "is_significant"}

    def test_normal_returns_sr_std(self) -> None:
        """With normal returns (skew=0, kurt=0), per-period SR std ≈ 1/sqrt(T-1)."""
        # Use periods_per_year=1 so returned sr_std is per-period
        result = probabilistic_sharpe_ratio(
            observed_sharpe=0.0,
            benchmark_sharpe=0.0,
            n_observations=101,
            skewness=0.0,
            kurtosis=0.0,
            periods_per_year=1,
        )
        expected_std = 1.0 / math.sqrt(100)
        assert result["sr_std"] == pytest.approx(expected_std, abs=1e-10)

    def test_sr_std_annualised(self) -> None:
        """Annualised sr_std = per-period sr_std * sqrt(periods_per_year)."""
        result_annual = probabilistic_sharpe_ratio(
            observed_sharpe=0.0,
            benchmark_sharpe=0.0,
            n_observations=101,
            skewness=0.0,
            kurtosis=0.0,
            periods_per_year=252,
        )
        expected_annual_std = 1.0 / math.sqrt(100) * math.sqrt(252)
        assert result_annual["sr_std"] == pytest.approx(expected_annual_std, abs=1e-10)


# ---------------------------------------------------------------------------
# CPCV
# ---------------------------------------------------------------------------


class TestCPCV:
    def test_synthetic_profitable_signal(self) -> None:
        """Equal-weight strategy on positive-drift returns → positive mean OOS Sharpe."""
        returns = _synthetic_returns(n_assets=3, n_days=1000, drift=0.001)
        result = cpcv_backtest(
            returns=returns,
            strategy_fn=_equal_weight_strategy,
            n_groups=6,
            n_test_groups=2,
            purge_window=3,
            embargo_pct=0.005,
        )
        assert result["mean_oos_sharpe"] > 0

    def test_n_splits_correct(self) -> None:
        """Number of splits should be C(n_groups, n_test_groups)."""
        returns = _synthetic_returns(n_days=500)
        result = cpcv_backtest(
            returns=returns,
            strategy_fn=_equal_weight_strategy,
            n_groups=8,
            n_test_groups=2,
            purge_window=2,
        )
        from math import comb

        assert result["n_splits"] == comb(8, 2)

    def test_return_keys(self) -> None:
        returns = _synthetic_returns(n_days=500)
        result = cpcv_backtest(
            returns=returns,
            strategy_fn=_equal_weight_strategy,
            n_groups=5,
            n_test_groups=2,
        )
        expected_keys = {
            "oos_returns",
            "per_split_sharpe",
            "per_split_is_sharpe",
            "mean_oos_sharpe",
            "std_oos_sharpe",
            "n_splits",
        }
        assert set(result.keys()) == expected_keys

    def test_oos_returns_nonempty(self) -> None:
        returns = _synthetic_returns(n_days=500)
        result = cpcv_backtest(
            returns=returns,
            strategy_fn=_equal_weight_strategy,
            n_groups=5,
            n_test_groups=2,
        )
        assert len(result["oos_returns"]) > 0
        for oos_ret in result["oos_returns"]:
            assert isinstance(oos_ret, pd.Series)
            assert len(oos_ret) > 0

    def test_sharpe_arrays_match_splits(self) -> None:
        returns = _synthetic_returns(n_days=500)
        result = cpcv_backtest(
            returns=returns,
            strategy_fn=_equal_weight_strategy,
            n_groups=5,
            n_test_groups=2,
        )
        assert len(result["per_split_sharpe"]) == result["n_splits"]
        assert len(result["per_split_is_sharpe"]) == result["n_splits"]

    def test_risk_free_rate_lowers_sharpe(self) -> None:
        """Positive rf should produce lower CPCV Sharpe than rf=0."""
        returns = _synthetic_returns(n_assets=3, n_days=1000, drift=0.001)
        result_rf0 = cpcv_backtest(
            returns=returns, strategy_fn=_equal_weight_strategy,
            n_groups=6, n_test_groups=2, risk_free_rate=0.0,
        )
        result_rf4 = cpcv_backtest(
            returns=returns, strategy_fn=_equal_weight_strategy,
            n_groups=6, n_test_groups=2, risk_free_rate=0.04,
        )
        assert result_rf4["mean_oos_sharpe"] < result_rf0["mean_oos_sharpe"]

    def test_strategy_fn_failure_handled(self) -> None:
        """Strategy that raises should be skipped, not crash."""
        def bad_strategy(train_returns: pd.DataFrame) -> pd.Series:
            raise ValueError("intentional failure")

        returns = _synthetic_returns(n_days=500)
        result = cpcv_backtest(
            returns=returns,
            strategy_fn=bad_strategy,
            n_groups=5,
            n_test_groups=2,
        )
        assert result["n_splits"] == 0


# ---------------------------------------------------------------------------
# PBO
# ---------------------------------------------------------------------------


class TestPBO:
    def test_random_noise_pbo_near_half(self) -> None:
        """Pure noise returns should give PBO near 0.5."""
        returns = _synthetic_returns(n_days=2000, drift=0.0, vol=0.01, seed=99)
        result = cpcv_backtest(
            returns=returns,
            strategy_fn=_equal_weight_strategy,
            n_groups=10,
            n_test_groups=2,
        )
        pbo = probability_of_backtest_overfitting(result)
        assert 0.1 <= pbo["pbo"] <= 0.9

    def test_return_keys(self) -> None:
        cpcv_results = {
            "per_split_sharpe": np.array([0.5, 0.8, -0.1, 0.3]),
            "per_split_is_sharpe": np.array([1.0, 1.2, 0.7, 0.9]),
        }
        result = probability_of_backtest_overfitting(cpcv_results)
        assert set(result.keys()) == {"pbo", "is_oos_rank_correlation", "degradation_ratio"}

    def test_pbo_range(self) -> None:
        cpcv_results = {
            "per_split_sharpe": np.array([0.5, 0.8, -0.1, 0.3, -0.2]),
            "per_split_is_sharpe": np.array([1.0, 1.2, 0.7, 0.9, 0.4]),
        }
        result = probability_of_backtest_overfitting(cpcv_results)
        assert 0.0 <= result["pbo"] <= 1.0

    def test_pbo_all_positive_oos(self) -> None:
        """All positive OOS Sharpe → PBO = 0."""
        cpcv_results = {
            "per_split_sharpe": np.array([0.5, 0.8, 0.3, 1.0]),
            "per_split_is_sharpe": np.array([1.0, 1.2, 0.9, 1.5]),
        }
        result = probability_of_backtest_overfitting(cpcv_results)
        assert result["pbo"] == 0.0

    def test_pbo_all_negative_oos(self) -> None:
        """All negative OOS Sharpe → PBO = 1."""
        cpcv_results = {
            "per_split_sharpe": np.array([-0.5, -0.8, -0.3, -1.0]),
            "per_split_is_sharpe": np.array([1.0, 1.2, 0.9, 1.5]),
        }
        result = probability_of_backtest_overfitting(cpcv_results)
        assert result["pbo"] == 1.0

    def test_empty_results(self) -> None:
        cpcv_results = {
            "per_split_sharpe": np.array([]),
            "per_split_is_sharpe": np.array([]),
        }
        result = probability_of_backtest_overfitting(cpcv_results)
        assert result["pbo"] == 1.0


# ---------------------------------------------------------------------------
# Minimum Backtest Length
# ---------------------------------------------------------------------------


class TestMinimumBacktestLength:
    def test_decreasing_sr_increases_minbtl(self) -> None:
        """Lower Sharpe should require longer track record."""
        btl_high = minimum_backtest_length(1.5, 0.0, 0.0)
        btl_mid = minimum_backtest_length(1.0, 0.0, 0.0)
        btl_low = minimum_backtest_length(0.5, 0.0, 0.0)
        assert btl_low["min_btl_observations"] > btl_mid["min_btl_observations"]
        assert btl_mid["min_btl_observations"] > btl_high["min_btl_observations"]

    def test_zero_sr_returns_inf(self) -> None:
        result = minimum_backtest_length(0.0, 0.0, 0.0)
        assert result["min_btl_observations"] == float("inf")
        assert result["min_btl_years"] == float("inf")

    def test_sufficient_flag_true(self) -> None:
        result = minimum_backtest_length(1.0, 0.0, 0.0, actual_observations=5000)
        assert result["actual_length_sufficient"] is True

    def test_sufficient_flag_false(self) -> None:
        result = minimum_backtest_length(0.3, 0.0, 0.0, actual_observations=10)
        assert result["actual_length_sufficient"] is False

    def test_sufficient_flag_none(self) -> None:
        result = minimum_backtest_length(1.0, 0.0, 0.0)
        assert result["actual_length_sufficient"] is None

    def test_return_keys(self) -> None:
        result = minimum_backtest_length(1.0, 0.0, 0.0)
        assert set(result.keys()) == {
            "min_btl_observations",
            "min_btl_years",
            "actual_length_sufficient",
        }

    def test_years_conversion(self) -> None:
        result = minimum_backtest_length(1.0, 0.0, 0.0)
        expected_years = result["min_btl_observations"] / 252
        assert result["min_btl_years"] == pytest.approx(expected_years, abs=1e-10)

    def test_annualised_sharpe_0_86_needs_years(self) -> None:
        """An 0.86 annualised Sharpe should require ~3-5 years, not 14 days."""
        result = minimum_backtest_length(0.86, 0.0, 0.0)
        # With normal returns (skew=0, kurt=0), daily SR = 0.86/sqrt(252) ≈ 0.054
        # MinBTL = 1 + (1.645/0.054)² ≈ 928 days ≈ 3.7 years
        assert result["min_btl_observations"] > 500
        assert result["min_btl_observations"] < 2000
        assert result["min_btl_years"] > 2.0
        assert result["min_btl_years"] < 8.0

    def test_annualised_sharpe_2_0_shorter(self) -> None:
        """A 2.0 annualised Sharpe should require roughly 150-300 days."""
        result = minimum_backtest_length(2.0, 0.0, 0.0)
        # daily SR = 2.0/sqrt(252) ≈ 0.126
        # MinBTL = 1 + (1.645/0.126)² ≈ 171 days
        assert result["min_btl_observations"] > 100
        assert result["min_btl_observations"] < 400

    def test_per_period_mode_backward_compat(self) -> None:
        """periods_per_year=1 treats SR as already per-period (old behaviour)."""
        result = minimum_backtest_length(1.0, 0.0, 0.0, periods_per_year=1)
        # SR=1.0 per-period, skew=0, kurt_excess=0 → std_kurt=3
        # sr_var_factor = 1 + (3-1)/4 * 1² = 1.5
        # MinBTL = 1 + 1.5 * (1.6449)² ≈ 5.058
        assert result["min_btl_observations"] == pytest.approx(5.058, abs=0.01)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_extreme_skewness(self) -> None:
        """Functions should not crash with extreme skewness."""
        dsr = deflated_sharpe_ratio(1.0, 1, 500, 5.0, 0.0)
        assert np.isfinite(dsr["dsr"])
        psr = probabilistic_sharpe_ratio(1.0, 0.0, 500, 5.0, 0.0)
        assert np.isfinite(psr["psr"])
        btl = minimum_backtest_length(1.0, 5.0, 0.0)
        assert np.isfinite(btl["min_btl_observations"])

    def test_extreme_kurtosis(self) -> None:
        """Functions should not crash with extreme kurtosis."""
        dsr = deflated_sharpe_ratio(1.0, 1, 500, 0.0, 50.0)
        assert np.isfinite(dsr["dsr"])
        psr = probabilistic_sharpe_ratio(1.0, 0.0, 500, 0.0, 50.0)
        assert np.isfinite(psr["psr"])
        btl = minimum_backtest_length(1.0, 0.0, 50.0)
        assert np.isfinite(btl["min_btl_observations"])

    def test_small_n_observations(self) -> None:
        """T=10 should not crash."""
        dsr = deflated_sharpe_ratio(1.0, 1, 10, 0.0, 0.0)
        assert np.isfinite(dsr["dsr"])
        psr = probabilistic_sharpe_ratio(1.0, 0.0, 10, 0.0, 0.0)
        assert np.isfinite(psr["psr"])

    def test_n_obs_one(self) -> None:
        """T=1 should return sr_std=0 and default DSR/PSR."""
        dsr = deflated_sharpe_ratio(1.0, 1, 1, 0.0, 0.0)
        assert dsr["sr_std"] == 0.0
        assert dsr["dsr"] == 0.5

    def test_negative_sharpe_minbtl(self) -> None:
        """Negative Sharpe should still return finite MinBTL."""
        result = minimum_backtest_length(-1.0, 0.0, 0.0)
        assert np.isfinite(result["min_btl_observations"])
        assert result["min_btl_observations"] > 0
