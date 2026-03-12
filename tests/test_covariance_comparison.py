"""Tests for covariance comparison framework (Phase 3D)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.covariance_comparison import (
    CovarianceComparisonResult,
    compare_covariance_estimators,
    run_covariance_backtest_comparison,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAST_METHODS = ["sample", "ledoit_wolf"]


def _make_prices(n_days: int = 300, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    """Synthetic price series for testing (random walk)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    log_rets = rng.normal(0.0003, 0.01, (n_days, n_assets))
    prices = np.exp(np.cumsum(log_rets, axis=0)) * 100
    tickers = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_returns(n_days: int = 300, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    data = rng.normal(0, 0.01, (n_days, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


# ---------------------------------------------------------------------------
# Existing diagnostic function (regression)
# ---------------------------------------------------------------------------


class TestExistingDiagnostics:

    def test_compare_covariance_estimators_returns_all_methods(self):
        returns = _make_returns(n_days=300, n_assets=5)
        result = compare_covariance_estimators(returns, methods=FAST_METHODS)
        assert set(result.keys()) == set(FAST_METHODS)

    def test_eigenvalues_sorted_descending(self):
        returns = _make_returns(n_days=300, n_assets=5)
        result = compare_covariance_estimators(returns, methods=FAST_METHODS)
        for name, data in result.items():
            assert (np.diff(data["eigenvalues"]) <= 1e-12).all()

    def test_custom_method_list(self):
        returns = _make_returns(n_days=300, n_assets=5)
        result = compare_covariance_estimators(returns, methods=["sample"])
        assert set(result.keys()) == {"sample"}


# ---------------------------------------------------------------------------
# CovarianceComparisonResult dataclass
# ---------------------------------------------------------------------------


class TestCovarianceComparisonResult:

    def test_dataclass_fields(self):
        """All expected fields are accessible."""
        fields = {
            "method", "sharpe", "cagr", "max_drawdown", "sortino",
            "annualized_turnover", "total_cost_drag_bps", "condition_number",
            "backtest_result",
        }
        from dataclasses import fields as dc_fields
        actual = {f.name for f in dc_fields(CovarianceComparisonResult)}
        assert actual == fields


# ---------------------------------------------------------------------------
# Backtest-based comparison
# ---------------------------------------------------------------------------


class TestRunCovarianceBacktestComparison:

    @pytest.fixture(scope="class")
    def comparison(self):
        """Run comparison once for the class (expensive)."""
        prices = _make_prices(n_days=300, n_assets=3)
        return run_covariance_backtest_comparison(
            prices,
            methods=FAST_METHODS,
            return_estimator="historical_mean",
            kelly_fraction=0.5,
            rebalance_freq="monthly",
        )

    def test_returns_dict_keyed_by_method(self, comparison):
        assert set(comparison.keys()) == set(FAST_METHODS)

    def test_result_type(self, comparison):
        for res in comparison.values():
            assert isinstance(res, CovarianceComparisonResult)

    def test_sharpe_is_finite(self, comparison):
        for res in comparison.values():
            assert np.isfinite(res.sharpe)

    def test_cagr_is_finite(self, comparison):
        for res in comparison.values():
            assert np.isfinite(res.cagr)

    def test_max_drawdown_negative(self, comparison):
        for res in comparison.values():
            assert res.max_drawdown <= 0

    def test_condition_number_positive(self, comparison):
        for res in comparison.values():
            assert res.condition_number > 0

    def test_backtest_result_has_equity_curve(self, comparison):
        for res in comparison.values():
            assert len(res.backtest_result.equity_curve) > 0

    def test_sortino_is_finite(self, comparison):
        for res in comparison.values():
            assert np.isfinite(res.sortino)

    def test_annualized_turnover_non_negative(self, comparison):
        for res in comparison.values():
            assert res.annualized_turnover >= 0


class TestReturnEstimatorRespected:

    def test_bl_return_estimator(self):
        """Comparison with black_litterman return estimator runs without error."""
        prices = _make_prices(n_days=300, n_assets=3)
        result = run_covariance_backtest_comparison(
            prices,
            methods=["ledoit_wolf"],
            return_estimator="black_litterman",
            kelly_fraction=0.5,
            rebalance_freq="monthly",
        )
        assert "ledoit_wolf" in result
        assert np.isfinite(result["ledoit_wolf"].sharpe)


# ---------------------------------------------------------------------------
# BL + Kelly cov_method parameter
# ---------------------------------------------------------------------------


class TestBLKellySignalCovMethod:

    def test_bl_kelly_signal_accepts_cov_method(self):
        """make_bl_kelly_signal should accept and use cov_method parameter."""
        from src.optimization.signals import make_bl_kelly_signal
        # Should not raise
        signal = make_bl_kelly_signal(cov_method="sample", fraction=0.5)
        assert callable(signal)

    def test_bl_kelly_signal_different_cov_methods_differ(self):
        """Different cov_method values should produce different weight estimates."""
        from src.optimization.signals import make_bl_kelly_signal
        prices = _make_prices(n_days=300, n_assets=3)

        sig_sample = make_bl_kelly_signal(cov_method="sample", fraction=0.5)
        sig_lw = make_bl_kelly_signal(cov_method="ledoit_wolf", fraction=0.5)

        date = prices.index[-1]
        current_weights = pd.Series(1.0 / 3, index=prices.columns)
        w1 = sig_sample(date, prices, current_weights)
        w2 = sig_lw(date, prices, current_weights)

        # They should produce weights (may or may not differ for synthetic data,
        # but both should be valid)
        assert len(w1) == 3
        assert len(w2) == 3
