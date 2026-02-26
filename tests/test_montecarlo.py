"""Tests for src/monte_carlo/simulation.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monte_carlo.simulation import MonteCarloResult, run_monte_carlo


def _sample_returns(n: int = 500, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    return pd.Series(np.random.normal(0.0008, 0.012, n))


class TestMonteCarloReproducibility:
    def test_same_seed_same_result(self):
        ret = _sample_returns()
        r1 = run_monte_carlo(ret, n_paths=100, n_days=52, seed=7)
        r2 = run_monte_carlo(ret, n_paths=100, n_days=52, seed=7)
        np.testing.assert_array_equal(r1.terminal_wealth, r2.terminal_wealth)

    def test_different_seeds_different_results(self):
        ret = _sample_returns()
        r1 = run_monte_carlo(ret, n_paths=100, n_days=52, seed=1)
        r2 = run_monte_carlo(ret, n_paths=100, n_days=52, seed=2)
        assert not np.array_equal(r1.terminal_wealth, r2.terminal_wealth)


class TestMonteCarloShape:
    def test_terminal_wealth_shape(self):
        ret = _sample_returns()
        result = run_monte_carlo(ret, n_paths=200, n_days=63, seed=42)
        assert result.terminal_wealth.shape == (200,)
        assert result.path_max_drawdowns.shape == (200,)

    def test_store_paths_shape(self):
        ret = _sample_returns()
        result = run_monte_carlo(ret, n_paths=50, n_days=30, seed=42, store_paths=True)
        assert result.equity_paths is not None
        assert result.equity_paths.shape == (50, 31)  # n_days+1

    def test_no_store_paths_is_none(self):
        ret = _sample_returns()
        result = run_monte_carlo(ret, n_paths=50, n_days=30, seed=42, store_paths=False)
        assert result.equity_paths is None


class TestMonteCarloStats:
    def test_all_paths_start_at_capital(self):
        ret = _sample_returns()
        capital = 1_000_000
        result = run_monte_carlo(
            ret,
            n_paths=100,
            n_days=50,
            seed=42,
            store_paths=True,
            initial_capital=capital,
        )
        np.testing.assert_array_equal(result.equity_paths[:, 0], capital)

    def test_positive_drift_prob_profit_high(self):
        """Strong positive daily returns → most paths profitable."""
        ret = pd.Series([0.005] * 252)  # 0.5% daily
        result = run_monte_carlo(ret, n_paths=1000, n_days=252, seed=42)
        assert result.prob_profit > 0.90

    def test_negative_drift_prob_profit_low(self):
        """Strong negative daily returns → most paths lose money."""
        ret = pd.Series([-0.005] * 252)
        result = run_monte_carlo(ret, n_paths=1000, n_days=252, seed=42)
        assert result.prob_profit < 0.10

    def test_drawdowns_non_positive(self):
        ret = _sample_returns()
        result = run_monte_carlo(ret, n_paths=200, n_days=63, seed=42)
        assert (result.path_max_drawdowns <= 0).all()

    def test_terminal_wealth_positive(self):
        ret = _sample_returns()
        result = run_monte_carlo(ret, n_paths=500, n_days=252, seed=42)
        assert (result.terminal_wealth > 0).all()

    def test_median_less_than_mean(self):
        """Log-normal distribution: median < mean due to positive skew."""
        ret = _sample_returns()
        result = run_monte_carlo(ret, n_paths=5000, n_days=252, seed=42)
        assert result.median_terminal_wealth < result.mean_terminal_wealth


class TestUnderwaterPeriods:
    def test_requires_store_paths(self):
        ret = _sample_returns()
        result = run_monte_carlo(ret, n_paths=50, n_days=30, seed=42, store_paths=False)
        with pytest.raises(ValueError, match="store_paths"):
            result.underwater_periods()

    def test_returns_array_of_correct_length(self):
        ret = _sample_returns()
        result = run_monte_carlo(ret, n_paths=50, n_days=30, seed=42, store_paths=True)
        periods = result.underwater_periods()
        assert len(periods) == 50

    def test_all_positive_drift_zero_underwater(self):
        """Paths with strong positive returns should never go underwater."""
        ret = pd.Series([0.01] * 100)  # 1% daily — never below start
        result = run_monte_carlo(ret, n_paths=200, n_days=50, seed=42, store_paths=True)
        periods = result.underwater_periods()
        assert (periods == 0).all()


class TestMultiAssetMode:
    def test_multiasset_produces_result(self):
        np.random.seed(42)
        dates = pd.bdate_range("2020-01-01", periods=300)
        asset_returns = pd.DataFrame(
            np.random.normal(0.0005, 0.01, (300, 3)),
            index=dates,
            columns=["A", "B", "C"],
        )
        port_returns = asset_returns.mean(axis=1)
        weights = pd.Series({"A": 0.33, "B": 0.33, "C": 0.34})

        result = run_monte_carlo(
            daily_returns=port_returns,
            weights=weights,
            asset_returns=asset_returns,
            n_paths=200,
            n_days=63,
            seed=42,
        )
        assert result.terminal_wealth.shape == (200,)
        assert (result.terminal_wealth > 0).all()
