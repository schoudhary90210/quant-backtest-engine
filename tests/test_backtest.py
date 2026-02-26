"""Tests for src/engine/backtest.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.backtest import (
    BacktestResult,
    PortfolioState,
    run_backtest,
    _should_rebalance,
)
from src.engine.costs import TransactionCostModel


# ---------------------------------------------------------------------------
# Helpers — simple signal functions for testing
# ---------------------------------------------------------------------------

def equal_weight_signal(date, prices, current_weights):
    """Always return equal weights across all assets."""
    n = len(current_weights)
    return pd.Series(1.0 / n, index=current_weights.index)


def single_asset_signal(date, prices, current_weights):
    """100% in the first asset."""
    weights = pd.Series(0.0, index=current_weights.index)
    weights.iloc[0] = 1.0
    return weights


def zero_weight_signal(date, prices, current_weights):
    """All cash — zero allocation."""
    return pd.Series(0.0, index=current_weights.index)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPortfolioState:

    def test_weights_with_holdings(self):
        state = PortfolioState(
            cash=0.0,
            holdings=pd.Series({"A": 600.0, "B": 400.0}),
        )
        state.update_equity()
        w = state.weights()
        assert abs(w["A"] - 0.6) < 1e-10
        assert abs(w["B"] - 0.4) < 1e-10

    def test_weights_zero_equity(self):
        state = PortfolioState(
            cash=0.0,
            holdings=pd.Series({"A": 0.0, "B": 0.0}),
        )
        state.update_equity()
        w = state.weights()
        assert (w == 0.0).all()

    def test_equity_includes_cash(self):
        state = PortfolioState(
            cash=100_000.0,
            holdings=pd.Series({"A": 500_000.0, "B": 400_000.0}),
        )
        state.update_equity()
        assert abs(state.equity - 1_000_000.0) < 1e-10


class TestShouldRebalance:

    def test_daily_always_true(self):
        d = pd.Timestamp("2020-06-15")
        assert _should_rebalance(d, pd.Timestamp("2020-06-12"), "daily") is True

    def test_monthly_same_month(self):
        d = pd.Timestamp("2020-06-15")
        prev = pd.Timestamp("2020-06-12")
        assert _should_rebalance(d, prev, "monthly") is False

    def test_monthly_new_month(self):
        d = pd.Timestamp("2020-07-01")
        prev = pd.Timestamp("2020-06-30")
        assert _should_rebalance(d, prev, "monthly") is True

    def test_monthly_first_day(self):
        d = pd.Timestamp("2020-01-02")
        assert _should_rebalance(d, None, "monthly") is True

    def test_unknown_freq_raises(self):
        with pytest.raises(ValueError, match="Unknown rebalance"):
            _should_rebalance(pd.Timestamp("2020-01-01"), None, "weekly")


class TestRunBacktest:

    def _make_prices(self, n_days=100, n_assets=3, drift=0.0005):
        """Create synthetic upward-drifting prices."""
        np.random.seed(42)
        dates = pd.bdate_range("2020-01-01", periods=n_days)
        tickers = [f"ASSET_{i}" for i in range(n_assets)]
        data = {}
        for t in tickers:
            rets = np.random.normal(drift, 0.015, n_days)
            data[t] = 100.0 * np.exp(np.cumsum(rets))
        return pd.DataFrame(data, index=dates)

    def test_equal_weight_basic(self):
        prices = self._make_prices()
        result = run_backtest(
            prices, equal_weight_signal,
            initial_capital=1_000_000,
            rebalance_freq="daily",
        )
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == 100
        assert len(result.daily_returns) == 100
        assert result.equity_curve.iloc[0] == pytest.approx(1_000_000, rel=0.01)

    def test_equity_never_negative(self):
        prices = self._make_prices(n_days=252, drift=-0.001)
        result = run_backtest(
            prices, equal_weight_signal,
            rebalance_freq="daily",
        )
        assert (result.equity_curve > 0).all()

    def test_single_asset_portfolio(self):
        prices = self._make_prices(n_assets=1)
        result = run_backtest(
            prices, single_asset_signal,
            rebalance_freq="daily",
        )
        assert len(result.positions.columns) == 1
        # Should track the single asset closely
        assert result.equity_curve.iloc[-1] > 0

    def test_zero_allocation_stays_at_capital(self):
        """All-cash portfolio should stay flat (minus any tiny float issues)."""
        prices = self._make_prices()
        result = run_backtest(
            prices, zero_weight_signal,
            initial_capital=1_000_000,
            rebalance_freq="daily",
            cost_model=TransactionCostModel(cost_bps=0, slippage_bps=0),
        )
        # With zero allocation and zero costs, equity should stay exactly at initial
        np.testing.assert_allclose(
            result.equity_curve.values,
            1_000_000.0,
            rtol=1e-10,
        )

    def test_monthly_rebalance_less_turnover(self):
        prices = self._make_prices(n_days=252)
        daily = run_backtest(prices, equal_weight_signal, rebalance_freq="daily")
        monthly = run_backtest(prices, equal_weight_signal, rebalance_freq="monthly")
        assert monthly.turnover.sum() < daily.turnover.sum()

    def test_costs_reduce_equity(self):
        prices = self._make_prices()
        no_cost = TransactionCostModel(cost_bps=0, slippage_bps=0)
        high_cost = TransactionCostModel(cost_bps=50, slippage_bps=25)

        result_free = run_backtest(
            prices, equal_weight_signal, rebalance_freq="daily", cost_model=no_cost,
        )
        result_expensive = run_backtest(
            prices, equal_weight_signal, rebalance_freq="daily", cost_model=high_cost,
        )
        assert result_expensive.equity_curve.iloc[-1] < result_free.equity_curve.iloc[-1]

    def test_empty_prices_raises(self):
        with pytest.raises(ValueError, match="empty"):
            run_backtest(pd.DataFrame(), equal_weight_signal)

    def test_result_properties(self):
        prices = self._make_prices()
        result = run_backtest(prices, equal_weight_signal, rebalance_freq="daily")
        # total_return, cagr, max_drawdown should all return floats
        assert isinstance(result.total_return, float)
        assert isinstance(result.cagr, float)
        assert isinstance(result.max_drawdown, float)
        assert result.max_drawdown <= 0.0  # drawdown is always negative or zero

    def test_positions_sum_to_approximately_one(self):
        """On rebalance days, weights should sum to ~1 for fully invested."""
        prices = self._make_prices()
        result = run_backtest(
            prices, equal_weight_signal, rebalance_freq="daily",
            cost_model=TransactionCostModel(cost_bps=0, slippage_bps=0),
        )
        weight_sums = result.positions.sum(axis=1)
        np.testing.assert_allclose(weight_sums.values, 1.0, atol=1e-6)
