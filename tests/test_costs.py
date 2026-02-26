"""Tests for src/engine/costs.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.costs import TransactionCostModel


class TestTransactionCostModel:
    """Tests for TransactionCostModel."""

    def test_zero_trade_zero_cost(self):
        model = TransactionCostModel(cost_bps=10, slippage_bps=5)
        trades = pd.Series({"A": 0.0, "B": 0.0})
        assert model.trading_cost(trades, 1_000_000) == 0.0

    def test_cost_scales_with_portfolio_value(self):
        model = TransactionCostModel(cost_bps=10, slippage_bps=0)
        trades = pd.Series({"A": 0.5})  # trade 50% of portfolio

        cost_1m = model.trading_cost(trades, 1_000_000)
        cost_2m = model.trading_cost(trades, 2_000_000)
        assert abs(cost_2m - 2.0 * cost_1m) < 1e-10

    def test_known_cost_calculation(self):
        """10bps cost + 5bps slippage on $100k traded = $15."""
        model = TransactionCostModel(cost_bps=10, slippage_bps=5)
        # Trade weights sum to $100k on a $1M portfolio = 10% turnover
        trades = pd.Series({"A": 0.10})
        cost = model.trading_cost(trades, 1_000_000)
        # total_traded = 0.10 * 1M = $100k
        # commission = 100k * 0.0010 = $100
        # slippage  = 100k * 0.0005 = $50
        assert abs(cost - 150.0) < 1e-10

    def test_symmetric_buy_sell(self):
        """Buying and selling the same amount should cost the same."""
        model = TransactionCostModel(cost_bps=10, slippage_bps=5)
        buy = pd.Series({"A": 0.20})
        sell = pd.Series({"A": -0.20})
        assert abs(model.trading_cost(buy, 1_000_000) -
                    model.trading_cost(sell, 1_000_000)) < 1e-10

    def test_numpy_array_input(self):
        model = TransactionCostModel(cost_bps=10, slippage_bps=5)
        trades = np.array([0.10, -0.10])
        cost = model.trading_cost(trades, 1_000_000)
        # total_traded = (0.10 + 0.10) * 1M = $200k
        # cost = 200k * (0.0010 + 0.0005) = $300
        assert abs(cost - 300.0) < 1e-10


class TestTurnover:
    """Tests for turnover calculation."""

    def test_no_trades(self):
        model = TransactionCostModel()
        assert model.turnover(pd.Series({"A": 0.0, "B": 0.0})) == 0.0

    def test_full_rebalance(self):
        """Going from 100% A to 100% B = 1.0 one-way turnover."""
        model = TransactionCostModel()
        # weight changes: A goes -1.0, B goes +1.0
        trades = pd.Series({"A": -1.0, "B": 1.0})
        assert abs(model.turnover(trades) - 1.0) < 1e-10

    def test_partial_rebalance(self):
        model = TransactionCostModel()
        trades = pd.Series({"A": 0.10, "B": -0.10})
        assert abs(model.turnover(trades) - 0.10) < 1e-10


class TestNetReturnAfterCosts:
    """Tests for net_return_after_costs."""

    def test_no_trade_preserves_return(self):
        model = TransactionCostModel(cost_bps=10, slippage_bps=5)
        trades = pd.Series({"A": 0.0})
        net = model.net_return_after_costs(0.01, trades, 1_000_000)
        assert abs(net - 0.01) < 1e-10

    def test_cost_reduces_return(self):
        model = TransactionCostModel(cost_bps=10, slippage_bps=5)
        trades = pd.Series({"A": 0.50})
        gross_ret = 0.01
        net = model.net_return_after_costs(gross_ret, trades, 1_000_000)
        assert net < gross_ret

    def test_negative_return_possible(self):
        """Large trades on small returns can produce negative net return."""
        model = TransactionCostModel(cost_bps=100, slippage_bps=50)
        trades = pd.Series({"A": 1.0})
        net = model.net_return_after_costs(0.001, trades, 1_000_000)
        assert net < 0.0
