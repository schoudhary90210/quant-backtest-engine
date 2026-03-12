"""Tests for src/engine/costs.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.costs import AlmgrenChrissCostModel, TransactionCostModel


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
        assert (
            abs(
                model.trading_cost(buy, 1_000_000) - model.trading_cost(sell, 1_000_000)
            )
            < 1e-10
        )

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


# ---------------------------------------------------------------------------
# Backward compatibility (proportional model ignores **kwargs)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_proportional_ignores_date_kwarg(self):
        """TransactionCostModel.trading_cost() should accept and ignore date=."""
        model = TransactionCostModel(cost_bps=10, slippage_bps=5)
        trades = pd.Series({"A": 0.10})
        cost_without = model.trading_cost(trades, 1_000_000)
        cost_with = model.trading_cost(trades, 1_000_000, date=pd.Timestamp("2020-01-01"))
        assert abs(cost_without - cost_with) < 1e-10

    def test_net_return_ignores_date_kwarg(self):
        model = TransactionCostModel(cost_bps=10, slippage_bps=5)
        trades = pd.Series({"A": 0.10})
        net_without = model.net_return_after_costs(0.01, trades, 1_000_000)
        net_with = model.net_return_after_costs(0.01, trades, 1_000_000, date=pd.Timestamp("2020-01-01"))
        assert abs(net_without - net_with) < 1e-10


# ---------------------------------------------------------------------------
# Almgren-Chriss cost model
# ---------------------------------------------------------------------------


def _make_ac_model(
    n_days: int = 100,
    n_assets: int = 2,
    base_price: float = 100.0,
    base_volume: float = 1e6,
    vol: float = 0.01,
) -> tuple[AlmgrenChrissCostModel, list[str], pd.DatetimeIndex]:
    """Helper to create a simple AC model with synthetic data."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]

    # Random walk prices
    log_rets = np.random.normal(0, vol, size=(n_days, n_assets))
    log_rets[0] = 0.0
    prices_arr = base_price * np.exp(np.cumsum(log_rets, axis=0))
    prices = pd.DataFrame(prices_arr, index=dates, columns=tickers)

    # Constant volume with some noise
    vol_arr = np.full((n_days, n_assets), base_volume) + np.random.normal(0, 1000, (n_days, n_assets))
    vol_arr = np.maximum(vol_arr, 1000)
    volumes = pd.DataFrame(vol_arr, index=dates, columns=tickers)

    model = AlmgrenChrissCostModel(
        prices=prices,
        volumes=volumes,
        half_spread_bps=5.0,
        eta=0.1,
        gamma=0.1,
        slippage_factor=0.5,
    )
    return model, tickers, dates


class TestAlmgrenChrissCostModel:
    def test_zero_trade_zero_cost(self):
        model, tickers, dates = _make_ac_model()
        cost = model.compute_cost(tickers[0], 0.0, dates[50])
        assert cost == 0.0

    def test_cost_positive_for_nonzero_trade(self):
        model, tickers, dates = _make_ac_model()
        cost = model.compute_cost(tickers[0], 100_000.0, dates[50])
        assert cost > 0.0

    def test_buy_sell_symmetric(self):
        """|buy| = |sell| costs (cost only depends on abs trade)."""
        model, tickers, dates = _make_ac_model()
        cost_buy = model.compute_cost(tickers[0], 50_000.0, dates[50])
        cost_sell = model.compute_cost(tickers[0], -50_000.0, dates[50])
        # compute_cost takes trade_dollars which is signed, but internally uses abs
        assert abs(cost_buy - cost_sell) < 1e-10

    def test_cost_breakdown_sums_to_total(self):
        model, tickers, dates = _make_ac_model()
        date = dates[50]
        trade = 100_000.0
        total = model.compute_cost(tickers[0], trade, date)
        breakdown = model.compute_cost_breakdown(tickers[0], trade, date)
        assert abs(total - sum(breakdown.values())) < 1e-10

    def test_nonlinear_scaling(self):
        """2x trade should cost more than 2x the cost of 1x trade (sqrt impact)."""
        model, tickers, dates = _make_ac_model()
        date = dates[50]
        cost_1x = model.compute_cost(tickers[0], 100_000.0, date)
        cost_2x = model.compute_cost(tickers[0], 200_000.0, date)
        assert cost_2x > 2.0 * cost_1x

    def test_cost_increases_with_volatility(self):
        """Higher vol environment should produce higher costs."""
        model_low, tickers, dates = _make_ac_model(vol=0.005)
        model_high, _, _ = _make_ac_model(vol=0.05)
        date = dates[50]
        cost_low = model_low.compute_cost(tickers[0], 100_000.0, date)
        cost_high = model_high.compute_cost(tickers[0], 100_000.0, date)
        assert cost_high > cost_low

    def test_trading_cost_requires_date(self):
        model, tickers, dates = _make_ac_model()
        trades = pd.Series({tickers[0]: 0.10, tickers[1]: -0.05})
        with pytest.raises(ValueError, match="requires date="):
            model.trading_cost(trades, 1_000_000)

    def test_trading_cost_with_date(self):
        model, tickers, dates = _make_ac_model()
        trades = pd.Series({tickers[0]: 0.10, tickers[1]: -0.05})
        cost = model.trading_cost(trades, 1_000_000, date=dates[50])
        assert cost > 0.0
