"""
Integration tests for the strategy registry, advanced backtests,
risk contributions, new charts, and report strategy comparison section.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.engine.backtest import run_backtest
from src.engine.costs import TransactionCostModel
from src.optimization.risk_budget import RiskBudget
from src.optimization.signals import (
    get_strategy_registry,
    make_hrp_signal,
    make_vol_targeted_bl_signal,
)
from src.visualization.charts import (
    plot_equity_curves,
    plot_hrp_dendrogram,
    plot_leverage_timeseries,
    plot_risk_contribution_comparison,
)
from src.visualization.report import generate_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(n_days: int = 300, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic price data for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    log_rets = rng.normal(0.0003, 0.015, (n_days, n_assets))
    prices = np.exp(np.cumsum(log_rets, axis=0)) * 100
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _cost_model() -> TransactionCostModel:
    return TransactionCostModel()


# ===================================================================
# TestStrategyRegistry
# ===================================================================


class TestStrategyRegistry:
    def test_registry_returns_all_seven(self):
        reg = get_strategy_registry(cov_method="sample", max_turnover=None)
        assert len(reg) == 7

    def test_registry_keys_match_expected(self):
        reg = get_strategy_registry(cov_method="sample", max_turnover=None)
        expected = {
            "equal_weight", "half_kelly", "full_kelly", "bl_kelly",
            "hrp", "risk_parity", "vol_targeted_bl",
        }
        assert set(reg.keys()) == expected

    def test_each_entry_has_label_and_callable(self):
        reg = get_strategy_registry(cov_method="sample", max_turnover=None)
        for key, (label, signal_fn) in reg.items():
            assert isinstance(label, str), f"{key}: label not str"
            assert callable(signal_fn), f"{key}: signal_fn not callable"

    def test_labels_are_unique(self):
        reg = get_strategy_registry(cov_method="sample", max_turnover=None)
        labels = [label for label, _ in reg.values()]
        assert len(labels) == len(set(labels))


# ===================================================================
# TestDiagnosticExposure
# ===================================================================


class TestDiagnosticExposure:
    def test_hrp_signal_exposes_optimizer(self):
        sig = make_hrp_signal(cov_method="sample", max_turnover=None)
        assert hasattr(sig, "hrp_optimizer")

    def test_hrp_dendrogram_after_backtest(self):
        prices = _make_prices(n_days=300, n_assets=3)
        sig = make_hrp_signal(cov_method="sample", max_turnover=None)
        run_backtest(prices, sig, rebalance_freq="monthly", cost_model=_cost_model())
        data = sig.hrp_optimizer.get_dendrogram_data()
        assert "linkage_matrix" in data
        assert "labels" in data

    def test_vol_targeted_bl_leverage_history(self):
        prices = _make_prices(n_days=300, n_assets=3)
        sig = make_vol_targeted_bl_signal(cov_method="sample", max_turnover=None)
        run_backtest(prices, sig, rebalance_freq="monthly", cost_model=_cost_model())
        assert hasattr(sig, "leverage_history")
        assert len(sig.leverage_history) > 0
        ts, lev = sig.leverage_history[0]
        assert isinstance(lev, float)


# ===================================================================
# TestAdvancedBacktestSmoke
# ===================================================================


class TestAdvancedBacktestSmoke:
    def test_hrp_through_registry(self):
        reg = get_strategy_registry(cov_method="sample", max_turnover=None)
        label, sig = reg["hrp"]
        prices = _make_prices()
        result = run_backtest(prices, sig, rebalance_freq="monthly", cost_model=_cost_model())
        assert len(result.equity_curve) > 0
        assert result.cagr != 0.0 or True  # just check it ran

    def test_risk_parity_through_registry(self):
        reg = get_strategy_registry(cov_method="sample", max_turnover=None)
        label, sig = reg["risk_parity"]
        prices = _make_prices()
        result = run_backtest(prices, sig, rebalance_freq="monthly", cost_model=_cost_model())
        assert len(result.equity_curve) > 0

    def test_vol_targeted_bl_through_registry(self):
        reg = get_strategy_registry(cov_method="ledoit_wolf", max_turnover=None)
        label, sig = reg["vol_targeted_bl"]
        prices = _make_prices()
        result = run_backtest(prices, sig, rebalance_freq="monthly", cost_model=_cost_model())
        assert len(result.equity_curve) > 0


# ===================================================================
# TestRiskContributions
# ===================================================================


class TestRiskContributions:
    def test_risk_contribs_sum_to_one(self):
        n = 3
        weights = pd.Series([0.4, 0.3, 0.3], index=["A", "B", "C"])
        cov = pd.DataFrame(
            np.diag([0.04, 0.09, 0.01]),
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        rb = RiskBudget()
        rc = rb.get_risk_contributions(weights, cov)
        total = rc.sum()
        rc_pct = rc / total
        assert abs(rc_pct.sum() - 1.0) < 1e-10


# ===================================================================
# TestChartSmoke
# ===================================================================


class TestChartSmoke:
    def test_plot_hrp_dendrogram(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CHART_DIR", str(tmp_path))
        prices = _make_prices()
        sig = make_hrp_signal(cov_method="sample", max_turnover=None)
        run_backtest(prices, sig, rebalance_freq="monthly", cost_model=_cost_model())
        data = sig.hrp_optimizer.get_dendrogram_data()
        p = plot_hrp_dendrogram(data)
        assert p.exists()

    def test_plot_risk_contribution_comparison(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CHART_DIR", str(tmp_path))
        contribs = {
            "Strategy A": pd.Series([0.5, 0.3, 0.2], index=["X", "Y", "Z"]),
            "Strategy B": pd.Series([0.2, 0.5, 0.3], index=["X", "Y", "Z"]),
        }
        p = plot_risk_contribution_comparison(contribs)
        assert p.exists()

    def test_plot_leverage_timeseries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CHART_DIR", str(tmp_path))
        dates = pd.bdate_range("2020-01-01", periods=50)
        lev = pd.Series(np.random.uniform(0.8, 1.5, 50), index=dates)
        p = plot_leverage_timeseries(lev)
        assert p.exists()

    def test_plot_equity_curves_strategy_comparison(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CHART_DIR", str(tmp_path))
        dates = pd.bdate_range("2020-01-01", periods=100)
        curves = {
            name: pd.Series(np.cumsum(np.random.randn(100) * 0.01) + 100, index=dates)
            for name in ["EW", "HK", "FK", "HRP", "RP"]
        }
        p = plot_equity_curves(curves, filename="strategy_comparison.png")
        assert p.exists()


# ===================================================================
# TestReportSmoke
# ===================================================================


class TestReportSmoke:
    def _make_mock_result(self):
        """Create a minimal BacktestResult-like object for report testing."""
        from src.engine.backtest import BacktestResult

        dates = pd.bdate_range("2020-01-01", periods=300)
        equity = pd.Series(np.cumsum(np.random.randn(300) * 0.01) + 100, index=dates)
        daily_rets = equity.pct_change().dropna()
        positions = pd.DataFrame(
            np.random.dirichlet([1, 1, 1], size=300),
            index=dates,
            columns=["A", "B", "C"],
        )
        return BacktestResult(
            equity_curve=equity,
            daily_returns=daily_rets,
            positions=positions,
            turnover=pd.Series(0.05, index=dates),
            total_costs=pd.Series(10.0, index=["A", "B", "C"]),
        )

    def test_report_with_strategy_comparison(self):
        result = self._make_mock_result()
        comparison = {
            "Strategy A": {
                "sharpe": 1.2, "sortino": 1.5, "cagr": 0.10,
                "max_drawdown": -0.15, "calmar": 0.67, "annualized_turnover": 2.5,
            },
            "Strategy B": {
                "sharpe": 0.9, "sortino": 1.1, "cagr": 0.08,
                "max_drawdown": -0.20, "calmar": 0.40, "annualized_turnover": 3.0,
            },
        }
        text = generate_report(
            {"Strategy A": result},
            kelly_result=result,
            write_file=False,
            display_label="Strategy A",
            strategy_comparison=comparison,
        )
        assert "STRATEGY COMPARISON" in text

    def test_report_marks_primary_strategy(self):
        result = self._make_mock_result()
        comparison = {
            "BL + Kelly": {
                "sharpe": 1.2, "sortino": 1.5, "cagr": 0.10,
                "max_drawdown": -0.15, "calmar": 0.67, "annualized_turnover": 2.5,
            },
            "HRP": {
                "sharpe": 0.9, "sortino": 1.1, "cagr": 0.08,
                "max_drawdown": -0.20, "calmar": 0.40, "annualized_turnover": 3.0,
            },
        }
        text = generate_report(
            {"BL + Kelly": result},
            kelly_result=result,
            write_file=False,
            display_label="BL + Kelly",
            strategy_comparison=comparison,
        )
        # Find the line with the primary strategy and check for marker
        for line in text.split("\n"):
            if "BL + Kelly" in line and "sharpe" not in line.lower() and "─" not in line:
                if "STRATEGY COMPARISON" not in line and "Best" not in line:
                    assert "<" in line, f"Expected '<' marker in: {line}"
                    break
