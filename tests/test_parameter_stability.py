"""
Tests for Phase 5D: Parameter stability analysis.

Uses small synthetic price data and a tiny parameter grid
so tests run in seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.backtest import BacktestResult
from src.validation.parameter_stability import parameter_stability_analysis
from src.visualization.charts import plot_parameter_stability
from src.visualization.report import generate_report


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_synthetic_prices(n_days: int = 900, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic price data for n_days business days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"A{i}" for i in range(n_assets)]
    rets = rng.normal(0.0003, 0.01, (n_days, n_assets))
    cum_rets = np.cumprod(1 + rets, axis=0) * 100
    prices = pd.DataFrame(cum_rets, index=dates, columns=tickers)
    return prices


def _trivial_strategy_factory(alpha=0.5, beta=1.0):
    """Return an equal-weight signal regardless of params."""
    def signal_fn(date, prices, current_weights):
        n = len(current_weights)
        return pd.Series(1.0 / n, index=current_weights.index)
    return signal_fn


def _make_mock_backtest_result(prices: pd.DataFrame) -> BacktestResult:
    """Minimal BacktestResult from prices."""
    rets = prices.pct_change().dropna().mean(axis=1)
    equity = (1 + rets).cumprod() * 100_000
    positions = pd.DataFrame(
        1.0 / len(prices.columns), index=rets.index, columns=prices.columns,
    )
    turnover = pd.Series(0.0, index=rets.index)
    total_costs = pd.Series(0.0, index=rets.index)
    return BacktestResult(
        equity_curve=equity,
        daily_returns=rets,
        positions=positions,
        turnover=turnover,
        total_costs=total_costs,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def tiny_prices():
    return _make_synthetic_prices(n_days=900, n_assets=3, seed=42)


@pytest.fixture()
def tiny_grid():
    return {"alpha": [0.3, 0.5, 0.7], "beta": [1.0, 2.0]}


@pytest.fixture()
def stability_result(tiny_prices, tiny_grid):
    return parameter_stability_analysis(
        tiny_prices,
        strategy_factory=_trivial_strategy_factory,
        param_grid=tiny_grid,
        window_years=1,
        step_months=3,
        rebalance_freq="monthly",
        cost_model=None,
    )


# ── TestOutputStructure ──────────────────────────────────────────────────────


class TestOutputStructure:

    def test_has_required_keys(self, stability_result):
        expected = {"parameter_timeseries", "parameter_cv", "sharpe_timeseries", "is_stable"}
        assert expected == set(stability_result.keys())

    def test_timeseries_is_dataframe(self, stability_result):
        assert isinstance(stability_result["parameter_timeseries"], pd.DataFrame)

    def test_sharpe_is_series(self, stability_result):
        assert isinstance(stability_result["sharpe_timeseries"], pd.Series)

    def test_lengths_match(self, stability_result):
        assert len(stability_result["parameter_timeseries"]) == len(stability_result["sharpe_timeseries"])

    def test_at_least_one_window(self, stability_result):
        assert len(stability_result["sharpe_timeseries"]) >= 1


# ── TestCVComputation ────────────────────────────────────────────────────────


class TestCVComputation:

    def test_cv_dict_has_numeric_params(self, stability_result):
        cv = stability_result["parameter_cv"]
        assert "alpha" in cv
        assert "beta" in cv

    def test_cv_values_nonnegative(self, stability_result):
        for v in stability_result["parameter_cv"].values():
            assert v >= 0.0


# ── TestSingleValueGridStability ─────────────────────────────────────────────


class TestSingleValueGridStability:

    def test_single_value_grid_is_stable(self, tiny_prices):
        """A grid with only one value per parameter should yield CV=0 → stable."""
        result = parameter_stability_analysis(
            tiny_prices,
            strategy_factory=_trivial_strategy_factory,
            param_grid={"alpha": [0.5], "beta": [1.0]},
            window_years=1,
            step_months=6,
            rebalance_freq="monthly",
        )
        assert result["is_stable"] is True
        for cv in result["parameter_cv"].values():
            assert cv < 1e-10

    def test_single_value_constant_timeseries(self, tiny_prices):
        result = parameter_stability_analysis(
            tiny_prices,
            strategy_factory=_trivial_strategy_factory,
            param_grid={"alpha": [0.5]},
            window_years=1,
            step_months=6,
            rebalance_freq="monthly",
        )
        ts = result["parameter_timeseries"]
        assert (ts["alpha"] == 0.5).all()


# ── TestNumericParameterColumns ──────────────────────────────────────────────


class TestNumericParameterColumns:

    def test_columns_match_grid(self, stability_result, tiny_grid):
        ts = stability_result["parameter_timeseries"]
        assert list(ts.columns) == list(tiny_grid.keys())


# ── TestChartSmoke ───────────────────────────────────────────────────────────


class TestChartSmoke:

    def test_produces_png(self, stability_result, tmp_path, monkeypatch):
        import src.visualization.charts as charts_mod
        monkeypatch.setattr(charts_mod, "CHART_DIR", tmp_path)
        path = plot_parameter_stability(
            stability_result["parameter_timeseries"],
            stability_result["sharpe_timeseries"],
        )
        assert path.exists()
        assert path.suffix == ".png"

    def test_single_param(self, tiny_prices, tmp_path, monkeypatch):
        import src.visualization.charts as charts_mod
        monkeypatch.setattr(charts_mod, "CHART_DIR", tmp_path)
        result = parameter_stability_analysis(
            tiny_prices,
            strategy_factory=_trivial_strategy_factory,
            param_grid={"alpha": [0.3, 0.5]},
            window_years=1,
            step_months=6,
            rebalance_freq="monthly",
        )
        path = plot_parameter_stability(
            result["parameter_timeseries"],
            result["sharpe_timeseries"],
        )
        assert path.exists()


# ── TestReportSmoke ──────────────────────────────────────────────────────────


class TestReportSmoke:

    def _make_minimal_results(self, prices):
        result = _make_mock_backtest_result(prices)
        return {"Test": result}, result

    def test_section_present(self, stability_result, tiny_prices):
        results, primary = self._make_minimal_results(tiny_prices)
        text = generate_report(
            results, kelly_result=primary, write_file=False,
            parameter_stability=stability_result,
        )
        assert "PARAMETER STABILITY" in text

    def test_cv_values_in_report(self, stability_result, tiny_prices):
        results, primary = self._make_minimal_results(tiny_prices)
        text = generate_report(
            results, kelly_result=primary, write_file=False,
            parameter_stability=stability_result,
        )
        for param_name in stability_result["parameter_cv"]:
            assert param_name in text

    def test_none_does_not_crash(self, tiny_prices):
        results, primary = self._make_minimal_results(tiny_prices)
        text = generate_report(
            results, kelly_result=primary, write_file=False,
            parameter_stability=None,
        )
        assert "PARAMETER STABILITY" not in text

    def test_stable_flag_in_report(self, stability_result, tiny_prices):
        results, primary = self._make_minimal_results(tiny_prices)
        text = generate_report(
            results, kelly_result=primary, write_file=False,
            parameter_stability=stability_result,
        )
        assert "STABLE" in text or "UNSTABLE" in text

    def test_lookback_appears_in_report(self, tiny_prices):
        """When the grid includes lookback, it should appear in the report."""
        def _factory_with_lookback(alpha=0.5, lookback=252):
            def signal_fn(date, prices, current_weights):
                n = len(current_weights)
                return pd.Series(1.0 / n, index=current_weights.index)
            return signal_fn

        result = parameter_stability_analysis(
            tiny_prices,
            strategy_factory=_factory_with_lookback,
            param_grid={"alpha": [0.3, 0.5], "lookback": [126, 252]},
            window_years=1,
            step_months=6,
            rebalance_freq="monthly",
        )
        assert "lookback" in result["parameter_cv"]
        assert "lookback" in result["parameter_timeseries"].columns

        results, primary = self._make_minimal_results(tiny_prices)
        text = generate_report(
            results, kelly_result=primary, write_file=False,
            parameter_stability=result,
        )
        assert "lookback" in text
