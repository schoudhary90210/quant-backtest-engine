"""
Tests for Phase 5C: Conditional strategy performance by HMM regime.

Uses synthetic returns and regime series — no HMM fitting required,
keeping tests fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.backtest import BacktestResult
from src.visualization.charts import plot_conditional_performance_heatmap
from src.visualization.report import generate_report


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_regime_series(n_days: int = 600, seed: int = 7) -> pd.Series:
    """Random regime labels for n_days business days."""
    rng = np.random.default_rng(seed)
    labels = ["low_vol", "mid_vol", "high_vol"]
    choices = rng.choice(labels, size=n_days)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    return pd.Series(choices, index=dates, name="regime")


def _make_strategy_returns(n_days: int = 600, drift: float = 0.0003, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, n_days)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    return pd.Series(rets, index=dates, name="returns")


def _build_conditional_perf(
    strategy_returns: dict[str, pd.Series],
    regimes: pd.Series,
    regime_order: list[str],
) -> dict:
    """Replicate the main.py conditional_perf construction logic using raw regimes."""
    long_rows = []
    sharpe_dict = {}
    for strat_label, rets in strategy_returns.items():
        common_idx = regimes.index.intersection(rets.index)
        reg = regimes.loc[common_idx]
        sr = rets.loc[common_idx]
        total_days = len(common_idx)

        strat_sharpe = {}
        for regime in regime_order:
            mask = reg == regime
            regime_rets = sr[mask]
            n_days = int(mask.sum())
            if n_days < 2:
                continue
            daily_mean = float(regime_rets.mean())
            daily_std = float(regime_rets.std())
            ann_ret = daily_mean * 252
            ann_vol = daily_std * np.sqrt(252)
            sharpe = (ann_ret - 0.04) / ann_vol if ann_vol > 1e-10 else 0.0
            max_dd = 0.0  # simplified for testing
            fraction = n_days / total_days if total_days > 0 else 0.0
            strat_sharpe[regime] = sharpe
            long_rows.append({
                "strategy": strat_label,
                "regime": regime,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "n_days": n_days,
                "fraction": fraction,
            })
        sharpe_dict[strat_label] = strat_sharpe

    long_form = pd.DataFrame(long_rows)
    conditional_sharpe = pd.DataFrame(sharpe_dict).T
    conditional_sharpe = conditional_sharpe.reindex(columns=regime_order)
    return {
        "long_form": long_form,
        "conditional_sharpe": conditional_sharpe,
        "regime_order": regime_order,
    }


def _make_mock_backtest_result(returns: pd.Series) -> BacktestResult:
    """Minimal BacktestResult from daily returns."""
    equity = (1 + returns).cumprod() * 100_000
    positions = pd.DataFrame(
        {"A": 0.5, "B": 0.5}, index=returns.index,
    )
    turnover = pd.Series(0.0, index=returns.index)
    total_costs = pd.Series(0.0, index=returns.index)
    return BacktestResult(
        equity_curve=equity,
        daily_returns=returns,
        positions=positions,
        turnover=turnover,
        total_costs=total_costs,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def regime_order():
    return ["low_vol", "mid_vol", "high_vol"]


@pytest.fixture()
def regimes():
    return _make_regime_series(600, seed=7)


@pytest.fixture()
def strategy_returns_map():
    return {
        "Strategy A": _make_strategy_returns(600, drift=0.0005, seed=10),
        "Strategy B": _make_strategy_returns(600, drift=0.0001, seed=20),
        "Strategy C": _make_strategy_returns(600, drift=0.0003, seed=30),
    }


@pytest.fixture()
def conditional_perf(strategy_returns_map, regimes, regime_order):
    return _build_conditional_perf(strategy_returns_map, regimes, regime_order)


# ── TestConditionalMatrixBuild ───────────────────────────────────────────────


class TestConditionalMatrixBuild:

    def test_sharpe_matrix_shape(self, conditional_perf, regime_order):
        cs = conditional_perf["conditional_sharpe"]
        assert cs.shape == (3, 3)  # 3 strategies × 3 regimes

    def test_long_form_columns(self, conditional_perf):
        lf = conditional_perf["long_form"]
        expected_cols = {"strategy", "regime", "sharpe", "max_drawdown", "n_days", "fraction"}
        assert expected_cols == set(lf.columns)

    def test_column_order_matches_regime_order(self, conditional_perf, regime_order):
        cs = conditional_perf["conditional_sharpe"]
        assert list(cs.columns) == regime_order

    def test_all_strategies_present(self, conditional_perf, strategy_returns_map):
        cs = conditional_perf["conditional_sharpe"]
        assert set(cs.index) == set(strategy_returns_map.keys())


# ── TestFractionAlignment ────────────────────────────────────────────────────


class TestFractionAlignment:

    def test_fractions_sum_to_one_per_strategy(self, conditional_perf):
        lf = conditional_perf["long_form"]
        for strat in lf["strategy"].unique():
            strat_rows = lf[lf["strategy"] == strat]
            total = strat_rows["fraction"].sum()
            assert abs(total - 1.0) < 1e-6, f"{strat} fractions sum to {total}"

    def test_n_days_match_regime_counts(self, conditional_perf, regimes, regime_order):
        lf = conditional_perf["long_form"]
        # All strategies share the same dates, so n_days per regime should match
        for regime in regime_order:
            expected = int((regimes == regime).sum())
            regime_rows = lf[lf["regime"] == regime]
            for _, row in regime_rows.iterrows():
                assert int(row["n_days"]) == expected


# ── TestHeatmapChartSmoke ────────────────────────────────────────────────────


class TestHeatmapChartSmoke:

    def test_produces_png(self, conditional_perf, tmp_path, monkeypatch):
        import src.visualization.charts as charts_mod
        monkeypatch.setattr(charts_mod, "CHART_DIR", tmp_path)
        cs = conditional_perf["conditional_sharpe"]
        path = plot_conditional_performance_heatmap(cs)
        assert path.exists()
        assert path.suffix == ".png"

    def test_handles_all_nan(self, tmp_path, monkeypatch):
        import src.visualization.charts as charts_mod
        monkeypatch.setattr(charts_mod, "CHART_DIR", tmp_path)
        df = pd.DataFrame(
            np.nan, index=["A", "B"], columns=["low_vol", "high_vol"],
        )
        path = plot_conditional_performance_heatmap(df)
        assert path.exists()

    def test_handles_all_zero(self, tmp_path, monkeypatch):
        import src.visualization.charts as charts_mod
        monkeypatch.setattr(charts_mod, "CHART_DIR", tmp_path)
        df = pd.DataFrame(
            0.0, index=["A"], columns=["low_vol", "mid_vol", "high_vol"],
        )
        path = plot_conditional_performance_heatmap(df)
        assert path.exists()


# ── TestReportSmoke ──────────────────────────────────────────────────────────


class TestReportSmoke:

    def _make_minimal_results(self):
        rets = _make_strategy_returns(200, seed=1)
        result = _make_mock_backtest_result(rets)
        return {"Test Strategy": result}

    def test_section_present(self, conditional_perf):
        results = self._make_minimal_results()
        primary = list(results.values())[0]
        text = generate_report(
            results, kelly_result=primary, write_file=False,
            conditional_performance=conditional_perf,
        )
        assert "CONDITIONAL PERFORMANCE BY REGIME" in text

    def test_strategies_appear(self, conditional_perf):
        results = self._make_minimal_results()
        primary = list(results.values())[0]
        text = generate_report(
            results, kelly_result=primary, write_file=False,
            conditional_performance=conditional_perf,
        )
        for strat_name in conditional_perf["conditional_sharpe"].index:
            assert strat_name in text

    def test_regime_labels_appear(self, conditional_perf):
        results = self._make_minimal_results()
        primary = list(results.values())[0]
        text = generate_report(
            results, kelly_result=primary, write_file=False,
            conditional_performance=conditional_perf,
        )
        for regime in ["LOW_VOL", "MID_VOL", "HIGH_VOL"]:
            assert regime in text

    def test_none_does_not_crash(self):
        results = self._make_minimal_results()
        primary = list(results.values())[0]
        text = generate_report(
            results, kelly_result=primary, write_file=False,
            conditional_performance=None,
        )
        assert "CONDITIONAL PERFORMANCE BY REGIME" not in text
