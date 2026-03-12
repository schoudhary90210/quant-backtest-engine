"""
Tests for HMM-based regime detection.

HMM-specific tests are guarded by ``requires_hmmlearn``.
Fallback tests always run regardless of whether hmmlearn is installed.
"""

from __future__ import annotations

import builtins

import numpy as np
import pandas as pd
import pytest

import config

# HMMRegimeDetector is safe to import without hmmlearn — it only does
# a lazy import of hmmlearn inside fit().
from src.risk.regime import HMMRegimeDetector

# Chart functions are also safe to import unconditionally.
from src.visualization.charts import plot_regime_overlay, plot_transition_matrix

# ── hmmlearn availability check ─────────────────────────────────────────────

try:
    import hmmlearn  # noqa: F401

    _HAS_HMMLEARN = True
except ImportError:
    _HAS_HMMLEARN = False

requires_hmmlearn = pytest.mark.skipif(
    not _HAS_HMMLEARN, reason="hmmlearn not installed"
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_regime_returns(n_per_regime: int = 500, seed: int = 42) -> pd.Series:
    """Two-block synthetic returns: low-vol then high-vol."""
    rng = np.random.default_rng(seed)
    low_vol = rng.normal(0.0003, 0.005, n_per_regime)
    high_vol = rng.normal(-0.0001, 0.025, n_per_regime)
    rets = np.concatenate([low_vol, high_vol])
    dates = pd.bdate_range("2015-01-01", periods=len(rets))
    return pd.Series(rets, index=dates, name="benchmark")


def _make_strategy_returns(n: int = 1000, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.01, n)
    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(rets, index=dates, name="strategy")


# ── HMM tests (guarded) ─────────────────────────────────────────────────────


@requires_hmmlearn
class TestHMMFit:
    def test_fit_returns_self(self):
        rets = _make_regime_returns()
        det = HMMRegimeDetector(n_states=2, feature_window=20)
        result = det.fit(rets)
        assert result is det

    def test_predict_returns_series(self):
        rets = _make_regime_returns()
        det = HMMRegimeDetector(n_states=2).fit(rets)
        pred = det.predict()
        assert isinstance(pred, pd.Series)
        assert len(pred) > 0

    def test_two_state_labels(self):
        rets = _make_regime_returns()
        det = HMMRegimeDetector(n_states=2).fit(rets)
        labels = set(det.predict().unique())
        assert labels == {"low_vol", "high_vol"}

    def test_three_state_labels(self):
        rets = _make_regime_returns()
        det = HMMRegimeDetector(n_states=3).fit(rets)
        labels = set(det.predict().unique())
        assert labels == {"low_vol", "mid_vol", "high_vol"}

    def test_auto_select_picks_2_or_3(self):
        rets = _make_regime_returns()
        det = HMMRegimeDetector(n_states=None).fit(rets)
        n = det.predict().nunique()
        assert n in (2, 3)


@requires_hmmlearn
class TestStateRecovery:
    def test_low_vol_block_detected(self):
        rets = _make_regime_returns(n_per_regime=500)
        det = HMMRegimeDetector(n_states=2, feature_window=20).fit(rets)
        pred = det.predict()
        # First block (after feature_window warm-up) should be mostly low_vol
        first_block = pred.iloc[: 500 - 20]
        low_frac = (first_block == "low_vol").mean()
        assert low_frac > 0.70, f"First block low_vol fraction = {low_frac:.2f}"

    def test_high_vol_block_detected(self):
        rets = _make_regime_returns(n_per_regime=500)
        det = HMMRegimeDetector(n_states=2, feature_window=20).fit(rets)
        pred = det.predict()
        # Second block should be mostly high_vol
        second_block = pred.iloc[500:]
        high_frac = (second_block == "high_vol").mean()
        assert high_frac > 0.70, f"Second block high_vol fraction = {high_frac:.2f}"


@requires_hmmlearn
class TestPredictLabelConsistency:
    """Verify predict() uses the stored state order from fit(), not sample-local ordering."""

    def test_predict_with_new_data_preserves_state_order(self):
        """predict(new_data) must not mutate the state order learned during fit()."""
        train = _make_regime_returns(n_per_regime=500, seed=42)
        det = HMMRegimeDetector(n_states=2, feature_window=20).fit(train)
        fit_order = det._state_order.copy()

        test = _make_regime_returns(n_per_regime=300, seed=99)
        _ = det.predict(test)

        np.testing.assert_array_equal(det._state_order, fit_order)

    def test_predict_none_vs_predict_same_data_consistent(self):
        """predict() and predict(same_training_data) must give the same labels."""
        rets = _make_regime_returns()
        det = HMMRegimeDetector(n_states=2, feature_window=20).fit(rets)

        labels_none = det.predict()
        labels_explicit = det.predict(rets)

        common = labels_none.index.intersection(labels_explicit.index)
        pd.testing.assert_series_equal(
            labels_none.loc[common], labels_explicit.loc[common]
        )


@requires_hmmlearn
class TestRelabeling:
    def test_state_params_sorted_by_vol_2state(self):
        rets = _make_regime_returns()
        det = HMMRegimeDetector(n_states=2).fit(rets)
        params = det.get_state_parameters()
        assert params["mean_vol"].iloc[0] <= params["mean_vol"].iloc[1]

    def test_state_params_sorted_by_vol_3state(self):
        rets = _make_regime_returns()
        det = HMMRegimeDetector(n_states=3).fit(rets)
        params = det.get_state_parameters()
        vols = params["mean_vol"].values
        assert all(vols[i] <= vols[i + 1] for i in range(len(vols) - 1))


@requires_hmmlearn
class TestTransitionMatrix:
    def test_rows_sum_to_one(self):
        rets = _make_regime_returns()
        det = HMMRegimeDetector(n_states=2).fit(rets)
        trans = det.get_transition_matrix()
        row_sums = trans.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_shape_matches_n_states(self):
        for n in (2, 3):
            rets = _make_regime_returns()
            det = HMMRegimeDetector(n_states=n).fit(rets)
            trans = det.get_transition_matrix()
            assert trans.shape == (n, n)

    def test_all_probs_nonnegative(self):
        rets = _make_regime_returns()
        det = HMMRegimeDetector(n_states=2).fit(rets)
        trans = det.get_transition_matrix()
        assert (trans.values >= 0).all()


@requires_hmmlearn
class TestRegimePerformance:
    def test_returns_dataframe(self):
        bench = _make_regime_returns()
        strat = _make_strategy_returns(n=len(bench))
        strat.index = bench.index  # align dates
        det = HMMRegimeDetector(n_states=2).fit(bench)
        perf = det.regime_performance(strat)
        assert isinstance(perf, pd.DataFrame)

    def test_expected_columns(self):
        bench = _make_regime_returns()
        strat = _make_strategy_returns(n=len(bench))
        strat.index = bench.index
        det = HMMRegimeDetector(n_states=2).fit(bench)
        perf = det.regime_performance(strat)
        expected = {"n_days", "fraction", "mean_return", "volatility", "sharpe", "max_drawdown"}
        assert expected == set(perf.columns)

    def test_fractions_sum_to_one(self):
        bench = _make_regime_returns()
        strat = _make_strategy_returns(n=len(bench))
        strat.index = bench.index
        det = HMMRegimeDetector(n_states=2).fit(bench)
        perf = det.regime_performance(strat)
        np.testing.assert_allclose(perf["fraction"].sum(), 1.0, atol=0.01)

    def test_n_days_positive(self):
        bench = _make_regime_returns()
        strat = _make_strategy_returns(n=len(bench))
        strat.index = bench.index
        det = HMMRegimeDetector(n_states=2).fit(bench)
        perf = det.regime_performance(strat)
        assert (perf["n_days"] > 0).all()

    def test_volatility_reasonable(self):
        bench = _make_regime_returns()
        strat = _make_strategy_returns(n=len(bench))
        strat.index = bench.index
        det = HMMRegimeDetector(n_states=2).fit(bench)
        perf = det.regime_performance(strat)
        assert (perf["volatility"] > 0).all()
        assert (perf["volatility"] < 2.0).all()


@requires_hmmlearn
class TestChartSmoke:
    def test_regime_overlay(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CHART_DIR", str(tmp_path))
        bench = _make_regime_returns()
        det = HMMRegimeDetector(n_states=2).fit(bench)
        equity = (1 + bench).cumprod() * 1_000_000
        path = plot_regime_overlay(equity, det.predict())
        assert path.exists()

    def test_transition_matrix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CHART_DIR", str(tmp_path))
        bench = _make_regime_returns()
        det = HMMRegimeDetector(n_states=2).fit(bench)
        path = plot_transition_matrix(det.get_transition_matrix())
        assert path.exists()


# ── Fallback tests (always run — no hmmlearn guard) ─────────────────────────


class TestReportRegimeOrdering:
    """Verify the primary HMM regime section uses canonical order, not perf.index order."""

    def test_report_renders_canonical_order(self):
        """When performance.index differs from regime_order, the report must use regime_order."""
        from src.engine.backtest import BacktestResult
        from src.visualization.report import generate_report

        n = 500
        dates = pd.bdate_range("2015-01-01", periods=n)
        daily_rets = pd.Series(
            np.random.default_rng(42).normal(0.0003, 0.01, n),
            index=dates,
        )
        equity = (1 + daily_rets).cumprod() * 1_000_000
        positions = pd.DataFrame(
            {"SPY": np.full(n, 0.5), "TLT": np.full(n, 0.5)}, index=dates,
        )
        costs = pd.Series(0.0, index=dates)
        turnover = pd.Series(0.0, index=dates)
        result = BacktestResult(
            equity_curve=equity,
            daily_returns=daily_rets,
            positions=positions,
            total_costs=costs,
            turnover=turnover,
        )

        # Simulate perf DataFrame sorted by volatility (wrong order: mid, low, high)
        perf = pd.DataFrame(
            {
                "n_days": [200, 200, 100],
                "fraction": [0.4, 0.4, 0.2],
                "mean_return": [0.05, 0.10, 0.02],
                "volatility": [0.10, 0.15, 0.25],
                "sharpe": [0.5, 1.0, 0.1],
                "max_drawdown": [-0.05, -0.10, -0.20],
            },
            index=["mid_vol", "low_vol", "high_vol"],
        )

        hmm_data = {
            "performance": perf,
            "n_states": 3,
            "transition_matrix": pd.DataFrame(
                np.eye(3) * 0.8 + 0.1 / 3,
                index=["low_vol", "mid_vol", "high_vol"],
                columns=["low_vol", "mid_vol", "high_vol"],
            ),
            "state_params": pd.DataFrame({"label": ["low_vol", "mid_vol", "high_vol"]}),
            "regime_order": ["low_vol", "mid_vol", "high_vol"],
            "regimes": pd.Series(["low_vol"] * n, index=dates),
        }

        text = generate_report(
            {"test": result},
            kelly_result=result,
            write_file=False,
            hmm_regime_data=hmm_data,
        )

        # Find positions of the three regime headers in the primary section
        low_pos = text.index("LOW_VOL")
        mid_pos = text.index("MID_VOL")
        high_pos = text.index("HIGH_VOL")
        assert low_pos < mid_pos < high_pos, (
            f"Expected LOW_VOL ({low_pos}) < MID_VOL ({mid_pos}) < HIGH_VOL ({high_pos})"
        )


class TestFallbackToLegacy:
    def test_report_falls_back_without_hmm_data(self):
        """When hmm_regime_data=None, report uses legacy regime labels."""
        from src.engine.backtest import BacktestResult
        from src.visualization.report import generate_report

        n = 500
        dates = pd.bdate_range("2015-01-01", periods=n)
        daily_rets = pd.Series(
            np.random.default_rng(42).normal(0.0003, 0.01, n),
            index=dates,
        )
        equity = (1 + daily_rets).cumprod() * 1_000_000
        positions = pd.DataFrame(
            {"SPY": np.full(n, 0.5), "TLT": np.full(n, 0.5)}, index=dates,
        )
        costs = pd.Series(0.0, index=dates, name="costs")
        turnover = pd.Series(0.0, index=dates)

        result = BacktestResult(
            equity_curve=equity,
            daily_returns=daily_rets,
            positions=positions,
            total_costs=costs,
            turnover=turnover,
        )

        text = generate_report(
            {"test": result},
            kelly_result=result,
            write_file=False,
            hmm_regime_data=None,
        )
        assert "Hidden Markov Model" not in text
        # Legacy labels
        assert any(label in text for label in ("BULL", "SIDEWAYS", "BEAR"))

    def test_unfitted_predict_raises(self):
        det = HMMRegimeDetector()
        with pytest.raises(RuntimeError):
            det.predict()

    def test_hmm_fit_import_failure_caught(self, monkeypatch):
        """Simulate hmmlearn being unavailable inside fit().

        Exercises the real fallback path: HMMRegimeDetector.fit() raises
        ImportError, and the try/except pattern (as used in main.py step 4c)
        catches it gracefully.
        """
        _real_import = builtins.__import__

        def _block_hmmlearn(name, *args, **kwargs):
            if "hmmlearn" in name:
                raise ImportError("mocked: hmmlearn not available")
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_hmmlearn)

        rets = _make_regime_returns()
        det = HMMRegimeDetector(n_states=2)

        # Replicate the try/except pattern from main.py step 4c
        hmm_regime_result = None
        try:
            det.fit(rets)
            hmm_regime_result = {"regimes": det.predict()}
        except Exception:
            pass  # fallback — exactly what main.py does

        assert hmm_regime_result is None

    def test_hmm_fit_runtime_failure_caught(self, monkeypatch):
        """When fitting itself raises at runtime, the fallback path catches it."""
        _real_import = builtins.__import__

        def _poison_hmmlearn(name, *args, **kwargs):
            mod = _real_import(name, *args, **kwargs)
            if name == "hmmlearn.hmm":
                # Replace GaussianHMM with a class that blows up on fit()
                class _BrokenHMM:
                    def __init__(self, **kwargs):
                        pass

                    def fit(self, X):
                        raise RuntimeError("mocked: convergence failure")

                mod.GaussianHMM = _BrokenHMM
            return mod

        monkeypatch.setattr(builtins, "__import__", _poison_hmmlearn)

        rets = _make_regime_returns()
        det = HMMRegimeDetector(n_states=2)

        hmm_regime_result = None
        try:
            det.fit(rets)
            hmm_regime_result = {"regimes": det.predict()}
        except Exception:
            pass

        assert hmm_regime_result is None
