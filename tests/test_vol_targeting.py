"""Tests for volatility targeting (Phase 4B)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.vol_targeting import VolatilityTargeter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diag_cov(variances: list[float], tickers: list[str]) -> pd.DataFrame:
    """Diagonal covariance from annualized variances."""
    return pd.DataFrame(np.diag(variances), index=tickers, columns=tickers)


def _make_prices(n_days: int = 300, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    log_rets = rng.normal(0, 0.01, (n_days, n_assets))
    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(n_assets)])


# ---------------------------------------------------------------------------
# get_leverage() — exact math tests
# ---------------------------------------------------------------------------


class TestGetLeverage:

    def test_exact_leverage_single_asset(self):
        """Single asset: port_vol = w * vol. leverage = target / (w * vol)."""
        # Annualized variance = 0.04 → vol = 0.20
        cov = _diag_cov([0.04], ["A"])
        w = pd.Series([1.0], index=["A"])
        vt = VolatilityTargeter(target_vol=0.10, max_leverage=5.0)
        lev = vt.get_leverage(w, cov)
        # port_vol = 1.0 * 0.20 = 0.20, leverage = 0.10 / 0.20 = 0.50
        assert pytest.approx(lev, abs=1e-10) == 0.50

    def test_exact_leverage_two_uncorrelated(self):
        """Two uncorrelated equal-weight assets."""
        # var_A = 0.04 (vol=20%), var_B = 0.01 (vol=10%)
        cov = _diag_cov([0.04, 0.01], ["A", "B"])
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        vt = VolatilityTargeter(target_vol=0.10, max_leverage=5.0)
        lev = vt.get_leverage(w, cov)
        # port_var = 0.25*0.04 + 0.25*0.01 = 0.0125
        # port_vol = sqrt(0.0125) ≈ 0.11180
        # leverage = 0.10 / 0.11180 ≈ 0.8944
        expected = 0.10 / np.sqrt(0.0125)
        assert pytest.approx(lev, abs=1e-10) == expected

    def test_leverage_capped_at_max(self):
        """Very low vol → leverage hits max_leverage cap."""
        cov = _diag_cov([0.0001], ["A"])  # vol = 1%
        w = pd.Series([1.0], index=["A"])
        vt = VolatilityTargeter(target_vol=0.10, max_leverage=2.0)
        lev = vt.get_leverage(w, cov)
        # Uncapped: 0.10 / 0.01 = 10.0, but capped at 2.0
        assert pytest.approx(lev) == 2.0

    def test_near_zero_vol_returns_one(self):
        """Near-zero portfolio vol → leverage = 1.0."""
        cov = _diag_cov([1e-25], ["A"])
        w = pd.Series([1.0], index=["A"])
        vt = VolatilityTargeter(target_vol=0.10, max_leverage=5.0)
        lev = vt.get_leverage(w, cov)
        assert lev == 1.0

    def test_zero_weights_returns_one(self):
        """Zero weights → port_var = 0 → leverage = 1.0."""
        cov = _diag_cov([0.04, 0.01], ["A", "B"])
        w = pd.Series([0.0, 0.0], index=["A", "B"])
        vt = VolatilityTargeter(target_vol=0.10, max_leverage=5.0)
        lev = vt.get_leverage(w, cov)
        assert lev == 1.0


# ---------------------------------------------------------------------------
# apply() — scaling tests
# ---------------------------------------------------------------------------


class TestApply:

    def test_apply_equals_weights_times_leverage(self):
        """apply() should return weights * get_leverage()."""
        cov = _diag_cov([0.04, 0.01, 0.0025], ["A", "B", "C"])
        w = pd.Series([0.4, 0.3, 0.3], index=["A", "B", "C"])
        vt = VolatilityTargeter(target_vol=0.10, max_leverage=5.0)
        lev = vt.get_leverage(w, cov)
        scaled = vt.apply(w, cov)
        expected = w * lev
        pd.testing.assert_series_equal(scaled, expected)

    def test_apply_high_vol_scales_down(self):
        """High-vol portfolio → leverage < 1 → weights shrink."""
        cov = _diag_cov([0.64], ["A"])  # vol = 80%
        w = pd.Series([1.0], index=["A"])
        vt = VolatilityTargeter(target_vol=0.10, max_leverage=5.0)
        scaled = vt.apply(w, cov)
        # leverage = 0.10 / 0.80 = 0.125
        assert pytest.approx(scaled["A"], abs=1e-10) == 0.125

    def test_apply_low_vol_scales_up(self):
        """Low-vol portfolio → leverage > 1 → weights grow."""
        cov = _diag_cov([0.0025], ["A"])  # vol = 5%
        w = pd.Series([1.0], index=["A"])
        vt = VolatilityTargeter(target_vol=0.10, max_leverage=5.0)
        scaled = vt.apply(w, cov)
        # leverage = 0.10 / 0.05 = 2.0
        assert pytest.approx(scaled["A"], abs=1e-10) == 2.0

    def test_apply_cap_enforced(self):
        """Leverage cap limits scaling even when vol is tiny."""
        cov = _diag_cov([0.0001], ["A"])  # vol = 1%
        w = pd.Series([1.0], index=["A"])
        vt = VolatilityTargeter(target_vol=0.10, max_leverage=2.0)
        scaled = vt.apply(w, cov)
        # Uncapped leverage = 10.0, capped at 2.0
        assert pytest.approx(scaled["A"], abs=1e-10) == 2.0

    def test_apply_near_zero_vol_unchanged(self):
        """Near-zero vol → leverage = 1.0 → weights unchanged."""
        cov = _diag_cov([1e-25, 1e-25], ["A", "B"])
        w = pd.Series([0.6, 0.4], index=["A", "B"])
        vt = VolatilityTargeter(target_vol=0.10, max_leverage=5.0)
        scaled = vt.apply(w, cov)
        pd.testing.assert_series_equal(scaled, w)


# ---------------------------------------------------------------------------
# Signal wrapper tests
# ---------------------------------------------------------------------------


class TestWrapVolTarget:

    def test_wrapped_signal_callable(self):
        from src.optimization.signals import wrap_vol_target

        def dummy(date, prices, current_weights):
            return pd.Series(1.0 / len(current_weights), index=current_weights.index)

        wrapped = wrap_vol_target(dummy, target_vol=0.10, max_leverage=2.0, cov_method="sample")
        assert callable(wrapped)

    def test_wrapped_signal_has_targeter(self):
        from src.optimization.signals import wrap_vol_target

        def dummy(date, prices, current_weights):
            return pd.Series(1.0 / len(current_weights), index=current_weights.index)

        wrapped = wrap_vol_target(dummy, target_vol=0.10, max_leverage=2.0, cov_method="sample")
        assert hasattr(wrapped, "targeter")
        assert isinstance(wrapped.targeter, VolatilityTargeter)

    def test_wrapper_populates_leverage_history(self):
        from src.optimization.signals import wrap_vol_target

        prices = _make_prices(n_days=300)

        def dummy(date, prices, current_weights):
            return pd.Series(1.0 / len(current_weights), index=current_weights.index)

        wrapped = wrap_vol_target(
            dummy, target_vol=0.10, max_leverage=5.0, lookback=63, cov_method="sample",
        )
        # Call several times
        cw = pd.Series(1.0 / 3, index=prices.columns)
        for d in prices.index[-5:]:
            wrapped(d, prices.loc[:d], cw)
        assert len(wrapped.leverage_history) == 5
        assert all(lev > 0 for _, lev in wrapped.leverage_history)


class TestVolTargetedBacktest:
    """Smoke test: run vol-targeted signal through run_backtest."""

    def test_vol_targeted_bl_backtest_runs(self):
        from src.engine.backtest import run_backtest
        from src.optimization.signals import make_vol_targeted_bl_signal

        prices = _make_prices(n_days=300, n_assets=3)
        signal = make_vol_targeted_bl_signal(
            lookback=60, cov_method="ledoit_wolf",
            target_vol=0.10, vol_max_leverage=2.0,
        )
        result = run_backtest(prices, signal, rebalance_freq="monthly")
        assert len(result.equity_curve) == len(prices)
        assert result.equity_curve.iloc[-1] > 0
