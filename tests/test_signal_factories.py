"""Smoke tests for HRP and risk-parity signal factories (Phase 4A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.backtest import run_backtest
from src.optimization.signals import make_hrp_signal, make_risk_parity_signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(n_days: int = 300, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    log_rets = rng.normal(0.0003, 0.01, (n_days, n_assets))
    prices = np.exp(np.cumsum(log_rets, axis=0)) * 100
    return pd.DataFrame(prices, index=dates, columns=[f"A{i}" for i in range(n_assets)])


# ---------------------------------------------------------------------------
# HRP signal factory
# ---------------------------------------------------------------------------


class TestHRPSignalFactory:

    def test_creates_callable(self):
        sig = make_hrp_signal(cov_method="sample")
        assert callable(sig)

    def test_returns_weights(self):
        prices = _make_prices()
        sig = make_hrp_signal(lookback=100, cov_method="sample")
        current = pd.Series(1.0 / 3, index=prices.columns)
        w = sig(prices.index[-1], prices, current)
        assert isinstance(w, pd.Series)
        assert len(w) == 3
        assert pytest.approx(w.sum(), abs=0.01) == 1.0
        assert (w >= -1e-10).all()

    def test_warmup_returns_equal_weight(self):
        """Before enough data, should return equal weight."""
        prices = _make_prices(n_days=50)
        sig = make_hrp_signal(lookback=252, cov_method="sample")
        current = pd.Series(1.0 / 3, index=prices.columns)
        w = sig(prices.index[-1], prices, current)
        np.testing.assert_allclose(w.values, 1.0 / 3, atol=1e-10)

    def test_backtest_smoke(self):
        """HRP signal should complete a full backtest without error."""
        prices = _make_prices(n_days=300, n_assets=3)
        sig = make_hrp_signal(lookback=100, cov_method="sample")
        result = run_backtest(prices, sig, rebalance_freq="monthly")
        assert len(result.equity_curve) == 300
        assert result.equity_curve.iloc[-1] > 0
        assert np.isfinite(result.cagr)

    def test_turnover_limiting(self):
        """With max_turnover, trades should be capped."""
        prices = _make_prices(n_days=300, n_assets=3)
        sig = make_hrp_signal(lookback=100, cov_method="sample", max_turnover=0.05)
        result = run_backtest(prices, sig, rebalance_freq="monthly")
        assert len(result.equity_curve) == 300


# ---------------------------------------------------------------------------
# Risk parity signal factory
# ---------------------------------------------------------------------------


class TestRiskParitySignalFactory:

    def test_creates_callable(self):
        sig = make_risk_parity_signal(cov_method="sample")
        assert callable(sig)

    def test_returns_weights(self):
        prices = _make_prices()
        sig = make_risk_parity_signal(lookback=100, cov_method="sample")
        current = pd.Series(1.0 / 3, index=prices.columns)
        w = sig(prices.index[-1], prices, current)
        assert isinstance(w, pd.Series)
        assert len(w) == 3
        assert pytest.approx(w.sum(), abs=0.01) == 1.0
        assert (w >= -1e-10).all()

    def test_warmup_returns_equal_weight(self):
        prices = _make_prices(n_days=50)
        sig = make_risk_parity_signal(lookback=252, cov_method="sample")
        current = pd.Series(1.0 / 3, index=prices.columns)
        w = sig(prices.index[-1], prices, current)
        np.testing.assert_allclose(w.values, 1.0 / 3, atol=1e-10)

    def test_backtest_smoke(self):
        """Risk parity signal should complete a full backtest without error."""
        prices = _make_prices(n_days=300, n_assets=3)
        sig = make_risk_parity_signal(lookback=100, cov_method="sample")
        result = run_backtest(prices, sig, rebalance_freq="monthly")
        assert len(result.equity_curve) == 300
        assert result.equity_curve.iloc[-1] > 0
        assert np.isfinite(result.cagr)

    def test_turnover_limiting(self):
        prices = _make_prices(n_days=300, n_assets=3)
        sig = make_risk_parity_signal(lookback=100, cov_method="sample", max_turnover=0.05)
        result = run_backtest(prices, sig, rebalance_freq="monthly")
        assert len(result.equity_curve) == 300
