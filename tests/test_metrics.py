"""Tests for src/risk/metrics.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compute_risk_report,
    conditional_var,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
    win_rate,
)


def _flat_returns(n: int = 252, daily: float = 0.001) -> pd.Series:
    """Returns with constant daily return."""
    return pd.Series([daily] * n)


def _random_returns(n: int = 500, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    return pd.Series(np.random.normal(0.0005, 0.01, n))


class TestAnnualizedReturn:
    def test_positive_drift(self):
        ret = _flat_returns(252, 0.001)
        ann = annualized_return(ret)
        expected = (1.001 ** 252) - 1
        assert abs(ann - expected) < 1e-10

    def test_empty_returns_zero(self):
        assert annualized_return(pd.Series(dtype=float)) == 0.0

    def test_all_negative(self):
        ret = pd.Series([-0.01] * 252)
        assert annualized_return(ret) < 0.0


class TestAnnualizedVolatility:
    def test_constant_returns_zero_vol(self):
        ret = _flat_returns()
        assert annualized_volatility(ret) < 1e-12

    def test_scales_with_daily_vol(self):
        np.random.seed(0)
        daily = pd.Series(np.random.normal(0, 0.01, 10000))
        ann_vol = annualized_volatility(daily)
        assert abs(ann_vol - 0.01 * np.sqrt(252)) < 0.005

    def test_single_obs_returns_zero(self):
        assert annualized_volatility(pd.Series([0.01])) == 0.0


class TestSharpeRatio:
    def test_all_positive_returns(self):
        np.random.seed(1)
        ret = pd.Series(np.abs(np.random.normal(0.005, 0.002, 252)))  # all positive
        sr = sharpe_ratio(ret, risk_free_rate=0.0)
        assert sr > 0.0

    def test_all_negative_returns(self):
        ret = _flat_returns(252, -0.005)
        sr = sharpe_ratio(ret, risk_free_rate=0.04)
        assert sr < 0.0

    def test_zero_vol_returns_zero(self):
        # Constant returns exactly at risk-free → zero excess → zero Sharpe
        daily_rf = 0.04 / 252
        ret = pd.Series([daily_rf] * 252)
        assert sharpe_ratio(ret, risk_free_rate=0.04) == 0.0

    def test_empty_returns_zero(self):
        assert sharpe_ratio(pd.Series(dtype=float)) == 0.0


class TestSortinoRatio:
    def test_no_negative_days_returns_inf(self):
        ret = _flat_returns(252, 0.005)
        sr = sortino_ratio(ret, risk_free_rate=0.0)
        assert sr == np.inf

    def test_all_negative_returns_negative(self):
        ret = _flat_returns(252, -0.005)
        sr = sortino_ratio(ret, risk_free_rate=0.0)
        assert sr < 0.0

    def test_greater_than_sharpe_for_positive_skew(self):
        np.random.seed(42)
        # Right-skewed returns: more big gains, fewer small losses
        ret = pd.Series(np.random.exponential(0.005, 500) - 0.002)
        sharpe = sharpe_ratio(ret, risk_free_rate=0.0)
        sortino = sortino_ratio(ret, risk_free_rate=0.0)
        assert sortino >= sharpe


class TestMaxDrawdown:
    def test_monotonically_increasing_zero_dd(self):
        ret = _flat_returns(100, 0.001)
        assert max_drawdown(ret) == pytest.approx(0.0, abs=1e-10)

    def test_known_drawdown(self):
        # Go up 10%, then down 20% → equity goes 1 → 1.1 → 0.88
        ret = pd.Series([0.10, -0.20])
        dd = max_drawdown(ret)
        expected = (0.88 - 1.1) / 1.1
        assert abs(dd - expected) < 1e-10

    def test_all_negative_large_drawdown(self):
        ret = pd.Series([-0.01] * 100)
        assert max_drawdown(ret) < -0.5

    def test_empty_returns_zero(self):
        assert max_drawdown(pd.Series(dtype=float)) == 0.0


class TestCalmarRatio:
    def test_positive_return_positive_calmar(self):
        ret = _random_returns()
        cal = calmar_ratio(ret)
        # With positive drift, calmar should be positive
        if annualized_return(ret) > 0 and max_drawdown(ret) < 0:
            assert cal > 0.0

    def test_zero_drawdown_returns_zero(self):
        ret = _flat_returns(252, 0.001)
        # monotonically increasing → zero drawdown → calmar = 0
        assert calmar_ratio(ret) == 0.0


class TestVaR:
    def test_var_is_negative_for_mixed_returns(self):
        ret = _random_returns()
        var = value_at_risk(ret, confidence=0.95)
        assert var < 0.0

    def test_var_95_worse_than_var_90(self):
        ret = _random_returns()
        var_95 = value_at_risk(ret, confidence=0.95)
        var_90 = value_at_risk(ret, confidence=0.90)
        assert var_95 <= var_90

    def test_all_positive_returns_positive_var(self):
        ret = _flat_returns(252, 0.01)
        assert value_at_risk(ret, 0.95) > 0.0


class TestCVaR:
    def test_cvar_worse_than_var(self):
        ret = _random_returns()
        var = value_at_risk(ret, 0.95)
        cvar = conditional_var(ret, 0.95)
        assert cvar <= var

    def test_all_same_returns(self):
        ret = _flat_returns(100, -0.005)
        var = value_at_risk(ret, 0.95)
        cvar = conditional_var(ret, 0.95)
        assert abs(cvar - var) < 1e-10


class TestWinRate:
    def test_all_positive(self):
        assert win_rate(_flat_returns(100, 0.001)) == 1.0

    def test_all_negative(self):
        assert win_rate(_flat_returns(100, -0.001)) == 0.0

    def test_half_half(self):
        ret = pd.Series([0.01, -0.01] * 50)
        assert abs(win_rate(ret) - 0.5) < 1e-10


class TestProfitFactor:
    def test_all_positive_returns_inf(self):
        assert profit_factor(_flat_returns(100, 0.001)) == np.inf

    def test_all_negative_returns_zero(self):
        assert profit_factor(_flat_returns(100, -0.001)) == 0.0

    def test_known_ratio(self):
        # 2 days up 0.02, 1 day down 0.01 → pf = 0.04 / 0.01 = 4.0
        ret = pd.Series([0.02, 0.02, -0.01])
        assert abs(profit_factor(ret) - 4.0) < 1e-10


class TestRiskReport:
    def test_all_fields_finite(self):
        ret = _random_returns()
        report = compute_risk_report(ret)
        assert np.isfinite(report.annualized_return)
        assert np.isfinite(report.annualized_volatility)
        assert np.isfinite(report.sharpe_ratio)
        assert np.isfinite(report.max_drawdown)
        assert np.isfinite(report.var_95)
        assert np.isfinite(report.cvar_95)
        assert 0.0 <= report.win_rate <= 1.0
