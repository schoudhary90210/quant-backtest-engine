"""Tests for src/risk/regime.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.regime import classify_regimes, regime_stats


def _vol_returns(annualized_vol: float, n: int = 252) -> pd.Series:
    """Construct returns with a known annualized volatility."""
    np.random.seed(0)
    daily_vol = annualized_vol / np.sqrt(252)
    return pd.Series(np.random.normal(0.0, daily_vol, n))


class TestClassifyRegimes:
    def test_low_vol_is_bull(self):
        ret = _vol_returns(0.05, n=500)   # 5% vol → bull
        regimes = classify_regimes(ret, window=60)
        valid = regimes.dropna()
        assert (valid == "bull").all()

    def test_high_vol_is_bear(self):
        ret = _vol_returns(0.40, n=500)   # 40% vol → bear
        regimes = classify_regimes(ret, window=60)
        valid = regimes.dropna()
        assert (valid == "bear").all()

    def test_nans_at_start(self):
        ret = _vol_returns(0.10, n=200)
        regimes = classify_regimes(ret, window=60)
        assert regimes.iloc[:59].isna().all()
        assert regimes.iloc[59:].notna().all()

    def test_regime_values_valid(self):
        ret = _vol_returns(0.20, n=300)
        regimes = classify_regimes(ret, window=60)
        valid_labels = {"bull", "sideways", "bear"}
        assert set(regimes.dropna().unique()).issubset(valid_labels)


class TestRegimeStats:
    def test_returns_dict_with_regime_keys(self):
        ret = pd.Series(np.random.normal(0.0005, 0.01, 500))
        stats = regime_stats(ret)
        assert isinstance(stats, dict)
        for key in stats:
            assert key in ("bull", "sideways", "bear")

    def test_fractions_sum_to_one(self):
        ret = pd.Series(np.random.normal(0.0005, 0.015, 600))
        stats = regime_stats(ret)
        total = sum(s.fraction for s in stats.values())
        assert abs(total - 1.0) < 1e-10

    def test_short_series_handled(self):
        # Too short for any regime stats (< window)
        ret = pd.Series([0.001] * 10)
        stats = regime_stats(ret, window=60)
        assert len(stats) == 0
