"""Tests for factor attribution — all offline with synthetic data."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest

from src.risk.factor_attribution import FACTOR_COLS, FactorAttribution


# ── Helpers ───────────────────────────────────────────────────────


def _make_synthetic_factors(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """Daily factor returns (already in decimal) with RF."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    factors = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0003, 0.01, n_days),
            "SMB": rng.normal(0.0001, 0.005, n_days),
            "HML": rng.normal(0.0001, 0.005, n_days),
            "RMW": rng.normal(0.00005, 0.003, n_days),
            "CMA": rng.normal(0.00005, 0.003, n_days),
            "Mom": rng.normal(0.0002, 0.008, n_days),
            "RF": np.full(n_days, 0.04 / 252),
        },
        index=dates,
    )
    return factors


def _make_market_returns(
    factors: pd.DataFrame, beta: float = 1.0, seed: int = 99
) -> pd.Series:
    """Strategy returns = beta * Mkt-RF + RF + noise."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, 0.002, len(factors))
    returns = beta * factors["Mkt-RF"] + factors["RF"] + noise
    return pd.Series(returns, index=factors.index, name="strategy")


# ── Pure Market Attribution ───────────────────────────────────────


class TestFactorAttributionPureMarket:
    def test_market_beta_close_to_one(self):
        factors = _make_synthetic_factors()
        returns = _make_market_returns(factors, beta=1.0)
        fa = FactorAttribution(factors_df=factors)
        result = fa.attribute(returns)

        assert abs(result["betas"]["Mkt-RF"] - 1.0) < 0.15
        assert abs(result["alpha_annual"]) < 0.05
        for f in FACTOR_COLS:
            assert np.isfinite(result["betas"][f])


# ── Metrics Validation ────────────────────────────────────────────


class TestFactorAttributionMetrics:
    def setup_method(self):
        self.factors = _make_synthetic_factors()
        rng = np.random.RandomState(77)
        self.returns = pd.Series(
            rng.normal(0.0005, 0.015, len(self.factors)),
            index=self.factors.index,
            name="strategy",
        )
        self.fa = FactorAttribution(factors_df=self.factors)
        self.result = self.fa.attribute(self.returns)

    def test_r_squared_bounds(self):
        assert 0 <= self.result["r_squared"] <= 1

    def test_adj_r_squared_le_r_squared(self):
        assert self.result["adj_r_squared"] <= self.result["r_squared"]

    def test_attributed_returns_finite(self):
        for f in FACTOR_COLS:
            assert np.isfinite(self.result["factor_attributed_returns"][f])

    def test_n_observations(self):
        assert self.result["n_observations"] == len(self.factors)

    def test_alpha_sharpe_finite(self):
        assert np.isfinite(self.result["alpha_sharpe"])

    def test_information_ratio_finite(self):
        assert np.isfinite(self.result["information_ratio"])

    def test_zero_residual_vol_no_inf(self):
        """Perfect fit (returns exactly match factors) should not produce inf/nan."""
        factors = _make_synthetic_factors(n_days=200)
        # Construct returns that are an exact linear combination of factors + RF
        perfect_returns = (
            1.0 * factors["Mkt-RF"]
            + 0.5 * factors["SMB"]
            + factors["RF"]
        )
        fa = FactorAttribution(factors_df=factors)
        result = fa.attribute(perfect_returns)
        assert np.isfinite(result["alpha_sharpe"])
        assert np.isfinite(result["information_ratio"])


# ── Rolling Attribution ───────────────────────────────────────────


class TestRollingAttribution:
    def test_rolling_shape_and_columns(self):
        factors = _make_synthetic_factors(n_days=500)
        returns = _make_market_returns(factors)
        fa = FactorAttribution(factors_df=factors)

        window = 100
        rolling = fa.rolling_attribution(returns, window=window)

        expected_cols = {"alpha_annual", "alpha_se_annual", "r_squared"} | set(
            FACTOR_COLS
        )
        assert set(rolling.columns) == expected_cols
        assert len(rolling) == len(factors) - window + 1
        assert not rolling.isna().any().any()

    def test_rolling_dates_match_window_end(self):
        """Each rolling result should be indexed at the last date in its window."""
        factors = _make_synthetic_factors(n_days=300)
        returns = _make_market_returns(factors)
        fa = FactorAttribution(factors_df=factors)

        window = 100
        rolling = fa.rolling_attribution(returns, window=window)

        common = returns.index.intersection(factors.index)
        # First window uses common[0:100], last data point is common[99]
        assert rolling.index[0] == common[window - 1]
        # Last window uses common[200:300], last data point is common[299]
        assert rolling.index[-1] == common[len(common) - 1]


# ── Cache Fallback ────────────────────────────────────────────────


class TestCacheFallback:
    def test_loads_from_cache_on_download_failure(self, monkeypatch, tmp_path):
        factors = _make_synthetic_factors()
        cache_dir = tmp_path / "factors"
        cache_dir.mkdir()
        factors.to_parquet(cache_dir / "ff5_momentum_daily.parquet")

        monkeypatch.setattr("config.FACTOR_CACHE_DIR", str(cache_dir))
        monkeypatch.setattr(
            "src.risk.factor_attribution.urllib.request.urlopen",
            MagicMock(side_effect=URLError("mocked")),
        )

        fa = FactorAttribution()
        result = fa.fetch_factors("2020-01-01", "2021-12-31")
        assert "Mkt-RF" in result.columns
        assert "Mom" in result.columns


class TestCacheNoFallback:
    def test_raises_when_no_cache(self, monkeypatch, tmp_path):
        cache_dir = tmp_path / "empty_cache"
        cache_dir.mkdir()

        monkeypatch.setattr("config.FACTOR_CACHE_DIR", str(cache_dir))
        monkeypatch.setattr(
            "src.risk.factor_attribution.urllib.request.urlopen",
            MagicMock(side_effect=URLError("mocked")),
        )

        fa = FactorAttribution()
        with pytest.raises(RuntimeError, match="Failed to fetch factor data"):
            fa.fetch_factors("2020-01-01", "2021-12-31")


# ── Chart Smoke Tests ─────────────────────────────────────────────


class TestChartSmoke:
    def test_plot_factor_exposures(self, monkeypatch, tmp_path):
        monkeypatch.setattr("config.CHART_DIR", str(tmp_path))
        from src.visualization.charts import plot_factor_exposures

        attribution = {
            "betas": {f: 0.1 * (i + 1) for i, f in enumerate(FACTOR_COLS)},
            "beta_tstats": {f: 2.5 * (i + 1) for i, f in enumerate(FACTOR_COLS)},
        }
        p = plot_factor_exposures(attribution)
        assert p.exists()

    def test_plot_rolling_alpha(self, monkeypatch, tmp_path):
        monkeypatch.setattr("config.CHART_DIR", str(tmp_path))
        from src.visualization.charts import plot_rolling_alpha

        dates = pd.bdate_range("2021-01-01", periods=200)
        rolling = pd.DataFrame(
            {
                "alpha_annual": np.random.normal(0.03, 0.01, 200),
                "alpha_se_annual": np.full(200, 0.02),
                "r_squared": np.random.uniform(0.3, 0.6, 200),
                **{f: np.random.normal(0, 0.5, 200) for f in FACTOR_COLS},
            },
            index=dates,
        )
        p = plot_rolling_alpha(rolling)
        assert p.exists()


# ── Report Smoke Test ─────────────────────────────────────────────


class TestReportSmoke:
    def test_report_with_factor_attribution(self, monkeypatch, tmp_path):
        monkeypatch.setattr("config.REPORT_PATH", str(tmp_path / "report.txt"))

        from src.engine.backtest import BacktestResult
        from src.visualization.report import generate_report

        dates = pd.bdate_range("2020-01-01", periods=100)
        equity = pd.Series(np.linspace(1e6, 1.1e6, 100), index=dates)
        returns = equity.pct_change().dropna()
        positions = pd.DataFrame(
            {"SPY": np.full(100, 0.5), "TLT": np.full(100, 0.5)}, index=dates
        )

        result = BacktestResult(
            equity_curve=equity,
            positions=positions,
            daily_returns=returns,
            total_costs=pd.Series(0.0, index=dates),
            turnover=pd.Series(0.0, index=dates),
        )

        fa = {
            "alpha_annual": 0.05,
            "alpha_tstat": 2.1,
            "alpha_pvalue": 0.035,
            "betas": {f: 0.1 for f in FACTOR_COLS},
            "beta_tstats": {f: 2.0 for f in FACTOR_COLS},
            "beta_pvalues": {f: 0.04 for f in FACTOR_COLS},
            "r_squared": 0.45,
            "adj_r_squared": 0.43,
            "factor_attributed_returns": {f: 0.02 for f in FACTOR_COLS},
            "residual_alpha": 0.05,
            "alpha_sharpe": 0.8,
            "information_ratio": 0.7,
            "n_observations": 500,
        }

        text = generate_report(
            {"Test": result},
            kelly_result=result,
            write_file=False,
            factor_attribution=fa,
        )
        assert "FACTOR ATTRIBUTION" in text
        assert "Alpha Sharpe" in text
        assert "Information Ratio" in text
