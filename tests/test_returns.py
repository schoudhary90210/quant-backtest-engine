"""Tests for src/data/returns.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.returns import (
    annualize_return,
    annualize_volatility,
    compute_log_returns,
    handle_missing,
    log_to_simple,
    simple_to_log,
    align_to_common_dates,
)


class TestComputeLogReturns:
    """Tests for compute_log_returns()."""

    def test_basic_log_returns(self, sample_prices: pd.DataFrame):
        ret = compute_log_returns(sample_prices)
        # Should have one fewer row than prices
        assert len(ret) == len(sample_prices) - 1
        assert list(ret.columns) == list(sample_prices.columns)

    def test_known_values(self):
        """Verify against hand-calculated log returns."""
        dates = pd.bdate_range("2020-01-01", periods=3)
        prices = pd.DataFrame({"A": [100.0, 110.0, 105.0]}, index=dates)
        ret = compute_log_returns(prices)

        expected = np.log(np.array([110.0 / 100.0, 105.0 / 110.0]))
        np.testing.assert_allclose(ret["A"].values, expected)

    def test_single_asset(self):
        dates = pd.bdate_range("2020-01-01", periods=10)
        prices = pd.DataFrame({"SOLO": np.linspace(100, 110, 10)}, index=dates)
        ret = compute_log_returns(prices)
        assert ret.shape == (9, 1)
        assert not ret.isna().any().any()

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_log_returns(pd.DataFrame())

    def test_negative_prices_raise(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        prices = pd.DataFrame({"BAD": [100.0, -50.0, 80.0]}, index=dates)
        with pytest.raises(ValueError, match="positive"):
            compute_log_returns(prices)

    def test_zero_price_raises(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        prices = pd.DataFrame({"ZERO": [100.0, 0.0, 80.0]}, index=dates)
        with pytest.raises(ValueError, match="positive"):
            compute_log_returns(prices)


class TestHandleMissing:
    """Tests for handle_missing()."""

    def test_drop_method(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        ret = pd.DataFrame({"A": [0.01, np.nan, 0.02, 0.03, np.nan]}, index=dates)
        cleaned = handle_missing(ret, method="drop")
        assert len(cleaned) == 3
        assert not cleaned.isna().any().any()

    def test_zero_method(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        ret = pd.DataFrame({"A": [0.01, np.nan, 0.02]}, index=dates)
        cleaned = handle_missing(ret, method="zero")
        assert len(cleaned) == 3
        assert cleaned["A"].iloc[1] == 0.0

    def test_ffill_method(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        ret = pd.DataFrame({"A": [0.01, np.nan, np.nan, 0.02]}, index=dates)
        cleaned = handle_missing(ret, method="ffill")
        assert len(cleaned) == 4
        assert cleaned["A"].iloc[1] == 0.01
        assert cleaned["A"].iloc[2] == 0.01

    def test_unknown_method_raises(self):
        ret = pd.DataFrame({"A": [0.01]})
        with pytest.raises(ValueError, match="Unknown method"):
            handle_missing(ret, method="magic")


class TestAlignToCommonDates:
    """Tests for align_to_common_dates()."""

    def test_drops_unaligned_dates(self):
        dates_a = pd.bdate_range("2020-01-01", periods=10)
        dates_b = pd.bdate_range("2020-01-03", periods=10)

        prices_a = pd.Series(np.linspace(100, 110, 10), index=dates_a, name="A")
        prices_b = pd.Series(np.linspace(200, 210, 10), index=dates_b, name="B")

        combined = pd.concat([prices_a, prices_b], axis=1)
        aligned = align_to_common_dates(combined)

        # After ffill(limit=5) and dropna, should have common dates
        assert not aligned.isna().any().any()
        assert len(aligned) <= len(combined)


class TestConversions:
    """Tests for simple_to_log and log_to_simple."""

    def test_roundtrip(self):
        simple = pd.DataFrame({"A": [0.05, -0.03, 0.02, 0.01]})
        log = simple_to_log(simple)
        recovered = log_to_simple(log)
        pd.testing.assert_frame_equal(recovered, simple, atol=1e-12)

    def test_log_to_simple_known(self):
        log_ret = pd.DataFrame({"A": [0.0]})
        simple = log_to_simple(log_ret)
        assert abs(simple["A"].iloc[0]) < 1e-12  # exp(0) - 1 = 0


class TestAnnualize:
    """Tests for annualize_return and annualize_volatility."""

    def test_annualize_return_scaling(self):
        daily = pd.Series([0.001] * 252)  # 0.1% daily
        ann = annualize_return(daily)
        assert abs(ann - 0.252) < 1e-10  # 0.001 * 252

    def test_annualize_volatility_scaling(self):
        np.random.seed(42)
        daily = pd.Series(np.random.normal(0, 0.01, 10000))
        ann_vol = annualize_volatility(daily)
        # Should be approximately 0.01 * sqrt(252) ≈ 0.1587
        assert abs(ann_vol - 0.01 * np.sqrt(252)) < 0.005

    def test_zero_variance_asset(self):
        """An asset with constant returns has zero volatility."""
        daily = pd.Series([0.001] * 100)
        assert annualize_volatility(daily) < 1e-12
