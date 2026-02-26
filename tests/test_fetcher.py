"""Tests for src/data/fetcher.py."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import config
from src.data import fetcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_download(prices: pd.Series):
    """Return a function that mimics yf.download for a single ticker."""

    def mock_download(ticker, start=None, end=None, auto_adjust=True, progress=False):
        df = pd.DataFrame({"Close": prices})
        df.index.name = "Date"
        return df

    return mock_download


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFetchSingle:
    """Tests for fetch_single()."""

    def test_returns_series_with_correct_name(self, tmp_path: Path):
        dates = pd.bdate_range("2020-01-01", periods=100)
        prices = pd.Series(np.random.uniform(100, 200, 100), index=dates, name="Close")

        with (
            patch.object(config, "CACHE_DIR", str(tmp_path)),
            patch("src.data.fetcher.yf.download", _make_mock_download(prices)),
        ):
            result = fetcher.fetch_single("TEST", use_cache=False)

        assert isinstance(result, pd.Series)
        assert result.name == "TEST"
        assert len(result) == 100

    def test_caching_roundtrip(self, tmp_path: Path):
        dates = pd.bdate_range("2020-01-01", periods=50)
        prices = pd.Series(np.linspace(100, 150, 50), index=dates, name="Close")

        with (
            patch.object(config, "CACHE_DIR", str(tmp_path)),
            patch("src.data.fetcher.yf.download", _make_mock_download(prices)),
        ):
            # First call: downloads and caches
            result1 = fetcher.fetch_single("CACHED", use_cache=True)
            assert (tmp_path / "CACHED.parquet").exists()

            # Second call: reads from cache (mock won't be called)
            result2 = fetcher.fetch_single("CACHED", use_cache=True)
            pd.testing.assert_series_equal(
                result1, result2, check_names=False, check_freq=False
            )

    def test_raises_on_empty_data(self, tmp_path: Path):
        def mock_empty(*args, **kwargs):
            return pd.DataFrame()

        with (
            patch.object(config, "CACHE_DIR", str(tmp_path)),
            patch("src.data.fetcher.yf.download", mock_empty),
        ):
            with pytest.raises(ValueError, match="No data returned"):
                fetcher.fetch_single("INVALID", use_cache=False)


class TestFetchPrices:
    """Tests for fetch_prices()."""

    def test_returns_aligned_dataframe(self, tmp_path: Path):
        dates = pd.bdate_range("2020-01-01", periods=100)

        def mock_download(ticker, **kwargs):
            prices = pd.Series(
                np.random.uniform(50, 200, 100), index=dates, name="Close"
            )
            df = pd.DataFrame({"Close": prices})
            df.index.name = "Date"
            return df

        with (
            patch.object(config, "CACHE_DIR", str(tmp_path)),
            patch("src.data.fetcher.yf.download", mock_download),
        ):
            result = fetcher.fetch_prices(tickers=["A", "B", "C"], use_cache=False)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["A", "B", "C"]
        assert len(result) == 100

    def test_handles_partial_failures(self, tmp_path: Path):
        dates = pd.bdate_range("2020-01-01", periods=50)
        call_count = 0

        def mock_download(ticker, **kwargs):
            nonlocal call_count
            call_count += 1
            if ticker == "BAD":
                return pd.DataFrame()  # triggers ValueError
            prices = pd.Series(
                np.random.uniform(50, 200, 50), index=dates, name="Close"
            )
            return pd.DataFrame({"Close": prices})

        with (
            patch.object(config, "CACHE_DIR", str(tmp_path)),
            patch("src.data.fetcher.yf.download", mock_download),
        ):
            result = fetcher.fetch_prices(
                tickers=["GOOD", "BAD", "ALSO_GOOD"], use_cache=False
            )

        # BAD should be skipped, but the other two should be present
        assert "GOOD" in result.columns
        assert "ALSO_GOOD" in result.columns
        assert "BAD" not in result.columns

    def test_raises_when_all_fail(self, tmp_path: Path):
        def mock_download(ticker, **kwargs):
            return pd.DataFrame()

        with (
            patch.object(config, "CACHE_DIR", str(tmp_path)),
            patch("src.data.fetcher.yf.download", mock_download),
        ):
            with pytest.raises(RuntimeError, match="No data fetched"):
                fetcher.fetch_prices(tickers=["X", "Y"], use_cache=False)

    def test_ffill_handles_short_gaps(self, tmp_path: Path):
        """Assets with 1-2 day gaps should be forward-filled."""
        dates = pd.bdate_range("2020-01-01", periods=50)

        def mock_download(ticker, **kwargs):
            prices = pd.Series(np.linspace(100, 150, 50), index=dates, name="Close")
            if ticker == "GAPPY":
                # Introduce NaN gaps of 2 days
                prices.iloc[10] = np.nan
                prices.iloc[11] = np.nan
            return pd.DataFrame({"Close": prices})

        with (
            patch.object(config, "CACHE_DIR", str(tmp_path)),
            patch("src.data.fetcher.yf.download", mock_download),
        ):
            result = fetcher.fetch_prices(tickers=["SOLID", "GAPPY"], use_cache=False)

        # After ffill(limit=5), no NaNs should remain
        assert not result.isna().any().any()


class TestClearCache:
    """Tests for clear_cache()."""

    def test_clears_parquet_files(self, tmp_path: Path):
        # Create fake cached files
        for name in ["A.parquet", "B.parquet"]:
            (tmp_path / name).write_text("fake")

        with patch.object(config, "CACHE_DIR", str(tmp_path)):
            count = fetcher.clear_cache()

        assert count == 2
        assert list(tmp_path.glob("*.parquet")) == []

    def test_returns_zero_when_no_cache(self, tmp_path: Path):
        with patch.object(config, "CACHE_DIR", str(tmp_path / "nonexistent")):
            assert fetcher.clear_cache() == 0
