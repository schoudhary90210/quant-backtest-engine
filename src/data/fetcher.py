"""
Yahoo Finance data ingestion.

Downloads adjusted close prices for the configured asset universe,
caches results as parquet files to avoid redundant API calls.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)


def _cache_path(ticker: str) -> Path:
    """Return the parquet cache file path for a given ticker."""
    cache_dir = Path(config.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize ticker for filesystem (e.g. BTC-USD → BTC-USD.parquet)
    return cache_dir / f"{ticker}.parquet"


def fetch_single(
    ticker: str,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    use_cache: bool = True,
) -> pd.Series:
    """
    Download adjusted close prices for a single ticker.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.
    use_cache : bool
        If True, read from / write to parquet cache.

    Returns
    -------
    pd.Series
        Adjusted close prices with DatetimeIndex, named by ticker.
    """
    cache_file = _cache_path(ticker)

    if use_cache and cache_file.exists():
        logger.info("Cache hit for %s", ticker)
        series = pd.read_parquet(cache_file).squeeze()
        series.name = ticker
        series.index = pd.DatetimeIndex(series.index)
        series.index.name = "Date"
        return series

    logger.info("Downloading %s from Yahoo Finance [%s → %s]", ticker, start, end)
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    # yfinance returns MultiIndex columns with (Price, Ticker) when single ticker
    # Flatten to just get the Close price
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = data["Close"]

    close = close.dropna()
    close.name = ticker
    close.index = pd.DatetimeIndex(close.index)
    close.index.name = "Date"

    if use_cache:
        close.to_frame().to_parquet(cache_file)
        logger.info("Cached %s → %s (%d rows)", ticker, cache_file, len(close))

    return close


def fetch_prices(
    tickers: list[str] | None = None,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted close prices for multiple tickers.

    Parameters
    ----------
    tickers : list[str] or None
        Ticker symbols. Defaults to config.ASSETS.
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.
    use_cache : bool
        If True, read from / write to parquet cache.

    Returns
    -------
    pd.DataFrame
        Columns = tickers, index = DatetimeIndex of trading dates.
        Forward-fills gaps up to 5 days, then drops remaining NaNs.
    """
    if tickers is None:
        tickers = config.ASSETS

    series_list: list[pd.Series] = []
    failed: list[str] = []

    for ticker in tickers:
        try:
            s = fetch_single(ticker, start=start, end=end, use_cache=use_cache)
            series_list.append(s)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", ticker, exc)
            failed.append(ticker)

    if not series_list:
        raise RuntimeError("No data fetched for any ticker")

    if failed:
        logger.warning("Failed tickers: %s", failed)

    prices = pd.concat(series_list, axis=1)
    prices.index = pd.DatetimeIndex(prices.index)
    prices.index.name = "Date"

    # Forward-fill short gaps (holidays, missing days), then drop remaining NaNs
    prices = prices.ffill(limit=5).dropna()

    logger.info(
        "Price matrix: %d dates × %d assets [%s → %s]",
        len(prices),
        len(prices.columns),
        prices.index[0].strftime("%Y-%m-%d"),
        prices.index[-1].strftime("%Y-%m-%d"),
    )

    return prices


def _volume_cache_path(ticker: str) -> Path:
    """Return the parquet cache file path for a ticker's volume data."""
    cache_dir = Path(config.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{ticker}_volume.parquet"


def fetch_single_volume(
    ticker: str,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    use_cache: bool = True,
) -> pd.Series:
    """
    Download daily volume for a single ticker.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start, end : str
        Date range in YYYY-MM-DD format.
    use_cache : bool
        If True, read from / write to parquet cache.

    Returns
    -------
    pd.Series
        Daily volume with DatetimeIndex, named by ticker.
    """
    cache_file = _volume_cache_path(ticker)

    if use_cache and cache_file.exists():
        logger.info("Volume cache hit for %s", ticker)
        series = pd.read_parquet(cache_file).squeeze()
        series.name = ticker
        series.index = pd.DatetimeIndex(series.index)
        series.index.name = "Date"
        return series

    logger.info("Downloading volume for %s [%s → %s]", ticker, start, end)
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No volume data returned for ticker '{ticker}'")

    if isinstance(data.columns, pd.MultiIndex):
        volume = data["Volume"]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
    else:
        volume = data["Volume"]

    volume = volume.dropna()
    volume.name = ticker
    volume.index = pd.DatetimeIndex(volume.index)
    volume.index.name = "Date"

    if use_cache:
        volume.to_frame().to_parquet(cache_file)
        logger.info("Cached volume %s → %s (%d rows)", ticker, cache_file, len(volume))

    return volume


def fetch_volumes(
    tickers: list[str] | None = None,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download daily volumes for multiple tickers.

    Parameters
    ----------
    tickers : list[str] or None
        Ticker symbols. Defaults to config.ASSETS.
    start, end : str
        Date range.
    use_cache : bool
        If True, read from / write to parquet cache.

    Returns
    -------
    pd.DataFrame
        Columns = tickers, index = DatetimeIndex.
        Forward-fills gaps up to 5 days, then drops remaining NaNs.
    """
    if tickers is None:
        tickers = config.ASSETS

    series_list: list[pd.Series] = []
    failed: list[str] = []

    for ticker in tickers:
        try:
            s = fetch_single_volume(ticker, start=start, end=end, use_cache=use_cache)
            series_list.append(s)
        except Exception as exc:
            logger.warning("Failed to fetch volume for %s: %s", ticker, exc)
            failed.append(ticker)

    if not series_list:
        raise RuntimeError("No volume data fetched for any ticker")

    if failed:
        logger.warning("Failed volume tickers: %s", failed)

    volumes = pd.concat(series_list, axis=1)
    volumes.index = pd.DatetimeIndex(volumes.index)
    volumes.index.name = "Date"
    volumes = volumes.ffill(limit=5).dropna()

    logger.info(
        "Volume matrix: %d dates × %d assets [%s → %s]",
        len(volumes),
        len(volumes.columns),
        volumes.index[0].strftime("%Y-%m-%d"),
        volumes.index[-1].strftime("%Y-%m-%d"),
    )

    return volumes


def clear_cache() -> int:
    """Remove all cached parquet files. Returns count of files deleted."""
    cache_dir = Path(config.CACHE_DIR)
    if not cache_dir.exists():
        return 0
    files = list(cache_dir.glob("*.parquet"))
    for f in files:
        f.unlink()
    logger.info("Cleared %d cached files", len(files))
    return len(files)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    prices = fetch_prices()
    logger.info("Price matrix shape: %s", prices.shape)
    logger.info("Date range: %s → %s", prices.index[0], prices.index[-1])
    logger.info("Assets: %s", prices.columns.tolist())
    logger.info("Sample (last 5 rows):\n%s", prices.tail())
