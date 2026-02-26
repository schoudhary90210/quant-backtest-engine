"""
Log-return calculation module.

Computes ln(P_t / P_{t-1}) from price data, handles missing values,
aligns all assets to common trading dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns = tickers, index = DatetimeIndex. Must contain positive values.

    Returns
    -------
    pd.DataFrame
        Log returns ln(P_t / P_{t-1}). First row is dropped (NaN from diff).
    """
    if prices.empty:
        raise ValueError("Price DataFrame is empty")

    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive for log returns")

    log_ret = np.log(prices / prices.shift(1))
    return log_ret.iloc[1:]  # drop first NaN row


def align_to_common_dates(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Align a multi-asset price DataFrame to common trading dates.

    Drops any date where *any* asset has a NaN after forward-filling
    up to 5 business days.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw price data, possibly with misaligned dates.

    Returns
    -------
    pd.DataFrame
        Price data on the intersection of available dates.
    """
    aligned = prices.ffill(limit=5).dropna()
    return aligned


def handle_missing(returns: pd.DataFrame, method: str = "drop") -> pd.DataFrame:
    """
    Handle missing values in a returns DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns, possibly with NaN values.
    method : str
        'drop' — drop rows with any NaN.
        'zero' — fill NaN with 0.0 (assume no return).
        'ffill' — forward-fill from previous return.

    Returns
    -------
    pd.DataFrame
        Cleaned returns.
    """
    if method == "drop":
        return returns.dropna()
    elif method == "zero":
        return returns.fillna(0.0)
    elif method == "ffill":
        return returns.ffill().dropna()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'drop', 'zero', or 'ffill'.")


def simple_to_log(simple_returns: pd.DataFrame) -> pd.DataFrame:
    """Convert simple returns to log returns: ln(1 + r)."""
    return np.log(1 + simple_returns)


def log_to_simple(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Convert log returns to simple returns: exp(r) - 1."""
    return np.exp(log_returns) - 1


def annualize_return(daily_log_returns: pd.Series, trading_days: int = 252) -> float:
    """Annualize mean daily log return."""
    return float(daily_log_returns.mean() * trading_days)


def annualize_volatility(
    daily_log_returns: pd.Series, trading_days: int = 252
) -> float:
    """Annualize daily log return volatility."""
    return float(daily_log_returns.std() * np.sqrt(trading_days))
