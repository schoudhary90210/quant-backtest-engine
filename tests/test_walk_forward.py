"""
Tests for walk-forward out-of-sample validation.

Key correctness properties verified:
  1. No future data leakage — weights at date t are identical whether or not
     the dataset contains observations after t.
  2. min_train_days is strictly enforced.
  3. Weight history dates are first trading days from the price index.
  4. Both IS and OOS RiskReports are computed correctly.
  5. WalkForwardResult contains all required fields.
  6. compare_is_oos() returns a string and correctly flags overfitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.metrics import RiskReport
from src.engine.backtest import BacktestResult
from src.validation.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    _monthly_rebalance_dates,
    run_walk_forward,
)
from src.validation.oos_metrics import compare_is_oos, OVERFITTING_THRESHOLD

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wf_prices() -> pd.DataFrame:
    """
    Synthetic price data for 4 assets over ~4 years (1008 business days).

    Used for all walk-forward tests.  min_train_days is set to 252 (1 year)
    in these tests so that the first valid OOS window appears around day 252,
    leaving ~756 days of genuine OOS history.
    """
    np.random.seed(7)
    dates = pd.bdate_range("2019-01-01", periods=1008)
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    data: dict[str, np.ndarray] = {}
    for ticker in tickers:
        daily_rets = np.random.normal(0.05 / 252, 0.18 / np.sqrt(252), 1008)
        data[ticker] = 100.0 * np.exp(np.cumsum(daily_rets))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def short_wf_config() -> WalkForwardConfig:
    """WalkForwardConfig with a small min_train_days suitable for unit tests."""
    return WalkForwardConfig(min_train_days=252)


# ---------------------------------------------------------------------------
# WalkForwardConfig defaults
# ---------------------------------------------------------------------------


def test_wf_config_defaults() -> None:
    cfg = WalkForwardConfig()
    assert cfg.min_train_days == 756
    assert cfg.rebalance_freq == "M"


# ---------------------------------------------------------------------------
# _monthly_rebalance_dates helper
# ---------------------------------------------------------------------------


def test_monthly_rebalance_dates_first_trading_days() -> None:
    """Each returned date is the first business day of its calendar month."""
    dates = pd.bdate_range("2022-01-01", periods=252)
    rebalance = _monthly_rebalance_dates(dates)

    assert isinstance(rebalance, pd.DatetimeIndex)
    assert len(rebalance) > 0

    for d in rebalance:
        # Must be in the original price index
        assert d in dates
        # Must be the first trading day of its month (no earlier date in same month)
        same_month = dates[(dates.year == d.year) & (dates.month == d.month)]
        assert d == same_month[0], f"{d} is not the first trading day of its month"


def test_monthly_rebalance_dates_count() -> None:
    """Number of rebalance dates equals the number of distinct calendar months."""
    dates = pd.bdate_range("2022-01-01", "2023-12-31")
    rebalance = _monthly_rebalance_dates(dates)
    n_months = len({(d.year, d.month) for d in dates})
    assert len(rebalance) == n_months


# ---------------------------------------------------------------------------
# No lookahead — core correctness test
# ---------------------------------------------------------------------------


def test_no_lookahead() -> None:
    """
    Weights at each rebalance date must be identical whether or not future
    price data exists in the dataset.

    Strategy: create prices_short (800 days).  Then append 200 days where
    one asset has extreme returns (2.0 = 200% per day) — this would produce
    dramatically different Kelly weights if the expanding window accidentally
    sees future data.  Weights at dates common to both runs must be equal.
    """
    all_dates = pd.bdate_range("2019-01-01", periods=1000)
    dates_short = all_dates[:800]
    dates_ext = all_dates[800:]

    np.random.seed(42)
    tickers = ["AAA", "BBB", "CCC"]

    # Normal prices for 800 days
    data_short: dict[str, np.ndarray] = {}
    for t in tickers:
        rets = np.random.normal(0.0002, 0.01, 800)
        data_short[t] = 100.0 * np.exp(np.cumsum(rets))
    prices_short = pd.DataFrame(data_short, index=dates_short)

    # Extended dataset: normal for first two assets, 200%/day for CCC
    data_ext: dict[str, np.ndarray] = {}
    for t in ["AAA", "BBB"]:
        rets = np.random.normal(0.0002, 0.01, 200)
        data_ext[t] = prices_short[t].iloc[-1] * np.exp(np.cumsum(rets))
    # CCC skyrockets — if this leaks into pre-extension weights it's obvious
    ccc_rets = np.full(200, 2.0)
    data_ext["CCC"] = prices_short["CCC"].iloc[-1] * np.exp(np.cumsum(ccc_rets))
    prices_ext = pd.DataFrame(data_ext, index=dates_ext)

    prices_full = pd.concat([prices_short, prices_ext])

    cfg = WalkForwardConfig(min_train_days=200)

    result_short = run_walk_forward(prices_short, cfg)
    result_full = run_walk_forward(prices_full, cfg)

    # Find dates present in both weight histories
    common_dates = result_short.weight_history.index.intersection(
        result_full.weight_history.index
    )
    assert len(common_dates) > 0, "No common rebalance dates to compare"

    for date in common_dates:
        w_short = result_short.weight_history.loc[date]
        w_full = result_full.weight_history.loc[date]
        try:
            pd.testing.assert_series_equal(
                w_short, w_full, check_names=False, rtol=1e-12
            )
        except AssertionError as exc:
            raise AssertionError(f"Lookahead detected at {date.date()}") from exc


# ---------------------------------------------------------------------------
# min_train_days enforcement
# ---------------------------------------------------------------------------


def test_min_train_days_respected(
    wf_prices: pd.DataFrame,
    short_wf_config: WalkForwardConfig,
) -> None:
    """Every weight history entry must have >= min_train_days of training data."""
    result = run_walk_forward(wf_prices, short_wf_config)

    log_returns = np.log(wf_prices / wf_prices.shift(1)).dropna()
    for date in result.weight_history.index:
        train_slice = log_returns.loc[:date]
        assert len(train_slice) >= short_wf_config.min_train_days, (
            f"Date {date.date()} has only {len(train_slice)} training days "
            f"(need {short_wf_config.min_train_days})"
        )


def test_insufficient_data_raises() -> None:
    """run_walk_forward raises ValueError when no window meets min_train_days."""
    dates = pd.bdate_range("2020-01-01", periods=100)
    prices = pd.DataFrame(
        {"A": 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))},
        index=dates,
    )
    with pytest.raises(ValueError, match="min_train_days"):
        run_walk_forward(prices, WalkForwardConfig(min_train_days=500))


# ---------------------------------------------------------------------------
# Weight history alignment
# ---------------------------------------------------------------------------


def test_weight_history_dates_in_price_index(
    wf_prices: pd.DataFrame,
    short_wf_config: WalkForwardConfig,
) -> None:
    """All weight history dates must exist in the price DataFrame index."""
    result = run_walk_forward(wf_prices, short_wf_config)
    for date in result.weight_history.index:
        assert (
            date in wf_prices.index
        ), f"Rebalance date {date.date()} not in price index"


def test_weight_history_columns_match_tickers(
    wf_prices: pd.DataFrame,
    short_wf_config: WalkForwardConfig,
) -> None:
    """Weight history columns must equal the price DataFrame columns."""
    result = run_walk_forward(wf_prices, short_wf_config)
    assert list(result.weight_history.columns) == list(wf_prices.columns)


def test_weight_history_non_negative(
    wf_prices: pd.DataFrame,
    short_wf_config: WalkForwardConfig,
) -> None:
    """Kelly weights are long-only — all entries must be >= 0."""
    result = run_walk_forward(wf_prices, short_wf_config)
    assert (result.weight_history >= 0).all().all()


# ---------------------------------------------------------------------------
# IS and OOS metrics
# ---------------------------------------------------------------------------


def test_is_oos_metrics_are_risk_reports(
    wf_prices: pd.DataFrame,
    short_wf_config: WalkForwardConfig,
) -> None:
    result = run_walk_forward(wf_prices, short_wf_config)
    assert isinstance(result.in_sample_metrics, RiskReport)
    assert isinstance(result.oos_metrics, RiskReport)


def test_is_oos_sharpe_finite(
    wf_prices: pd.DataFrame,
    short_wf_config: WalkForwardConfig,
) -> None:
    result = run_walk_forward(wf_prices, short_wf_config)
    assert np.isfinite(result.in_sample_metrics.sharpe_ratio)
    assert np.isfinite(result.oos_metrics.sharpe_ratio)


# ---------------------------------------------------------------------------
# WalkForwardResult fields
# ---------------------------------------------------------------------------


def test_wf_result_fields(
    wf_prices: pd.DataFrame,
    short_wf_config: WalkForwardConfig,
) -> None:
    result = run_walk_forward(wf_prices, short_wf_config)
    assert isinstance(result.in_sample_result, BacktestResult)
    assert isinstance(result.oos_result, BacktestResult)
    assert isinstance(result.weight_history, pd.DataFrame)
    assert isinstance(result.oos_start_date, pd.Timestamp)
    assert len(result.weight_history) > 0


def test_oos_start_date_in_weight_history(
    wf_prices: pd.DataFrame,
    short_wf_config: WalkForwardConfig,
) -> None:
    """oos_start_date must be the first entry in weight_history."""
    result = run_walk_forward(wf_prices, short_wf_config)
    assert result.oos_start_date == result.weight_history.index[0]


def test_in_sample_result_reused(
    wf_prices: pd.DataFrame,
    short_wf_config: WalkForwardConfig,
) -> None:
    """If an IS result is passed in, it must be returned unchanged."""
    from src.optimization.signals import make_kelly_signal
    from src.engine.backtest import run_backtest

    is_signal = make_kelly_signal(fraction=0.5)
    is_result = run_backtest(wf_prices, is_signal, rebalance_freq="monthly")

    wf = run_walk_forward(wf_prices, short_wf_config, in_sample_result=is_result)
    assert wf.in_sample_result is is_result


# ---------------------------------------------------------------------------
# compare_is_oos
# ---------------------------------------------------------------------------


def _make_risk_report(sharpe: float) -> RiskReport:
    """Helper to build a minimal RiskReport with a given Sharpe ratio."""
    return RiskReport(
        annualized_return=0.10,
        annualized_volatility=0.15,
        sharpe_ratio=sharpe,
        sortino_ratio=sharpe * 1.1,
        max_drawdown=-0.20,
        calmar_ratio=0.50,
        var_95=-0.02,
        cvar_95=-0.03,
        win_rate=0.55,
        profit_factor=1.5,
    )


def test_compare_is_oos_returns_string() -> None:
    is_m = _make_risk_report(sharpe=1.0)
    oos_m = _make_risk_report(sharpe=0.8)
    result = compare_is_oos(is_m, oos_m)
    assert isinstance(result, str)
    assert len(result) > 0


def test_compare_is_oos_no_overfitting_flag() -> None:
    """OOS Sharpe >= threshold → '[OK]' in output."""
    is_m = _make_risk_report(sharpe=1.0)
    oos_m = _make_risk_report(sharpe=0.6)  # 60% of IS — above 50% threshold
    table = compare_is_oos(is_m, oos_m)
    assert "[OK]" in table
    assert "[WARN]" not in table


def test_compare_is_oos_overfitting_warning() -> None:
    """OOS Sharpe < threshold → '[WARN]' in output."""
    is_m = _make_risk_report(sharpe=1.0)
    oos_m = _make_risk_report(sharpe=0.3)  # 30% of IS — below 50% threshold
    table = compare_is_oos(is_m, oos_m)
    assert "[WARN]" in table
    assert "[OK]" not in table


def test_overfitting_threshold_boundary() -> None:
    """Exactly at the threshold (50%) should NOT trigger the warning."""
    is_m = _make_risk_report(sharpe=1.0)
    oos_m = _make_risk_report(sharpe=OVERFITTING_THRESHOLD)
    table = compare_is_oos(is_m, oos_m)
    # At exactly 50%, ratio < threshold is False → no warning
    assert "[OK]" in table
