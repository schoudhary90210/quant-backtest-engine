"""
Event-driven backtester core loop.

Daily rebalance cycle: generate signals → calculate target weights →
apply transaction costs and slippage → update portfolio state → log metrics.
Tracks equity curve, daily returns, positions, and turnover.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd

import config
from src.engine.costs import TransactionCostModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal protocol — any callable or object providing target weights
# ---------------------------------------------------------------------------

class SignalProvider(Protocol):
    """Protocol for strategy signal generators."""

    def __call__(
        self,
        date: pd.Timestamp,
        prices: pd.DataFrame,
        current_weights: pd.Series,
    ) -> pd.Series:
        """Return target portfolio weights for the given date."""
        ...


# ---------------------------------------------------------------------------
# Portfolio state snapshot
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    """Mutable state of the portfolio at a point in time."""

    cash: float
    holdings: pd.Series  # dollar value per asset
    equity: float = 0.0

    def weights(self) -> pd.Series:
        """Current portfolio weights (0 if equity is zero)."""
        if self.equity <= 0:
            return self.holdings * 0.0
        return self.holdings / self.equity

    def update_equity(self) -> None:
        """Recalculate total equity = cash + sum(holdings)."""
        self.equity = self.cash + self.holdings.sum()


# ---------------------------------------------------------------------------
# Backtest result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Container for all backtest outputs."""

    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    positions: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    total_costs: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def total_return(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        return self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1.0

    @property
    def cagr(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        if years <= 0:
            return 0.0
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) ** (1 / years) - 1

    @property
    def max_drawdown(self) -> float:
        peak = self.equity_curve.cummax()
        drawdown = (self.equity_curve - peak) / peak
        return float(drawdown.min())


# ---------------------------------------------------------------------------
# Rebalance schedule helpers
# ---------------------------------------------------------------------------

def _should_rebalance(
    date: pd.Timestamp,
    prev_date: pd.Timestamp | None,
    freq: str,
) -> bool:
    """Determine whether to rebalance on this date."""
    if freq == "daily":
        return True
    if freq == "monthly":
        if prev_date is None:
            return True
        return date.month != prev_date.month
    raise ValueError(f"Unknown rebalance frequency: '{freq}'")


# ---------------------------------------------------------------------------
# Core backtester
# ---------------------------------------------------------------------------

def run_backtest(
    prices: pd.DataFrame,
    signal_fn: Callable[..., pd.Series] | SignalProvider,
    initial_capital: float = config.INITIAL_CAPITAL,
    rebalance_freq: str = config.REBALANCE_FREQ,
    cost_model: TransactionCostModel | None = None,
) -> BacktestResult:
    """
    Run an event-driven backtest.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices. Columns = tickers, index = DatetimeIndex.
    signal_fn : callable
        ``signal_fn(date, prices_up_to_date, current_weights) → target_weights``
        Must return a pd.Series indexed by ticker with target weights summing to ≤ 1.
    initial_capital : float
        Starting portfolio value.
    rebalance_freq : str
        'daily' or 'monthly'.
    cost_model : TransactionCostModel or None
        If None, uses default from config.

    Returns
    -------
    BacktestResult
    """
    if prices.empty:
        raise ValueError("Price DataFrame is empty")

    if cost_model is None:
        cost_model = TransactionCostModel()

    tickers = prices.columns.tolist()
    dates = prices.index

    # --- Initialize state ---
    state = PortfolioState(
        cash=initial_capital,
        holdings=pd.Series(0.0, index=tickers),
    )
    state.update_equity()

    # --- Result accumulators ---
    equity_log: dict[pd.Timestamp, float] = {}
    return_log: dict[pd.Timestamp, float] = {}
    position_log: dict[pd.Timestamp, pd.Series] = {}
    turnover_log: dict[pd.Timestamp, float] = {}
    cost_log: dict[pd.Timestamp, float] = {}

    prev_date: pd.Timestamp | None = None
    prev_equity = initial_capital

    for i, date in enumerate(dates):
        # ── 1. Mark-to-market: update holdings with today's price changes ──
        if i > 0:
            daily_price_returns = prices.iloc[i] / prices.iloc[i - 1] - 1.0
            state.holdings = state.holdings * (1.0 + daily_price_returns)
            state.update_equity()

        # ── 2. Check rebalance schedule ──
        if _should_rebalance(date, prev_date, rebalance_freq):
            current_weights = state.weights()

            # ── 3. Get target weights from signal ──
            prices_so_far = prices.iloc[: i + 1]
            target_weights = signal_fn(date, prices_so_far, current_weights)

            # Ensure target_weights is aligned
            target_weights = target_weights.reindex(tickers, fill_value=0.0)

            # ── 4. Compute trades ──
            trade_weights = target_weights - current_weights

            # ── 5. Apply costs ──
            cost_dollars = cost_model.trading_cost(trade_weights, state.equity)
            turn = cost_model.turnover(trade_weights)

            # ── 6. Execute rebalance ──
            state.cash -= cost_dollars
            state.update_equity()

            # Reallocate holdings to target weights (after cost deduction)
            state.holdings = target_weights * state.equity
            state.cash = state.equity - state.holdings.sum()
            state.update_equity()

            turnover_log[date] = turn
            cost_log[date] = cost_dollars
        else:
            turnover_log[date] = 0.0
            cost_log[date] = 0.0

        # ── 7. Log daily state ──
        equity_log[date] = state.equity
        position_log[date] = state.weights().copy()

        daily_ret = (state.equity / prev_equity - 1.0) if prev_equity > 0 else 0.0
        return_log[date] = daily_ret
        prev_equity = state.equity
        prev_date = date

    # --- Build result ---
    result = BacktestResult(
        equity_curve=pd.Series(equity_log, name="equity"),
        daily_returns=pd.Series(return_log, name="return"),
        positions=pd.DataFrame(position_log).T,
        turnover=pd.Series(turnover_log, name="turnover"),
        total_costs=pd.Series(cost_log, name="costs"),
    )

    logger.info(
        "Backtest complete: %d days | Total return: %.2f%% | Max DD: %.2f%% | CAGR: %.2f%%",
        len(dates),
        result.total_return * 100,
        result.max_drawdown * 100,
        result.cagr * 100,
    )

    return result
