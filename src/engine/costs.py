"""
Transaction cost models.

Provides two models:
  - TransactionCostModel: proportional commission + slippage (default).
  - AlmgrenChrissCostModel: nonlinear multi-component market impact
    (Almgren & Chriss, 2001) with spread, temporary impact, permanent
    impact, and volatility-driven slippage.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


class TransactionCostModel:
    """
    Models trading friction: proportional commission/spread and slippage.

    Parameters
    ----------
    cost_bps : float
        Round-trip transaction cost in basis points.
    slippage_bps : float
        Slippage in basis points, applied proportionally to trade size.
    """

    def __init__(
        self,
        cost_bps: float = config.TRANSACTION_COST_BPS,
        slippage_bps: float = config.SLIPPAGE_BPS,
    ) -> None:
        self.cost_rate = cost_bps / 10_000  # bps → decimal
        self.slippage_rate = slippage_bps / 10_000

    def trading_cost(
        self,
        trade_weights: pd.Series | np.ndarray,
        portfolio_value: float,
        **kwargs: object,
    ) -> float:
        """
        Compute total dollar cost of a rebalance.

        Parameters
        ----------
        trade_weights : pd.Series or np.ndarray
            Signed weight changes (target_weight - current_weight) per asset.
        portfolio_value : float
            Current portfolio market value.

        Returns
        -------
        float
            Total dollar cost (always non-negative).
        """
        trade_notional = np.abs(trade_weights) * portfolio_value
        if isinstance(trade_notional, pd.Series):
            total_traded = trade_notional.sum()
        else:
            total_traded = float(np.sum(trade_notional))

        commission = total_traded * self.cost_rate
        slippage = total_traded * self.slippage_rate
        return commission + slippage

    def turnover(self, trade_weights: pd.Series | np.ndarray) -> float:
        """
        Compute one-way turnover as sum of absolute weight changes / 2.

        Parameters
        ----------
        trade_weights : pd.Series or np.ndarray
            Signed weight changes per asset.

        Returns
        -------
        float
            One-way turnover (0 = no trades, 1 = full portfolio turnover).
        """
        return float(np.sum(np.abs(trade_weights))) / 2.0

    def net_return_after_costs(
        self,
        gross_return: float,
        trade_weights: pd.Series | np.ndarray,
        portfolio_value: float,
        **kwargs: object,
    ) -> float:
        """
        Deduct trading costs from a gross portfolio return.

        Parameters
        ----------
        gross_return : float
            Pre-cost portfolio return for the period.
        trade_weights : pd.Series or np.ndarray
            Signed weight changes this period.
        portfolio_value : float
            Portfolio value before the trade.

        Returns
        -------
        float
            Net return after subtracting cost drag.
        """
        cost = self.trading_cost(trade_weights, portfolio_value)
        cost_drag = cost / portfolio_value if portfolio_value > 0 else 0.0
        return gross_return - cost_drag


# ---------------------------------------------------------------------------
# Almgren-Chriss nonlinear market impact model
# ---------------------------------------------------------------------------


class AlmgrenChrissCostModel:
    """
    Multi-component transaction cost model based on Almgren & Chriss (2001).

    Four cost components per trade:
        1. Spread:      half_spread * |trade_dollars|
        2. Temp impact: eta * sigma * sqrt(|shares|/ADV) * |trade_dollars|
        3. Perm impact: gamma * (|shares|/ADV) * |trade_dollars|
        4. Slippage:    slippage_factor * sigma * |trade_dollars|

    Parameters
    ----------
    prices : pd.DataFrame
        Daily close prices (DatetimeIndex x tickers).
    volumes : pd.DataFrame
        Daily volumes (DatetimeIndex x tickers).
    half_spread_bps : float
        Half-spread in basis points.
    eta : float
        Temporary impact coefficient.
    gamma : float
        Permanent impact coefficient.
    slippage_factor : float
        Volatility-proportional slippage multiplier.
    adv_window : int
        Rolling window for average daily volume.
    vol_window : int
        Rolling window for daily return volatility.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        half_spread_bps: float = config.SPREAD_BPS_DEFAULT,
        eta: float = config.IMPACT_ETA,
        gamma: float = config.IMPACT_GAMMA,
        slippage_factor: float = config.SLIPPAGE_FACTOR,
        adv_window: int = config.ADV_WINDOW,
        vol_window: int = 20,
    ) -> None:
        self.half_spread = half_spread_bps / 10_000
        self.eta = eta
        self.gamma = gamma
        self.slippage_factor = slippage_factor
        self.prices = prices
        self.volumes = volumes

        # Pre-compute rolling ADV and daily volatility
        self.adv: pd.DataFrame = volumes.rolling(adv_window, min_periods=1).mean()
        log_ret = np.log(prices / prices.shift(1))
        self.daily_vol: pd.DataFrame = log_ret.rolling(vol_window, min_periods=2).std()

    @staticmethod
    def _lookup(
        df: pd.DataFrame, ticker: str, date: pd.Timestamp, fallback: float
    ) -> float:
        """Safe date-aligned lookup with fallback for warm-up period."""
        if ticker not in df.columns:
            return fallback
        col = df[ticker].loc[:date].dropna()
        if col.empty:
            return fallback
        return float(col.iloc[-1])

    def compute_cost(self, ticker: str, trade_dollars: float, date: pd.Timestamp) -> float:
        """Compute total dollar cost for a single-asset trade."""
        return sum(self.compute_cost_breakdown(ticker, trade_dollars, date).values())

    def compute_cost_breakdown(
        self, ticker: str, trade_dollars: float, date: pd.Timestamp
    ) -> dict[str, float]:
        """
        Compute per-component dollar costs for a single-asset trade.

        Returns dict with keys: spread, temp_impact, perm_impact, slippage.
        """
        abs_trade = abs(trade_dollars)
        if abs_trade < 1e-10:
            return {"spread": 0.0, "temp_impact": 0.0, "perm_impact": 0.0, "slippage": 0.0}

        price = self._lookup(self.prices, ticker, date, fallback=100.0)
        sigma = self._lookup(self.daily_vol, ticker, date, fallback=0.01)
        adv = self._lookup(self.adv, ticker, date, fallback=1e6)

        shares = abs_trade / price if price > 0 else 0.0
        participation = shares / adv if adv > 0 else 0.0

        # 1. Half-spread cost
        spread = self.half_spread * abs_trade

        # 2. Temporary impact — sqrt of participation rate (nonlinear)
        temp_impact = self.eta * sigma * np.sqrt(participation) * abs_trade

        # 3. Permanent impact — linear in participation rate
        perm_impact = self.gamma * participation * abs_trade

        # 4. Volatility-driven slippage
        slippage = self.slippage_factor * sigma * abs_trade

        return {
            "spread": spread,
            "temp_impact": temp_impact,
            "perm_impact": perm_impact,
            "slippage": slippage,
        }

    def trading_cost(
        self,
        trade_weights: pd.Series | np.ndarray,
        portfolio_value: float,
        **kwargs: object,
    ) -> float:
        """
        Compute total dollar cost of a rebalance.

        Requires ``date`` keyword argument for date-aligned lookups.
        Falls back to spread-only cost if trade_weights is a numpy array.
        """
        date = kwargs.get("date")
        if date is None:
            raise ValueError(
                "AlmgrenChrissCostModel.trading_cost() requires date= keyword argument"
            )

        if isinstance(trade_weights, np.ndarray):
            # Numpy array fallback — spread-only approximation
            total_traded = float(np.sum(np.abs(trade_weights))) * portfolio_value
            return total_traded * self.half_spread

        total_cost = 0.0
        for ticker, weight_change in trade_weights.items():
            trade_dollars = abs(weight_change) * portfolio_value
            if trade_dollars < 1e-10:
                continue
            total_cost += self.compute_cost(ticker, trade_dollars, date)
        return total_cost

    def turnover(self, trade_weights: pd.Series | np.ndarray) -> float:
        """Compute one-way turnover as sum of absolute weight changes / 2."""
        return float(np.sum(np.abs(trade_weights))) / 2.0

    def net_return_after_costs(
        self,
        gross_return: float,
        trade_weights: pd.Series | np.ndarray,
        portfolio_value: float,
        **kwargs: object,
    ) -> float:
        """Deduct trading costs from a gross portfolio return."""
        cost = self.trading_cost(trade_weights, portfolio_value, **kwargs)
        cost_drag = cost / portfolio_value if portfolio_value > 0 else 0.0
        return gross_return - cost_drag
