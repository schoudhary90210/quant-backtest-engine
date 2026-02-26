"""
Transaction cost model.

Proportional costs (commission + spread) and slippage model
proportional to trade size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


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
    ):
        self.cost_rate = cost_bps / 10_000  # bps → decimal
        self.slippage_rate = slippage_bps / 10_000

    def trading_cost(
        self,
        trade_weights: pd.Series | np.ndarray,
        portfolio_value: float,
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
