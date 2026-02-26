"""
Equal-weight (1/N) benchmark strategy.

Simple uniform allocation across all assets. Serves as a baseline
for comparing optimizer performance.
"""

from __future__ import annotations

import pandas as pd


def equal_weight_signal(
    date: pd.Timestamp,
    prices: pd.DataFrame,
    current_weights: pd.Series,
) -> pd.Series:
    """
    Return equal weights for all assets.

    Compatible with the backtester's SignalProvider protocol.
    """
    n = len(current_weights)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / n, index=current_weights.index)
