"""
Monte Carlo forward simulation.

Generates correlated return paths using multivariate normal draws
calibrated to historical return statistics. Tracks terminal wealth,
max drawdown distribution, and underwater period lengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config


@dataclass
class MonteCarloResult:
    """
    Output of a Monte Carlo simulation.

    Attributes
    ----------
    terminal_wealth : np.ndarray
        Final portfolio values across all paths (shape: n_paths).
    path_max_drawdowns : np.ndarray
        Maximum drawdown per path (shape: n_paths, negative values).
    equity_paths : np.ndarray or None
        Full equity curves if store_paths=True (shape: n_paths × n_days+1).
    initial_capital : float
        Starting portfolio value used in simulation.
    n_paths : int
        Number of simulated paths.
    n_days : int
        Simulation horizon in trading days.
    """

    terminal_wealth: np.ndarray
    path_max_drawdowns: np.ndarray
    initial_capital: float
    n_paths: int
    n_days: int
    equity_paths: np.ndarray | None = field(default=None, repr=False)

    # ── Summary statistics ──

    @property
    def median_terminal_wealth(self) -> float:
        return float(np.median(self.terminal_wealth))

    @property
    def mean_terminal_wealth(self) -> float:
        return float(np.mean(self.terminal_wealth))

    @property
    def terminal_wealth_var_95(self) -> float:
        """5th percentile of terminal wealth (worst-case scenario)."""
        return float(np.percentile(self.terminal_wealth, 5))

    @property
    def prob_profit(self) -> float:
        """Fraction of paths ending above initial capital."""
        return float(np.mean(self.terminal_wealth > self.initial_capital))

    @property
    def prob_loss_25pct(self) -> float:
        """Fraction of paths losing more than 25% of capital."""
        return float(np.mean(self.terminal_wealth < self.initial_capital * 0.75))

    @property
    def median_max_drawdown(self) -> float:
        return float(np.median(self.path_max_drawdowns))

    @property
    def var_95_max_drawdown(self) -> float:
        """95th percentile of max drawdown severity (worst 5% of paths)."""
        return float(np.percentile(self.path_max_drawdowns, 5))

    def underwater_periods(self, threshold: float = 0.0) -> np.ndarray:
        """
        Compute max consecutive days below initial capital for each path.

        Requires equity_paths to be stored (store_paths=True).

        Parameters
        ----------
        threshold : float
            Fraction below initial capital to count as underwater.
            0.0 = any loss, 0.10 = more than 10% loss.

        Returns
        -------
        np.ndarray
            Max underwater days per path (shape: n_paths).
        """
        if self.equity_paths is None:
            raise ValueError("equity_paths not stored; rerun with store_paths=True")

        floor = self.initial_capital * (1.0 - threshold)
        # Bool array: True where below floor
        underwater = self.equity_paths < floor  # shape: n_paths × (n_days+1)

        # Max consecutive True run per path
        def max_consecutive(row: np.ndarray) -> int:
            max_run = cur = 0
            for v in row:
                cur = cur + 1 if v else 0
                max_run = max(max_run, cur)
            return max_run

        return np.array([max_consecutive(row) for row in underwater])

    def __str__(self) -> str:
        lines = [
            f"  Paths:                {self.n_paths:>8,}",
            f"  Horizon:              {self.n_days:>8} days",
            f"  Initial Capital:      ${self.initial_capital:>12,.0f}",
            f"  Median Terminal W.:   ${self.median_terminal_wealth:>12,.0f}",
            f"  Mean Terminal W.:     ${self.mean_terminal_wealth:>12,.0f}",
            f"  5th Pct Terminal W.:  ${self.terminal_wealth_var_95:>12,.0f}",
            f"  P(Profit):            {self.prob_profit * 100:>7.1f}%",
            f"  P(Loss >25%):         {self.prob_loss_25pct * 100:>7.1f}%",
            f"  Median Max Drawdown:  {self.median_max_drawdown * 100:>7.1f}%",
            f"  95th Pct MaxDD:       {self.var_95_max_drawdown * 100:>7.1f}%",
        ]
        return "\n".join(lines)


def _compute_path_max_drawdown(equity_paths: np.ndarray) -> np.ndarray:
    """
    Vectorized max drawdown computation across all paths.

    Parameters
    ----------
    equity_paths : np.ndarray
        Shape (n_paths, n_days+1). First column = initial capital.

    Returns
    -------
    np.ndarray
        Max drawdown per path (shape: n_paths, negative values).
    """
    # Running peak across time axis
    peaks = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns = (equity_paths - peaks) / peaks
    return drawdowns.min(axis=1)


def run_monte_carlo(
    daily_returns: pd.Series,
    weights: pd.Series | None = None,
    asset_returns: pd.DataFrame | None = None,
    cov_matrix: pd.DataFrame | None = None,
    n_paths: int = config.MC_NUM_PATHS,
    n_days: int = config.MC_HORIZON_DAYS,
    initial_capital: float = config.INITIAL_CAPITAL,
    seed: int = config.MC_SEED,
    store_paths: bool = False,
) -> MonteCarloResult:
    """
    Run a vectorized Monte Carlo forward simulation.

    Two modes:
    1. **Portfolio mode** (daily_returns only): treats the series as a
       univariate portfolio return stream. Fits mu + sigma to historical data
       and simulates iid normal draws.

    2. **Multi-asset mode** (weights + asset_returns OR weights + cov_matrix):
       draws correlated daily returns from a multivariate normal calibrated
       to the historical asset return distribution, then computes portfolio
       returns using the fixed weight vector.

    Parameters
    ----------
    daily_returns : pd.Series
        Historical daily portfolio returns (used for mu estimation in both modes).
    weights : pd.Series or None
        Asset weights for multi-asset mode.
    asset_returns : pd.DataFrame or None
        Historical asset-level returns for covariance estimation.
    cov_matrix : pd.DataFrame or None
        Pre-computed covariance matrix (annualized). If provided, skips
        estimation from asset_returns.
    n_paths : int
        Number of Monte Carlo paths.
    n_days : int
        Simulation horizon in trading days.
    initial_capital : float
        Starting portfolio value.
    seed : int
        Random seed for reproducibility.
    store_paths : bool
        If True, stores full equity curves in result (memory-intensive for
        large n_paths).

    Returns
    -------
    MonteCarloResult
    """
    rng = np.random.default_rng(seed)

    # ── Estimate daily mu and sigma from historical portfolio returns ──
    mu_daily = float(daily_returns.mean())
    sigma_daily = float(daily_returns.std())

    if weights is not None and (asset_returns is not None or cov_matrix is not None):
        # ── Multi-asset correlated simulation ──
        w = weights.values  # shape (N,)

        if cov_matrix is not None:
            daily_cov = cov_matrix.values / 252
        else:
            daily_cov = asset_returns.cov().values

        daily_mu_assets = (
            asset_returns.mean().values
            if asset_returns is not None
            else np.full(len(w), mu_daily)
        )

        # Draw shape: (n_paths, n_days, N)
        draws = rng.multivariate_normal(
            mean=daily_mu_assets,
            cov=daily_cov,
            size=(n_paths, n_days),
        )
        # Portfolio returns: (n_paths, n_days)
        port_returns = draws @ w

    else:
        # ── Univariate simulation ──
        port_returns = rng.normal(
            loc=mu_daily,
            scale=sigma_daily,
            size=(n_paths, n_days),
        )

    # ── Build equity curves ──
    # Shape: (n_paths, n_days+1) with col 0 = initial_capital
    growth_factors = 1.0 + port_returns
    cum_returns = np.cumprod(growth_factors, axis=1)
    equity_matrix = np.empty((n_paths, n_days + 1))
    equity_matrix[:, 0] = initial_capital
    equity_matrix[:, 1:] = initial_capital * cum_returns

    terminal_wealth = equity_matrix[:, -1]
    path_mdd = _compute_path_max_drawdown(equity_matrix)

    return MonteCarloResult(
        terminal_wealth=terminal_wealth,
        path_max_drawdowns=path_mdd,
        initial_capital=initial_capital,
        n_paths=n_paths,
        n_days=n_days,
        equity_paths=equity_matrix if store_paths else None,
    )
