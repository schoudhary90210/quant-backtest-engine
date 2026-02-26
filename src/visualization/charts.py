"""
Visualization module.

Generates equity curves, drawdown plots, rolling Sharpe,
Monte Carlo fan, monthly heatmap, and weight-evolution charts.
All charts are saved as 300-DPI PNGs to output/charts/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

import config

CHART_DIR = Path(config.CHART_DIR)
DPI = config.CHART_DPI

COLORS = {
    "half_kelly": "#2196F3",  # blue
    "full_kelly": "#9C27B0",  # purple
    "equal_weight": "#FF9800",  # orange
    "drawdown": "#F44336",  # red
    "mc_median": "#1E88E5",
    "mc_band": "#90CAF9",
    "grid": "#E0E0E0",
}


def _save(fig: plt.Figure, name: str) -> Path:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. Equity curve comparison
# ---------------------------------------------------------------------------


def plot_equity_curves(
    results: dict[str, pd.Series],
    filename: str = "equity_curves.png",
) -> Path:
    """
    Plot normalised equity curves for multiple strategies.

    Parameters
    ----------
    results : dict[str, pd.Series]
        Strategy name → equity curve Series (DatetimeIndex, dollar values).
    filename : str
        Output file name.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    color_map = {
        "Half-Kelly": COLORS["half_kelly"],
        "Full Kelly": COLORS["full_kelly"],
        "Equal Weight": COLORS["equal_weight"],
    }

    for name, equity in results.items():
        normalised = equity / equity.iloc[0]
        color = color_map.get(name, "#607D8B")
        ax.plot(normalised.index, normalised, label=name, color=color, linewidth=1.6)

    ax.axhline(1.0, color="#9E9E9E", linewidth=0.8, linestyle="--")
    ax.set_title(
        "Portfolio Equity Curves (Normalised to 1.0)", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("Portfolio Value (×)")
    ax.set_xlabel("Date")
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}×"))
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 2. Monte Carlo fan chart
# ---------------------------------------------------------------------------


def plot_monte_carlo_fan(
    equity_paths: np.ndarray,
    initial_capital: float,
    filename: str = "monte_carlo_fan.png",
) -> Path:
    """
    Plot Monte Carlo equity path fan with percentile bands.

    Parameters
    ----------
    equity_paths : np.ndarray
        Shape (n_paths, n_days+1). Requires store_paths=True.
    initial_capital : float
        Starting value for normalisation baseline.
    filename : str
        Output file name.
    """
    # Normalise to multiples of initial capital
    norm = equity_paths / initial_capital
    days = np.arange(norm.shape[1])

    p5 = np.percentile(norm, 5, axis=0)
    p25 = np.percentile(norm, 25, axis=0)
    p50 = np.percentile(norm, 50, axis=0)
    p75 = np.percentile(norm, 75, axis=0)
    p95 = np.percentile(norm, 95, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(
        days, p5, p95, alpha=0.15, color=COLORS["mc_band"], label="5–95th pct"
    )
    ax.fill_between(
        days, p25, p75, alpha=0.30, color=COLORS["mc_band"], label="25–75th pct"
    )
    ax.plot(days, p50, color=COLORS["mc_median"], linewidth=2.0, label="Median")
    ax.axhline(
        1.0, color="#9E9E9E", linewidth=0.8, linestyle="--", label="Initial capital"
    )

    ax.set_title(
        "Monte Carlo Forward Simulation — Half-Kelly (1 Year)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value (×)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}×"))
    ax.legend(framealpha=0.9)
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 3. Rolling Sharpe ratio
# ---------------------------------------------------------------------------


def plot_rolling_sharpe(
    returns_dict: dict[str, pd.Series],
    window: int = config.ROLLING_SHARPE_WINDOW,
    risk_free_rate: float = config.RISK_FREE_RATE,
    filename: str = "rolling_sharpe.png",
) -> Path:
    """
    Plot rolling annualised Sharpe ratio for one or more strategies.

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        Strategy name → daily returns Series.
    window : int
        Rolling window in trading days.
    risk_free_rate : float
        Annualised risk-free rate.
    filename : str
        Output file name.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    color_map = {
        "Half-Kelly": COLORS["half_kelly"],
        "Full Kelly": COLORS["full_kelly"],
        "Equal Weight": COLORS["equal_weight"],
    }

    for name, returns in returns_dict.items():
        excess = returns - risk_free_rate / 252
        rolling = (
            excess.rolling(window).mean() / excess.rolling(window).std() * np.sqrt(252)
        )
        color = color_map.get(name, "#607D8B")
        ax.plot(rolling.index, rolling, label=name, color=color, linewidth=1.4)

    ax.axhline(0.0, color="#9E9E9E", linewidth=0.8, linestyle="--")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(framealpha=0.9)
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 4. Drawdown chart
# ---------------------------------------------------------------------------


def plot_drawdowns(
    results: dict[str, pd.Series],
    filename: str = "drawdowns.png",
) -> Path:
    """
    Plot underwater equity (drawdown from peak) for multiple strategies.

    Parameters
    ----------
    results : dict[str, pd.Series]
        Strategy name → equity curve Series.
    filename : str
        Output file name.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    color_map = {
        "Half-Kelly": COLORS["half_kelly"],
        "Full Kelly": COLORS["full_kelly"],
        "Equal Weight": COLORS["equal_weight"],
    }

    for name, equity in results.items():
        peak = equity.cummax()
        dd = (equity - peak) / peak * 100
        color = color_map.get(name, "#607D8B")
        ax.fill_between(dd.index, dd, 0, alpha=0.25, color=color)
        ax.plot(dd.index, dd, label=name, color=color, linewidth=1.2)

    ax.set_title("Portfolio Drawdown from Peak", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(framealpha=0.9)
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 5. Monthly returns heatmap
# ---------------------------------------------------------------------------


def plot_monthly_heatmap(
    daily_returns: pd.Series,
    strategy_name: str = "Half-Kelly",
    filename: str = "monthly_heatmap.png",
) -> Path:
    """
    Calendar heatmap of monthly returns (years × months).

    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns with DatetimeIndex.
    strategy_name : str
        Used in chart title.
    filename : str
        Output file name.
    """
    monthly = (1 + daily_returns).resample("ME").prod().sub(1).mul(100)
    monthly.index = monthly.index.to_period("M")

    years = sorted(monthly.index.year.unique())
    months = list(range(1, 13))
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    grid = pd.DataFrame(index=years, columns=months, dtype=float)
    for period, val in monthly.items():
        grid.loc[period.year, period.month] = val

    fig, ax = plt.subplots(figsize=(14, max(4, len(years) * 0.55)))

    vmax = (
        max(abs(grid.values[~np.isnan(grid.values)]))
        if grid.notna().any().any()
        else 10
    )
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(grid.values.astype(float), cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)

    for i, year in enumerate(years):
        for j, month in enumerate(months):
            val = grid.loc[year, month]
            if not np.isnan(val):
                text_color = "black" if abs(val) < vmax * 0.6 else "white"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=text_color,
                    fontweight="bold",
                )

    plt.colorbar(im, ax=ax, label="Monthly Return (%)", shrink=0.8)
    ax.set_title(
        f"Monthly Returns Heatmap — {strategy_name}", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 6. Weight allocation over time (stacked area)
# ---------------------------------------------------------------------------


def plot_weight_evolution(
    positions: pd.DataFrame,
    strategy_name: str = "Half-Kelly",
    filename: str = "weight_evolution.png",
    resample: str = "ME",
) -> Path:
    """
    Stacked area chart of portfolio weight allocation over time.

    Parameters
    ----------
    positions : pd.DataFrame
        DatetimeIndex × tickers DataFrame of portfolio weights.
    strategy_name : str
        Used in chart title.
    filename : str
        Output file name.
    resample : str
        Pandas offset string for resampling (default 'ME' = month-end).
    """
    # Resample to reduce density
    pos = positions.resample(resample).last().dropna(how="all")

    # Sort columns by mean weight descending for legibility
    col_order = pos.mean().sort_values(ascending=False).index.tolist()
    pos = pos[col_order]

    cmap = plt.cm.get_cmap("tab20", len(col_order))
    colors = [cmap(i) for i in range(len(col_order))]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(pos.index, pos.values.T, labels=col_order, colors=colors, alpha=0.85)

    ax.set_title(
        f"Portfolio Weight Allocation — {strategy_name}", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylim(0, 1)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 7. In-sample vs out-of-sample equity curves
# ---------------------------------------------------------------------------


def plot_is_vs_oos_equity(
    is_equity: pd.Series,
    oos_equity: pd.Series,
    oos_start_date: pd.Timestamp | None = None,
    filename: str = "is_vs_oos_equity.png",
) -> Path:
    """
    Overlay IS and OOS equity curves with an optional OOS-start marker.

    Both curves are normalised to 1.0 at their own first date so that
    differences in starting capital do not distort the comparison.

    Parameters
    ----------
    is_equity : pd.Series
        In-sample equity curve (standard Kelly backtest).
    oos_equity : pd.Series
        Out-of-sample equity curve (walk-forward backtest).
    oos_start_date : pd.Timestamp or None
        Date of the first valid walk-forward rebalance.  A vertical dotted
        line is drawn here when provided.
    filename : str
        Output file name.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    is_norm = is_equity / is_equity.iloc[0]
    oos_norm = oos_equity / oos_equity.iloc[0]

    ax.plot(
        is_norm.index,
        is_norm,
        label="In-Sample (Half-Kelly)",
        color=COLORS["half_kelly"],
        linewidth=1.6,
    )
    ax.plot(
        oos_norm.index,
        oos_norm,
        label="Out-of-Sample (Walk-Forward)",
        color="#4CAF50",
        linewidth=1.6,
        linestyle="--",
    )

    if oos_start_date is not None:
        ax.axvline(
            oos_start_date,
            color="#9E9E9E",
            linewidth=1.0,
            linestyle=":",
            label=f"OOS start ({oos_start_date.strftime('%Y-%m')})",
        )

    ax.axhline(1.0, color="#9E9E9E", linewidth=0.8, linestyle="--")
    ax.set_title(
        "In-Sample vs Out-of-Sample Equity Curves",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Portfolio Value (×)")
    ax.set_xlabel("Date")
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}×"))
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 8. Return estimator comparison
# ---------------------------------------------------------------------------


def plot_return_estimator_comparison(
    equity_curves: dict[str, pd.Series],
    metrics: dict[str, tuple[float, float]],
    filename: str = "return_estimator_comparison.png",
) -> Path:
    """
    Overlay equity curves for different return estimators with metric labels.

    Parameters
    ----------
    equity_curves : dict[str, pd.Series]
        Strategy name → equity curve (DatetimeIndex, dollar values).
    metrics : dict[str, tuple[float, float]]
        Strategy name → (cagr, sharpe) pre-computed floats shown in the legend.
    filename : str
        Output file name.
    """
    palette = {
        "Equal Weight": COLORS["equal_weight"],
        "Historical Mean + Kelly": COLORS["half_kelly"],
        "BL + Kelly": "#4CAF50",  # green
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, equity in equity_curves.items():
        norm = equity / equity.iloc[0]
        color = palette.get(name, "#607D8B")
        cagr, sharpe = metrics.get(name, (float("nan"), float("nan")))
        label = f"{name}  (CAGR {cagr * 100:.1f}%  Sharpe {sharpe:.2f})"
        ax.plot(norm.index, norm, label=label, color=color, linewidth=1.6)

    ax.axhline(1.0, color="#9E9E9E", linewidth=0.8, linestyle="--")
    ax.set_title(
        "Return Estimator Comparison — Historical Mean vs Black-Litterman",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Portfolio Value (×)")
    ax.set_xlabel("Date")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}×"))
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    fig.tight_layout()
    return _save(fig, filename)
