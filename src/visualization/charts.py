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
    "half_kelly": "#2196F3",      # blue
    "full_kelly": "#9C27B0",      # purple
    "equal_weight": "#FF9800",    # orange
    "bl_kelly": "#4CAF50",        # green
    "hrp": "#E91E63",             # pink
    "risk_parity": "#00BCD4",     # cyan
    "vol_targeted_bl": "#795548", # brown
    "drawdown": "#F44336",        # red
    "mc_median": "#1E88E5",
    "mc_band": "#90CAF9",
    "grid": "#E0E0E0",
    "regime_low_vol": "#4CAF50",   # green
    "regime_mid_vol": "#FFC107",   # amber
    "regime_high_vol": "#F44336",  # red
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
        "BL + Kelly": COLORS["bl_kelly"],
        "HRP": COLORS["hrp"],
        "Risk Parity": COLORS["risk_parity"],
        "Vol-Targeted BL": COLORS["vol_targeted_bl"],
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
        "BL + Kelly": COLORS["bl_kelly"],
        "HRP": COLORS["hrp"],
        "Risk Parity": COLORS["risk_parity"],
        "Vol-Targeted BL": COLORS["vol_targeted_bl"],
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
        "BL + Kelly": COLORS["bl_kelly"],
        "HRP": COLORS["hrp"],
        "Risk Parity": COLORS["risk_parity"],
        "Vol-Targeted BL": COLORS["vol_targeted_bl"],
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


# ---------------------------------------------------------------------------
# 9. Cost sensitivity chart
# ---------------------------------------------------------------------------


def plot_cost_sensitivity(
    cost_sensitivity: list[dict[str, float]],
    strategy_name: str = "Half-Kelly",
    filename: str = "cost_sensitivity.png",
) -> Path:
    """
    Line chart: Sharpe ratio (y) vs cost multiplier (x).

    Parameters
    ----------
    cost_sensitivity : list[dict]
        Each dict has 'multiplier' and 'sharpe' keys.
    strategy_name : str
        Used in chart title.
    filename : str
        Output file name.
    """
    mults = [d["multiplier"] for d in cost_sensitivity]
    sharpes = [d["sharpe"] for d in cost_sensitivity]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mults, sharpes, color=COLORS["half_kelly"], linewidth=2.0, marker="o", markersize=6)
    ax.axvline(1.0, color="#9E9E9E", linewidth=1.0, linestyle="--", label="Base cost (1×)")

    ax.set_title(
        f"Cost Sensitivity — {strategy_name}", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Cost Multiplier (×)")
    ax.set_ylabel("Annualised Sharpe Ratio")
    ax.legend(framealpha=0.9)
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 10. Turnover over time
# ---------------------------------------------------------------------------


def plot_turnover_over_time(
    turnover_series: pd.Series,
    strategy_name: str = "Half-Kelly",
    filename: str = "turnover_over_time.png",
) -> Path:
    """
    Bar chart of turnover at each rebalance date with rolling average overlay.

    Parameters
    ----------
    turnover_series : pd.Series
        One-way turnover per date (many entries will be 0 on non-rebalance days).
    strategy_name : str
        Used in chart title.
    filename : str
        Output file name.
    """
    # Filter to non-zero entries (rebalance dates)
    active = turnover_series[turnover_series > 0]
    if active.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No rebalance events", ha="center", va="center", fontsize=14)
        fig.tight_layout()
        return _save(fig, filename)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(active.index, active.values, color=COLORS["half_kelly"], alpha=0.6, width=15)

    # Rolling 6-period average
    if len(active) >= 6:
        rolling_avg = active.rolling(6, min_periods=1).mean()
        ax.plot(
            rolling_avg.index, rolling_avg.values,
            color="#F44336", linewidth=2.0, label="6-period rolling avg",
        )

    ax.set_title(
        f"Turnover per Rebalance — {strategy_name}", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("One-Way Turnover")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(framealpha=0.9)
    ax.grid(color=COLORS["grid"], linewidth=0.5, axis="y")
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 11. Cost breakdown (Almgren-Chriss only)
# ---------------------------------------------------------------------------


def plot_cost_breakdown(
    cost_breakdowns: dict[str, dict[str, float]],
    filename: str = "cost_breakdown.png",
) -> Path:
    """
    Stacked bar chart of per-asset cost breakdown.

    Parameters
    ----------
    cost_breakdowns : dict[str, dict[str, float]]
        Ticker → {spread, temp_impact, perm_impact, slippage} in dollars.
    filename : str
        Output file name.
    """
    tickers = list(cost_breakdowns.keys())
    components = ["spread", "temp_impact", "perm_impact", "slippage"]
    component_colors = ["#2196F3", "#FF9800", "#F44336", "#9C27B0"]
    component_labels = ["Spread", "Temp. Impact", "Perm. Impact", "Slippage"]

    data = {comp: [cost_breakdowns[t].get(comp, 0.0) for t in tickers] for comp in components}

    fig, ax = plt.subplots(figsize=(max(10, len(tickers) * 0.8), 5))
    x = np.arange(len(tickers))
    bottom = np.zeros(len(tickers))

    for comp, color, label in zip(components, component_colors, component_labels):
        vals = np.array(data[comp])
        ax.bar(x, vals, bottom=bottom, color=color, label=label, alpha=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_title("Cost Breakdown by Asset (Almgren-Chriss)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Cost ($)")
    ax.legend(framealpha=0.9)
    ax.grid(color=COLORS["grid"], linewidth=0.5, axis="y")
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 12. CPCV out-of-sample Sharpe distribution
# ---------------------------------------------------------------------------


def plot_cpcv_oos_sharpe(
    per_split_sharpe: np.ndarray,
    filename: str = "cpcv_oos_sharpe.png",
    wf_sharpe: float | None = None,
) -> Path:
    """
    Histogram of CPCV out-of-sample Sharpe ratios across all splits.

    Parameters
    ----------
    per_split_sharpe : np.ndarray
        OOS Sharpe ratio for each CPCV split.
    filename : str
        Output file name.
    wf_sharpe : float or None
        Walk-forward OOS Sharpe ratio.  If provided, drawn as a reference line.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        per_split_sharpe,
        bins=min(15, max(5, len(per_split_sharpe) // 3)),
        color=COLORS["half_kelly"],
        alpha=0.75,
        edgecolor="white",
    )
    ax.axvline(0, color="#F44336", linewidth=1.5, linestyle="--", label="Sharpe = 0")
    mean_sr = float(np.mean(per_split_sharpe))
    ax.axvline(
        mean_sr, color="#4CAF50", linewidth=1.5, linestyle="-", label=f"Mean = {mean_sr:.2f}"
    )

    if wf_sharpe is not None:
        ax.axvline(
            wf_sharpe, color="#2196F3", linewidth=1.5, linestyle="--",
            label=f"Walk-Forward OOS = {wf_sharpe:.2f}",
        )

    pbo = float((per_split_sharpe < 0).mean())
    ax.set_title(
        f"CPCV Out-of-Sample Sharpe Distribution (PBO = {pbo:.1%})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Annualised Sharpe Ratio")
    ax.set_ylabel("Frequency")
    ax.legend(framealpha=0.9)
    ax.grid(color=COLORS["grid"], linewidth=0.5, axis="y")
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 13. Eigenvalue distribution with Marchenko-Pastur overlay
# ---------------------------------------------------------------------------


def plot_eigenvalue_distribution(
    eigenvalues: np.ndarray,
    n_assets: int,
    n_observations: int,
    filename: str = "eigenvalue_mp.png",
) -> Path:
    """
    Histogram of sample correlation eigenvalues with MP PDF overlay.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of the sample correlation matrix.
    n_assets : int
        Number of assets (N).
    n_observations : int
        Number of observations (T).
    filename : str
        Output file name.
    """
    from src.optimization.covariance import marchenko_pastur_bounds, marchenko_pastur_pdf

    lambda_minus, lambda_plus = marchenko_pastur_bounds(n_assets, n_observations)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram (density-normalised so it overlays with the PDF)
    ax.hist(
        eigenvalues,
        bins=max(5, len(eigenvalues) // 2),
        density=True,
        color=COLORS["half_kelly"],
        alpha=0.65,
        edgecolor="white",
        label="Sample eigenvalues",
    )

    # MP PDF
    x = np.linspace(max(0.001, lambda_minus * 0.5), lambda_plus * 1.5, 500)
    mp_pdf = marchenko_pastur_pdf(x, n_assets, n_observations)
    ax.plot(x, mp_pdf, color="#F44336", linewidth=2.0, label="Marchenko-Pastur PDF")

    # Signal/noise boundary
    ax.axvline(
        lambda_plus,
        color="#4CAF50",
        linewidth=1.5,
        linestyle="--",
        label=f"$\\lambda_+$ = {lambda_plus:.2f}",
    )

    n_signal = int((eigenvalues > lambda_plus).sum())
    ax.set_title(
        f"Correlation Eigenvalues vs Marchenko-Pastur "
        f"(N={n_assets}, T={n_observations}, signal={n_signal})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.legend(framealpha=0.9)
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 14. Covariance estimator comparison
# ---------------------------------------------------------------------------


def plot_covariance_comparison(
    comparison: dict[str, dict],
    filename: str = "cov_estimator_comparison.png",
) -> Path:
    """
    Overlay eigenvalue spectra from multiple covariance estimators.

    Parameters
    ----------
    comparison : dict[str, dict]
        Output of ``compare_covariance_estimators()``.  Each value must
        contain an ``"eigenvalues"`` key (sorted descending) and a
        ``"condition_number"`` key.
    filename : str
        Output file name.
    """
    palette = {
        "sample": "#607D8B",
        "ledoit_wolf": "#2196F3",
        "ledoit_wolf_nonlinear": "#9C27B0",
        "ewma": "#FF9800",
        "rmt_denoised": "#4CAF50",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: eigenvalue spectra
    for name, data in comparison.items():
        eigs = data["eigenvalues"]
        color = palette.get(name, "#9C27B0")
        ax1.plot(
            range(1, len(eigs) + 1),
            eigs,
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=color,
            label=name,
        )

    ax1.set_title("Eigenvalue Spectra", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Eigenvalue Index")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_yscale("log")
    ax1.legend(framealpha=0.9)
    ax1.grid(color=COLORS["grid"], linewidth=0.5)

    # Right panel: condition number bar chart
    names = list(comparison.keys())
    cond_nums = [comparison[n]["condition_number"] for n in names]
    bar_colors = [palette.get(n, "#9C27B0") for n in names]

    ax2.barh(names, cond_nums, color=bar_colors, alpha=0.85)
    ax2.set_title("Condition Number", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Condition Number (lower = better)")
    ax2.invert_yaxis()
    ax2.grid(color=COLORS["grid"], linewidth=0.5, axis="x")

    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 15. DCC-GARCH: time-varying correlation heatmaps
# ---------------------------------------------------------------------------


def plot_dcc_correlation_heatmaps(
    dcc_estimator,
    dates: list[pd.Timestamp],
    filename: str = "dcc_correlation_heatmaps.png",
) -> Path:
    """
    Correlation heatmaps at multiple dates from a DCC-GARCH model.

    Parameters
    ----------
    dcc_estimator : DCCGarchEstimator
        Fitted DCC model (must have ``get_covariance()``).
    dates : list[pd.Timestamp]
        Dates at which to snapshot the conditional correlation.
    filename : str
        Output file name.
    """
    n = len(dates)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, date in zip(axes, dates):
        cov = dcc_estimator.get_covariance(date)
        vols = np.sqrt(np.diag(cov.values))
        vols_safe = np.where(vols > 1e-15, vols, 1e-15)
        corr = cov.values / np.outer(vols_safe, vols_safe)
        np.fill_diagonal(corr, 1.0)

        cmap = plt.cm.RdYlBu_r
        im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
        tickers = list(cov.index)
        ax.set_xticks(range(len(tickers)))
        ax.set_xticklabels(tickers, rotation=90, fontsize=6)
        ax.set_yticks(range(len(tickers)))
        ax.set_yticklabels(tickers, fontsize=6)
        ax.set_title(pd.Timestamp(date).strftime("%Y-%m-%d"), fontsize=10, fontweight="bold")

    fig.suptitle(
        "DCC-GARCH Conditional Correlations", fontsize=14, fontweight="bold", y=1.02
    )
    fig.colorbar(im, ax=axes, shrink=0.8, label="Correlation")
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 16. DCC-GARCH: pairwise conditional correlation time series
# ---------------------------------------------------------------------------


def plot_dcc_correlation_timeseries(
    dcc_estimator,
    pairs: list[tuple[str, str]],
    filename: str = "dcc_correlation_timeseries.png",
) -> Path:
    """
    Time series of conditional pairwise correlations from DCC-GARCH.

    Parameters
    ----------
    dcc_estimator : DCCGarchEstimator
        Fitted DCC model.
    pairs : list[tuple[str, str]]
        Asset pairs, e.g. [("SPY", "TLT"), ("SPY", "GLD")].
    filename : str
        Output file name.
    """
    pair_colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for idx, (a_i, a_j) in enumerate(pairs):
        rho = dcc_estimator.get_correlation_timeseries(a_i, a_j)
        color = pair_colors[idx % len(pair_colors)]
        ax.plot(rho.index, rho.values, label=f"{a_i} / {a_j}", color=color, linewidth=1.2)

    ax.axhline(0, color="#9E9E9E", linewidth=0.8, linestyle="--")
    ax.set_title("DCC-GARCH Conditional Correlations", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1, 1)
    ax.legend(framealpha=0.9)
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 17. Covariance backtest comparison
# ---------------------------------------------------------------------------

_COV_PALETTE = {
    "sample": "#607D8B",
    "ledoit_wolf": "#2196F3",
    "ledoit_wolf_nonlinear": "#9C27B0",
    "ewma": "#FF9800",
    "rmt_denoised": "#4CAF50",
    "dcc_garch": "#E91E63",
}


def plot_covariance_backtest_comparison(
    comparison: dict,
    filename: str = "cov_backtest_comparison.png",
) -> Path:
    """
    2x2 chart comparing backtest performance across covariance estimators.

    Parameters
    ----------
    comparison : dict[str, CovarianceComparisonResult]
        Output of ``run_covariance_backtest_comparison()``.
    filename : str
        Output file name.
    """
    # Sort methods by Sharpe descending
    sorted_methods = sorted(comparison.keys(), key=lambda m: comparison[m].sharpe, reverse=True)
    colors = [_COV_PALETTE.get(m, "#757575") for m in sorted_methods]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Top-left: Normalised equity curves ──
    ax = axes[0, 0]
    for m in sorted_methods:
        eq = comparison[m].backtest_result.equity_curve
        normalised = eq / eq.iloc[0]
        ax.plot(normalised.index, normalised.values,
                label=m, color=_COV_PALETTE.get(m, "#757575"), linewidth=1.3)
    ax.set_title("Equity Curves by Covariance Estimator", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(color=COLORS["grid"], linewidth=0.5)

    # ── Top-right: Sharpe bar chart ──
    ax = axes[0, 1]
    sharpes = [comparison[m].sharpe for m in sorted_methods]
    bars = ax.barh(sorted_methods, sharpes, color=colors, alpha=0.85)
    for i, m in enumerate(sorted_methods):
        if m == config.COV_METHOD:
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(2)
    ax.set_title("Sharpe Ratio", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sharpe Ratio")
    ax.invert_yaxis()
    ax.grid(color=COLORS["grid"], linewidth=0.5, axis="x")

    # ── Bottom-left: Max Drawdown bar chart ──
    ax = axes[1, 0]
    max_dds = [comparison[m].max_drawdown * 100 for m in sorted_methods]
    bars = ax.barh(sorted_methods, max_dds, color=colors, alpha=0.85)
    for i, m in enumerate(sorted_methods):
        if m == config.COV_METHOD:
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(2)
    ax.set_title("Max Drawdown", fontsize=13, fontweight="bold")
    ax.set_xlabel("Max Drawdown (%)")
    ax.invert_yaxis()
    ax.grid(color=COLORS["grid"], linewidth=0.5, axis="x")

    # ── Bottom-right: Summary table ──
    ax = axes[1, 1]
    ax.axis("off")
    col_labels = ["Method", "Sharpe", "CAGR", "Max DD", "Turnover", "Cond #"]
    table_data = []
    for m in sorted_methods:
        r = comparison[m]
        row = [
            m,
            f"{r.sharpe:.3f}",
            f"{r.cagr * 100:.1f}%",
            f"{r.max_drawdown * 100:.1f}%",
            f"{r.annualized_turnover:.2f}x",
            f"{r.condition_number:.0f}",
        ]
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#E0E0E0")
        table[0, j].set_text_props(fontweight="bold")
    ax.set_title("Performance Summary", fontsize=13, fontweight="bold", pad=20)

    fig.suptitle("Covariance Estimator Comparison", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 18. HRP dendrogram
# ---------------------------------------------------------------------------


def plot_hrp_dendrogram(
    dendrogram_data: dict,
    filename: str = "hrp_dendrogram.png",
) -> Path:
    """
    Plot the HRP hierarchical clustering dendrogram.

    Parameters
    ----------
    dendrogram_data : dict
        Output of ``HierarchicalRiskParity.get_dendrogram_data()``.
        Must contain ``linkage_matrix`` and ``labels``.
    filename : str
        Output file name.
    """
    from scipy.cluster.hierarchy import dendrogram

    linkage_matrix = dendrogram_data["linkage_matrix"]
    labels = dendrogram_data["labels"]

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=45,
        leaf_font_size=10,
        ax=ax,
    )
    ax.set_title("HRP Dendrogram — Asset Clustering", fontsize=14, fontweight="bold")
    ax.set_ylabel("Distance")
    ax.grid(color=COLORS["grid"], linewidth=0.5, axis="y")
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 19. Risk contribution comparison
# ---------------------------------------------------------------------------


def plot_risk_contribution_comparison(
    contribs: dict[str, pd.Series],
    filename: str = "risk_contributions.png",
) -> Path:
    """
    Stacked horizontal bar chart of per-asset risk contributions by strategy.

    Parameters
    ----------
    contribs : dict[str, pd.Series]
        Strategy label -> pd.Series of per-asset percentage contributions.
    filename : str
        Output file name.
    """
    # Collect all unique asset names across strategies
    all_assets: list[str] = []
    for rc in contribs.values():
        for a in rc.index:
            if a not in all_assets:
                all_assets.append(a)

    cmap = plt.cm.get_cmap("tab20", len(all_assets))
    asset_colors = {a: cmap(i) for i, a in enumerate(all_assets)}

    strategies = list(contribs.keys())
    fig, ax = plt.subplots(figsize=(12, max(4, len(strategies) * 1.2)))

    for i, strat in enumerate(strategies):
        rc = contribs[strat]
        left = 0.0
        for asset in all_assets:
            val = float(rc.get(asset, 0.0)) * 100  # to percentage
            ax.barh(i, val, left=left, color=asset_colors[asset], height=0.6)
            left += val

    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)
    ax.set_xlabel("Risk Contribution (%)")
    ax.set_title("Risk Contribution by Strategy", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Legend for assets
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=asset_colors[a], label=a) for a in all_assets]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
        framealpha=0.9,
    )
    ax.grid(color=COLORS["grid"], linewidth=0.5, axis="x")
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 20. HMM regime overlay
# ---------------------------------------------------------------------------


def plot_regime_overlay(
    equity_curve: pd.Series,
    regimes: pd.Series,
    filename: str = "regime_overlay.png",
) -> Path:
    """
    Plot equity curve with HMM regime-shaded background.

    Parameters
    ----------
    equity_curve : pd.Series
        Dollar equity curve (DatetimeIndex).
    regimes : pd.Series
        Regime labels (e.g. 'low_vol', 'high_vol') from HMMRegimeDetector.
    filename : str
        Output file name.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    norm = equity_curve / equity_curve.iloc[0]
    ax.plot(norm.index, norm.values, color=COLORS["bl_kelly"], linewidth=1.4, label="Equity")

    # Align regimes to equity dates via reindex with ffill
    aligned = regimes.reindex(norm.index, method="ffill")

    ymin, ymax = norm.min() * 0.95, norm.max() * 1.05
    for label in sorted(aligned.dropna().unique()):
        color_key = f"regime_{label}"
        color = COLORS.get(color_key, "#9E9E9E")
        mask = aligned == label
        ax.fill_between(
            norm.index,
            ymin,
            ymax,
            where=mask,
            alpha=0.15,
            color=color,
            label=label,
            step="pre",
        )

    ax.set_ylim(ymin, ymax)
    ax.set_title("Equity Curve with HMM Regime Overlay", fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value (×)")
    ax.set_xlabel("Date")
    ax.legend(framealpha=0.9)
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 21. HMM transition matrix
# ---------------------------------------------------------------------------


def plot_transition_matrix(
    transition_df: pd.DataFrame,
    filename: str = "regime_transition_matrix.png",
) -> Path:
    """
    Heatmap of HMM regime transition probabilities.

    Parameters
    ----------
    transition_df : pd.DataFrame
        Square DataFrame with regime labels as index/columns, values in [0, 1].
    filename : str
        Output file name.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(transition_df.values, cmap="Blues", vmin=0, vmax=1)

    labels = list(transition_df.index)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = transition_df.values[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=12, color=text_color, fontweight="bold")

    ax.set_title("HMM Regime Transition Probabilities", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Probability")
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 21b. Conditional performance heatmap (Sharpe by strategy × regime)
# ---------------------------------------------------------------------------


def plot_conditional_performance_heatmap(
    conditional_sharpe: pd.DataFrame,
    filename: str = "conditional_performance_heatmap.png",
) -> Path:
    """
    Diverging heatmap of Sharpe ratio by strategy (rows) × HMM regime (columns).

    Parameters
    ----------
    conditional_sharpe : pd.DataFrame
        Index = strategy names, columns = regime labels, values = Sharpe ratios.
    filename : str
        Output file name.
    """
    data = conditional_sharpe.values.astype(float)
    n_rows, n_cols = data.shape

    # Determine symmetric color range
    finite_vals = data[np.isfinite(data)]
    abs_max = max(np.abs(finite_vals).max(), 1e-6) if finite_vals.size > 0 else 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    fig_width = max(6, 2 + n_cols * 1.8)
    fig_height = max(4, 1.5 + n_rows * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")

    # Axis labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(list(conditional_sharpe.columns), fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(list(conditional_sharpe.index), fontsize=10)

    # Annotate each cell
    cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isfinite(val):
                rgba = cmap(norm(val))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=11, color=text_color, fontweight="bold")

    ax.set_title("Conditional Sharpe Ratio by HMM Regime", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Sharpe Ratio")
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 22. Leverage time series (vol targeting)
# ---------------------------------------------------------------------------


def plot_factor_exposures(
    attribution: dict,
    filename: str = "factor_exposures.png",
) -> Path:
    """
    Horizontal bar chart of factor betas with significance indicators.

    Parameters
    ----------
    attribution : dict
        Output of ``FactorAttribution.attribute()``.
    filename : str
        Output file name.
    """
    factors = list(attribution["betas"].keys())
    betas = [attribution["betas"][f] for f in factors]
    tstats = [attribution["beta_tstats"][f] for f in factors]

    # Compute SE from t-stats (guard near-zero)
    errors = []
    for b, t in zip(betas, tstats):
        if abs(t) < 1e-6:
            errors.append(0.0)
        else:
            errors.append(abs(b / t) * 2)  # ±2 SE

    bar_colors = [
        COLORS["half_kelly"] if b >= 0 else COLORS["drawdown"] for b in betas
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(factors))
    ax.barh(y_pos, betas, xerr=errors, color=bar_colors, alpha=0.85,
            edgecolor="white", capsize=4)
    ax.axvline(0, color="#9E9E9E", linewidth=1.0)

    # Annotate significance
    labels = []
    for f, t in zip(factors, tstats):
        label = f"{f} *" if abs(t) > 2 else f
        labels.append(label)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontweight="bold")
    ax.set_title("Factor Exposures — Primary Strategy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Beta (factor loading)")
    ax.grid(color=COLORS["grid"], linewidth=0.5, axis="x")
    fig.tight_layout()
    return _save(fig, filename)


def plot_rolling_alpha(
    rolling_attr: pd.DataFrame,
    filename: str = "rolling_alpha.png",
) -> Path:
    """
    Rolling annualized alpha with confidence band.

    Parameters
    ----------
    rolling_attr : pd.DataFrame
        Output of ``FactorAttribution.rolling_attribution()``.
    filename : str
        Output file name.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    alpha = rolling_attr["alpha_annual"] * 100  # to percentage

    if "alpha_se_annual" in rolling_attr.columns:
        se = rolling_attr["alpha_se_annual"] * 100
        upper = alpha + 2 * se
        lower = alpha - 2 * se
        ax.fill_between(
            rolling_attr.index, lower, upper,
            color="#4CAF50", alpha=0.2, label="±2 SE",
        )

    ax.plot(
        rolling_attr.index, alpha,
        color=COLORS["bl_kelly"], linewidth=1.4, label="Rolling Alpha",
    )
    ax.axhline(0, color="#9E9E9E", linewidth=1.0, linestyle="--")

    ax.set_title(
        "Rolling Annualized Alpha (252-day window)", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Alpha (%)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(framealpha=0.9)
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    fig.tight_layout()
    return _save(fig, filename)


def plot_leverage_timeseries(
    leverage_series: pd.Series,
    filename: str = "vol_target_leverage.png",
) -> Path:
    """
    Line chart of vol-targeted leverage over time.

    Parameters
    ----------
    leverage_series : pd.Series
        DatetimeIndex -> leverage multiplier.
    filename : str
        Output file name.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        leverage_series.index,
        leverage_series.values,
        color=COLORS["vol_targeted_bl"],
        linewidth=1.4,
    )
    ax.axhline(1.0, color="#9E9E9E", linewidth=1.0, linestyle="--", label="Baseline (1.0×)")
    ax.axhline(
        config.VOL_TARGET_MAX_LEVERAGE,
        color="#F44336",
        linewidth=1.0,
        linestyle="--",
        label=f"Max leverage ({config.VOL_TARGET_MAX_LEVERAGE:.1f}×)",
    )

    ax.set_title("Vol-Targeted BL — Leverage Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Leverage Multiplier")
    ax.legend(framealpha=0.9)
    ax.grid(color=COLORS["grid"], linewidth=0.5)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# 24. Parameter stability
# ---------------------------------------------------------------------------


def plot_parameter_stability(
    parameter_timeseries: pd.DataFrame,
    sharpe_timeseries: pd.Series,
    filename: str = "parameter_stability.png",
) -> Path:
    """
    Two-panel chart: optimal parameter trajectories (top) and
    best-window Sharpe (bottom) over rolling windows.

    Parameters
    ----------
    parameter_timeseries : pd.DataFrame
        Index = window end date, columns = parameter names.
    sharpe_timeseries : pd.Series
        Index = window end date, values = best Sharpe.
    filename : str
        Output file name.
    """
    n_params = len(parameter_timeseries.columns)
    fig, axes = plt.subplots(
        n_params + 1, 1,
        figsize=(12, 2.5 * (n_params + 1)),
        sharex=True,
    )
    if n_params + 1 == 1:
        axes = [axes]

    param_colors = list(COLORS.values())[:n_params]
    for i, col in enumerate(parameter_timeseries.columns):
        ax = axes[i]
        vals = pd.to_numeric(parameter_timeseries[col], errors="coerce")
        color = param_colors[i % len(param_colors)]
        ax.plot(vals.index, vals.values, marker="o", markersize=4,
                linewidth=1.4, color=color, label=col)
        ax.set_ylabel(col, fontsize=10)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(color=COLORS["grid"], linewidth=0.5)

    # Bottom panel: Sharpe
    ax_sharpe = axes[-1]
    ax_sharpe.plot(
        sharpe_timeseries.index, sharpe_timeseries.values,
        marker="s", markersize=4, linewidth=1.4,
        color=COLORS["half_kelly"], label="Best Sharpe",
    )
    ax_sharpe.axhline(0, color="#9E9E9E", linewidth=1.0, linestyle="--")
    ax_sharpe.set_ylabel("Sharpe", fontsize=10)
    ax_sharpe.set_xlabel("Window End Date")
    ax_sharpe.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_sharpe.grid(color=COLORS["grid"], linewidth=0.5)

    fig.suptitle(
        "Parameter Stability — Rolling Optimal Parameters",
        fontsize=14, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    return _save(fig, filename)
