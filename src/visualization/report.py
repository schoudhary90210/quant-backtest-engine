"""
Summary report generator.

Prints and saves a full text report covering: risk metrics per strategy,
regime breakdown, Kelly weight table, and top-3 drawdown periods.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.engine.backtest import BacktestResult
from src.risk.metrics import RiskReport, compute_risk_report
from src.risk.regime import regime_stats

REPORT_PATH = Path(config.REPORT_PATH)
SEP = "=" * 66
SEP2 = "-" * 66


def _top_drawdown_periods(equity: pd.Series, n: int = 3) -> list[dict]:
    """
    Find the top-N distinct drawdown periods by peak-to-trough depth.

    Returns a list of dicts with keys: start, trough, end, depth, duration.
    """
    peak = equity.cummax()
    dd = (equity - peak) / peak

    results = []
    in_dd = False
    dd_start = None
    dd_peak_val = None

    for date, val in dd.items():
        if val < 0 and not in_dd:
            in_dd = True
            dd_start = date
            dd_peak_val = equity[date]
        elif val == 0 and in_dd:
            # Recovery — record the period
            segment = dd[dd_start:date]
            trough_date = segment.idxmin()
            depth = float(segment.min())
            duration = (date - dd_start).days
            results.append(
                {
                    "start": dd_start,
                    "trough": trough_date,
                    "end": date,
                    "depth": depth,
                    "duration_days": duration,
                }
            )
            in_dd = False

    # Handle still-in-drawdown at end
    if in_dd:
        segment = dd[dd_start:]
        trough_date = segment.idxmin()
        depth = float(segment.min())
        results.append(
            {
                "start": dd_start,
                "trough": trough_date,
                "end": equity.index[-1],
                "depth": depth,
                "duration_days": (equity.index[-1] - dd_start).days,
            }
        )

    results.sort(key=lambda x: x["depth"])
    return results[:n]


def generate_report(
    backtest_results: dict[str, BacktestResult],
    kelly_result: BacktestResult,
    write_file: bool = True,
) -> str:
    """
    Generate a comprehensive text report.

    Parameters
    ----------
    backtest_results : dict[str, BacktestResult]
        Mapping of strategy name → BacktestResult.
    kelly_result : BacktestResult
        The Half-Kelly result used for regime and weight analysis.
    write_file : bool
        If True, save to config.REPORT_PATH.

    Returns
    -------
    str
        Full report text.
    """
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add(SEP)
    add("  QUANTITATIVE BACKTESTING ENGINE — FULL REPORT")
    add(SEP)
    add(
        f"  Date range: {kelly_result.equity_curve.index[0].date()} → "
        f"{kelly_result.equity_curve.index[-1].date()}"
    )
    add(f"  Initial capital: ${config.INITIAL_CAPITAL:,.0f}")
    add()

    # ── Risk metrics per strategy ──
    add(SEP)
    add("  RISK METRICS BY STRATEGY")
    add(SEP)

    for name, result in backtest_results.items():
        report: RiskReport = compute_risk_report(result.daily_returns)
        add()
        add(f"  ┌─ {name} " + "─" * max(0, 50 - len(name)) + "┐")
        add(f"  │  Total Return:         {result.total_return * 100:>8.2f}%")
        add(f"  │  CAGR:                 {result.cagr * 100:>8.2f}%")
        add(f"  │  Annualised Return:    {report.annualized_return * 100:>8.2f}%")
        add(f"  │  Annualised Vol:       {report.annualized_volatility * 100:>8.2f}%")
        add(f"  │  Sharpe Ratio:         {report.sharpe_ratio:>8.3f}")
        add(f"  │  Sortino Ratio:        {report.sortino_ratio:>8.3f}")
        add(f"  │  Max Drawdown:         {report.max_drawdown * 100:>8.2f}%")
        add(f"  │  Calmar Ratio:         {report.calmar_ratio:>8.3f}")
        add(f"  │  VaR (95%):            {report.var_95 * 100:>8.2f}%")
        add(f"  │  CVaR (95%):           {report.cvar_95 * 100:>8.2f}%")
        add(f"  │  Win Rate:             {report.win_rate * 100:>8.2f}%")
        add(f"  │  Profit Factor:        {report.profit_factor:>8.3f}")
        add(f"  │  Total Costs:         ${result.total_costs.sum():>10,.0f}")
        add(f"  └" + "─" * 54 + "┘")

    # ── Regime breakdown ──
    add()
    add(SEP)
    add("  REGIME ANALYSIS — Half-Kelly")
    add(SEP)

    stats = regime_stats(kelly_result.daily_returns)
    for regime_name in ("bull", "sideways", "bear"):
        if regime_name not in stats:
            continue
        s = stats[regime_name]
        r = s.report
        add()
        add(
            f"  {regime_name.upper()} ({s.n_days} days, {s.fraction * 100:.1f}% of period)"
        )
        add(
            f"    Sharpe:  {r.sharpe_ratio:>6.3f}  |  CAGR:    {r.annualized_return * 100:>6.2f}%"
        )
        add(
            f"    MaxDD:   {r.max_drawdown * 100:>6.2f}%  |  Sortino: {r.sortino_ratio:>6.3f}"
        )
        add(
            f"    VaR 95%: {r.var_95 * 100:>6.2f}%  |  CVaR:    {r.cvar_95 * 100:>6.2f}%"
        )

    # ── Kelly weight table ──
    add()
    add(SEP)
    add("  AVERAGE KELLY WEIGHTS BY ASSET")
    add(SEP)
    add()

    if not kelly_result.positions.empty:
        mean_w = (
            kelly_result.positions.replace(0, np.nan)
            .mean()
            .sort_values(ascending=False)
            .dropna()
        )
        add(f"  {'Asset':<12}  {'Avg Weight':>10}  {'Max Weight':>10}")
        add(f"  {'─'*12}  {'─'*10}  {'─'*10}")
        for ticker, avg in mean_w.items():
            max_w = kelly_result.positions[ticker].max()
            add(f"  {ticker:<12}  {avg * 100:>9.2f}%  {max_w * 100:>9.2f}%")

    # ── Top drawdowns ──
    add()
    add(SEP)
    add("  TOP 3 DRAWDOWN PERIODS — Half-Kelly")
    add(SEP)
    add()

    periods = _top_drawdown_periods(kelly_result.equity_curve, n=3)
    for i, p in enumerate(periods, 1):
        add(
            f"  #{i}  Peak→Trough: {p['start'].date()} → {p['trough'].date()}"
            f"  (Recovery: {p['end'].date()})"
        )
        add(
            f"       Depth: {p['depth'] * 100:.2f}%   Duration: {p['duration_days']} days"
        )
        add()

    add(SEP)

    text = "\n".join(lines)
    print(text)

    if write_file:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(text)
        print(f"\n  Report saved → {REPORT_PATH}")

    return text
