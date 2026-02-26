"""
In-sample vs out-of-sample performance comparison.

compare_is_oos() prints a side-by-side metric table and flags potential
overfitting when the OOS Sharpe ratio falls below 50% of the IS Sharpe ratio.
"""

from __future__ import annotations

from src.risk.metrics import RiskReport

OVERFITTING_THRESHOLD = 0.50  # warn if OOS Sharpe < 50% of IS Sharpe


def compare_is_oos(
    is_metrics: RiskReport,
    oos_metrics: RiskReport,
) -> str:
    """
    Generate a side-by-side IS vs OOS performance comparison table.

    Prints the table to stdout and returns it as a string so callers can
    log or save it.  Also emits an overfitting warning when the OOS Sharpe
    ratio is less than ``OVERFITTING_THRESHOLD`` times the IS Sharpe ratio.

    Parameters
    ----------
    is_metrics : RiskReport
        In-sample risk report (full-period standard backtest).
    oos_metrics : RiskReport
        Out-of-sample risk report (walk-forward backtest).

    Returns
    -------
    str
        Formatted comparison table.
    """
    w = 62  # total line width
    lines: list[str] = [
        "",
        "=" * w,
        "  In-Sample vs Out-of-Sample Performance Comparison",
        "=" * w,
        f"  {'Metric':<26} {'IS':>10}  {'OOS':>10}  {'Delta':>10}",
        "  " + "-" * (w - 2),
    ]

    def _pct(v: float) -> str:
        return f"{v * 100:>9.2f}%"

    def _ratio(v: float) -> str:
        return f"{v:>10.3f}"

    rows: list[tuple[str, float, float, object]] = [
        ("CAGR", is_metrics.annualized_return, oos_metrics.annualized_return, _pct),
        ("Sharpe Ratio", is_metrics.sharpe_ratio, oos_metrics.sharpe_ratio, _ratio),
        ("Sortino Ratio", is_metrics.sortino_ratio, oos_metrics.sortino_ratio, _ratio),
        ("Max Drawdown", is_metrics.max_drawdown, oos_metrics.max_drawdown, _pct),
        ("Calmar Ratio", is_metrics.calmar_ratio, oos_metrics.calmar_ratio, _ratio),
        (
            "Ann. Volatility",
            is_metrics.annualized_volatility,
            oos_metrics.annualized_volatility,
            _pct,
        ),
        ("VaR 95%", is_metrics.var_95, oos_metrics.var_95, _pct),
        ("Win Rate", is_metrics.win_rate, oos_metrics.win_rate, _pct),
    ]

    for label, is_val, oos_val, fmt in rows:
        delta = oos_val - is_val
        lines.append(f"  {label:<26} {fmt(is_val)}  {fmt(oos_val)}  {fmt(delta)}")

    lines.append("  " + "-" * (w - 2))

    # Overfitting check
    if is_metrics.sharpe_ratio > 0:
        ratio = oos_metrics.sharpe_ratio / is_metrics.sharpe_ratio
        if ratio < OVERFITTING_THRESHOLD:
            lines.append(
                f"\n  [WARN] OOS Sharpe is {ratio:.0%} of IS Sharpe "
                f"(threshold: {OVERFITTING_THRESHOLD:.0%}) — potential overfitting"
            )
        else:
            lines.append(
                f"\n  [OK]   OOS Sharpe is {ratio:.0%} of IS Sharpe "
                f"(threshold: {OVERFITTING_THRESHOLD:.0%}) — no overfitting detected"
            )

    lines.append("=" * w)

    table = "\n".join(lines)
    print(table)
    return table
