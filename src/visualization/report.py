"""
Summary report generator.

Prints and saves a full text report covering: risk metrics per strategy,
regime breakdown, Kelly weight table, and top-3 drawdown periods.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.engine.backtest import BacktestResult
from src.risk.metrics import RiskReport, compute_risk_report
from src.risk.regime import regime_stats

logger = logging.getLogger(__name__)

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
    stat_validation: dict | None = None,
    turnover_data: dict | None = None,
    display_label: str = "Half-Kelly",
    cov_comparison: dict | None = None,
    strategy_comparison: dict | None = None,
    factor_attribution: dict | None = None,
    rolling_factor_attr: pd.DataFrame | None = None,
    hmm_regime_data: dict | None = None,
    conditional_performance: dict | None = None,
    parameter_stability: dict | None = None,
) -> str:
    """
    Generate a comprehensive text report.

    Parameters
    ----------
    backtest_results : dict[str, BacktestResult]
        Mapping of strategy name → BacktestResult.
    kelly_result : BacktestResult
        The primary result used for regime and weight analysis.
    write_file : bool
        If True, save to config.REPORT_PATH.
    display_label : str
        Label for the primary strategy (e.g. "BL + Kelly", "Half-Kelly").

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

    # ── Strategy comparison table ──
    if strategy_comparison is not None:
        add()
        add(SEP)
        add("  STRATEGY COMPARISON")
        add(SEP)
        add()

        sorted_strats = sorted(
            strategy_comparison.keys(),
            key=lambda s: strategy_comparison[s]["sharpe"],
            reverse=True,
        )

        add(f"  {'Strategy':<20s}  {'Sharpe':>7s}  {'Sortino':>8s}  {'CAGR':>7s}  {'Max DD':>7s}  {'Calmar':>7s}  {'Turnover':>9s}")
        add(f"  {'─' * 20}  {'─' * 7}  {'─' * 8}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 9}")

        for label in sorted_strats:
            s = strategy_comparison[label]
            marker = "  <" if label == display_label else ""
            add(
                f"  {label:<20s}  {s['sharpe']:>7.3f}  {s['sortino']:>8.3f}"
                f"  {s['cagr'] * 100:>6.1f}%  {s['max_drawdown'] * 100:>6.1f}%"
                f"  {s['calmar']:>7.3f}  {s['annualized_turnover']:>8.2f}x{marker}"
            )

        best = sorted_strats[0]
        add()
        add(f"  Best Sharpe: {best} ({strategy_comparison[best]['sharpe']:.3f})")

    # ── Regime breakdown ──
    add()
    add(SEP)
    add(f"  REGIME ANALYSIS — {display_label}")
    add(SEP)

    if hmm_regime_data is not None:
        perf = hmm_regime_data["performance"]
        n_states = hmm_regime_data["n_states"]
        regime_order = hmm_regime_data.get("regime_order", list(perf.index))
        add()
        add(f"  Method: Hidden Markov Model ({n_states}-state, benchmark={config.REGIME_BENCHMARK})")
        add()
        for regime_label in regime_order:
            if regime_label not in perf.index:
                continue
            row = perf.loc[regime_label]
            add(f"  {regime_label.upper()} ({int(row['n_days'])} days, {row['fraction'] * 100:.1f}%)")
            add(f"    Sharpe:  {row['sharpe']:>6.3f}  |  Ann. Return: {row['mean_return'] * 100:>6.2f}%")
            add(f"    MaxDD:   {row['max_drawdown'] * 100:>6.2f}%  |  Volatility:  {row['volatility'] * 100:>6.2f}%")
            add()
        # Transition matrix
        trans = hmm_regime_data["transition_matrix"]
        add("  Transition Probabilities:")
        header = "    " + " " * 12 + "  ".join(f"{c:>10s}" for c in trans.columns)
        add(header)
        for row_label in trans.index:
            vals = "  ".join(f"{trans.loc[row_label, c]:>10.3f}" for c in trans.columns)
            add(f"    {row_label:<12s}{vals}")
    else:
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

    # ── Conditional performance by regime ──
    if conditional_performance is not None:
        add()
        add(SEP)
        add("  CONDITIONAL PERFORMANCE BY REGIME")
        add(SEP)
        add()
        long_form = conditional_performance["long_form"]
        regime_order = conditional_performance["regime_order"]
        for regime in regime_order:
            regime_rows = long_form[long_form["regime"] == regime]
            if regime_rows.empty:
                continue
            first = regime_rows.iloc[0]
            add(f"  {regime.upper()} ({int(first['n_days'])} days, {first['fraction'] * 100:.1f}% of time)")
            add(f"  {'Strategy':<22s}  {'Sharpe':>7s}  {'Max DD':>7s}")
            add(f"  {'─' * 22}  {'─' * 7}  {'─' * 7}")
            for _, row in regime_rows.iterrows():
                add(f"  {row['strategy']:<22s}  {row['sharpe']:>7.3f}  {row['max_drawdown'] * 100:>6.2f}%")
            add()

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

    # ── Statistical validation ──
    if stat_validation is not None:
        add()
        add(SEP)
        oos_tag = " — Out-of-Sample" if stat_validation.get("oos", False) else ""
        add(f"  STATISTICAL VALIDATION{oos_tag}")
        add(SEP)

        dsr = stat_validation.get("dsr")
        if dsr:
            add()
            add("  Deflated Sharpe Ratio (DSR)")
            tag = "[PASS]" if dsr["is_significant"] else "[FAIL]"
            add(f"    DSR:              {dsr['dsr']:>8.4f}   {tag}")
            add(
                f"    E[max(SR)]:       {dsr['expected_max_sr']:>8.4f}"
                f"   (correction for {config.N_STRATEGY_TRIALS} trial(s))"
            )
            add(f"    SR std error:     {dsr['sr_std']:>8.4f}")

        psr = stat_validation.get("psr")
        if psr:
            add()
            add("  Probabilistic Sharpe Ratio (PSR)")
            tag = "[PASS]" if psr["is_significant"] else "[FAIL]"
            add(f"    PSR:              {psr['psr']:>8.4f}   {tag}")
            add(f"    SR std error:     {psr['sr_std']:>8.4f}")

        minbtl = stat_validation.get("minbtl")
        if minbtl:
            add()
            add("  Minimum Backtest Length (MinBTL)")
            add(
                f"    Required:         {minbtl['min_btl_observations']:>8.0f} days"
                f" ({minbtl['min_btl_years']:.1f} years)"
            )
            suff = minbtl.get("actual_length_sufficient")
            if suff is not None:
                label = "Yes [PASS]" if suff else "No [FAIL]"
                add(f"    Sufficient:       {label:>14}")

        cpcv = stat_validation.get("cpcv")
        if cpcv:
            add()
            add("  Combinatorial Purged Cross-Validation (CPCV)")
            add("    (static weight estimation, Ledoit-Wolf covariance — not full walk-forward/costed path)")
            add(f"    Splits:           {cpcv['n_splits']:>8d}")
            add(f"    Mean OOS Sharpe:  {cpcv['mean_oos_sharpe']:>8.3f}")
            add(f"    Std OOS Sharpe:   {cpcv['std_oos_sharpe']:>8.3f}")

        pbo = stat_validation.get("pbo")
        if pbo:
            add()
            add("  Probability of Backtest Overfitting (PBO)")
            tag = "[WARN]" if pbo["pbo"] > 0.5 else "[OK]"
            add(f"    PBO:              {pbo['pbo']:>8.3f}   {tag}")
            add(f"    IS/OOS Rank Corr: {pbo['is_oos_rank_correlation']:>8.3f}")
            add(f"    Degradation:      {pbo['degradation_ratio']:>8.3f}")

    # ── Turnover & cost analysis ──
    if turnover_data is not None:
        add()
        add(SEP)
        add("  TURNOVER & COST ANALYSIS")
        add(SEP)
        add()
        add(f"    Annualised Turnover:   {turnover_data['annualized_turnover']:>8.2f}×")
        add(f"    Avg Holding Period:    {turnover_data['avg_holding_period_days']:>8.0f} days")
        add(f"    Cost Drag:            {turnover_data['total_cost_drag_bps']:>8.1f} bps/yr")

        cs = turnover_data.get("cost_sensitivity")
        if cs:
            add()
            add("    Cost Sensitivity (Sharpe at different cost levels):")
            add(f"    {'Multiplier':>12}  {'Sharpe':>8}")
            add(f"    {'─'*12}  {'─'*8}")
            for entry in cs:
                marker = " ◄" if entry["multiplier"] == 1.0 else ""
                add(f"    {entry['multiplier']:>11.1f}×  {entry['sharpe']:>8.3f}{marker}")

    # ── Covariance estimator comparison ──
    if cov_comparison is not None:
        add()
        add(SEP)
        add("  COVARIANCE ESTIMATOR COMPARISON")
        add(SEP)
        add()

        # Sort by Sharpe descending
        sorted_methods = sorted(
            cov_comparison.keys(),
            key=lambda m: cov_comparison[m].sharpe,
            reverse=True,
        )

        add(f"  {'Method':<25s}  {'Sharpe':>7s}  {'CAGR':>7s}  {'Max DD':>7s}  {'Turnover':>9s}  {'Cond #':>8s}")
        add(f"  {'─' * 25}  {'─' * 7}  {'─' * 7}  {'─' * 7}  {'─' * 9}  {'─' * 8}")
        for m in sorted_methods:
            r = cov_comparison[m]
            marker = "  < default" if m == config.COV_METHOD else ""
            add(
                f"  {m:<25s}  {r.sharpe:>7.3f}  {r.cagr * 100:>6.1f}%  {r.max_drawdown * 100:>6.1f}%"
                f"  {r.annualized_turnover:>8.2f}x  {r.condition_number:>8.0f}{marker}"
            )

        # Best Sharpe / Best Condition
        best_sharpe_m = sorted_methods[0]
        best_cond_m = min(cov_comparison.keys(), key=lambda m: cov_comparison[m].condition_number)
        add()
        add(f"  Best Sharpe:    {best_sharpe_m} ({cov_comparison[best_sharpe_m].sharpe:.3f})")
        add(f"  Best Condition: {best_cond_m} ({cov_comparison[best_cond_m].condition_number:.0f})")

    # ── Factor attribution ──
    if factor_attribution is not None:
        add()
        add(SEP)
        add(f"  FACTOR ATTRIBUTION — {display_label}")
        add(SEP)
        add()

        fa = factor_attribution
        sig = "[PASS]" if fa["alpha_pvalue"] < 0.05 else "[FAIL]"
        alpha_pct = fa["alpha_annual"] * 100
        add(f"    Annualized Alpha:    {alpha_pct:>8.2f}%   (t={fa['alpha_tstat']:.2f}, p={fa['alpha_pvalue']:.3f})  {sig}")
        add(f"    R²:                  {fa['r_squared']:>8.3f}")
        add(f"    Adjusted R²:         {fa['adj_r_squared']:>8.3f}")
        add(f"    Alpha Sharpe:        {fa['alpha_sharpe']:>8.3f}")
        add(f"    Information Ratio:   {fa['information_ratio']:>8.3f}")
        add(f"    Observations:        {fa['n_observations']:>8d}")
        add()

        add(f"    {'Factor':<10s}  {'Beta':>8s}  {'t-stat':>8s}  {'p-value':>8s}  {'Attributed':>10s}")
        add(f"    {'─' * 10}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 10}")
        for f in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]:
            beta = fa["betas"][f]
            tstat = fa["beta_tstats"][f]
            pval = fa["beta_pvalues"][f]
            attr_ret = fa["factor_attributed_returns"][f]
            star = " *" if abs(tstat) > 2 else ""
            add(f"    {f:<10s}  {beta:>8.3f}  {tstat:>8.2f}  {pval:>8.3f}  {attr_ret * 100:>9.2f}%{star}")

    # ── Parameter stability ──
    if parameter_stability is not None:
        add()
        add(SEP)
        add(f"  PARAMETER STABILITY — {display_label}")
        add(SEP)
        add()
        cv_dict = parameter_stability["parameter_cv"]
        is_stable = parameter_stability["is_stable"]
        n_windows = len(parameter_stability["sharpe_timeseries"])
        add(f"    Windows evaluated:  {n_windows:>6d}")
        add()
        add(f"    {'Parameter':<20s}  {'CV':>6s}  {'Stable':>7s}")
        add(f"    {'─' * 20}  {'─' * 6}  {'─' * 7}")
        for param_name, cv in cv_dict.items():
            stable_flag = "Yes" if cv < 0.50 else "No"
            add(f"    {param_name:<20s}  {cv:>6.3f}  {stable_flag:>7s}")
        add()
        overall = "STABLE" if is_stable else "UNSTABLE"
        add(f"    Overall: {overall} (all CVs {'<' if is_stable else '>='} 0.50)")

    # ── Top drawdowns ──
    add()
    add(SEP)
    add(f"  TOP 3 DRAWDOWN PERIODS — {display_label}")
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
    logger.info("\n%s", text)

    if write_file:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(text)
        logger.info("Report saved → %s", REPORT_PATH)

    return text
