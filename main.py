"""
Quantitative Backtesting & Portfolio Optimization Engine
=========================================================
Single entry point. Runs the full pipeline end-to-end:

    1. Fetch & cache price data
    2. Run backtests (Equal Weight, Half-Kelly, Full Kelly, BL + Kelly)
    3. Walk-forward out-of-sample validation  [if WALK_FORWARD=True]
    4. Statistical validation (DSR, PSR, CPCV, PBO, MinBTL)
    5. Run Monte Carlo simulation
    6. Generate all charts
    7. Generate and save text report

Flags
-----
WALK_FORWARD : bool
    Run anchored walk-forward OOS validation (adds ~2 min for 15-year data).
RETURN_ESTIMATOR : str
    "black_litterman" (default) or "historical_mean".  Controls which return
    estimator is used for the *primary* Kelly backtest and walk-forward loop.
    Both estimators are always run and compared in the return-estimator chart.

Usage:
    source venv/bin/activate && python main.py
"""

import logging
import math
from pathlib import Path

import pandas as pd

import config
from src.data.fetcher import fetch_prices, fetch_volumes
from src.engine.backtest import BacktestResult, run_backtest
from src.engine.costs import AlmgrenChrissCostModel, TransactionCostModel
from src.monte_carlo.simulation import run_monte_carlo
from src.optimization.equal_weight import equal_weight_signal
from src.optimization.black_litterman import BlackLittermanConfig
from src.optimization.signals import get_strategy_registry, make_bl_kelly_signal, make_kelly_signal
from src.optimization.risk_budget import RiskBudget
from src.optimization.covariance import get_covariance_estimator
from src.risk.metrics import compute_risk_report, sharpe_ratio, turnover_analysis
from src.validation.oos_metrics import compare_is_oos
from src.validation.walk_forward import WalkForwardConfig, run_walk_forward
from src.data.returns import compute_log_returns
from src.validation.statistical_tests import (
    cpcv_backtest,
    deflated_sharpe_ratio,
    minimum_backtest_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from src.optimization.covariance_comparison import run_covariance_backtest_comparison
from src.risk.factor_attribution import FactorAttribution
from src.visualization.charts import (
    plot_cost_breakdown,
    plot_cost_sensitivity,
    plot_covariance_backtest_comparison,
    plot_cpcv_oos_sharpe,
    plot_drawdowns,
    plot_equity_curves,
    plot_factor_exposures,
    plot_hrp_dendrogram,
    plot_is_vs_oos_equity,
    plot_leverage_timeseries,
    plot_monthly_heatmap,
    plot_monte_carlo_fan,
    plot_return_estimator_comparison,
    plot_risk_contribution_comparison,
    plot_regime_overlay,
    plot_rolling_alpha,
    plot_rolling_sharpe,
    plot_conditional_performance_heatmap,
    plot_parameter_stability,
    plot_transition_matrix,
    plot_turnover_over_time,
    plot_weight_evolution,
)
from src.visualization.report import generate_report

# ── Pipeline flags ────────────────────────────────────────────────────────
# Set WALK_FORWARD=False for fast iteration (skips the ~2-min WF loop).
WALK_FORWARD = True
# "black_litterman" uses BL posterior as expected-return input to Kelly.
# "historical_mean" uses raw annualised log-return mean (original approach).
RETURN_ESTIMATOR = "black_litterman"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 62)
    logger.info("  Quant Backtest Engine — Full Pipeline")
    logger.info("=" * 62)

    # ── 1. Data ──────────────────────────────────────────────────
    logger.info("[1/7] Fetching price data...")
    try:
        prices = fetch_prices()
    except Exception as exc:
        logger.error(
            "Failed to load price data: %s\n"
            "  Check network connectivity or verify cache exists at '%s'.",
            exc, config.CACHE_DIR,
        )
        raise SystemExit(1)
    logger.info("      %d assets × %d trading days", len(prices.columns), len(prices))

    # ── 1b. Build cost model ──────────────────────────────────────
    cost_model: TransactionCostModel | AlmgrenChrissCostModel
    if config.COST_MODEL == "almgren_chriss":
        try:
            logger.info("      Fetching volume data for Almgren-Chriss model...")
            volumes = fetch_volumes()
            # Align volumes to price dates
            common_idx = prices.index.intersection(volumes.index)
            volumes = volumes.loc[common_idx]
            prices_aligned = prices.loc[common_idx]
            cost_model = AlmgrenChrissCostModel(
                prices=prices_aligned,
                volumes=volumes,
                half_spread_bps=config.SPREAD_BPS_DEFAULT,
                eta=config.IMPACT_ETA,
                gamma=config.IMPACT_GAMMA,
                slippage_factor=config.SLIPPAGE_FACTOR,
                adv_window=config.ADV_WINDOW,
            )
            logger.info("      Cost model: Almgren-Chriss (nonlinear market impact)")
        except Exception as exc:
            logger.warning(
                "Failed to load volume data: %s\n"
                "  Falling back to proportional cost model.",
                exc,
            )
            cost_model = TransactionCostModel()
    else:
        cost_model = TransactionCostModel()
        logger.info("      Cost model: Proportional (%d + %d bps)", config.TRANSACTION_COST_BPS, config.SLIPPAGE_BPS)

    # ── 2. Backtests ─────────────────────────────────────────────
    logger.info("[2/7] Running backtests...")

    # Historical Mean + Kelly (original approach)
    hk_signal = make_kelly_signal(
        fraction=0.5, cov_method=config.COV_METHOD, max_turnover=config.MAX_TURNOVER,
    )
    fk_signal = make_kelly_signal(
        fraction=1.0, cov_method=config.COV_METHOD, max_turnover=config.MAX_TURNOVER,
    )

    # Black-Litterman + Kelly (new approach)
    bl_signal = make_bl_kelly_signal(fraction=0.5, cov_method=config.COV_METHOD, max_turnover=config.MAX_TURNOVER)

    ew_result = run_backtest(prices, equal_weight_signal, rebalance_freq="monthly", cost_model=cost_model)
    hk_result = run_backtest(prices, hk_signal, rebalance_freq="monthly", cost_model=cost_model)
    fk_result = run_backtest(prices, fk_signal, rebalance_freq="monthly", cost_model=cost_model)
    bl_result = run_backtest(prices, bl_signal, rebalance_freq="monthly", cost_model=cost_model)

    backtest_results = {
        "Equal Weight": ew_result,
        "Half-Kelly": hk_result,
        "Full Kelly": fk_result,
        "BL + Kelly": bl_result,
    }

    hk_metrics = compute_risk_report(hk_result.daily_returns)
    bl_metrics = compute_risk_report(bl_result.daily_returns)
    ew_metrics = compute_risk_report(ew_result.daily_returns)

    logger.info(
        "      Equal Weight  — CAGR %.1f%%  Sharpe %.3f",
        ew_result.cagr * 100,
        ew_metrics.sharpe_ratio,
    )
    logger.info(
        "      Half-Kelly    — CAGR %.1f%%  Sharpe %.3f",
        hk_result.cagr * 100,
        hk_metrics.sharpe_ratio,
    )
    logger.info("      Full Kelly    — CAGR %.1f%%", fk_result.cagr * 100)
    logger.info(
        "      BL + Kelly    — CAGR %.1f%%  Sharpe %.3f",
        bl_result.cagr * 100,
        bl_metrics.sharpe_ratio,
    )

    # Choose which IS result drives the walk-forward comparison
    primary_result = bl_result if RETURN_ESTIMATOR == "black_litterman" else hk_result
    display_label = "BL + Kelly" if RETURN_ESTIMATOR == "black_litterman" else "Half-Kelly"

    # ── 2a. Strategy comparison ───────────────────────────────────
    registry = get_strategy_registry(
        cov_method=config.COV_METHOD, max_turnover=config.MAX_TURNOVER,
    )

    # Reuse already-computed core results
    _precomputed = {
        "equal_weight": ew_result,
        "half_kelly": hk_result,
        "full_kelly": fk_result,
        "bl_kelly": bl_result,
    }

    strategy_results: dict[str, BacktestResult] = {}
    strategy_signals: dict[str, object] = {}

    for strat_id in config.STRATEGIES_TO_RUN:
        if strat_id not in registry:
            logger.warning("Unknown strategy ID '%s', skipping.", strat_id)
            continue
        label, signal_fn = registry[strat_id]
        strategy_signals[strat_id] = signal_fn

        if strat_id in _precomputed:
            strategy_results[label] = _precomputed[strat_id]
        else:
            logger.info("      Running backtest: %s ...", label)
            result = run_backtest(
                prices, signal_fn,
                rebalance_freq=config.REBALANCE_FREQ,
                cost_model=cost_model,
            )
            strategy_results[label] = result
            logger.info(
                "      %s — CAGR %.1f%%", label, result.cagr * 100,
            )

    # Build compact comparison summary
    strategy_comparison: dict[str, dict] = {}
    for label, result in strategy_results.items():
        report = compute_risk_report(result.daily_returns)
        ta = turnover_analysis(result)
        strategy_comparison[label] = {
            "sharpe": report.sharpe_ratio,
            "sortino": report.sortino_ratio,
            "cagr": result.cagr,
            "max_drawdown": report.max_drawdown,
            "calmar": report.calmar_ratio,
            "annualized_turnover": ta["annualized_turnover"],
        }

    # Risk contributions for advanced strategies
    _RC_STRAT_IDS = ["bl_kelly", "hrp", "risk_parity", "vol_targeted_bl"]
    rc_strat_ids = [
        s for s in _RC_STRAT_IDS
        if s in strategy_signals and registry[s][0] in strategy_results
    ]
    risk_contribs: dict[str, pd.Series] = {}

    if rc_strat_ids:
        cov_est = get_covariance_estimator(config.COV_METHOD)
        # DCC-GARCH needs a longer window
        rc_lookback = (
            max(config.LOOKBACK_WINDOW, config.DCC_GARCH_WINDOW)
            if config.COV_METHOD == "dcc_garch"
            else config.LOOKBACK_WINDOW
        )
        recent_prices = prices.iloc[-(rc_lookback + 1) :]
        recent_rets = compute_log_returns(recent_prices)
        common_cov = cov_est.estimate(recent_rets) * 252

        rb_helper = RiskBudget()
        for sid in rc_strat_ids:
            label = registry[sid][0]
            result = strategy_results[label]
            final_weights = result.positions.iloc[-1]
            rc = rb_helper.get_risk_contributions(final_weights, common_cov)
            total_var = rc.sum()
            risk_contribs[label] = rc / total_var if total_var > 1e-15 else rc * 0.0

    # ── 2b. Covariance estimator comparison ──────────────────────
    cov_comparison = None
    if config.ENABLE_COVARIANCE_COMPARISON:
        logger.info(
            "[2b/7] Running covariance estimator comparison (%d methods)...",
            len(config.COV_COMPARISON_METHODS),
        )
        cov_comparison = run_covariance_backtest_comparison(
            prices,
            methods=config.COV_COMPARISON_METHODS,
            return_estimator=RETURN_ESTIMATOR,
            kelly_fraction=0.5,
            rebalance_freq=config.REBALANCE_FREQ,
            cost_model=cost_model,
            max_turnover=config.MAX_TURNOVER,
        )
        for method, res in cov_comparison.items():
            logger.info(
                "      %-25s Sharpe %.3f  CAGR %.1f%%  CondNum %.0f",
                method, res.sharpe, res.cagr * 100, res.condition_number,
            )

    # ── 3. Walk-Forward Validation ───────────────────────────────
    wf_result = None
    if WALK_FORWARD:
        logger.info(
            "[3/7] Walk-forward OOS validation (estimator=%s)...", RETURN_ESTIMATOR
        )
        wf_result = run_walk_forward(
            prices,
            wf_config=WalkForwardConfig(),
            in_sample_result=primary_result,
            return_estimator=RETURN_ESTIMATOR,
            cost_model=cost_model,
        )
        logger.info(
            "      OOS start: %s | IS Sharpe %.3f | OOS Sharpe %.3f",
            wf_result.oos_start_date.date(),
            wf_result.in_sample_metrics.sharpe_ratio,
            wf_result.oos_metrics.sharpe_ratio,
        )
        compare_is_oos(wf_result.in_sample_metrics, wf_result.oos_metrics)

    # ── 4. Statistical Validation ──────────────────────────────────
    logger.info("[4/7] Statistical validation...")

    # Use OOS returns from walk-forward (sliced to post-oos_start_date)
    if wf_result is not None:
        oos_daily = wf_result.oos_result.daily_returns.loc[wf_result.oos_start_date:]
        ew_oos_daily = ew_result.daily_returns.loc[wf_result.oos_start_date:]
    else:
        oos_daily = primary_result.daily_returns
        ew_oos_daily = ew_result.daily_returns

    oos_sharpe = sharpe_ratio(oos_daily)
    oos_skew = float(oos_daily.skew())
    oos_kurt = float(oos_daily.kurtosis())  # excess kurtosis
    oos_nobs = len(oos_daily)

    ew_oos_sharpe = sharpe_ratio(ew_oos_daily)

    dsr_result = deflated_sharpe_ratio(
        observed_sharpe=oos_sharpe,
        n_trials=config.N_STRATEGY_TRIALS,
        n_observations=oos_nobs,
        skewness=oos_skew,
        kurtosis=oos_kurt,
    )
    logger.info(
        "      DSR: %.4f (significant: %s)", dsr_result["dsr"], dsr_result["is_significant"]
    )

    psr_result = probabilistic_sharpe_ratio(
        observed_sharpe=oos_sharpe,
        benchmark_sharpe=ew_oos_sharpe,
        n_observations=oos_nobs,
        skewness=oos_skew,
        kurtosis=oos_kurt,
    )
    logger.info(
        "      PSR: %.4f (significant: %s)", psr_result["psr"], psr_result["is_significant"]
    )

    minbtl_result = minimum_backtest_length(
        observed_sharpe=oos_sharpe,
        skewness=oos_skew,
        kurtosis=oos_kurt,
        actual_observations=oos_nobs,
    )
    logger.info(
        "      MinBTL: %.0f days (%.1f years) | Sufficient: %s",
        minbtl_result["min_btl_observations"],
        minbtl_result["min_btl_years"],
        minbtl_result["actual_length_sufficient"],
    )

    cpcv_result = None
    pbo_result = None
    if config.ENABLE_CPCV:
        log_returns = compute_log_returns(prices)

        def _cpcv_strategy_fn(train_returns: pd.DataFrame) -> pd.Series:
            """BL + Kelly weight estimation for CPCV splits."""
            from src.optimization.black_litterman import BlackLittermanEstimator
            from src.optimization.covariance import LedoitWolfCovariance
            from src.optimization.kelly import kelly_weights

            # CPCV uses Ledoit-Wolf regardless of config.COV_METHOD —
            # short training splits need a numerically stable estimator.
            cov = LedoitWolfCovariance().estimate(train_returns) * 252
            mu = BlackLittermanEstimator().estimate(train_returns, cov)
            return kelly_weights(
                mu, cov,
                risk_free_rate=config.RISK_FREE_RATE,
                fraction=config.KELLY_FRACTION,
                max_weight=config.MAX_POSITION_WEIGHT,
                max_leverage=config.MAX_LEVERAGE,
            )

        cpcv_result = cpcv_backtest(
            returns=log_returns,
            strategy_fn=_cpcv_strategy_fn,
            n_groups=config.CPCV_N_GROUPS,
            n_test_groups=config.CPCV_N_TEST_GROUPS,
            purge_window=config.CPCV_PURGE_WINDOW,
            embargo_pct=config.CPCV_EMBARGO_PCT,
            risk_free_rate=config.RISK_FREE_RATE,
        )
        logger.info(
            "      CPCV: %d splits | Mean OOS Sharpe: %.3f +/- %.3f",
            cpcv_result["n_splits"],
            cpcv_result["mean_oos_sharpe"],
            cpcv_result["std_oos_sharpe"],
        )

        pbo_result = probability_of_backtest_overfitting(cpcv_result)
        logger.info(
            "      PBO: %.3f | Rank corr: %.3f | Degradation: %.3f",
            pbo_result["pbo"],
            pbo_result["is_oos_rank_correlation"],
            pbo_result["degradation_ratio"],
        )

    stat_validation = {
        "dsr": dsr_result,
        "psr": psr_result,
        "minbtl": minbtl_result,
        "cpcv": cpcv_result,
        "pbo": pbo_result,
        "oos": wf_result is not None,
    }

    # ── 4b. Factor attribution ──────────────────────────────────
    factor_attr = None
    rolling_factor_attr = None
    if config.ENABLE_FACTOR_ATTRIBUTION:
        logger.info("[4b/7] Factor attribution (FF5 + Momentum)...")
        try:
            fa_engine = FactorAttribution()
            start_str = primary_result.equity_curve.index[0].strftime("%Y-%m-%d")
            end_str = primary_result.equity_curve.index[-1].strftime("%Y-%m-%d")
            fa_engine.fetch_factors(start_str, end_str)

            factor_attr = fa_engine.attribute(primary_result.daily_returns)
            logger.info(
                "      Alpha: %.2f%% (t=%.2f, p=%.3f) | R²: %.3f",
                factor_attr["alpha_annual"] * 100,
                factor_attr["alpha_tstat"],
                factor_attr["alpha_pvalue"],
                factor_attr["r_squared"],
            )

            rolling_factor_attr = fa_engine.rolling_attribution(
                primary_result.daily_returns,
                window=config.FACTOR_ROLLING_WINDOW,
            )
            logger.info("      Rolling attribution: %d windows", len(rolling_factor_attr))
        except Exception as exc:
            logger.warning("Factor attribution failed: %s — skipping.", exc)

    # ── 4c. Regime detection (HMM) ──────────────────────────────────
    hmm_regime_result = None
    conditional_perf = None
    if config.REGIME_DETECTOR == "hmm":
        logger.info("[4c/7] HMM regime detection (benchmark=%s)...", config.REGIME_BENCHMARK)
        try:
            from src.risk.regime import HMMRegimeDetector
            benchmark_returns = prices[config.REGIME_BENCHMARK].pct_change().dropna()
            detector = HMMRegimeDetector(
                n_states=config.HMM_N_STATES,
                feature_window=config.HMM_FEATURE_WINDOW,
                random_state=config.HMM_RANDOM_STATE,
            )
            detector.fit(benchmark_returns)
            hmm_perf = detector.regime_performance(primary_result.daily_returns)
            hmm_regime_result = {
                "performance": hmm_perf,
                "regimes": detector.predict(),
                "transition_matrix": detector.get_transition_matrix(),
                "state_params": detector.get_state_parameters(),
                "n_states": detector.predict().nunique(),
                "regime_order": list(detector.get_transition_matrix().index),
            }
            logger.info(
                "      %d-state HMM: %s",
                hmm_regime_result["n_states"],
                ", ".join(
                    f"{r}: {int(hmm_perf.loc[r, 'n_days'])}d" for r in hmm_perf.index
                ),
            )

            # ── 4c-ii. Conditional performance: all strategies by regime ──
            if strategy_results:
                regime_order = list(detector.get_transition_matrix().index)
                long_rows = []
                sharpe_dict = {}
                for strat_label, strat_result in strategy_results.items():
                    perf_df = detector.regime_performance(strat_result.daily_returns)
                    sharpe_dict[strat_label] = {
                        regime: perf_df.loc[regime, "sharpe"]
                        for regime in regime_order if regime in perf_df.index
                    }
                    for regime in regime_order:
                        if regime not in perf_df.index:
                            continue
                        row = perf_df.loc[regime]
                        long_rows.append({
                            "strategy": strat_label,
                            "regime": regime,
                            "sharpe": row["sharpe"],
                            "max_drawdown": row["max_drawdown"],
                            "n_days": int(row["n_days"]),
                            "fraction": row["fraction"],
                        })
                long_form = pd.DataFrame(long_rows)
                conditional_sharpe = pd.DataFrame(sharpe_dict).T
                conditional_sharpe = conditional_sharpe.reindex(columns=regime_order)
                conditional_perf = {
                    "long_form": long_form,
                    "conditional_sharpe": conditional_sharpe,
                    "regime_order": regime_order,
                }
                logger.info("      Conditional performance: %d strategies × %d regimes",
                            len(strategy_results), len(regime_order))
        except Exception as exc:
            logger.warning("HMM regime detection failed: %s — falling back to threshold.", exc)

    # ── 4d. Parameter stability ────────────────────────────────────
    param_stability_result = None
    if config.ENABLE_PARAMETER_STABILITY:
        logger.info("[4d/7] Parameter stability analysis...")
        try:
            from src.validation.parameter_stability import parameter_stability_analysis

            def _bl_strategy_factory(fraction, tau, view_confidence, lookback):
                bl_cfg = BlackLittermanConfig(tau=tau, view_confidence=view_confidence)
                return make_bl_kelly_signal(
                    fraction=fraction,
                    lookback=lookback,
                    cov_method=config.COV_METHOD,
                    max_turnover=config.MAX_TURNOVER,
                    bl_config=bl_cfg,
                )

            _param_grid = {
                "fraction": [0.3, 0.5, 0.7],
                "tau": [0.025, 0.05, 0.1],
                "view_confidence": [0.15, 0.25, 0.35],
                "lookback": [189, 252],
            }

            param_stability_result = parameter_stability_analysis(
                prices,
                strategy_factory=_bl_strategy_factory,
                param_grid=_param_grid,
                window_years=config.PARAM_STABILITY_WINDOW_YEARS,
                step_months=config.PARAM_STABILITY_STEP_MONTHS,
                rebalance_freq=config.REBALANCE_FREQ,
                cost_model=cost_model,
            )
            stable_tag = "STABLE" if param_stability_result["is_stable"] else "UNSTABLE"
            logger.info(
                "      Parameter stability: %s (CVs: %s)",
                stable_tag,
                ", ".join(
                    f"{k}={v:.2f}"
                    for k, v in param_stability_result["parameter_cv"].items()
                ),
            )
        except Exception as exc:
            logger.warning("Parameter stability analysis failed: %s — skipping.", exc)

    # ── 5. Monte Carlo ───────────────────────────────────────────
    logger.info(
        "[5/7] Running Monte Carlo (%d paths × %d days)...",
        config.MC_NUM_PATHS,
        config.MC_HORIZON_DAYS,
    )
    mc_result = run_monte_carlo(
        daily_returns=primary_result.daily_returns,
        n_paths=config.MC_NUM_PATHS,
        n_days=config.MC_HORIZON_DAYS,
        initial_capital=config.INITIAL_CAPITAL,
        seed=config.MC_SEED,
        store_paths=True,
    )
    logger.info(
        "      P(profit)=%.1f%%  Median terminal $%s",
        mc_result.prob_profit * 100,
        f"{mc_result.median_terminal_wealth:,.0f}",
    )

    # ── 6. Charts ────────────────────────────────────────────────
    logger.info("[6/7] Generating charts → %s", config.CHART_DIR)
    Path(config.CHART_DIR).mkdir(parents=True, exist_ok=True)

    # Core strategy equity curves (EW / HK / FK for the main comparison)
    core_equity_map = {
        "Equal Weight": ew_result.equity_curve,
        "Half-Kelly": hk_result.equity_curve,
        "Full Kelly": fk_result.equity_curve,
    }
    returns_map = {
        "Equal Weight": ew_result.daily_returns,
        "Half-Kelly": hk_result.daily_returns,
        "Full Kelly": fk_result.daily_returns,
    }

    saved = []

    p = plot_equity_curves(core_equity_map)
    saved.append(p)
    logger.info("      + %s", p.name)

    p = plot_monte_carlo_fan(mc_result.equity_paths, config.INITIAL_CAPITAL)
    saved.append(p)
    logger.info("      + %s", p.name)

    p = plot_rolling_sharpe(returns_map)
    saved.append(p)
    logger.info("      + %s", p.name)

    p = plot_drawdowns(core_equity_map)
    saved.append(p)
    logger.info("      + %s", p.name)

    p = plot_monthly_heatmap(primary_result.daily_returns, strategy_name=display_label)
    saved.append(p)
    logger.info("      + %s", p.name)

    p = plot_weight_evolution(primary_result.positions, strategy_name=display_label)
    saved.append(p)
    logger.info("      + %s", p.name)

    if wf_result is not None:
        p = plot_is_vs_oos_equity(
            wf_result.in_sample_result.equity_curve,
            wf_result.oos_result.equity_curve,
            oos_start_date=wf_result.oos_start_date,
        )
        saved.append(p)
        logger.info("      + %s", p.name)

    # Return estimator comparison: EW vs Historical Mean + Kelly vs BL + Kelly
    estimator_equity = {
        "Equal Weight": ew_result.equity_curve,
        "Historical Mean + Kelly": hk_result.equity_curve,
        "BL + Kelly": bl_result.equity_curve,
    }
    estimator_metrics = {
        "Equal Weight": (ew_result.cagr, ew_metrics.sharpe_ratio),
        "Historical Mean + Kelly": (hk_result.cagr, hk_metrics.sharpe_ratio),
        "BL + Kelly": (bl_result.cagr, bl_metrics.sharpe_ratio),
    }
    p = plot_return_estimator_comparison(estimator_equity, estimator_metrics)
    saved.append(p)
    logger.info("      + %s", p.name)

    if cov_comparison is not None:
        p = plot_covariance_backtest_comparison(cov_comparison)
        saved.append(p)
        logger.info("      + %s", p.name)

    if cpcv_result is not None:
        wf_oos_sharpe = oos_sharpe if wf_result is not None else None
        p = plot_cpcv_oos_sharpe(cpcv_result["per_split_sharpe"], wf_sharpe=wf_oos_sharpe)
        saved.append(p)
        logger.info("      + %s", p.name)

    # Turnover & cost analysis charts
    turnover_data = turnover_analysis(primary_result)
    logger.info(
        "      Turnover: %.2f×/yr | Cost drag: %.1f bps/yr",
        turnover_data["annualized_turnover"],
        turnover_data["total_cost_drag_bps"],
    )

    p = plot_cost_sensitivity(turnover_data["cost_sensitivity"])
    saved.append(p)
    logger.info("      + %s", p.name)

    p = plot_turnover_over_time(primary_result.turnover)
    saved.append(p)
    logger.info("      + %s", p.name)

    # Cost breakdown chart (Almgren-Chriss only)
    if config.COST_MODEL == "almgren_chriss" and isinstance(cost_model, AlmgrenChrissCostModel):
        # Compute cost breakdown per asset from last rebalance trade sizes
        last_rebal_dates = primary_result.turnover[primary_result.turnover > 0]
        if not last_rebal_dates.empty:
            last_date = last_rebal_dates.index[-1]
            last_weights = primary_result.positions.loc[last_date]
            # Pre-rebalance weights = positions on the day before last rebalance
            pos_idx = primary_result.positions.index
            last_loc = pos_idx.get_loc(last_date)
            if last_loc > 0:
                prev_weights = primary_result.positions.iloc[last_loc - 1]
            else:
                prev_weights = pd.Series(0.0, index=last_weights.index)
            trade_weights = last_weights - prev_weights.reindex(last_weights.index, fill_value=0.0)
            pv = primary_result.equity_curve.loc[last_date]
            cost_breakdowns = {}
            for ticker in trade_weights.index:
                tw = abs(trade_weights[ticker])
                if tw > 1e-6:
                    trade_dollars = tw * pv
                    cost_breakdowns[ticker] = cost_model.compute_cost_breakdown(
                        ticker, trade_dollars, last_date
                    )
            if cost_breakdowns:
                p = plot_cost_breakdown(cost_breakdowns)
                saved.append(p)
                logger.info("      + %s", p.name)

    # Strategy comparison charts
    if len(strategy_results) >= 2:
        all_equity = {name: res.equity_curve for name, res in strategy_results.items()}
        p = plot_equity_curves(all_equity, filename="strategy_comparison.png")
        saved.append(p)
        logger.info("      + %s", p.name)

    if "hrp" in strategy_signals and hasattr(strategy_signals["hrp"], "hrp_optimizer"):
        try:
            dendro_data = strategy_signals["hrp"].hrp_optimizer.get_dendrogram_data()
            p = plot_hrp_dendrogram(dendro_data)
            saved.append(p)
            logger.info("      + %s", p.name)
        except RuntimeError:
            pass

    if risk_contribs:
        p = plot_risk_contribution_comparison(risk_contribs)
        saved.append(p)
        logger.info("      + %s", p.name)

    if "vol_targeted_bl" in strategy_signals:
        vt_signal = strategy_signals["vol_targeted_bl"]
        if hasattr(vt_signal, "leverage_history") and vt_signal.leverage_history:
            lev_series = pd.Series(
                dict(vt_signal.leverage_history), name="leverage",
            ).sort_index()
            p = plot_leverage_timeseries(lev_series)
            saved.append(p)
            logger.info("      + %s", p.name)

    # Regime charts (HMM)
    if hmm_regime_result is not None:
        p = plot_regime_overlay(primary_result.equity_curve, hmm_regime_result["regimes"])
        saved.append(p); logger.info("      + %s", p.name)
        p = plot_transition_matrix(hmm_regime_result["transition_matrix"])
        saved.append(p); logger.info("      + %s", p.name)

    if conditional_perf is not None:
        p = plot_conditional_performance_heatmap(conditional_perf["conditional_sharpe"])
        saved.append(p); logger.info("      + %s", p.name)

    if param_stability_result is not None:
        p = plot_parameter_stability(
            param_stability_result["parameter_timeseries"],
            param_stability_result["sharpe_timeseries"],
        )
        saved.append(p); logger.info("      + %s", p.name)

    # Factor attribution charts
    if factor_attr is not None:
        p = plot_factor_exposures(factor_attr)
        saved.append(p)
        logger.info("      + %s", p.name)

    if rolling_factor_attr is not None and len(rolling_factor_attr) > 0:
        p = plot_rolling_alpha(rolling_factor_attr)
        saved.append(p)
        logger.info("      + %s", p.name)

    logger.info("      %d charts saved.", len(saved))

    # ── 7. Report ────────────────────────────────────────────────
    logger.info("[7/7] Generating report → %s", config.REPORT_PATH)
    generate_report(
        strategy_results,
        kelly_result=primary_result,
        stat_validation=stat_validation,
        turnover_data=turnover_data,
        display_label=display_label,
        cov_comparison=cov_comparison,
        strategy_comparison=strategy_comparison,
        factor_attribution=factor_attr,
        rolling_factor_attr=rolling_factor_attr,
        hmm_regime_data=hmm_regime_result,
        conditional_performance=conditional_perf,
        parameter_stability=param_stability_result,
    )

    # ── Done ─────────────────────────────────────────────────────
    logger.info("Pipeline complete.")
    logger.info("      Charts: %s", config.CHART_DIR)
    logger.info("      Report: %s", config.REPORT_PATH)


if __name__ == "__main__":
    main()
