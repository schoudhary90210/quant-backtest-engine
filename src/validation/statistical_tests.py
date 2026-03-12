"""
Statistical validation tests for backtesting.

Implements five tests from the López de Prado literature to detect
overfitting and quantify the statistical significance of backtest results:

- Deflated Sharpe Ratio (DSR) — Bailey & López de Prado (2014)
- Probabilistic Sharpe Ratio (PSR) — Bailey & López de Prado (2012)
- Combinatorial Purged Cross-Validation (CPCV) — López de Prado (2018)
- Probability of Backtest Overfitting (PBO) — Bailey et al. (2017)
- Minimum Backtest Length (MinBTL) — Bailey & López de Prado

References
----------
Bailey, D. H. & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier.
Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio.
Bailey, D. H. et al. (2017). The Probability of Backtest Overfitting.
López de Prado, M. (2018). Advances in Financial Machine Learning.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
_EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sr_std(sr: float, skewness: float, kurtosis: float, n_obs: int) -> float:
    """Standard error of the **per-period** Sharpe ratio estimator.

    The Bailey & López de Prado formula requires the per-period (e.g. daily)
    Sharpe ratio — callers must de-annualise before calling this function.

    Parameters
    ----------
    sr : float
        Observed **per-period** Sharpe ratio (e.g. daily mean / daily std).
    skewness : float
        Skewness of the return series.
    kurtosis : float
        **Excess** kurtosis (normal = 0).  Converted internally to standard
        kurtosis (normal = 3) for the Bailey & López de Prado formula.
    n_obs : int
        Number of return observations.

    Returns
    -------
    float
        Standard deviation of the per-period SR estimator.
    """
    if n_obs < 2:
        return 0.0
    # Convert excess kurtosis to standard kurtosis, then apply (std_kurt - 1)/4
    std_kurt = kurtosis + 3.0
    variance = (1.0 - skewness * sr + (std_kurt - 1.0) / 4.0 * sr ** 2) / (n_obs - 1)
    if variance < 0:
        return 0.0
    return math.sqrt(variance)


def _annualised_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio from daily returns."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / TRADING_DAYS
    vol = excess.std()
    if vol == 0.0:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(TRADING_DAYS))


# ---------------------------------------------------------------------------
# 1. Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float,
    kurtosis: float,
    periods_per_year: int = 252,
) -> dict[str, float | bool]:
    """Deflated Sharpe Ratio — Bailey & López de Prado (2014).

    Corrects the observed Sharpe ratio for selection bias when multiple
    strategy variants have been tested.

    Parameters
    ----------
    observed_sharpe : float
        Observed Sharpe ratio (annualised by default).
    n_trials : int
        Number of independent strategy trials tested.
    n_observations : int
        Number of return observations (e.g. trading days).
    skewness : float
        Skewness of the return series.
    kurtosis : float
        Excess kurtosis of the return series (normal = 0).
    periods_per_year : int
        Observation frequency.  Set to 252 (default) when ``observed_sharpe``
        is annualised.  Set to 1 when ``observed_sharpe`` is already per-period.

    Returns
    -------
    dict
        ``dsr`` — probability that the true Sharpe exceeds zero after
        correcting for multiple testing.
        ``expected_max_sr`` — expected maximum annualised Sharpe under the null.
        ``sr_std`` — standard error of the annualised SR estimator.
        ``is_significant`` — True if dsr > 0.95.
    """
    # Convert to per-period SR for the Bailey & López de Prado formula
    sqrt_ppy = math.sqrt(periods_per_year)
    sr_pp = observed_sharpe / sqrt_ppy if periods_per_year > 1 else observed_sharpe

    # Per-period standard error of SR estimator
    sr_std_pp = _sr_std(sr_pp, skewness, kurtosis, n_observations)

    # Expected maximum SR under the null (all strategies have true SR = 0)
    # E[max(SR̂)] = σ(SR̂) · z_max, where z_max = E[max of N standard normals]
    if n_trials <= 1:
        expected_max_sr_pp = 0.0
    else:
        n = float(n_trials)
        z_max = (
            (1.0 - _EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n)
            + _EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n * math.e))
        )
        expected_max_sr_pp = sr_std_pp * z_max

    if sr_std_pp > 0:
        dsr = float(norm.cdf((sr_pp - expected_max_sr_pp) / sr_std_pp))
    else:
        dsr = 0.5

    return {
        "dsr": dsr,
        "expected_max_sr": expected_max_sr_pp * sqrt_ppy,
        "sr_std": sr_std_pp * sqrt_ppy,
        "is_significant": dsr > 0.95,
    }


# ---------------------------------------------------------------------------
# 2. Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_observations: int,
    skewness: float,
    kurtosis: float,
    periods_per_year: int = 252,
) -> dict[str, float | bool]:
    """Probabilistic Sharpe Ratio — Bailey & López de Prado (2012).

    Probability that the true Sharpe ratio exceeds a benchmark, accounting
    for non-normal returns.

    Parameters
    ----------
    observed_sharpe : float
        Observed Sharpe ratio (annualised by default).
    benchmark_sharpe : float
        Benchmark Sharpe ratio (same units as ``observed_sharpe``).
    n_observations : int
        Number of return observations.
    skewness : float
        Skewness of the return series.
    kurtosis : float
        Excess kurtosis of the return series (normal = 0).
    periods_per_year : int
        Observation frequency.  Set to 252 (default) when Sharpe ratios
        are annualised.  Set to 1 when already per-period.

    Returns
    -------
    dict
        ``psr`` — probability that the true Sharpe exceeds the benchmark.
        ``sr_std`` — standard error of the annualised SR estimator.
        ``is_significant`` — True if psr > 0.95.
    """
    sqrt_ppy = math.sqrt(periods_per_year)
    sr_pp = observed_sharpe / sqrt_ppy if periods_per_year > 1 else observed_sharpe
    bench_pp = benchmark_sharpe / sqrt_ppy if periods_per_year > 1 else benchmark_sharpe

    sr_std_pp = _sr_std(sr_pp, skewness, kurtosis, n_observations)

    if sr_std_pp > 0:
        psr = float(norm.cdf((sr_pp - bench_pp) / sr_std_pp))
    else:
        psr = 0.5

    return {
        "psr": psr,
        "sr_std": sr_std_pp * sqrt_ppy,
        "is_significant": psr > 0.95,
    }


# ---------------------------------------------------------------------------
# 3. Combinatorial Purged Cross-Validation
# ---------------------------------------------------------------------------


def cpcv_backtest(
    returns: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame], pd.Series],
    n_groups: int = 10,
    n_test_groups: int = 2,
    purge_window: int = 5,
    embargo_pct: float = 0.01,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Combinatorial Purged Cross-Validation — López de Prado (2018).

    Generates combinatorial train/test splits with purging and embargo
    to prevent information leakage in time-series backtesting.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns (DatetimeIndex × tickers).
    strategy_fn : callable
        ``strategy_fn(train_returns) -> pd.Series`` of portfolio weights.
    n_groups : int
        Number of chronological blocks.
    n_test_groups : int
        Number of groups held out for testing per split.
    purge_window : int
        Trading days purged from training at train/test boundaries.
    embargo_pct : float
        Fraction of total observations embargoed after test groups.

    Returns
    -------
    dict
        ``oos_returns`` — list of OOS return Series per split.
        ``per_split_sharpe`` — OOS Sharpe per split (np.ndarray).
        ``per_split_is_sharpe`` — IS Sharpe per split (np.ndarray).
        ``mean_oos_sharpe`` — mean OOS Sharpe across splits.
        ``std_oos_sharpe`` — std OOS Sharpe across splits.
        ``n_splits`` — number of completed splits.
    """
    T = len(returns)
    embargo_size = int(math.floor(embargo_pct * T))

    # Split into chronological groups
    group_boundaries = np.array_split(np.arange(T), n_groups)

    all_combos = list(combinations(range(n_groups), n_test_groups))
    logger.info(
        "CPCV: %d splits (C(%d,%d)), T=%d, purge=%d, embargo=%d",
        len(all_combos),
        n_groups,
        n_test_groups,
        T,
        purge_window,
        embargo_size,
    )

    oos_returns_list: list[pd.Series] = []
    oos_sharpes: list[float] = []
    is_sharpes: list[float] = []

    for combo in all_combos:
        test_set = set(combo)

        # Build test indices
        test_idx: list[int] = []
        for g in combo:
            test_idx.extend(group_boundaries[g].tolist())

        # Build train indices with purge and embargo
        train_idx: list[int] = []
        for g in range(n_groups):
            if g in test_set:
                continue
            indices = group_boundaries[g].tolist()

            # Purge: remove rows from end of this group if next group is test
            if g + 1 < n_groups and (g + 1) in test_set:
                indices = indices[: max(0, len(indices) - purge_window)]

            # Purge: remove rows from start of this group if previous group is test
            if g - 1 >= 0 and (g - 1) in test_set:
                indices = indices[purge_window:]

            # Embargo: remove rows from start of this group if previous group is test
            if g - 1 >= 0 and (g - 1) in test_set and embargo_size > 0:
                indices = indices[embargo_size:]

            train_idx.extend(indices)

        # Skip if training set too small
        if len(train_idx) < 30:
            logger.warning("CPCV split %s: training set too small (%d), skipping", combo, len(train_idx))
            continue

        train_returns = returns.iloc[train_idx]
        test_returns = returns.iloc[test_idx]

        # Run strategy on training data to get weights
        try:
            weights = strategy_fn(train_returns)
        except Exception:
            logger.warning("CPCV split %s: strategy_fn failed, skipping", combo)
            continue

        # Align weights with available columns
        common = weights.index.intersection(test_returns.columns)
        if len(common) == 0:
            continue
        weights = weights[common]
        w_sum = weights.sum()
        if w_sum == 0:
            continue

        # Compute portfolio returns
        oos_port = (test_returns[common] * weights).sum(axis=1)
        is_port = (train_returns[common] * weights).sum(axis=1)

        oos_returns_list.append(oos_port)
        oos_sharpes.append(_annualised_sharpe(oos_port, risk_free_rate))
        is_sharpes.append(_annualised_sharpe(is_port, risk_free_rate))

    oos_sharpe_arr = np.array(oos_sharpes)
    is_sharpe_arr = np.array(is_sharpes)

    return {
        "oos_returns": oos_returns_list,
        "per_split_sharpe": oos_sharpe_arr,
        "per_split_is_sharpe": is_sharpe_arr,
        "mean_oos_sharpe": float(oos_sharpe_arr.mean()) if len(oos_sharpe_arr) > 0 else 0.0,
        "std_oos_sharpe": float(oos_sharpe_arr.std()) if len(oos_sharpe_arr) > 0 else 0.0,
        "n_splits": len(oos_sharpes),
    }


# ---------------------------------------------------------------------------
# 4. Probability of Backtest Overfitting
# ---------------------------------------------------------------------------


def probability_of_backtest_overfitting(
    cpcv_results: dict[str, Any],
) -> dict[str, float]:
    """Probability of Backtest Overfitting — Bailey et al. (2017).

    Quantifies overfitting risk using CPCV split results.  For a single
    strategy, PBO is the fraction of splits where OOS Sharpe < 0.

    Parameters
    ----------
    cpcv_results : dict
        Output from :func:`cpcv_backtest`.

    Returns
    -------
    dict
        ``pbo`` — fraction of splits with negative OOS Sharpe.
        ``is_oos_rank_correlation`` — Spearman rank correlation between
        IS and OOS Sharpe across splits.
        ``degradation_ratio`` — mean(OOS Sharpe / IS Sharpe).
    """
    oos = cpcv_results["per_split_sharpe"]
    is_sr = cpcv_results["per_split_is_sharpe"]

    n = len(oos)
    if n == 0:
        return {"pbo": 1.0, "is_oos_rank_correlation": 0.0, "degradation_ratio": 0.0}

    pbo = float((oos < 0).mean())

    # Spearman rank correlation
    if n >= 3:
        corr, _ = spearmanr(is_sr, oos)
        rank_corr = float(corr) if np.isfinite(corr) else 0.0
    else:
        rank_corr = 0.0

    # Degradation ratio: mean of OOS/IS for splits where IS != 0
    nonzero_mask = is_sr != 0
    if nonzero_mask.any():
        degradation = float(np.mean(oos[nonzero_mask] / is_sr[nonzero_mask]))
    else:
        degradation = 0.0

    return {
        "pbo": pbo,
        "is_oos_rank_correlation": rank_corr,
        "degradation_ratio": degradation,
    }


# ---------------------------------------------------------------------------
# 5. Minimum Backtest Length
# ---------------------------------------------------------------------------


def minimum_backtest_length(
    observed_sharpe: float,
    skewness: float,
    kurtosis: float,
    alpha: float = 0.05,
    actual_observations: int | None = None,
    periods_per_year: int = 252,
) -> dict[str, float | bool | None]:
    """Minimum Backtest Length — Bailey & López de Prado.

    Minimum track record length required for a Sharpe ratio to be
    statistically significant at the given confidence level.

    Parameters
    ----------
    observed_sharpe : float
        Observed Sharpe ratio (annualised by default).
    skewness : float
        Skewness of the return series.
    kurtosis : float
        Excess kurtosis (normal = 0).
    alpha : float
        Significance level (default 0.05 for 95% confidence).
    actual_observations : int or None
        Actual number of observations.  If provided, the result includes
        whether the actual length is sufficient.
    periods_per_year : int
        Observation frequency.  Set to 252 (default) when ``observed_sharpe``
        is annualised.  Set to 1 when already per-period.

    Returns
    -------
    dict
        ``min_btl_observations`` — minimum observations required.
        ``min_btl_years`` — minimum years (assuming ``periods_per_year``
        trading days per year).
        ``actual_length_sufficient`` — bool or None.
    """
    if observed_sharpe == 0:
        return {
            "min_btl_observations": float("inf"),
            "min_btl_years": float("inf"),
            "actual_length_sufficient": False if actual_observations is not None else None,
        }

    # Convert to per-period SR for the Bailey & López de Prado formula
    sr_pp = observed_sharpe / math.sqrt(periods_per_year) if periods_per_year > 1 else observed_sharpe

    z_alpha = float(norm.ppf(1.0 - alpha))

    # Convert excess kurtosis to standard kurtosis
    std_kurt = kurtosis + 3.0

    sr_var_factor = 1.0 - skewness * sr_pp + (std_kurt - 1.0) / 4.0 * sr_pp ** 2
    min_btl = 1.0 + sr_var_factor * (z_alpha / sr_pp) ** 2

    min_btl_years = min_btl / periods_per_year

    if actual_observations is not None:
        sufficient: bool | None = actual_observations >= min_btl
    else:
        sufficient = None

    return {
        "min_btl_observations": min_btl,
        "min_btl_years": min_btl_years,
        "actual_length_sufficient": sufficient,
    }
