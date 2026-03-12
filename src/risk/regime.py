"""
Market regime detection.

Classifies market regimes (bull/sideways/bear) based on
rolling volatility thresholds. Computes performance metrics
segmented by regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from src.risk.metrics import compute_risk_report, RiskReport

TRADING_DAYS = 252
REGIME_WINDOW = 60  # rolling days for vol estimation


@dataclass
class RegimeStats:
    """Performance summary for a single regime."""

    name: str
    n_days: int
    fraction: float
    report: RiskReport

    def __str__(self) -> str:
        return (
            f"  Regime: {self.name.upper()}  "
            f"({self.n_days} days, {self.fraction * 100:.1f}% of period)\n"
            + str(self.report)
        )


def classify_regimes(
    daily_returns: pd.Series,
    window: int = REGIME_WINDOW,
    thresholds: dict[str, float] | None = None,
) -> pd.Series:
    """
    Classify each trading day into a market regime.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio or benchmark returns.
    window : int
        Rolling window in trading days for volatility estimation.
    thresholds : dict or None
        Volatility thresholds (annualized). Defaults to config values.
        Keys: 'bull' (upper bound for bull), 'sideways' (upper bound for sideways).

    Returns
    -------
    pd.Series
        String labels ('bull', 'sideways', 'bear') indexed like daily_returns.
        First `window` days are NaN (insufficient history).
    """
    if thresholds is None:
        thresholds = config.REGIME_VOL_THRESHOLDS

    bull_threshold = thresholds["bull"]
    sideways_threshold = thresholds["sideways"]

    rolling_vol = daily_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)

    regimes = pd.Series(index=daily_returns.index, dtype=object)
    regimes[rolling_vol < bull_threshold] = "bull"
    regimes[(rolling_vol >= bull_threshold) & (rolling_vol < sideways_threshold)] = (
        "sideways"
    )
    regimes[rolling_vol >= sideways_threshold] = "bear"

    return regimes


def regime_stats(
    daily_returns: pd.Series,
    window: int = REGIME_WINDOW,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> dict[str, RegimeStats]:
    """
    Compute risk metrics segmented by market regime.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns.
    window : int
        Rolling window for regime classification.
    risk_free_rate : float
        Annualized risk-free rate.

    Returns
    -------
    dict[str, RegimeStats]
        Keys: 'bull', 'sideways', 'bear'. Only includes regimes with data.
    """
    regimes = classify_regimes(daily_returns, window)
    valid = regimes.dropna()
    total_days = len(valid)

    results: dict[str, RegimeStats] = {}
    for name in ("bull", "sideways", "bear"):
        mask = valid == name
        regime_returns = daily_returns.loc[valid.index[mask]]

        if len(regime_returns) < 2:
            continue

        report = compute_risk_report(regime_returns, risk_free_rate=risk_free_rate)
        results[name] = RegimeStats(
            name=name,
            n_days=len(regime_returns),
            fraction=len(regime_returns) / total_days if total_days > 0 else 0.0,
            report=report,
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# HMM-based regime detection
# ──────────────────────────────────────────────────────────────────────────────


class HMMRegimeDetector:
    """
    Hidden Markov Model regime detector.

    Fits a Gaussian HMM on benchmark returns (e.g. SPY) to classify market
    regimes without look-ahead bias. Strategy performance is then segmented
    by the benchmark-derived regimes.
    """

    _LABELS_2 = ("low_vol", "high_vol")
    _LABELS_3 = ("low_vol", "mid_vol", "high_vol")

    def __init__(
        self,
        n_states: int | None = None,
        feature_window: int = 20,
        random_state: int = 42,
    ) -> None:
        self._n_states = n_states
        self._feature_window = feature_window
        self._random_state = random_state
        self._model = None
        self._fitted = False
        self._regime_series: pd.Series | None = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None
        self._state_order: np.ndarray | None = None

    # ── private helpers ──────────────────────────────────────────────────

    def _build_features(
        self, returns: pd.Series
    ) -> tuple[np.ndarray, pd.DatetimeIndex]:
        """Build feature matrix: [daily return, rolling annualised vol]."""
        rolling_vol = (
            returns.rolling(self._feature_window).std() * np.sqrt(TRADING_DAYS)
        )
        features = pd.DataFrame(
            {"ret": returns, "vol": rolling_vol}, index=returns.index
        )
        features = features.dropna()
        return features.values, features.index

    def _standardize(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Per-column z-score normalisation."""
        if fit:
            self._scaler_mean = features.mean(axis=0)
            self._scaler_std = features.std(axis=0)
            # Guard against zero-variance features
            self._scaler_std = np.where(
                self._scaler_std < 1e-10, 1.0, self._scaler_std
            )
        return (features - self._scaler_mean) / self._scaler_std

    @staticmethod
    def _compute_bic(model, X: np.ndarray) -> float:
        n, d = X.shape
        K = model.n_components
        n_params = (K - 1) + K * (K - 1) + K * d + K * d * (d + 1) // 2
        log_likelihood = model.score(X)
        return -2 * log_likelihood + n_params * np.log(n)

    def _learn_state_order(self, raw_states: np.ndarray, X: np.ndarray) -> None:
        """Compute and store state ordering by ascending mean volatility (col 1).

        Called once during fit(). The resulting ``_state_order`` is then reused
        by ``_apply_state_labels`` for all future predictions so that labels
        remain consistent across samples.
        """
        K = self._model.n_components
        vol_means = np.array(
            [X[raw_states == s, 1].mean() for s in range(K)]
        )
        self._state_order = np.argsort(vol_means)

    def _apply_state_labels(self, raw_states: np.ndarray) -> np.ndarray:
        """Map raw HMM state indices to string labels using the stored order."""
        if self._state_order is None:
            raise RuntimeError(
                "State ordering not available. Call fit() first."
            )
        K = self._model.n_components
        labels = self._LABELS_2 if K == 2 else self._LABELS_3
        mapping = {int(self._state_order[i]): labels[i] for i in range(K)}
        return np.array([mapping[s] for s in raw_states])

    # ── public API ───────────────────────────────────────────────────────

    def fit(self, benchmark_returns: pd.Series) -> "HMMRegimeDetector":
        """Fit the HMM on benchmark returns."""
        from hmmlearn.hmm import GaussianHMM

        X, valid_index = self._build_features(benchmark_returns)
        X_std = self._standardize(X, fit=True)

        def _fit_hmm(n_components: int) -> GaussianHMM:
            model = GaussianHMM(
                n_components=n_components,
                covariance_type="full",
                n_iter=100,
                random_state=self._random_state,
            )
            model.fit(X_std)
            return model

        if self._n_states is not None:
            self._model = _fit_hmm(self._n_states)
        else:
            model_2 = _fit_hmm(2)
            model_3 = _fit_hmm(3)
            bic_2 = self._compute_bic(model_2, X_std)
            bic_3 = self._compute_bic(model_3, X_std)
            self._model = model_2 if bic_2 <= bic_3 else model_3

        raw_states = self._model.predict(X_std)
        self._learn_state_order(raw_states, X_std)
        labels = self._apply_state_labels(raw_states)
        self._regime_series = pd.Series(labels, index=valid_index)
        self._fitted = True
        return self

    def predict(self, benchmark_returns: pd.Series | None = None) -> pd.Series:
        """Return regime labels. If no new returns, return fitted regimes."""
        if not self._fitted:
            raise RuntimeError(
                "HMMRegimeDetector must be fitted before calling predict()."
            )
        if benchmark_returns is None:
            return self._regime_series

        X, valid_index = self._build_features(benchmark_returns)
        X_std = self._standardize(X, fit=False)
        raw_states = self._model.predict(X_std)
        labels = self._apply_state_labels(raw_states)
        return pd.Series(labels, index=valid_index)

    def regime_performance(
        self,
        strategy_returns: pd.Series,
        risk_free_rate: float = config.RISK_FREE_RATE,
    ) -> pd.DataFrame:
        """Compute performance metrics for each regime from strategy returns."""
        regimes = self.predict()
        common_idx = regimes.index.intersection(strategy_returns.index)
        regimes = regimes.loc[common_idx]
        strat_rets = strategy_returns.loc[common_idx]
        total_days = len(common_idx)

        rows = []
        for label in sorted(regimes.unique()):
            mask = regimes == label
            regime_rets = strat_rets[mask]
            n_days = int(mask.sum())
            daily_mean = float(regime_rets.mean())
            daily_std = float(regime_rets.std())
            ann_ret = daily_mean * 252
            ann_vol = daily_std * np.sqrt(252)
            sharpe = (
                (ann_ret - risk_free_rate) / ann_vol if ann_vol > 1e-10 else 0.0
            )

            # Max drawdown from contiguous blocks
            max_dd = 0.0
            in_block = False
            block_start = None
            for i in range(len(common_idx)):
                if mask.iloc[i]:
                    if not in_block:
                        in_block = True
                        block_start = i
                elif in_block:
                    block_rets = strat_rets.iloc[block_start:i]
                    equity = (1 + block_rets).cumprod()
                    dd = (equity / equity.cummax() - 1).min()
                    if dd < max_dd:
                        max_dd = dd
                    in_block = False
            # Handle final block
            if in_block:
                block_rets = strat_rets.iloc[block_start:]
                equity = (1 + block_rets).cumprod()
                dd = (equity / equity.cummax() - 1).min()
                if dd < max_dd:
                    max_dd = dd

            rows.append(
                {
                    "n_days": n_days,
                    "fraction": n_days / total_days if total_days > 0 else 0.0,
                    "mean_return": ann_ret,
                    "volatility": ann_vol,
                    "sharpe": sharpe,
                    "max_drawdown": max_dd,
                }
            )

        labels = sorted(regimes.unique())
        df = pd.DataFrame(rows, index=labels)
        return df.sort_values("volatility", ascending=True)

    def get_transition_matrix(self) -> pd.DataFrame:
        """Return the HMM transition probability matrix, reordered by regime."""
        if not self._fitted:
            raise RuntimeError("HMMRegimeDetector must be fitted first.")
        K = self._model.n_components
        labels = self._LABELS_2 if K == 2 else self._LABELS_3
        raw_matrix = self._model.transmat_
        reordered = raw_matrix[np.ix_(self._state_order, self._state_order)]
        return pd.DataFrame(reordered, index=labels, columns=labels)

    def get_state_parameters(self) -> pd.DataFrame:
        """Return un-standardised state means (return, vol) per regime."""
        if not self._fitted:
            raise RuntimeError("HMMRegimeDetector must be fitted first.")
        K = self._model.n_components
        labels = self._LABELS_2 if K == 2 else self._LABELS_3

        raw_means = self._model.means_  # (K, d)
        real_means = raw_means * self._scaler_std + self._scaler_mean
        reordered = real_means[self._state_order]

        return pd.DataFrame(
            {
                "label": list(labels),
                "mean_return": reordered[:, 0],
                "mean_vol": reordered[:, 1],
            }
        )
