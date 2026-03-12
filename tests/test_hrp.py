"""Tests for Hierarchical Risk Parity optimizer (Phase 4A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.hrp import HierarchicalRiskParity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(
    n_days: int = 500, n_assets: int = 5, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    data = rng.normal(0, 0.01, (n_days, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def _make_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * 252


def _make_clustered_data(
    n_days: int = 500, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create returns with two clusters:
    - Cluster 1 (A0, A1): highly correlated, high vol
    - Cluster 2 (A2, A3): highly correlated, low vol
    - A4: independent, medium vol
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    # Cluster 1: correlated high-vol
    factor1 = rng.normal(0, 0.02, n_days)
    a0 = factor1 + rng.normal(0, 0.003, n_days)
    a1 = factor1 + rng.normal(0, 0.003, n_days)

    # Cluster 2: correlated low-vol
    factor2 = rng.normal(0, 0.005, n_days)
    a2 = factor2 + rng.normal(0, 0.001, n_days)
    a3 = factor2 + rng.normal(0, 0.001, n_days)

    # Independent medium vol
    a4 = rng.normal(0, 0.01, n_days)

    data = np.column_stack([a0, a1, a2, a3, a4])
    returns = pd.DataFrame(data, index=dates, columns=["A0", "A1", "A2", "A3", "A4"])
    cov = returns.cov() * 252
    return returns, cov


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


class TestHRPBasic:

    def test_weights_sum_to_one(self):
        returns = _make_returns()
        cov = _make_covariance(returns)
        hrp = HierarchicalRiskParity()
        w = hrp.optimize(returns, cov)
        assert pytest.approx(w.sum(), abs=1e-10) == 1.0

    def test_weights_non_negative(self):
        returns = _make_returns()
        cov = _make_covariance(returns)
        hrp = HierarchicalRiskParity()
        w = hrp.optimize(returns, cov)
        assert (w >= -1e-15).all()

    def test_output_index_matches_tickers(self):
        returns = _make_returns(n_assets=4)
        returns.columns = ["SPY", "TLT", "GLD", "BTC"]
        cov = _make_covariance(returns)
        hrp = HierarchicalRiskParity()
        w = hrp.optimize(returns, cov)
        assert list(w.index) == ["SPY", "TLT", "GLD", "BTC"]

    def test_weights_differ_from_equal(self):
        """HRP should not produce exactly equal weights on nontrivial data."""
        returns = _make_returns(n_assets=5)
        cov = _make_covariance(returns)
        hrp = HierarchicalRiskParity()
        w = hrp.optimize(returns, cov)
        ew = pd.Series(1.0 / 5, index=w.index)
        assert not np.allclose(w.values, ew.values, atol=1e-4)

    def test_single_asset(self):
        returns = _make_returns(n_assets=1)
        cov = _make_covariance(returns)
        hrp = HierarchicalRiskParity()
        w = hrp.optimize(returns, cov)
        assert len(w) == 1
        assert pytest.approx(w.iloc[0]) == 1.0

    def test_two_assets(self):
        returns = _make_returns(n_assets=2)
        cov = _make_covariance(returns)
        hrp = HierarchicalRiskParity()
        w = hrp.optimize(returns, cov)
        assert len(w) == 2
        assert pytest.approx(w.sum(), abs=1e-10) == 1.0
        assert (w >= 0).all()

    def test_empty(self):
        returns = pd.DataFrame()
        cov = pd.DataFrame()
        hrp = HierarchicalRiskParity()
        w = hrp.optimize(returns, cov)
        assert len(w) == 0


# ---------------------------------------------------------------------------
# Cluster allocation
# ---------------------------------------------------------------------------


class TestHRPClusterAllocation:

    def test_low_vol_cluster_gets_more_weight(self):
        """Low-vol correlated cluster should get more combined weight
        than high-vol correlated cluster via inverse-variance bisection."""
        returns, cov = _make_clustered_data()
        hrp = HierarchicalRiskParity()
        w = hrp.optimize(returns, cov)

        high_vol_cluster = w["A0"] + w["A1"]
        low_vol_cluster = w["A2"] + w["A3"]

        assert low_vol_cluster > high_vol_cluster, (
            f"Low-vol cluster ({low_vol_cluster:.3f}) should exceed "
            f"high-vol cluster ({high_vol_cluster:.3f})"
        )


# ---------------------------------------------------------------------------
# IVP-based cluster variance regression test
# ---------------------------------------------------------------------------


class TestHRPClusterVarianceIVP:
    """Regression test: cluster variance must use inverse-variance portfolio,
    not equal weights.  A cluster with one high-var and one low-var asset has
    lower IVP variance than equal-weight variance.  This changes how allocation
    is split between clusters."""

    def test_ivp_cluster_variance_lower_than_equal_weight(self):
        """Directly verify _cluster_variance uses IVP, not equal weights."""
        # Construct a 3x3 covariance with heterogeneous diagonal
        # Asset 0: very high var (0.16), Asset 1: low var (0.01), Asset 2: medium (0.04)
        # Cluster {0, 1}: EW var = [0.5,0.5] @ [[0.16,0],[0,0.01]] @ [0.5,0.5] = 0.0425
        #                  IVP: inv_var = [1/0.16, 1/0.01] = [6.25, 100]
        #                       w = [6.25/106.25, 100/106.25] ≈ [0.0588, 0.9412]
        #                       var = 0.0588^2*0.16 + 0.9412^2*0.01 ≈ 0.0094
        cov = np.diag([0.16, 0.01, 0.04])
        indices = [0, 1]

        ivp_var = HierarchicalRiskParity._cluster_variance(cov, indices)

        # Equal-weight variance for comparison
        sub_cov = cov[np.ix_(indices, indices)]
        ew = np.array([0.5, 0.5])
        ew_var = float(ew @ sub_cov @ ew)

        assert ivp_var < ew_var, (
            f"IVP var ({ivp_var:.6f}) must be < EW var ({ew_var:.6f})"
        )
        # IVP should be substantially lower (~0.0094 vs 0.0425)
        assert ivp_var < ew_var * 0.5

    def test_heterogeneous_cluster_shifts_allocation(self):
        """When a cluster has very heterogeneous variances, IVP-based bisection
        should allocate differently than EW-based bisection.

        Setup: 4 assets in 2 clusters of 2.
        - Cluster L: one huge-var (0.64) + one tiny-var (0.0001) asset
        - Cluster R: two medium-var (0.04 each) assets
        Under EW cluster var, cluster L looks very risky → gets little weight.
        Under IVP cluster var, cluster L concentrates on the tiny-var asset → low var → gets more.
        """
        tickers = ["BIG", "TINY", "MED1", "MED2"]
        # Diagonal cov — no cross-correlation to keep clustering simple
        cov_vals = np.diag([0.64, 0.0001, 0.04, 0.04])
        cov_df = pd.DataFrame(cov_vals, index=tickers, columns=tickers)
        returns_df = pd.DataFrame(
            np.zeros((10, 4)), columns=tickers,
            index=pd.bdate_range("2020-01-01", periods=10),
        )

        hrp = HierarchicalRiskParity()
        w = hrp.optimize(returns_df, cov_df)

        # Under IVP, TINY should get substantial weight because the
        # cluster's IVP concentrates on TINY, making cluster L's var low
        assert w["TINY"] > 0.2, (
            f"TINY weight ({w['TINY']:.3f}) should be substantial under IVP"
        )
        # BIG should get very little since IVP within {BIG, TINY} gives BIG ~0.02%
        assert w["BIG"] < w["TINY"], (
            f"BIG ({w['BIG']:.3f}) should be < TINY ({w['TINY']:.3f})"
        )


# ---------------------------------------------------------------------------
# Dendrogram data
# ---------------------------------------------------------------------------


class TestHRPDendrogram:

    def test_dendrogram_populated_after_optimize(self):
        returns = _make_returns()
        cov = _make_covariance(returns)
        hrp = HierarchicalRiskParity()
        hrp.optimize(returns, cov)
        data = hrp.get_dendrogram_data()
        assert "linkage_matrix" in data
        assert "labels" in data
        assert "distance_matrix" in data
        assert data["linkage_matrix"].shape[1] == 4  # scipy linkage format

    def test_dendrogram_raises_before_optimize(self):
        hrp = HierarchicalRiskParity()
        with pytest.raises(RuntimeError):
            hrp.get_dendrogram_data()


# ---------------------------------------------------------------------------
# Linkage methods
# ---------------------------------------------------------------------------


class TestHRPLinkage:

    @pytest.mark.parametrize("method", ["ward", "single", "complete", "average"])
    def test_different_linkage_methods(self, method):
        returns = _make_returns()
        cov = _make_covariance(returns)
        hrp = HierarchicalRiskParity(linkage_method=method)
        w = hrp.optimize(returns, cov)
        assert pytest.approx(w.sum(), abs=1e-10) == 1.0
        assert (w >= -1e-15).all()
