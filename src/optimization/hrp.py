"""
Hierarchical Risk Parity (HRP) portfolio optimizer.

Implements the HRP algorithm of López de Prado (2016):

1. Build a correlation-based distance matrix from the covariance matrix.
2. Hierarchical clustering (agglomerative, default: Ward linkage).
3. Quasi-diagonalize the covariance matrix using the dendrogram leaf order.
4. Recursive bisection: split allocation between left/right clusters
   inversely proportional to each cluster's inverse-variance portfolio (IVP)
   variance.

Output is a long-only, fully-invested (sum-to-1) weight vector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage


class HierarchicalRiskParity:
    """
    HRP portfolio optimizer.

    Parameters
    ----------
    linkage_method : str
        Linkage method for ``scipy.cluster.hierarchy.linkage``.
        Common choices: ``"ward"``, ``"single"``, ``"complete"``, ``"average"``.
    """

    def __init__(self, linkage_method: str = "ward") -> None:
        self._linkage_method = linkage_method
        self._dendrogram_data: dict | None = None

    def optimize(
        self,
        returns: pd.DataFrame,
        covariance: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute HRP weights.

        Parameters
        ----------
        returns : pd.DataFrame
            T x N log-return matrix (used only for its column index).
        covariance : pd.DataFrame
            N x N covariance matrix indexed by ticker.

        Returns
        -------
        pd.Series
            Long-only weights summing to 1, indexed by ticker.
        """
        tickers = covariance.columns.tolist()
        cov = covariance.values.copy()
        n = len(tickers)

        if n == 0:
            return pd.Series(dtype=float)
        if n == 1:
            return pd.Series(1.0, index=tickers)

        # ── 1. Correlation-based distance matrix ──
        vols = np.sqrt(np.diag(cov))
        vols_safe = np.where(vols > 1e-15, vols, 1e-15)
        corr = cov / np.outer(vols_safe, vols_safe)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        dist = np.sqrt(0.5 * (1.0 - corr))

        # ── 2. Hierarchical clustering ──
        # Convert distance matrix to condensed form for scipy
        condensed = dist[np.triu_indices(n, k=1)]
        link = linkage(condensed, method=self._linkage_method)

        # Store dendrogram data
        self._dendrogram_data = {
            "linkage_matrix": link,
            "labels": tickers,
            "distance_matrix": dist,
        }

        # ── 3. Quasi-diagonalize ──
        sorted_idx = leaves_list(link).tolist()

        # ── 4. Recursive bisection ──
        weights = self._recursive_bisect(cov, sorted_idx)

        result = pd.Series(0.0, index=tickers)
        for i, w in zip(sorted_idx, weights):
            result.iloc[i] = w

        return result

    def _recursive_bisect(
        self,
        cov: np.ndarray,
        sorted_idx: list[int],
    ) -> list[float]:
        """Recursive bisection using IVP-based cluster variance."""
        n = len(sorted_idx)
        if n == 1:
            return [1.0]

        # Allocate 1.0 to the full set, then recursively split
        weights = np.ones(n)
        items = list(range(n))

        # Stack-based iterative bisection
        stack = [(items, 1.0)]

        while stack:
            cluster, alloc = stack.pop()
            if len(cluster) == 1:
                weights[cluster[0]] = alloc
                continue

            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            # Allocation inversely proportional to each cluster's IVP variance
            left_var = self._cluster_variance(cov, [sorted_idx[i] for i in left])
            right_var = self._cluster_variance(cov, [sorted_idx[i] for i in right])

            total_inv_var = 1.0 / max(left_var, 1e-15) + 1.0 / max(right_var, 1e-15)
            left_alloc = (1.0 / max(left_var, 1e-15)) / total_inv_var
            right_alloc = 1.0 - left_alloc

            stack.append((left, alloc * left_alloc))
            stack.append((right, alloc * right_alloc))

        return weights.tolist()

    @staticmethod
    def _cluster_variance(cov: np.ndarray, indices: list[int]) -> float:
        """Variance of the cluster's inverse-variance portfolio (IVP).

        Weights within the cluster are proportional to 1/diag(sub_cov),
        following López de Prado (2016).  This gives lower variance than
        equal-weighting when assets in the cluster have heterogeneous risk.
        """
        sub_cov = cov[np.ix_(indices, indices)]
        diag = np.diag(sub_cov)
        # Inverse-variance weights; guard against near-zero variance
        inv_var = np.where(diag > 1e-15, 1.0 / diag, 1e15)
        ivp = inv_var / inv_var.sum()
        return float(ivp @ sub_cov @ ivp)

    def get_dendrogram_data(self) -> dict:
        """
        Return dendrogram data from the last ``optimize()`` call.

        Returns
        -------
        dict
            Keys: ``linkage_matrix``, ``labels``, ``distance_matrix``.

        Raises
        ------
        RuntimeError
            If ``optimize()`` has not been called yet.
        """
        if self._dendrogram_data is None:
            raise RuntimeError("Call optimize() before get_dendrogram_data()")
        return self._dendrogram_data
