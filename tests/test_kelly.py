"""Tests for src/optimization/kelly.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.kelly import kelly_weights


class TestKellyBasic:

    def test_single_asset(self):
        """Single asset: f* = (mu - r) / sigma^2."""
        mu = pd.Series({"A": 0.10})
        cov = pd.DataFrame({"A": [0.04]}, index=["A"])  # vol = 20%
        w = kelly_weights(mu, cov, risk_free_rate=0.04, fraction=1.0,
                          max_weight=10.0, max_leverage=10.0)
        # f* = (0.10 - 0.04) / 0.04 = 1.5
        assert abs(w["A"] - 1.5) < 1e-10

    def test_half_kelly_halves_weight(self):
        mu = pd.Series({"A": 0.10})
        cov = pd.DataFrame({"A": [0.04]}, index=["A"])
        full = kelly_weights(mu, cov, risk_free_rate=0.04, fraction=1.0,
                             max_weight=10.0, max_leverage=100.0)
        half = kelly_weights(mu, cov, risk_free_rate=0.04, fraction=0.5,
                             max_weight=10.0, max_leverage=100.0)
        assert abs(half["A"] - full["A"] * 0.5) < 1e-10

    def test_quarter_kelly(self):
        mu = pd.Series({"A": 0.10})
        cov = pd.DataFrame({"A": [0.04]}, index=["A"])
        full = kelly_weights(mu, cov, risk_free_rate=0.04, fraction=1.0,
                             max_weight=10.0, max_leverage=100.0)
        quarter = kelly_weights(mu, cov, risk_free_rate=0.04, fraction=0.25,
                                max_weight=10.0, max_leverage=100.0)
        assert abs(quarter["A"] - full["A"] * 0.25) < 1e-10

    def test_two_uncorrelated_assets(self):
        """Two uncorrelated assets with same stats → equal weights."""
        mu = pd.Series({"A": 0.10, "B": 0.10})
        cov = pd.DataFrame(
            np.diag([0.04, 0.04]), index=["A", "B"], columns=["A", "B"]
        )
        w = kelly_weights(mu, cov, risk_free_rate=0.04, fraction=1.0,
                          max_weight=10.0, max_leverage=100.0)
        assert abs(w["A"] - w["B"]) < 1e-10


class TestKellyConstraints:

    def test_position_cap(self):
        mu = pd.Series({"A": 0.20})
        cov = pd.DataFrame({"A": [0.01]}, index=["A"])  # vol = 10%
        # Raw Kelly = (0.20 - 0.04) / 0.01 = 16.0 → should be capped at 0.40
        w = kelly_weights(mu, cov, risk_free_rate=0.04, fraction=1.0,
                          max_weight=0.40, max_leverage=100.0)
        assert abs(w["A"] - 0.40) < 1e-10

    def test_leverage_cap(self):
        mu = pd.Series({"A": 0.20, "B": 0.20, "C": 0.20})
        cov = pd.DataFrame(
            np.diag([0.01, 0.01, 0.01]),
            index=["A", "B", "C"], columns=["A", "B", "C"],
        )
        w = kelly_weights(mu, cov, risk_free_rate=0.04, fraction=1.0,
                          max_weight=10.0, max_leverage=1.5)
        gross = w.abs().sum()
        assert gross <= 1.5 + 1e-10

    def test_negative_excess_return_gets_zero(self):
        """Assets with mu < risk_free should get zero weight."""
        mu = pd.Series({"GOOD": 0.10, "BAD": 0.02})
        cov = pd.DataFrame(
            np.diag([0.04, 0.04]),
            index=["GOOD", "BAD"], columns=["GOOD", "BAD"],
        )
        w = kelly_weights(mu, cov, risk_free_rate=0.04, fraction=0.5)
        assert w["BAD"] == 0.0
        assert w["GOOD"] > 0.0

    def test_all_negative_excess_returns_zero(self):
        mu = pd.Series({"A": 0.01, "B": 0.02})
        cov = pd.DataFrame(
            np.diag([0.04, 0.04]),
            index=["A", "B"], columns=["A", "B"],
        )
        w = kelly_weights(mu, cov, risk_free_rate=0.04, fraction=0.5)
        assert (w == 0.0).all()

    def test_empty_returns_empty(self):
        mu = pd.Series(dtype=float)
        cov = pd.DataFrame()
        w = kelly_weights(mu, cov, risk_free_rate=0.04, fraction=0.5)
        assert len(w) == 0


class TestKellySingularCovariance:

    def test_singular_covariance_falls_back(self):
        """Perfectly correlated assets produce a singular matrix."""
        mu = pd.Series({"A": 0.10, "B": 0.10})
        # Singular: rows are identical
        cov = pd.DataFrame(
            [[0.04, 0.04], [0.04, 0.04]],
            index=["A", "B"], columns=["A", "B"],
        )
        # Should not raise — falls back to lstsq
        w = kelly_weights(mu, cov, risk_free_rate=0.04, fraction=0.5)
        assert isinstance(w, pd.Series)
        assert len(w) == 2
        # Weights should be finite
        assert np.isfinite(w).all()

    def test_near_singular_stable(self):
        """Near-singular matrix should still produce reasonable weights."""
        mu = pd.Series({"A": 0.10, "B": 0.10})
        cov = pd.DataFrame(
            [[0.04, 0.0399], [0.0399, 0.04]],
            index=["A", "B"], columns=["A", "B"],
        )
        w = kelly_weights(mu, cov, risk_free_rate=0.04, fraction=0.5)
        assert np.isfinite(w).all()
        assert (w >= 0).all()
