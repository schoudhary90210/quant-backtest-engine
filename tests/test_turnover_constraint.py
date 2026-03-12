"""Tests for turnover constraint in kelly_weights()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.kelly import kelly_weights


def _simple_inputs(
    n: int = 3,
    mu_val: float = 0.10,
    vol_val: float = 0.20,
) -> tuple[pd.Series, pd.DataFrame]:
    """Create simple expected returns and covariance for testing."""
    tickers = [f"A{i}" for i in range(n)]
    mu = pd.Series(mu_val, index=tickers)
    cov = pd.DataFrame(
        np.diag([vol_val**2] * n),
        index=tickers,
        columns=tickers,
    )
    return mu, cov


class TestTurnoverConstraint:
    def test_no_constraint_when_none(self):
        """None max_turnover = unconstrained (same as baseline)."""
        mu, cov = _simple_inputs()
        w_base = kelly_weights(mu, cov, fraction=0.5, max_leverage=1.5)
        w_none = kelly_weights(
            mu, cov, fraction=0.5, max_leverage=1.5,
            current_weights=pd.Series(0.0, index=mu.index),
            max_turnover=None,
        )
        np.testing.assert_array_almost_equal(w_base.values, w_none.values)

    def test_turnover_constraint_applied(self):
        """Tight constraint limits trades relative to unconstrained."""
        mu, cov = _simple_inputs()
        current = pd.Series(0.0, index=mu.index)

        w_unconstrained = kelly_weights(mu, cov, fraction=0.5, max_leverage=1.5)
        w_constrained = kelly_weights(
            mu, cov, fraction=0.5, max_leverage=1.5,
            current_weights=current,
            max_turnover=0.05,
        )

        unconstrained_turnover = float(np.sum(np.abs(w_unconstrained.values - current.values))) / 2
        constrained_turnover = float(np.sum(np.abs(w_constrained.values - current.values))) / 2

        # Constraint should reduce turnover if unconstrained exceeds limit
        if unconstrained_turnover > 0.05:
            assert constrained_turnover <= 0.05 + 1e-10
            assert constrained_turnover < unconstrained_turnover

    def test_turnover_within_limit(self):
        """||w_new - w_old||_1 / 2 <= max_turnover."""
        mu, cov = _simple_inputs()
        current = pd.Series([0.2, 0.3, 0.1], index=mu.index)
        max_to = 0.10

        w = kelly_weights(
            mu, cov, fraction=0.5, max_leverage=1.5,
            current_weights=current,
            max_turnover=max_to,
        )
        actual_turnover = float(np.sum(np.abs(w.values - current.values))) / 2
        assert actual_turnover <= max_to + 1e-10

    def test_constraint_preserves_direction(self):
        """Constrained weights should lie between current and unconstrained."""
        mu, cov = _simple_inputs()
        current = pd.Series(0.0, index=mu.index)

        w_unc = kelly_weights(mu, cov, fraction=0.5, max_leverage=1.5)
        w_con = kelly_weights(
            mu, cov, fraction=0.5, max_leverage=1.5,
            current_weights=current,
            max_turnover=0.05,
        )

        # Each constrained weight should be between current (0) and unconstrained
        for ticker in mu.index:
            lo = min(current[ticker], w_unc[ticker])
            hi = max(current[ticker], w_unc[ticker])
            assert w_con[ticker] >= lo - 1e-10
            assert w_con[ticker] <= hi + 1e-10

    def test_zero_turnover_keeps_current(self):
        """max_turnover=0 → no change from current weights."""
        mu, cov = _simple_inputs()
        current = pd.Series([0.2, 0.3, 0.1], index=mu.index)

        w = kelly_weights(
            mu, cov, fraction=0.5, max_leverage=1.5,
            current_weights=current,
            max_turnover=0.0,
        )
        np.testing.assert_array_almost_equal(w.values, current.values)
