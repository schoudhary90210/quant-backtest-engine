"""Tests for risk budgeting / risk parity optimizer (Phase 4A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.risk_budget import RiskBudget

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_covariance(
    n_assets: int = 4, seed: int = 42, n_days: int = 500
) -> pd.DataFrame:
    """Synthetic covariance from random returns (annualized)."""
    rng = np.random.default_rng(seed)
    tickers = [f"A{i}" for i in range(n_assets)]
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    data = rng.normal(0, 0.01, (n_days, n_assets))
    returns = pd.DataFrame(data, index=dates, columns=tickers)
    return returns.cov() * 252


def _make_diagonal_covariance() -> pd.DataFrame:
    """Diagonal covariance with known different variances."""
    tickers = ["A", "B", "C"]
    # Annualised variances: 0.04 (20% vol), 0.01 (10% vol), 0.0025 (5% vol)
    cov = np.diag([0.04, 0.01, 0.0025])
    return pd.DataFrame(cov, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# Equal risk contribution (risk parity)
# ---------------------------------------------------------------------------


class TestEqualRiskContribution:

    def test_weights_sum_to_one(self):
        cov = _make_covariance()
        rb = RiskBudget()
        w = rb.optimize(cov)
        assert pytest.approx(w.sum(), abs=1e-8) == 1.0

    def test_weights_non_negative(self):
        cov = _make_covariance()
        rb = RiskBudget()
        w = rb.optimize(cov)
        assert (w >= -1e-10).all()

    def test_risk_contributions_approximately_equal(self):
        """Under equal risk parity, all assets should contribute equally."""
        cov = _make_diagonal_covariance()
        rb = RiskBudget()
        w = rb.optimize(cov)

        rc = rb.get_risk_contributions(w, cov)
        port_var = float(w.values @ cov.values @ w.values)
        rc_frac = rc / port_var

        # Each should be ~1/3
        np.testing.assert_allclose(rc_frac.values, 1.0 / 3, atol=0.01)

    def test_lower_vol_asset_gets_more_weight(self):
        """In risk parity, lower-vol assets get higher weight."""
        cov = _make_diagonal_covariance()
        rb = RiskBudget()
        w = rb.optimize(cov)
        # C has lowest vol (5%), A has highest (20%)
        assert w["C"] > w["B"] > w["A"]

    def test_output_index_matches_tickers(self):
        cov = _make_covariance(n_assets=3)
        cov.index = ["SPY", "TLT", "GLD"]
        cov.columns = ["SPY", "TLT", "GLD"]
        rb = RiskBudget()
        w = rb.optimize(cov)
        assert list(w.index) == ["SPY", "TLT", "GLD"]

    def test_single_asset(self):
        cov = pd.DataFrame({"X": [0.04]}, index=["X"])
        rb = RiskBudget()
        w = rb.optimize(cov)
        assert pytest.approx(w["X"]) == 1.0

    def test_empty(self):
        cov = pd.DataFrame()
        rb = RiskBudget()
        w = rb.optimize(cov)
        assert len(w) == 0


# ---------------------------------------------------------------------------
# Custom budgets
# ---------------------------------------------------------------------------


class TestCustomBudgets:

    def test_custom_budgets_approximately_matched(self):
        """Custom budgets should be approximately realised."""
        cov = _make_diagonal_covariance()
        budgets = {"A": 0.5, "B": 0.3, "C": 0.2}
        rb = RiskBudget(budgets=budgets)
        w = rb.optimize(cov)

        rc = rb.get_risk_contributions(w, cov)
        port_var = float(w.values @ cov.values @ w.values)
        rc_frac = rc / port_var

        np.testing.assert_allclose(rc_frac["A"], 0.5, atol=0.02)
        np.testing.assert_allclose(rc_frac["B"], 0.3, atol=0.02)
        np.testing.assert_allclose(rc_frac["C"], 0.2, atol=0.02)

    def test_custom_budgets_sum_to_one(self):
        cov = _make_diagonal_covariance()
        budgets = {"A": 0.5, "B": 0.3, "C": 0.2}
        rb = RiskBudget(budgets=budgets)
        w = rb.optimize(cov)
        assert pytest.approx(w.sum(), abs=1e-8) == 1.0

    def test_higher_budget_gets_higher_rc(self):
        """Asset with 50% budget should have higher risk contribution than 20%."""
        cov = _make_covariance(n_assets=3)
        tickers = list(cov.index)
        budgets = {tickers[0]: 0.5, tickers[1]: 0.3, tickers[2]: 0.2}
        rb = RiskBudget(budgets=budgets)
        w = rb.optimize(cov)
        rc = rb.get_risk_contributions(w, cov)
        assert rc[tickers[0]] > rc[tickers[2]]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestBudgetValidation:

    def test_negative_budget_raises(self):
        cov = _make_diagonal_covariance()
        budgets = {"A": -0.1, "B": 0.6, "C": 0.5}
        rb = RiskBudget(budgets=budgets)
        with pytest.raises(ValueError, match="non-negative"):
            rb.optimize(cov)

    def test_budgets_not_summing_to_one_raises(self):
        cov = _make_diagonal_covariance()
        budgets = {"A": 0.3, "B": 0.3, "C": 0.3}  # sums to 0.9
        rb = RiskBudget(budgets=budgets)
        with pytest.raises(ValueError, match="sum to 1"):
            rb.optimize(cov)

    def test_missing_ticker_in_budgets_raises(self):
        cov = _make_diagonal_covariance()
        budgets = {"A": 0.5, "B": 0.5}  # missing C
        rb = RiskBudget(budgets=budgets)
        with pytest.raises(ValueError, match="missing"):
            rb.optimize(cov)


# ---------------------------------------------------------------------------
# Risk contributions helper
# ---------------------------------------------------------------------------


class TestRiskContributions:

    def test_rc_sums_to_portfolio_variance(self):
        """Risk contributions sum to portfolio variance (w^T Cov w)."""
        cov = _make_covariance(n_assets=4)
        rb = RiskBudget()
        w = rb.optimize(cov)
        rc = rb.get_risk_contributions(w, cov)
        port_var = float(w.values @ cov.values @ w.values)
        assert pytest.approx(rc.sum(), abs=1e-10) == port_var

    def test_rc_all_non_negative(self):
        cov = _make_covariance(n_assets=4)
        rb = RiskBudget()
        w = rb.optimize(cov)
        rc = rb.get_risk_contributions(w, cov)
        # Small negative values possible near optimizer bounds
        assert (rc >= -1e-8).all()
