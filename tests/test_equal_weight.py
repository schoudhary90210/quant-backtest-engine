"""Tests for src/optimization/equal_weight.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.optimization.equal_weight import equal_weight_signal


class TestEqualWeight:

    def test_uniform_weights(self):
        date = pd.Timestamp("2020-06-15")
        prices = pd.DataFrame({"A": [100], "B": [200], "C": [300]})
        current = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})

        w = equal_weight_signal(date, prices, current)
        assert abs(w["A"] - 1.0 / 3) < 1e-10
        assert abs(w["B"] - 1.0 / 3) < 1e-10
        assert abs(w["C"] - 1.0 / 3) < 1e-10

    def test_single_asset(self):
        date = pd.Timestamp("2020-06-15")
        prices = pd.DataFrame({"SOLO": [100]})
        current = pd.Series({"SOLO": 1.0})

        w = equal_weight_signal(date, prices, current)
        assert abs(w["SOLO"] - 1.0) < 1e-10

    def test_weights_sum_to_one(self):
        date = pd.Timestamp("2020-06-15")
        tickers = [f"T{i}" for i in range(12)]
        prices = pd.DataFrame({t: [100] for t in tickers})
        current = pd.Series({t: 1.0 / 12 for t in tickers})

        w = equal_weight_signal(date, prices, current)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_empty_returns_empty(self):
        date = pd.Timestamp("2020-06-15")
        prices = pd.DataFrame()
        current = pd.Series(dtype=float)

        w = equal_weight_signal(date, prices, current)
        assert len(w) == 0
