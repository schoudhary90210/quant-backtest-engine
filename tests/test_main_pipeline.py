"""Tests for graceful failure handling in main.py pipeline."""

from __future__ import annotations

import pytest

from src.engine.costs import TransactionCostModel


def test_price_fetch_failure_exits_cleanly(monkeypatch):
    """If fetch_prices raises, main() should exit with code 1."""
    import main

    monkeypatch.setattr(
        main, "fetch_prices",
        lambda: (_ for _ in ()).throw(RuntimeError("no network")),
    )
    with pytest.raises(SystemExit) as exc_info:
        main.main()
    assert exc_info.value.code == 1


def test_volume_fetch_fallback_to_proportional(monkeypatch, caplog):
    """If fetch_volumes raises, pipeline should fall back to proportional cost model."""
    import main
    from types import SimpleNamespace
    from pathlib import Path as _Path

    _DUMMY_MC = SimpleNamespace(
        prob_profit=0.5,
        median_terminal_wealth=1_000_000,
        equity_paths=None,
    )
    _DUMMY_PATH = _Path("/dev/null")

    def _raise_no_volumes(*args, **kwargs):
        raise RuntimeError("no volumes")

    monkeypatch.setattr(main, "fetch_volumes", _raise_no_volumes)

    # Capture the cost_model passed to run_backtest
    captured_cost_models = []
    _real_run_backtest = main.run_backtest

    def _spy_run_backtest(*args, **kwargs):
        captured_cost_models.append(
            kwargs.get("cost_model", args[3] if len(args) > 3 else None)
        )
        return _real_run_backtest(*args, **kwargs)

    monkeypatch.setattr(main, "run_backtest", _spy_run_backtest)

    # Stub heavy stages to keep test fast
    monkeypatch.setattr(main, "WALK_FORWARD", False)
    monkeypatch.setattr(main.config, "ENABLE_CPCV", False)
    monkeypatch.setattr(main.config, "ENABLE_FACTOR_ATTRIBUTION", False)
    monkeypatch.setattr(main.config, "ENABLE_PARAMETER_STABILITY", False)
    monkeypatch.setattr(main.config, "REGIME_DETECTOR", "threshold")
    monkeypatch.setattr(main, "run_monte_carlo", lambda **kw: _DUMMY_MC)
    monkeypatch.setattr(main, "generate_report", lambda *a, **kw: "")

    # Stub chart functions to return dummy Path
    for chart_fn in [
        "plot_equity_curves", "plot_monte_carlo_fan", "plot_rolling_sharpe",
        "plot_drawdowns", "plot_monthly_heatmap", "plot_weight_evolution",
        "plot_return_estimator_comparison", "plot_cost_sensitivity",
        "plot_turnover_over_time",
    ]:
        monkeypatch.setattr(main, chart_fn, lambda *a, **kw: _DUMMY_PATH)

    main.main()

    # Assert all captured cost_models are TransactionCostModel (proportional)
    assert len(captured_cost_models) > 0
    assert all(
        isinstance(cm, TransactionCostModel)
        for cm in captured_cost_models
        if cm is not None
    )
    # Assert warning log mentions fallback
    assert any(
        "Falling back to proportional cost model" in r.message
        for r in caplog.records
    )
