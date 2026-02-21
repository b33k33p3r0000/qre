"""
Integration test: Full QRE pipeline without live API.

Uses synthetic data to verify all modules work together:
data -> strategy -> backtest -> metrics -> results.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qre.config import MIN_TRADES_YEAR_HARD, MIN_WARMUP_BARS, STARTING_EQUITY
from qre.core.backtest import simulate_trades_fast
from qre.core.metrics import aggregate_mc_results, calculate_metrics, monte_carlo_validation
from qre.core.strategy import MACDRSIStrategy
from qre.io import save_json, save_trades_csv
from qre.optimize import build_objective, compute_awf_splits
from qre.report import build_equity_curve, build_drawdown_curve, generate_report, save_report


def make_synthetic_data(n_bars: int = 5000) -> dict:
    """Create synthetic 1H OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    close = 40000 + np.cumsum(np.random.randn(n_bars) * 50)
    high = close + np.abs(np.random.randn(n_bars)) * 100
    low = close - np.abs(np.random.randn(n_bars)) * 100
    open_ = close + np.random.randn(n_bars) * 30

    return {
        "1h": pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close},
            index=dates,
        ),
    }


def _run_pipeline(data):
    """Helper: run full strategy -> backtest pipeline."""
    strategy = MACDRSIStrategy()
    params = strategy.get_default_params()
    buy, sell = strategy.precompute_signals(data, params)
    result = simulate_trades_fast("BTC/USDC", data, buy, sell)
    return strategy, params, buy, sell, result


class TestFullPipeline:
    """End-to-end: strategy -> backtest -> metrics -> penalties."""

    @pytest.fixture
    def data(self):
        return make_synthetic_data(5000)

    def test_strategy_produces_signals(self, data):
        """Strategy precompute_signals returns 1D buy/sell arrays."""
        strategy = MACDRSIStrategy()
        params = strategy.get_default_params()
        buy, sell = strategy.precompute_signals(data, params)
        assert buy.ndim == 1
        assert sell.ndim == 1
        assert len(buy) == len(data["1h"])

    def test_backtest_produces_trades(self, data):
        """Backtest with default params produces trades."""
        _, _, _, _, result = _run_pipeline(data)
        assert len(result.trades) > 0

    def test_metrics_from_backtest(self, data):
        """Metrics calculation works on backtest output."""
        _, _, _, _, result = _run_pipeline(data)
        metrics = calculate_metrics(result.trades, result.backtest_days, start_equity=STARTING_EQUITY)
        assert metrics.equity > 0
        assert metrics.trades > 0

    def test_hard_constraints_on_metrics(self, data):
        """Hard constraints pass for valid backtest output."""
        _, _, _, _, result = _run_pipeline(data)
        metrics = calculate_metrics(result.trades, result.backtest_days, start_equity=STARTING_EQUITY)
        assert metrics.trades_per_year >= MIN_TRADES_YEAR_HARD, (
            f"trades_per_year {metrics.trades_per_year:.0f} < {MIN_TRADES_YEAR_HARD}"
        )

    def test_io_save_and_load(self, data, tmp_path):
        """Results can be saved and loaded correctly."""
        _, _, _, _, result = _run_pipeline(data)

        json_path = tmp_path / "best_params.json"
        csv_path = tmp_path / "trades.csv"
        save_json(json_path, {"equity": 51000.0, "trades": len(result.trades)})
        save_trades_csv(csv_path, result.trades)

        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["equity"] == 51000.0
        assert csv_path.stat().st_size > 0

    def test_awf_splits_work_with_data(self, data):
        """AWF splits can be computed for synthetic data."""
        total_bars = len(data["1h"])
        splits = compute_awf_splits(total_bars, n_splits=3)
        assert splits is not None
        assert len(splits) == 3

    def test_objective_returns_score(self, data):
        """build_objective returns a callable that produces a float score."""
        splits = compute_awf_splits(len(data["1h"]), n_splits=2)
        objective = build_objective("BTC/USDC", data, splits)

        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10, show_progress_bar=False)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        assert len(completed) > 0


class TestReportIntegration:
    """Report generation works with real backtest output."""

    def test_report_generates_html(self, tmp_path):
        """HTML report generates with Plotly charts from real trades."""
        data = make_synthetic_data(5000)
        _, _, _, _, result = _run_pipeline(data)
        metrics = calculate_metrics(result.trades, result.backtest_days, start_equity=STARTING_EQUITY)

        report_params = {
            "symbol": "BTC/USDC",
            "equity": metrics.equity,
            "trades": metrics.trades,
            "sharpe": round(metrics.sharpe_ratio, 4),
            "max_drawdown": round(metrics.max_drawdown, 2),
            "win_rate": round(metrics.win_rate / 100, 4),
            "total_pnl_pct": round(metrics.total_pnl_pct, 2),
            "start_equity": STARTING_EQUITY,
        }
        trades_dicts = [t._asdict() if hasattr(t, '_asdict') else t for t in result.trades]

        report_path = tmp_path / "report.html"
        save_report(report_path, report_params, trades_dicts)

        assert report_path.exists()
        html = report_path.read_text()
        assert "BTC/USDC" in html
        assert "plotly" in html.lower()
        assert "equity-combo-chart" in html

    def test_equity_curve_from_real_trades(self):
        """Equity curve starts at start_equity and has correct length."""
        data = make_synthetic_data(5000)
        _, _, _, _, result = _run_pipeline(data)
        trades_dicts = [t._asdict() if hasattr(t, '_asdict') else t for t in result.trades]
        curve = build_equity_curve(trades_dicts, STARTING_EQUITY)
        assert curve[0] == STARTING_EQUITY
        assert len(curve) == len(trades_dicts) + 1

    def test_drawdown_curve_non_positive(self):
        """Drawdown values are always <= 0."""
        data = make_synthetic_data(5000)
        _, _, _, _, result = _run_pipeline(data)
        trades_dicts = [t._asdict() if hasattr(t, '_asdict') else t for t in result.trades]
        curve = build_equity_curve(trades_dicts, STARTING_EQUITY)
        dd = build_drawdown_curve(curve)
        assert all(d <= 0 for d in dd)


class TestMonteCarlo:
    def test_mc_on_synthetic_trades(self):
        """Monte Carlo validation works on synthetic trades."""
        data = make_synthetic_data(5000)
        _, _, _, _, result = _run_pipeline(data)

        if len(result.trades) >= 30:
            mc = monte_carlo_validation(result.trades, n_simulations=100, seed=42)
            assert mc.sharpe_ci_low <= mc.sharpe_ci_high
        else:
            pytest.skip("Not enough trades for MC validation")

    def test_aggregate_mc_from_splits(self):
        """Aggregate MC takes worst-case across splits."""
        data = make_synthetic_data(5000)
        _, _, _, _, result = _run_pipeline(data)

        if len(result.trades) < 30:
            pytest.skip("Not enough trades")

        trades_dicts = [t._asdict() if hasattr(t, '_asdict') else t for t in result.trades]
        mc1 = monte_carlo_validation(trades_dicts, n_simulations=100, seed=42)
        mc2 = monte_carlo_validation(trades_dicts, n_simulations=100, seed=99)
        agg = aggregate_mc_results([mc1, mc2])
        assert agg.sharpe_ci_low <= mc1.sharpe_ci_low or agg.sharpe_ci_low <= mc2.sharpe_ci_low
        assert agg.confidence_level in ("HIGH", "MEDIUM", "LOW")
