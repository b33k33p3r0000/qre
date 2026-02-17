"""Unit tests for QRE metrics."""

import numpy as np
import pytest

from qre.core.metrics import (
    MonteCarloResult,
    aggregate_mc_results,
    calculate_metrics,
    MetricsResult,
    monte_carlo_validation,
)


def make_trades(n=20, win_rate=0.6, avg_pnl=10.0):
    """Helper: generate synthetic trades."""
    trades = []
    for i in range(n):
        is_win = i < int(n * win_rate)
        pnl = abs(avg_pnl) if is_win else -abs(avg_pnl) * 0.8
        trades.append({
            "entry_ts": f"2025-01-{(i % 28) + 1:02d}T00:00:00",
            "exit_ts": f"2025-01-{(i % 28) + 1:02d}T08:00:00",
            "pnl_abs": pnl,
            "hold_bars": 8,
            "entry_bar_idx": i * 10,
            "exit_bar_idx": i * 10 + 8,
        })
    return trades


class TestCalculateMetrics:
    def test_returns_metrics_result(self):
        """calculate_metrics returns MetricsResult dataclass."""
        result = calculate_metrics(make_trades(), backtest_days=365)
        assert isinstance(result, MetricsResult)

    def test_equity_calculation(self):
        """Final equity = start_equity + sum(pnl)."""
        trades = [
            {"entry_ts": "2025-01-01T00:00:00", "exit_ts": "2025-01-01T08:00:00",
             "pnl_abs": 100.0, "hold_bars": 8, "entry_bar_idx": 0, "exit_bar_idx": 8},
            {"entry_ts": "2025-01-02T00:00:00", "exit_ts": "2025-01-02T08:00:00",
             "pnl_abs": -50.0, "hold_bars": 8, "entry_bar_idx": 24, "exit_bar_idx": 32},
        ]
        result = calculate_metrics(trades, backtest_days=365, start_equity=10000)
        assert result.equity == 10050.0
        assert result.total_pnl == 50.0

    def test_drawdown_negative(self):
        """Max drawdown should be negative or zero."""
        result = calculate_metrics(make_trades(), backtest_days=365)
        assert result.max_drawdown <= 0

    def test_win_rate(self):
        """Win rate calculated correctly."""
        trades = make_trades(n=10, win_rate=0.7)
        result = calculate_metrics(trades, backtest_days=365)
        assert result.win_rate == 70.0

    def test_no_trades_returns_zero_metrics(self):
        """Empty trades list returns zero metrics."""
        result = calculate_metrics([], backtest_days=365)
        assert result.trades == 0
        assert result.total_pnl == 0.0

    def test_sharpe_positive_for_profitable(self):
        """Sharpe ratio should be positive for consistently profitable trades."""
        trades = make_trades(n=50, win_rate=0.7, avg_pnl=20.0)
        result = calculate_metrics(trades, backtest_days=365)
        assert result.sharpe_ratio >= 0

    def test_start_equity_used_correctly(self):
        """Verify start_equity is account level, not position level."""
        trades = make_trades(n=10, win_rate=0.5, avg_pnl=100.0)
        r1 = calculate_metrics(trades, backtest_days=365, start_equity=20000)
        r2 = calculate_metrics(trades, backtest_days=365, start_equity=4000)
        # Same trades, different start_equity → different drawdown %
        # Higher start = smaller drawdown percentage
        assert abs(r1.max_drawdown) < abs(r2.max_drawdown)


class TestMonteCarloValidation:
    def test_monte_carlo_returns_result(self):
        """Monte Carlo validation returns MonteCarloResult."""
        trades = make_trades(n=50, win_rate=0.6)
        result = monte_carlo_validation(trades, n_simulations=100)
        assert hasattr(result, "sharpe_mean")

    def test_monte_carlo_confidence_intervals(self):
        """CI low <= mean <= CI high."""
        trades = make_trades(n=50, win_rate=0.6)
        result = monte_carlo_validation(trades, n_simulations=100)
        assert result.sharpe_ci_low <= result.sharpe_mean <= result.sharpe_ci_high


class TestAggregateMCResults:
    def _make_mc(self, sharpe_ci_low, confidence, robustness):
        return MonteCarloResult(
            sharpe_mean=1.0, sharpe_std=0.5,
            sharpe_ci_low=sharpe_ci_low, sharpe_ci_high=2.0,
            max_dd_mean=-5.0, max_dd_std=1.0,
            max_dd_ci_low=-8.0, max_dd_ci_high=-3.0,
            win_rate_mean=55.0, win_rate_ci_low=45.0, win_rate_ci_high=65.0,
            confidence_level=confidence, n_simulations=1000,
            robustness_score=robustness,
        )

    def test_empty_returns_insufficient_data(self):
        result = aggregate_mc_results([])
        assert result.confidence_level == "INSUFFICIENT_DATA"
        assert result.n_simulations == 0

    def test_single_result_passthrough(self):
        mc = self._make_mc(0.5, "HIGH", 0.8)
        result = aggregate_mc_results([mc])
        assert result.sharpe_ci_low == 0.5
        assert result.confidence_level == "HIGH"
        assert result.robustness_score == 0.8

    def test_takes_worst_confidence(self):
        mc1 = self._make_mc(0.5, "HIGH", 0.8)
        mc2 = self._make_mc(-0.2, "LOW", 0.3)
        result = aggregate_mc_results([mc1, mc2])
        assert result.confidence_level == "LOW"

    def test_takes_min_sharpe_ci_low(self):
        mc1 = self._make_mc(0.5, "HIGH", 0.8)
        mc2 = self._make_mc(0.1, "MEDIUM", 0.6)
        result = aggregate_mc_results([mc1, mc2])
        assert result.sharpe_ci_low == 0.1

    def test_takes_min_robustness(self):
        mc1 = self._make_mc(0.5, "HIGH", 0.8)
        mc2 = self._make_mc(0.3, "MEDIUM", 0.4)
        result = aggregate_mc_results([mc1, mc2])
        assert result.robustness_score == 0.4
