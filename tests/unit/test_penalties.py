# tests/unit/test_penalties.py
"""Unit tests for QRE simplified penalties."""

import pytest

from qre.penalties import (
    apply_all_penalties,
    apply_divergence_penalty,
    apply_drawdown_penalty,
    apply_overtrading_penalty,
    apply_short_hold_penalty,
    check_trade_count_hard_constraint,
)


class TestTradeCountHardConstraint:
    def test_passes_with_enough_trades(self):
        passed, reason = check_trade_count_hard_constraint(100, test_trades=20)
        assert passed is True

    def test_fails_below_min_trades_year(self):
        passed, reason = check_trade_count_hard_constraint(50, test_trades=20)
        assert passed is False
        assert "trades_per_year" in reason.lower() or "80" in reason

    def test_fails_below_min_test_trades(self):
        passed, reason = check_trade_count_hard_constraint(100, test_trades=5)
        assert passed is False

    def test_none_test_trades_skips_test_check(self):
        passed, _ = check_trade_count_hard_constraint(100, test_trades=None)
        assert passed is True


class TestOvertradingPenalty:
    def test_no_penalty_below_max(self):
        result = apply_overtrading_penalty(50000.0, trades_per_year=300)
        assert result == 50000.0

    def test_penalty_above_max(self):
        result = apply_overtrading_penalty(50000.0, trades_per_year=600)
        assert result < 50000.0

    def test_max_penalty_capped(self):
        result = apply_overtrading_penalty(50000.0, trades_per_year=10000)
        assert result >= 50000.0 * 0.85  # Max 15% penalty


class TestShortHoldPenalty:
    def test_no_penalty_below_threshold(self):
        result = apply_short_hold_penalty(50000.0, short_hold_ratio=0.05)
        assert result == 50000.0

    def test_penalty_above_threshold(self):
        result = apply_short_hold_penalty(50000.0, short_hold_ratio=0.30)
        assert result < 50000.0


class TestDrawdownPenalty:
    def test_no_penalty_small_drawdown(self):
        result = apply_drawdown_penalty(50000.0, max_drawdown=-3.0)
        assert result == 50000.0

    def test_penalty_large_drawdown(self):
        result = apply_drawdown_penalty(50000.0, max_drawdown=-15.0)
        assert result < 50000.0

    def test_penalty_increases_with_drawdown(self):
        r1 = apply_drawdown_penalty(50000.0, max_drawdown=-12.0)
        r2 = apply_drawdown_penalty(50000.0, max_drawdown=-20.0)
        assert r2 < r1


class TestDivergencePenalty:
    def test_no_penalty_good_ratio(self):
        result = apply_divergence_penalty(50000.0, 48000.0, test_sharpe=1.5, test_trades=20)
        assert result > 0

    def test_hard_fail_negative_test_sharpe(self):
        result = apply_divergence_penalty(50000.0, 48000.0, test_sharpe=-1.0, test_trades=20)
        assert result == 0.0

    def test_hard_fail_too_few_test_trades(self):
        result = apply_divergence_penalty(50000.0, 48000.0, test_sharpe=1.5, test_trades=3)
        assert result == 0.0


class TestApplyAllPenalties:
    def test_returns_equity_when_no_penalties(self):
        result = apply_all_penalties(
            equity=51000.0,
            trades_per_year=150,
            short_hold_ratio=0.05,
            max_drawdown=-3.0,
            monthly_returns=[100, 200, 50, 150, -30, 80, 100, 200, -50, 100, 150, 200],
        )
        assert result > 0

    def test_returns_zero_for_hard_fail(self):
        result = apply_all_penalties(
            equity=51000.0,
            trades_per_year=30,  # Below MIN_TRADES_YEAR_HARD (80)
            short_hold_ratio=0.05,
            max_drawdown=-3.0,
            monthly_returns=[100],
        )
        assert result == 0.0

    def test_returns_reasons_when_requested(self):
        result, reasons = apply_all_penalties(
            equity=51000.0,
            trades_per_year=150,
            short_hold_ratio=0.30,  # Will trigger short hold penalty
            max_drawdown=-3.0,
            monthly_returns=[100],
            return_reasons=True,
        )
        assert isinstance(reasons, list)
        assert any("short_hold" in r for r in reasons)

    def test_divergence_hard_fail(self):
        result = apply_all_penalties(
            equity=51000.0,
            trades_per_year=150,
            short_hold_ratio=0.05,
            max_drawdown=-3.0,
            monthly_returns=[100],
            train_equity=51000.0,
            test_equity=48000.0,
            test_sharpe=-2.0,  # Negative test sharpe = hard fail
            test_trades=20,
        )
        assert result == 0.0
