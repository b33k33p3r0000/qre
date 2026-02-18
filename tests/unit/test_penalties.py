"""Unit tests for QRE penalties (only hard constraints remain)."""

import pytest

from qre.penalties import check_hard_constraints


class TestHardConstraints:
    def test_passes_with_enough_trades(self):
        passed, reason = check_hard_constraints(trades_per_year=100, test_trades=20)
        assert passed is True
        assert reason == "OK"

    def test_fails_below_min_trades_year(self):
        passed, reason = check_hard_constraints(trades_per_year=20, test_trades=20)
        assert passed is False
        assert "trades_per_year" in reason

    def test_fails_below_min_test_trades(self):
        passed, reason = check_hard_constraints(trades_per_year=100, test_trades=2)
        assert passed is False
        assert "test_trades" in reason

    def test_none_test_trades_skips_check(self):
        passed, _ = check_hard_constraints(trades_per_year=100, test_trades=None)
        assert passed is True


class TestSoftPenaltiesRemoved:
    def test_no_soft_penalty_functions(self):
        """All soft penalty functions removed."""
        import qre.penalties as mod
        for name in [
            "apply_low_test_trades_penalty",
            "apply_rsi_asymmetry_penalty",
            "apply_overtrading_penalty",
            "apply_sol_low_trade_penalty",
            "apply_all_penalties",
            "apply_short_hold_penalty",
            "apply_drawdown_penalty",
            "apply_divergence_penalty",
        ]:
            assert not hasattr(mod, name), f"{name} should be removed"
