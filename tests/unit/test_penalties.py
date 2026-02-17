"""Unit tests for QRE simplified penalties (post-redesign)."""

import pytest

from qre.penalties import (
    apply_all_penalties,
    apply_low_test_trades_penalty,
    apply_overtrading_penalty,
    apply_rsi_asymmetry_penalty,
    apply_sol_low_trade_penalty,
    check_trade_count_hard_constraint,
)


class TestTradeCountHardConstraint:
    def test_passes_with_enough_trades(self):
        passed, reason = check_trade_count_hard_constraint(100, test_trades=20)
        assert passed is True

    def test_fails_below_min_trades_year(self):
        passed, reason = check_trade_count_hard_constraint(20, test_trades=20)
        assert passed is False

    def test_fails_below_min_test_trades(self):
        passed, reason = check_trade_count_hard_constraint(100, test_trades=2)
        assert passed is False

    def test_none_test_trades_skips_check(self):
        passed, _ = check_trade_count_hard_constraint(100, test_trades=None)
        assert passed is True


class TestLowTestTradesPenalty:
    def test_no_penalty_above_threshold(self):
        result = apply_low_test_trades_penalty(50000.0, test_trades=20)
        assert result == 50000.0

    def test_penalty_below_threshold(self):
        result = apply_low_test_trades_penalty(50000.0, test_trades=10)
        assert result == 50000.0 * 0.85

    def test_no_penalty_at_threshold(self):
        result = apply_low_test_trades_penalty(50000.0, test_trades=15)
        assert result == 50000.0

    def test_none_test_trades_no_penalty(self):
        result = apply_low_test_trades_penalty(50000.0, test_trades=None)
        assert result == 50000.0


class TestRSIAsymmetryPenalty:
    def test_symmetric_no_penalty(self):
        params = {"rsi_lower": 30, "rsi_upper": 70}  # |30-30|=0
        result = apply_rsi_asymmetry_penalty(50000.0, params)
        assert result == 50000.0

    def test_mild_asymmetry_no_penalty(self):
        params = {"rsi_lower": 38, "rsi_upper": 72}  # |38-28|=10 < 15
        result = apply_rsi_asymmetry_penalty(50000.0, params)
        assert result == 50000.0

    def test_strong_asymmetry_penalized(self):
        params = {"rsi_lower": 20, "rsi_upper": 60}  # |20-40|=20 > 15
        result = apply_rsi_asymmetry_penalty(50000.0, params)
        assert result == 50000.0 * 0.95

    def test_no_params_no_penalty(self):
        result = apply_rsi_asymmetry_penalty(50000.0, params=None)
        assert result == 50000.0

    def test_at_threshold_no_penalty(self):
        params = {"rsi_lower": 25, "rsi_upper": 60}  # |25-40|=15 = threshold
        result = apply_rsi_asymmetry_penalty(50000.0, params)
        assert result == 50000.0


class TestOvertradingPenalty:
    def test_no_penalty_below_max(self):
        result = apply_overtrading_penalty(50000.0, trades_per_year=300)
        assert result == 50000.0

    def test_penalty_above_max(self):
        result = apply_overtrading_penalty(50000.0, trades_per_year=600)
        assert result < 50000.0

    def test_max_penalty_capped(self):
        result = apply_overtrading_penalty(50000.0, trades_per_year=10000)
        assert result >= 50000.0 * 0.85


class TestSolLowTradePenalty:
    def test_sol_below_threshold_penalized(self):
        result = apply_sol_low_trade_penalty(50000.0, 40, symbol="SOLUSDC")
        assert result == 50000.0 * 0.85

    def test_sol_above_threshold_no_penalty(self):
        result = apply_sol_low_trade_penalty(50000.0, 80, symbol="SOLUSDC")
        assert result == 50000.0

    def test_sol_at_threshold_no_penalty(self):
        result = apply_sol_low_trade_penalty(50000.0, 50, symbol="SOLUSDC")
        assert result == 50000.0

    def test_btc_below_threshold_no_penalty(self):
        result = apply_sol_low_trade_penalty(50000.0, 40, symbol="BTCUSDC")
        assert result == 50000.0

    def test_no_symbol_no_penalty(self):
        result = apply_sol_low_trade_penalty(50000.0, 40, symbol=None)
        assert result == 50000.0

    def test_case_insensitive(self):
        result = apply_sol_low_trade_penalty(50000.0, 40, symbol="sol/usdc")
        assert result == 50000.0 * 0.85


class TestNoRemovedPenalties:
    def test_no_removed_functions(self):
        """short_hold, drawdown, divergence penalties removed."""
        import qre.penalties as mod
        assert not hasattr(mod, "apply_short_hold_penalty")
        assert not hasattr(mod, "apply_drawdown_penalty")
        assert not hasattr(mod, "apply_divergence_penalty")


class TestApplyAllPenalties:
    def test_simplified_signature(self):
        result = apply_all_penalties(
            equity=51000.0,
            trades_per_year=150,
            test_trades=20,
        )
        assert result > 0

    def test_returns_zero_for_hard_fail(self):
        result = apply_all_penalties(
            equity=51000.0,
            trades_per_year=20,
            test_trades=20,
        )
        assert result == 0.0

    def test_returns_reasons(self):
        result, reasons = apply_all_penalties(
            equity=51000.0,
            trades_per_year=600,
            test_trades=20,
            return_reasons=True,
        )
        assert isinstance(reasons, list)
        assert any("overtrading" in r for r in reasons)

    def test_no_penalties_ok(self):
        result, reasons = apply_all_penalties(
            equity=51000.0,
            trades_per_year=150,
            test_trades=20,
            return_reasons=True,
        )
        assert result == 51000.0
        assert any("OK" in r for r in reasons)

    def test_sol_penalty_in_apply_all(self):
        result, reasons = apply_all_penalties(
            equity=50000.0,
            trades_per_year=45,
            test_trades=10,
            symbol="SOLUSDC",
            return_reasons=True,
        )
        # low_test_trades (-15%) + sol_low_trades (-15%) = 0.85 * 0.85
        assert result == pytest.approx(50000.0 * 0.85 * 0.85)
        assert any("sol_low_trades" in r for r in reasons)
        assert any("low_test_trades" in r for r in reasons)

    def test_btc_no_sol_penalty(self):
        result = apply_all_penalties(
            equity=50000.0,
            trades_per_year=45,
            test_trades=10,
            symbol="BTCUSDC",
        )
        # low_test_trades penalty only (no SOL penalty for BTC)
        assert result == pytest.approx(50000.0 * 0.85)
