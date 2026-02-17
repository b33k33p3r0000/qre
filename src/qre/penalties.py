"""
QRE Penalties (Simplified)
==========================
Two penalty types + per-symbol soft penalty:
  1. Trade count hard constraint (min trades/year, min test trades)
  2. Overtrading penalty (>500 trades/yr -> up to 15% penalty)
  3. SOL low-trade penalty (<60 trades/yr -> -15%)
"""

import logging
from typing import List, Optional, Tuple, Union

from qre.config import (
    MAX_TRADES_YEAR,
    MIN_TRADES_TEST_HARD,
    MIN_TRADES_YEAR_HARD,
    OVERTRADING_PENALTY,
    SOL_LOW_TRADE_PENALTY,
    SOL_MIN_TRADES_YEAR,
)

logger = logging.getLogger("qre.penalties")


def check_trade_count_hard_constraint(
    trades_per_year: float,
    test_trades: Optional[int] = None,
) -> Tuple[bool, str]:
    """Hard constraint: minimum trades per year and in test set."""
    if trades_per_year < MIN_TRADES_YEAR_HARD:
        return False, f"trades_per_year {trades_per_year:.0f} < {MIN_TRADES_YEAR_HARD}"
    if test_trades is not None and test_trades < MIN_TRADES_TEST_HARD:
        return False, f"test_trades {test_trades} < {MIN_TRADES_TEST_HARD}"
    return True, "OK"


def apply_overtrading_penalty(equity: float, trades_per_year: float) -> float:
    """Soft penalty for overtrading (>500 trades/year). Max 15% penalty."""
    if trades_per_year <= MAX_TRADES_YEAR:
        return equity
    excess = trades_per_year - MAX_TRADES_YEAR
    penalty_pct = min(excess / MAX_TRADES_YEAR * OVERTRADING_PENALTY, OVERTRADING_PENALTY)
    return equity * (1.0 - penalty_pct)


def apply_sol_low_trade_penalty(
    equity: float, trades_per_year: float, symbol: Optional[str] = None,
) -> float:
    """Soft penalty: SOL with <60 trades/year gets -15%."""
    if symbol and "SOL" in symbol.upper() and trades_per_year < SOL_MIN_TRADES_YEAR:
        return equity * (1.0 - SOL_LOW_TRADE_PENALTY)
    return equity


def apply_all_penalties(
    equity: float,
    trades_per_year: float,
    test_trades: Optional[int] = None,
    symbol: Optional[str] = None,
    return_reasons: bool = False,
) -> Union[float, Tuple[float, List[str]]]:
    """Apply all penalties. Returns penalized equity."""
    reasons: List[str] = []

    # Hard constraint
    passed, reason = check_trade_count_hard_constraint(trades_per_year, test_trades)
    if not passed:
        reasons.append(f"HARD_FAIL:trade_count:{reason}")
        if return_reasons:
            return 0.0, reasons
        return 0.0

    result = equity

    # SOL low-trade penalty
    before = result
    result = apply_sol_low_trade_penalty(result, trades_per_year, symbol)
    if result < before:
        reasons.append(f"PENALTY:sol_low_trades:{trades_per_year:.0f}_trades/year:-{SOL_LOW_TRADE_PENALTY*100:.0f}%")

    # Overtrading
    before = result
    result = apply_overtrading_penalty(result, trades_per_year)
    if result < before:
        penalty_pct = (1 - result / before) * 100
        reasons.append(f"PENALTY:overtrading:{trades_per_year:.0f}_trades/year:-{penalty_pct:.0f}%")

    if not reasons:
        reasons.append("OK:no_penalties")

    if return_reasons:
        return result, reasons
    return result
