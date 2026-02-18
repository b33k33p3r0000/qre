"""
QRE Penalties — Hard Constraints Only
======================================
Soft penalties removed in v4.2 (Sharpe objective handles risk naturally).
Only hard constraints remain: minimum trades per year and per test split.
"""

from typing import Optional, Tuple

from qre.config import MIN_TRADES_TEST_HARD, MIN_TRADES_YEAR_HARD


def check_hard_constraints(
    trades_per_year: float,
    test_trades: Optional[int] = None,
) -> Tuple[bool, str]:
    """Hard constraint: minimum trades per year and in test set.

    Returns (passed, reason) tuple.
    """
    if trades_per_year < MIN_TRADES_YEAR_HARD:
        return False, f"trades_per_year {trades_per_year:.0f} < {MIN_TRADES_YEAR_HARD}"
    if test_trades is not None and test_trades < MIN_TRADES_TEST_HARD:
        return False, f"test_trades {test_trades} < {MIN_TRADES_TEST_HARD}"
    return True, "OK"
