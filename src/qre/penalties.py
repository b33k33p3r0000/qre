# src/qre/penalties.py
"""
QRE Penalties
=============
Simplified penalty system for optimization scoring.

Penalties guide Optuna away from overfitted/unreliable strategies.
5 penalty types (vs 10 in optimizer):
  1. Trade count hard constraint (min 80/yr, min 15 test)
  2. Overtrading penalty (>500 trades/yr -> up to 15% penalty)
  3. Short hold penalty (>15% short trades -> up to 25% penalty)
  4. Drawdown penalty (exceed limit -> up to 70% penalty)
  5. Divergence penalty (test vs train -> hard fail or up to 75% penalty)

Removed from optimizer: correlation, regime, quality gates, consistency.
"""

import logging
from typing import List, Optional, Tuple, Union

from qre.config import (
    DIVERGENCE_PENALTY_FACTOR,
    DRAWDOWN_PENALTY_FACTOR,
    MAX_ACCEPTABLE_DRAWDOWN,
    MAX_SHORT_HOLD_RATIO,
    MAX_TRADES_YEAR,
    MIN_TEST_SHARPE,
    MIN_TEST_TRADES,
    MIN_TEST_TRAIN_RATIO,
    MIN_TRADES_TEST_HARD,
    MIN_TRADES_YEAR_HARD,
    OVERTRADING_PENALTY,
    SHORT_HOLD_PENALTY,
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


def apply_short_hold_penalty(equity: float, short_hold_ratio: float) -> float:
    """Soft penalty if too many trades held < MIN_HOLD_BARS. Max 25% penalty."""
    if short_hold_ratio <= MAX_SHORT_HOLD_RATIO:
        return equity

    excess = short_hold_ratio - MAX_SHORT_HOLD_RATIO
    penalty_pct = min(excess / (1.0 - MAX_SHORT_HOLD_RATIO) * SHORT_HOLD_PENALTY, SHORT_HOLD_PENALTY)
    return equity * (1.0 - penalty_pct)


def apply_drawdown_penalty(equity: float, max_drawdown: float) -> float:
    """Soft penalty for excessive drawdown. Max 70% penalty.

    max_drawdown comes as negative percentage (e.g., -15.0 means 15% drawdown).
    Convert to decimal: abs(-15.0) / 100.0 = 0.15, then compare with
    MAX_ACCEPTABLE_DRAWDOWN (0.10 = 10%).
    """
    dd_pct = abs(max_drawdown) / 100.0 if abs(max_drawdown) > 1 else abs(max_drawdown)
    if dd_pct <= MAX_ACCEPTABLE_DRAWDOWN:
        return equity

    excess = dd_pct - MAX_ACCEPTABLE_DRAWDOWN
    penalty_pct = min(excess * DRAWDOWN_PENALTY_FACTOR, 0.70)
    return equity * (1.0 - penalty_pct)


def apply_divergence_penalty(
    train_equity: float,
    test_equity: float,
    test_sharpe: float,
    test_trades: int,
) -> float:
    """
    Overfitting detection: penalize if test performance diverges from train.

    Hard fails:
    - test_sharpe < 0.0 -> return 0.0
    - test_trades < 5 -> return 0.0

    Soft penalty: if test/train equity ratio < 0.50 -> up to 75% penalty.

    Returns multiplicative factor (0.0 to 1.0).
    """
    if test_sharpe < MIN_TEST_SHARPE:
        return 0.0

    if test_trades < MIN_TEST_TRADES:
        return 0.0

    if train_equity <= 0:
        return 0.0

    ratio = test_equity / train_equity
    if ratio >= MIN_TEST_TRAIN_RATIO:
        return 1.0

    shortfall = MIN_TEST_TRAIN_RATIO - ratio
    penalty_pct = min(shortfall * DIVERGENCE_PENALTY_FACTOR, 0.75)
    return 1.0 - penalty_pct


def apply_all_penalties(
    equity: float,
    trades_per_year: float,
    short_hold_ratio: float,
    max_drawdown: float,
    monthly_returns: List[float],
    train_equity: Optional[float] = None,
    test_equity: Optional[float] = None,
    test_sharpe: Optional[float] = None,
    test_trades: Optional[int] = None,
    return_reasons: bool = False,
) -> Union[float, Tuple[float, List[str]]]:
    """
    Apply all penalties in sequence. Returns penalized equity.

    Order:
    0. Hard constraint (trade count) -> 0.0 if failed
    1. Overtrading penalty
    2. Short hold penalty
    3. Drawdown penalty
    4. Divergence penalty (can hard fail)
    """
    reasons: List[str] = []

    # 0. Hard constraint
    passed, reason = check_trade_count_hard_constraint(trades_per_year, test_trades)
    if not passed:
        reasons.append(f"HARD_FAIL:trade_count:{reason}")
        if return_reasons:
            return 0.0, reasons
        return 0.0

    result = equity

    # 1. Overtrading
    before = result
    result = apply_overtrading_penalty(result, trades_per_year)
    if result < before:
        penalty_pct = (1 - result / before) * 100
        reasons.append(f"PENALTY:overtrading:{trades_per_year:.0f}_trades/year:-{penalty_pct:.0f}%")

    # 2. Short hold
    before = result
    result = apply_short_hold_penalty(result, short_hold_ratio)
    if result < before:
        penalty_pct = (1 - result / before) * 100
        reasons.append(f"PENALTY:short_hold:{short_hold_ratio * 100:.0f}%_ratio:-{penalty_pct:.0f}%")

    # 3. Drawdown
    before = result
    result = apply_drawdown_penalty(result, max_drawdown)
    if result < before:
        penalty_pct = (1 - result / before) * 100
        reasons.append(f"PENALTY:drawdown:{max_drawdown:.1f}%_DD:-{penalty_pct:.0f}%")

    # 4. Divergence
    if all(x is not None for x in [train_equity, test_equity, test_sharpe, test_trades]):
        factor = apply_divergence_penalty(train_equity, test_equity, test_sharpe, test_trades)
        if factor == 0.0:
            if test_sharpe < MIN_TEST_SHARPE:
                reasons.append(f"HARD_FAIL:test_sharpe:{test_sharpe:.2f}<{MIN_TEST_SHARPE}")
            elif test_trades < MIN_TEST_TRADES:
                reasons.append(f"HARD_FAIL:test_trades:{test_trades}<{MIN_TEST_TRADES}")
            if return_reasons:
                return 0.0, reasons
            return 0.0
        elif factor < 1.0:
            ratio = test_equity / train_equity if train_equity > 0 else 0
            penalty_pct = (1 - factor) * 100
            reasons.append(f"PENALTY:divergence:test/train={ratio:.2f}:-{penalty_pct:.0f}%")
        result = result * factor

    if not reasons:
        reasons.append("OK:no_penalties")

    if return_reasons:
        return result, reasons
    return result
