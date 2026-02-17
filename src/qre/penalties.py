"""
QRE Penalties (Simplified)
==========================
Penalty types:
  1. Trade count hard constraint (min trades/year, min test trades)
  2. Low test trades soft penalty (<15 test trades -> -15%)
  3. RSI asymmetry penalty (asymmetry > 15 -> -5%)
  4. Overtrading penalty (>500 trades/yr -> up to 15% penalty)
  5. SOL low-trade penalty (<60 trades/yr -> -15%)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from qre.config import (
    LOW_TEST_TRADES_PENALTY,
    MAX_TRADES_YEAR,
    MIN_TRADES_TEST_HARD,
    MIN_TRADES_TEST_SOFT,
    MIN_TRADES_YEAR_HARD,
    OVERTRADING_PENALTY,
    RSI_ASYMMETRY_PENALTY,
    RSI_ASYMMETRY_THRESHOLD,
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


def apply_low_test_trades_penalty(equity: float, test_trades: Optional[int] = None) -> float:
    """Soft penalty: test splits with <15 trades get -15% (Sharpe unreliable)."""
    if test_trades is not None and test_trades < MIN_TRADES_TEST_SOFT:
        return equity * (1.0 - LOW_TEST_TRADES_PENALTY)
    return equity


def apply_rsi_asymmetry_penalty(equity: float, params: Optional[Dict[str, Any]] = None) -> float:
    """Soft penalty: RSI zones with asymmetry > 15 get -5% (long/short bias)."""
    if params is None:
        return equity
    rsi_lower = params.get("rsi_lower")
    rsi_upper = params.get("rsi_upper")
    if rsi_lower is None or rsi_upper is None:
        return equity
    asymmetry = abs(rsi_lower - (100 - rsi_upper))
    if asymmetry > RSI_ASYMMETRY_THRESHOLD:
        return equity * (1.0 - RSI_ASYMMETRY_PENALTY)
    return equity


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
    params: Optional[Dict[str, Any]] = None,
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

    # Low test trades penalty
    before = result
    result = apply_low_test_trades_penalty(result, test_trades)
    if result < before:
        reasons.append(f"PENALTY:low_test_trades:{test_trades}_trades<{MIN_TRADES_TEST_SOFT}:-{LOW_TEST_TRADES_PENALTY*100:.0f}%")

    # RSI asymmetry penalty
    before = result
    result = apply_rsi_asymmetry_penalty(result, params)
    if result < before:
        asym = abs(params["rsi_lower"] - (100 - params["rsi_upper"]))
        reasons.append(f"PENALTY:rsi_asymmetry:{asym:.0f}>{RSI_ASYMMETRY_THRESHOLD}:-{RSI_ASYMMETRY_PENALTY*100:.0f}%")

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
