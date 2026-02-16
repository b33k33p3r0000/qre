"""
Backtest Engine (Chio Extreme)
==============================
Numba-accelerated backtesting with Long+Short support.

v3.0: Simplified for Chio Extreme strategy.
- 1D buy/sell signal arrays (no 2D vote matrices)
- Long + Short positions with flipping
- Catastrophic stop for both directions
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

from qre.config import (
    BACKTEST_POSITION_PCT,
    BASE_TF,
    CATASTROPHIC_STOP_PCT,
    FEE,
    LONG_ONLY,
    MIN_HOLD_HOURS,
    MIN_WARMUP_BARS,
    STARTING_EQUITY,
    get_slippage,
)

logger = logging.getLogger("qre.backtest")


@dataclass
class BacktestResult:
    """Backtest result."""

    equity: float
    trades: List[Dict[str, Any]]
    backtest_days: int


# =============================================================================
# UTILITY
# =============================================================================


def precompute_timeframe_indices(base_timestamps: np.ndarray, tf_timestamps: np.ndarray) -> np.ndarray:
    """Map base timeframe indices to higher timeframe indices."""
    indices = np.searchsorted(tf_timestamps, base_timestamps, side="right") - 1
    indices = np.clip(indices, 0, len(tf_timestamps) - 1)
    return indices.astype(np.int32)


# =============================================================================
# NUMBA TRADING LOOP
# =============================================================================


@njit(cache=True)
def trading_loop_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    buy_signal: np.ndarray,
    sell_signal: np.ndarray,
    min_hold: int,
    position_pct: float,
    slippage: float,
    fee: float,
    start_idx: int,
    end_idx: int,
    catastrophic_stop_pct: float,
    long_only: bool,
) -> Tuple[float, np.ndarray, int]:
    """
    Numba trading loop with Long+Short support.

    Returns:
        (final_equity, trades_array, n_trades)
        trades_array columns: [entry_idx, exit_idx, entry_price, exit_price,
                               pnl_abs, pnl_pct, exit_reason, size, capital_at_entry, direction]
        exit_reason: 0=signal, 1=catastrophic_stop, 2=force_close
        direction: +1=long, -1=short
    """
    cash = STARTING_EQUITY
    position = 0  # 0=flat, 1=long, -1=short
    position_size = 0.0
    entry_bar_idx = 0
    entry_price = 0.0
    capital_at_entry = 0.0

    max_trades = (end_idx - start_idx) // 2 + 1
    trades = np.zeros((max_trades, 10), dtype=np.float64)
    n_trades = 0

    for bar in range(start_idx, end_idx):
        current_price = close[bar]
        current_high = high[bar]
        current_low = low[bar]

        # === CATASTROPHIC STOP (highest priority) ===
        if position == 1:  # long
            if current_low / entry_price - 1.0 <= -catastrophic_stop_pct:
                stop_price = entry_price * (1.0 - catastrophic_stop_pct)
                exit_price = stop_price * (1.0 - slippage)
                fee_cost = exit_price * position_size * fee
                sell_proceeds = position_size * exit_price - fee_cost
                pnl = sell_proceeds - capital_at_entry
                pnl_pct = pnl / capital_at_entry if capital_at_entry > 0 else 0.0

                trades[n_trades, 0] = entry_bar_idx
                trades[n_trades, 1] = bar
                trades[n_trades, 2] = entry_price
                trades[n_trades, 3] = exit_price
                trades[n_trades, 4] = pnl
                trades[n_trades, 5] = pnl_pct
                trades[n_trades, 6] = 1  # catastrophic_stop
                trades[n_trades, 7] = position_size
                trades[n_trades, 8] = capital_at_entry
                trades[n_trades, 9] = 1  # long
                n_trades += 1

                cash += sell_proceeds
                position = 0
                position_size = 0.0
                continue

        elif position == -1:  # short
            if current_high / entry_price - 1.0 >= catastrophic_stop_pct:
                stop_price = entry_price * (1.0 + catastrophic_stop_pct)
                exit_price = stop_price * (1.0 + slippage)
                net_entry_rev = position_size * entry_price * (1.0 - fee)
                net_exit_cost = position_size * exit_price * (1.0 + fee)
                pnl = net_entry_rev - net_exit_cost
                pnl_pct = pnl / capital_at_entry if capital_at_entry > 0 else 0.0

                trades[n_trades, 0] = entry_bar_idx
                trades[n_trades, 1] = bar
                trades[n_trades, 2] = entry_price
                trades[n_trades, 3] = exit_price
                trades[n_trades, 4] = pnl
                trades[n_trades, 5] = pnl_pct
                trades[n_trades, 6] = 1  # catastrophic_stop
                trades[n_trades, 7] = position_size
                trades[n_trades, 8] = capital_at_entry
                trades[n_trades, 9] = -1  # short
                n_trades += 1

                cash += capital_at_entry + pnl
                position = 0
                position_size = 0.0
                continue

        # === SIGNAL EXIT + FLIP ===
        bars_held = bar - entry_bar_idx if position != 0 else 0
        can_exit = bars_held >= min_hold

        if position == 1 and sell_signal[bar] and can_exit:
            # Close long
            exit_price = current_price * (1.0 - slippage)
            fee_cost = exit_price * position_size * fee
            sell_proceeds = position_size * exit_price - fee_cost
            pnl = sell_proceeds - capital_at_entry
            pnl_pct = pnl / capital_at_entry if capital_at_entry > 0 else 0.0

            trades[n_trades, 0] = entry_bar_idx
            trades[n_trades, 1] = bar
            trades[n_trades, 2] = entry_price
            trades[n_trades, 3] = exit_price
            trades[n_trades, 4] = pnl
            trades[n_trades, 5] = pnl_pct
            trades[n_trades, 6] = 0  # signal
            trades[n_trades, 7] = position_size
            trades[n_trades, 8] = capital_at_entry
            trades[n_trades, 9] = 1  # long
            n_trades += 1

            cash += sell_proceeds
            position = 0
            position_size = 0.0

            # Flip to short
            if not long_only and cash > 0:
                entry_price = current_price * (1.0 - slippage)
                capital_at_entry = cash * position_pct
                position_size = capital_at_entry / (entry_price * (1.0 + fee))
                cash -= capital_at_entry
                entry_bar_idx = bar
                position = -1

        elif position == -1 and buy_signal[bar] and can_exit:
            # Close short
            exit_price = current_price * (1.0 + slippage)
            net_entry_rev = position_size * entry_price * (1.0 - fee)
            net_exit_cost = position_size * exit_price * (1.0 + fee)
            pnl = net_entry_rev - net_exit_cost
            pnl_pct = pnl / capital_at_entry if capital_at_entry > 0 else 0.0

            trades[n_trades, 0] = entry_bar_idx
            trades[n_trades, 1] = bar
            trades[n_trades, 2] = entry_price
            trades[n_trades, 3] = exit_price
            trades[n_trades, 4] = pnl
            trades[n_trades, 5] = pnl_pct
            trades[n_trades, 6] = 0  # signal
            trades[n_trades, 7] = position_size
            trades[n_trades, 8] = capital_at_entry
            trades[n_trades, 9] = -1  # short
            n_trades += 1

            cash += capital_at_entry + pnl
            position = 0
            position_size = 0.0

            # Flip to long
            if cash > 0:
                entry_price = current_price * (1.0 + slippage)
                capital_at_entry = cash * position_pct
                position_size = capital_at_entry / (entry_price * (1.0 + fee))
                cash -= capital_at_entry
                entry_bar_idx = bar
                position = 1

        # === OPEN NEW POSITION (if flat) ===
        elif position == 0:
            if buy_signal[bar] and cash > 0:
                entry_price = current_price * (1.0 + slippage)
                capital_at_entry = cash * position_pct
                position_size = capital_at_entry / (entry_price * (1.0 + fee))
                cash -= capital_at_entry
                entry_bar_idx = bar
                position = 1
            elif sell_signal[bar] and not long_only and cash > 0:
                entry_price = current_price * (1.0 - slippage)
                capital_at_entry = cash * position_pct
                position_size = capital_at_entry / (entry_price * (1.0 + fee))
                cash -= capital_at_entry
                entry_bar_idx = bar
                position = -1

    # === FORCE CLOSE AT END ===
    if position != 0 and position_size > 0:
        final_idx = end_idx - 1
        if position == 1:
            exit_price = close[final_idx] * (1.0 - slippage)
            fee_cost = exit_price * position_size * fee
            sell_proceeds = position_size * exit_price - fee_cost
            pnl = sell_proceeds - capital_at_entry
            pnl_pct = pnl / capital_at_entry if capital_at_entry > 0 else 0.0
            cash += sell_proceeds
            direction = 1.0
        else:  # short
            exit_price = close[final_idx] * (1.0 + slippage)
            net_entry_rev = position_size * entry_price * (1.0 - fee)
            net_exit_cost = position_size * exit_price * (1.0 + fee)
            pnl = net_entry_rev - net_exit_cost
            pnl_pct = pnl / capital_at_entry if capital_at_entry > 0 else 0.0
            cash += capital_at_entry + pnl
            direction = -1.0

        trades[n_trades, 0] = entry_bar_idx
        trades[n_trades, 1] = final_idx
        trades[n_trades, 2] = entry_price
        trades[n_trades, 3] = exit_price
        trades[n_trades, 4] = pnl
        trades[n_trades, 5] = pnl_pct
        trades[n_trades, 6] = 2  # force_close
        trades[n_trades, 7] = position_size
        trades[n_trades, 8] = capital_at_entry
        trades[n_trades, 9] = direction
        n_trades += 1

    return cash, trades[:n_trades], n_trades


# =============================================================================
# MAIN BACKTEST FUNCTION
# =============================================================================


def simulate_trades_fast(
    symbol: str,
    data: Dict[str, pd.DataFrame],
    buy_signal: np.ndarray,
    sell_signal: np.ndarray,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    long_only: Optional[bool] = None,
) -> BacktestResult:
    """
    Backtest with 1D buy/sell signals. Supports Long+Short.

    Args:
        symbol: Trading pair (for slippage lookup).
        data: Dict with at least "1h" key containing OHLCV DataFrame.
        buy_signal: 1D boolean array (n_bars,) — True where buy signal fires.
        sell_signal: 1D boolean array (n_bars,) — True where sell signal fires.
        start_idx: Start index for backtest (default: MIN_WARMUP_BARS).
        end_idx: End index for backtest (default: len(data)).
        long_only: Override LONG_ONLY config flag.

    Returns:
        BacktestResult with equity, trades list, and backtest_days.
    """
    if long_only is None:
        long_only = LONG_ONLY

    base = data[BASE_TF]

    if len(base) < MIN_WARMUP_BARS:
        logger.warning(f"Not enough data ({len(base)} bars)")
        return BacktestResult(equity=0.0, trades=[], backtest_days=0)

    actual_start = start_idx if start_idx is not None else MIN_WARMUP_BARS
    actual_end = end_idx if end_idx is not None else len(base)

    backtest_start = base.index[actual_start]
    backtest_end = base.index[min(actual_end - 1, len(base) - 1)]
    backtest_days = (backtest_end - backtest_start).total_seconds() / (24 * 3600)

    close = base["close"].values.astype(np.float64)
    high_arr = base["high"].values.astype(np.float64)
    low_arr = base["low"].values.astype(np.float64)

    slippage = get_slippage(symbol)

    final_equity, trades_arr, n_trades = trading_loop_numba(
        close=close,
        high=high_arr,
        low=low_arr,
        buy_signal=buy_signal.astype(np.bool_),
        sell_signal=sell_signal.astype(np.bool_),
        min_hold=MIN_HOLD_HOURS,
        position_pct=BACKTEST_POSITION_PCT,
        slippage=slippage,
        fee=FEE,
        start_idx=actual_start,
        end_idx=actual_end,
        catastrophic_stop_pct=CATASTROPHIC_STOP_PCT,
        long_only=long_only,
    )

    reason_map = {0: "signal", 1: "catastrophic_stop", 2: "force_close"}
    direction_map = {1: "long", -1: "short"}
    trades = []

    for i in range(n_trades):
        entry_idx = int(trades_arr[i, 0])
        exit_idx = int(trades_arr[i, 1])
        direction_code = int(trades_arr[i, 9])

        trades.append(
            {
                "entry_ts": base.index[entry_idx].isoformat(),
                "entry_price": round(trades_arr[i, 2], 6),
                "exit_ts": base.index[exit_idx].isoformat(),
                "exit_price": round(trades_arr[i, 3], 6),
                "hold_bars": exit_idx - entry_idx,
                "size": round(trades_arr[i, 7], 8),
                "capital_at_entry": round(trades_arr[i, 8], 2),
                "pnl_abs": round(trades_arr[i, 4], 2),
                "pnl_pct": trades_arr[i, 5],
                "symbol": symbol,
                "reason": reason_map.get(int(trades_arr[i, 6]), "unknown"),
                "direction": direction_map.get(direction_code, "unknown"),
            }
        )

    return BacktestResult(equity=float(final_equity), trades=trades, backtest_days=int(backtest_days))
