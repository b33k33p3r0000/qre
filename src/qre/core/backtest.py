#!/usr/bin/env python3
"""
Backtest Engine (Optimized)
===========================
Numba-accelerated backtesting for 10-50x speedup.

Key optimizations:
- Pre-compute all indicators and crossover signals as NumPy arrays
- Numba JIT-compiled trading loop
- Vectorized RSI gate computation
- Minimize Python object creation in hot path

v9.3: Performance optimized version
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Fallback: identity decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range

from qre.config import (
    BACKTEST_POSITION_PCT,
    BASE_TF,
    CATASTROPHIC_STOP_PCT,
    FEE,
    MIN_HOLD_BARS,
    MIN_WARMUP_BARS,
    RSI_LENGTH,
    STARTING_EQUITY,  # v10.2: Use config instead of hardcoded value
    STOCH_LENGTH,
    TF_LIST,
    get_slippage,
)
from qre.core.indicators import rsi, stochrsi

logger = logging.getLogger("qre.backtest")


@dataclass
class BacktestResult:
    """Vysledek backtestu."""

    equity: float
    trades: List[Dict[str, Any]]
    backtest_days: int


# =============================================================================
# PRE-COMPUTATION HELPERS
# =============================================================================


def precompute_timeframe_indices(base_timestamps: np.ndarray, tf_timestamps: np.ndarray) -> np.ndarray:
    """
    Pre-compute index mapping from base timeframe to higher timeframe.

    For each bar in base_tf, find the corresponding index in tf.
    Uses vectorized searchsorted for efficiency.

    Args:
        base_timestamps: Timestamps of base timeframe as int64 (ns)
        tf_timestamps: Timestamps of higher timeframe as int64 (ns)

    Returns:
        Array of indices into tf for each base bar
    """
    # searchsorted finds insertion point, we want last bar <= current
    indices = np.searchsorted(tf_timestamps, base_timestamps, side="right") - 1
    # Clip to valid range
    indices = np.clip(indices, 0, len(tf_timestamps) - 1)
    return indices.astype(np.int32)


def precompute_crossover_signals(
    k_line: np.ndarray, d_line: np.ndarray, low_threshold: float, high_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute buy and sell crossover signals for a timeframe.

    Args:
        k_line: StochRSI %K values
        d_line: StochRSI %D values
        low_threshold: Buy threshold
        high_threshold: Sell threshold

    Returns:
        (buy_signals, sell_signals) as boolean arrays
    """
    n = len(k_line)
    if n < 2:
        return np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_)

    # Previous values (shifted by 1)
    k_prev = np.roll(k_line, 1)
    d_prev = np.roll(d_line, 1)
    k_prev[0] = np.nan
    d_prev[0] = np.nan

    # Buy crossover: K crosses above D, K < low_threshold
    # K < 0.6 prevents buying in neutral/overbought zone (StochRSI standard practice)
    buy_signals = (
        (k_prev < d_prev)
        & (k_line > d_line)
        & (k_line < low_threshold)
        & (k_line < 0.6)  # Max K for buy - avoid neutral/overbought entries
        & ~np.isnan(k_line)
        & ~np.isnan(d_line)
        & ~np.isnan(k_prev)
        & ~np.isnan(d_prev)
    )

    # Sell crossover: K crosses below D, K > high_threshold
    # K > 0.4 prevents selling in neutral/oversold zone (StochRSI standard practice)
    sell_signals = (
        (k_prev > d_prev)
        & (k_line < d_line)
        & (k_line > high_threshold)
        & (k_line > 0.4)  # Min K for sell - avoid neutral/oversold exits
        & ~np.isnan(k_line)
        & ~np.isnan(d_line)
        & ~np.isnan(k_prev)
        & ~np.isnan(d_prev)
    )

    return buy_signals, sell_signals


def precompute_rsi_gate(rsi_values: np.ndarray, gate_threshold: float) -> np.ndarray:
    """
    Pre-compute RSI gate signals.

    Args:
        rsi_values: RSI values
        gate_threshold: Threshold for gate activation

    Returns:
        Boolean array where RSI > threshold
    """
    return (~np.isnan(rsi_values)) & (rsi_values > gate_threshold)


# =============================================================================
# NUMBA-ACCELERATED TRADING LOOP
# =============================================================================


@njit(cache=True)
def trading_loop_numba(
    # Price data (base timeframe)
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    # Pre-computed signals per timeframe (shape: n_timeframes x n_bars)
    buy_votes_per_tf: np.ndarray,
    sell_votes_per_tf: np.ndarray,
    # RSI gate (shape: 4 x n_bars) - one per gate timeframe
    rsi_gate_signals: np.ndarray,
    # Parameters
    required_buy_votes: int,
    min_sell_votes: int,
    min_hold: int,
    position_pct: float,
    trailing_stop_pct: float,
    slippage: float,
    fee: float,
    # Range
    start_idx: int,
    end_idx: int,
    # v6.0: Plugin arrays
    filter_mask: np.ndarray,  # bool (n_bars,) - plugin filters
    gate_mask: np.ndarray,  # bool (n_bars,) - plugin gates
    position_mult: np.ndarray,  # float (n_bars,) - plugin position sizers
    # v12.10: Stuck trade protection
    max_hold_hours: int,  # e.g., 120 (5 days)
    # v10.0: Catastrophic stop for black swan protection
    catastrophic_stop_pct: float,  # e.g., 0.09 (-9% from entry)
) -> Tuple[float, np.ndarray, int]:
    """
    Numba-accelerated trading loop.

    Returns:
        (final_equity, trades_array, n_trades)
        trades_array has columns: [entry_idx, exit_idx, entry_price, exit_price,
                                   pnl_abs, pnl_pct, exit_reason, size, capital_at_entry]
        exit_reason: 0=signal, 1=trailing_stop, 2=eod, 3=catastrophic_stop, 4=max_hold
    """
    starting_equity = STARTING_EQUITY  # v10.2: Use config value ($20,000)
    cash = starting_equity
    position_size = 0.0
    in_position = False

    entry_bar_idx = 0
    entry_price = 0.0
    capital_at_entry = 0.0
    high_water_mark = 0.0

    # Pre-allocate trades array (max possible trades = n_bars/2)
    max_trades = (end_idx - start_idx) // 2 + 1
    trades = np.zeros((max_trades, 9), dtype=np.float64)  # 9 columns: see docstring
    n_trades = 0

    n_timeframes = buy_votes_per_tf.shape[0]

    for bar_idx in range(start_idx, end_idx):
        current_price = close[bar_idx]
        current_high = high[bar_idx]
        current_low = low[bar_idx]

        # Update high water mark
        if in_position and current_high > high_water_mark:
            high_water_mark = current_high

        # Count votes across timeframes
        buy_votes = 0
        sell_votes = 0
        for tf_idx in range(n_timeframes):
            if buy_votes_per_tf[tf_idx, bar_idx]:
                buy_votes += 1
            if sell_votes_per_tf[tf_idx, bar_idx]:
                sell_votes += 1

        # RSI gate check (any of 4 gates passes)
        rsi_gate_ok = False
        for gate_idx in range(4):
            if rsi_gate_signals[gate_idx, bar_idx]:
                rsi_gate_ok = True
                break

        # Decision (v6.0: plugin filters & gates)
        buy_signal = (buy_votes >= required_buy_votes) and rsi_gate_ok and filter_mask[bar_idx] and gate_mask[bar_idx]
        sell_signal = sell_votes >= min_sell_votes

        bars_in_position = bar_idx - entry_bar_idx if in_position else 0
        can_sell = bars_in_position >= min_hold

        # BUY (v6.0: plugin position multiplier)
        if buy_signal and not in_position and cash > 0:
            buy_price = current_price * (1 + slippage)
            effective_position_pct = position_pct * position_mult[bar_idx]
            capital_to_risk = cash * effective_position_pct
            position_size = capital_to_risk / (buy_price * (1 + fee))

            if position_size > 0:
                entry_bar_idx = bar_idx
                entry_price = buy_price
                capital_at_entry = capital_to_risk
                cash = cash - capital_to_risk
                in_position = True
                high_water_mark = current_high

        # TRAILING STOP
        if in_position and trailing_stop_pct > 0 and high_water_mark > 0:
            stop_price = high_water_mark * (1 - trailing_stop_pct)
            if current_low <= stop_price:
                sell_price = stop_price * (1 - slippage)
                fee_cost = sell_price * position_size * fee
                sell_proceeds = position_size * sell_price - fee_cost

                pnl_absolute = sell_proceeds - capital_at_entry
                pnl_percent = pnl_absolute / capital_at_entry if capital_at_entry > 0 else 0.0

                # Record trade
                trades[n_trades, 0] = entry_bar_idx
                trades[n_trades, 1] = bar_idx
                trades[n_trades, 2] = entry_price
                trades[n_trades, 3] = sell_price
                trades[n_trades, 4] = pnl_absolute
                trades[n_trades, 5] = pnl_percent
                trades[n_trades, 6] = 1  # trailing stop
                trades[n_trades, 7] = position_size
                trades[n_trades, 8] = capital_at_entry
                n_trades += 1

                cash = cash + sell_proceeds
                position_size = 0.0
                in_position = False

        # CATASTROPHIC STOP CHECK (v10.0)
        if in_position and catastrophic_stop_pct > 0:
            unrealized_pnl_pct = (current_low / entry_price) - 1.0
            if unrealized_pnl_pct <= -catastrophic_stop_pct:
                catastrophic_stop_price = entry_price * (1 - catastrophic_stop_pct)
                sell_price = catastrophic_stop_price * (1 - slippage)
                fee_cost = sell_price * position_size * fee
                sell_proceeds = position_size * sell_price - fee_cost

                pnl_absolute = sell_proceeds - capital_at_entry
                pnl_percent = pnl_absolute / capital_at_entry if capital_at_entry > 0 else 0.0

                trades[n_trades, 0] = entry_bar_idx
                trades[n_trades, 1] = bar_idx
                trades[n_trades, 2] = entry_price
                trades[n_trades, 3] = sell_price
                trades[n_trades, 4] = pnl_absolute
                trades[n_trades, 5] = pnl_percent
                trades[n_trades, 6] = 3  # catastrophic_stop
                trades[n_trades, 7] = position_size
                trades[n_trades, 8] = capital_at_entry
                n_trades += 1

                cash = cash + sell_proceeds
                position_size = 0.0
                in_position = False

        # MAX HOLD CHECK (v12.10: Stuck Trade Protection)
        if in_position and max_hold_hours > 0 and (bar_idx - entry_bar_idx) >= max_hold_hours:
            sell_price = current_price * (1 - slippage)
            fee_cost = sell_price * position_size * fee
            sell_proceeds = position_size * sell_price - fee_cost

            pnl_absolute = sell_proceeds - capital_at_entry
            pnl_percent = pnl_absolute / capital_at_entry if capital_at_entry > 0 else 0.0

            # Record trade
            trades[n_trades, 0] = entry_bar_idx
            trades[n_trades, 1] = bar_idx
            trades[n_trades, 2] = entry_price
            trades[n_trades, 3] = sell_price
            trades[n_trades, 4] = pnl_absolute
            trades[n_trades, 5] = pnl_percent
            trades[n_trades, 6] = 4  # max_hold
            trades[n_trades, 7] = position_size
            trades[n_trades, 8] = capital_at_entry
            n_trades += 1

            cash = cash + sell_proceeds
            position_size = 0.0
            in_position = False

        # SELL (signal)
        elif sell_signal and in_position and position_size > 0 and can_sell:
            sell_price = current_price * (1 - slippage)
            fee_cost = sell_price * position_size * fee
            sell_proceeds = position_size * sell_price - fee_cost

            pnl_absolute = sell_proceeds - capital_at_entry
            pnl_percent = pnl_absolute / capital_at_entry if capital_at_entry > 0 else 0.0

            # Record trade
            trades[n_trades, 0] = entry_bar_idx
            trades[n_trades, 1] = bar_idx
            trades[n_trades, 2] = entry_price
            trades[n_trades, 3] = sell_price
            trades[n_trades, 4] = pnl_absolute
            trades[n_trades, 5] = pnl_percent
            trades[n_trades, 6] = 0  # signal
            trades[n_trades, 7] = position_size
            trades[n_trades, 8] = capital_at_entry
            n_trades += 1

            cash = cash + sell_proceeds
            position_size = 0.0
            in_position = False

    # Force close at end
    if in_position and position_size > 0:
        final_idx = end_idx - 1
        final_price = close[final_idx]
        sell_price = final_price * (1 - slippage)
        fee_cost = sell_price * position_size * fee
        sell_proceeds = position_size * sell_price - fee_cost

        pnl_absolute = sell_proceeds - capital_at_entry
        pnl_percent = pnl_absolute / capital_at_entry if capital_at_entry > 0 else 0.0

        trades[n_trades, 0] = entry_bar_idx
        trades[n_trades, 1] = final_idx
        trades[n_trades, 2] = entry_price
        trades[n_trades, 3] = sell_price
        trades[n_trades, 4] = pnl_absolute
        trades[n_trades, 5] = pnl_percent
        trades[n_trades, 6] = 2  # EOD
        trades[n_trades, 7] = position_size
        trades[n_trades, 8] = capital_at_entry
        n_trades += 1

        cash = cash + sell_proceeds

    return cash, trades[:n_trades], n_trades


# =============================================================================
# MAIN BACKTEST FUNCTION (OPTIMIZED)
# =============================================================================


def simulate_trades_fast(
    symbol: str,
    data: Dict[str, pd.DataFrame],
    params: Dict[str, Any],
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    # v6.0: Plugin arrays (optional - defaults to np.ones for backward compat)
    filter_mask: Optional[np.ndarray] = None,
    gate_mask: Optional[np.ndarray] = None,
    position_mult: Optional[np.ndarray] = None,
    # v12.0: Strategy precomputed signals (optional - uses inline computation if None)
    precomputed_buy_signals: Optional[np.ndarray] = None,
    precomputed_sell_signals: Optional[np.ndarray] = None,
    precomputed_rsi_gates: Optional[np.ndarray] = None,
) -> BacktestResult:
    """
    Optimized backtest using pre-computation and Numba JIT.

    ~10-50x faster than the original simulate_trades().

    Args:
        symbol: Trading par
        data: Dict s OHLCV daty
        params: Parametry strategie
        start_idx: Pocatecni index pro backtest
        end_idx: Koncovy index
        precomputed_buy_signals: (v12.0) Optional pre-computed buy signals from strategy
        precomputed_sell_signals: (v12.0) Optional pre-computed sell signals from strategy
        precomputed_rsi_gates: (v12.0) Optional pre-computed RSI gates from strategy

    Returns:
        BacktestResult
    """
    base = data[BASE_TF]

    if len(base) < MIN_WARMUP_BARS:
        logger.warning(f"Malo dat ({len(base)} radku)")
        return BacktestResult(equity=0.0, trades=[], backtest_days=0)

    # Walk-forward support
    actual_start = start_idx if start_idx is not None else MIN_WARMUP_BARS
    actual_end = end_idx if end_idx is not None else len(base)

    # Backtest period
    backtest_start = base.index[actual_start]
    backtest_end = base.index[min(actual_end - 1, len(base) - 1)]
    backtest_days = (backtest_end - backtest_start).total_seconds() / (24 * 3600)

    # === PARAMETERS ===
    k_smooth = int(params.get("kB", 3))
    d_smooth = int(params.get("dB", 3))
    buy_threshold_pct = float(params.get("p_buy", 0.26))
    min_sell_votes = int(params.get("k_sell", 1))
    min_hold = int(params.get("min_hold", MIN_HOLD_BARS))
    max_hold_hours = int(params.get("max_hold_hours", 0))  # v12.10: Stuck trade protection
    position_pct = BACKTEST_POSITION_PCT  # Statickych 25%
    trailing_stop_pct = float(params.get("trailing_stop_pct", 0.0))

    low_thresholds = {tf: float(params.get(f"low_{'24h' if tf == '1d' else tf}", 0.2)) for tf in TF_LIST}
    high_thresholds = {tf: float(params.get(f"high_{'24h' if tf == '1d' else tf}", 0.8)) for tf in TF_LIST}

    rsi_gate_24h = float(params.get("rsi_gate_24h", 50))
    rsi_gate_12h = float(params.get("rsi_gate_12h", 50))
    rsi_gate_8h = float(params.get("rsi_gate_8h", 50))
    rsi_gate_6h = float(params.get("rsi_gate_6h", 50))

    slippage = get_slippage(symbol)
    total_timeframes = len(TF_LIST)
    required_buy_votes = int(math.ceil(buy_threshold_pct * total_timeframes))

    # === EXTRACT NUMPY ARRAYS ===
    close = base["close"].values.astype(np.float64)
    high_arr = base["high"].values.astype(np.float64)
    low_arr = base["low"].values.astype(np.float64)
    base_ts = base.index.values.astype(np.int64)  # Timestamps as int64
    n_bars = len(base)

    # === PRE-COMPUTE OR USE PROVIDED SIGNALS (v12.0) ===
    if precomputed_buy_signals is not None and precomputed_sell_signals is not None:
        # Use strategy-provided signals
        buy_votes_per_tf = precomputed_buy_signals
        sell_votes_per_tf = precomputed_sell_signals
    else:
        # Compute signals inline (legacy behavior)
        buy_votes_per_tf = np.zeros((total_timeframes, n_bars), dtype=np.bool_)
        sell_votes_per_tf = np.zeros((total_timeframes, n_bars), dtype=np.bool_)

        for tf_idx, tf in enumerate(TF_LIST):
            if tf not in data or len(data[tf]) == 0:
                continue

            df_tf = data[tf]

            # Compute StochRSI for this timeframe
            k_line, d_line = stochrsi(df_tf["close"], STOCH_LENGTH, STOCH_LENGTH, k_smooth, d_smooth)
            k_vals = k_line.values.astype(np.float64)
            d_vals = d_line.values.astype(np.float64)

            # Compute crossover signals on the TF's own bars
            tf_buy, tf_sell = precompute_crossover_signals(k_vals, d_vals, low_thresholds[tf], high_thresholds[tf])

            # Map TF indices to base TF indices
            tf_ts = df_tf.index.values.astype(np.int64)
            base_to_tf_idx = precompute_timeframe_indices(base_ts, tf_ts)

            # Propagate signals: for each base bar, get the signal from corresponding TF bar
            for bar_idx in range(n_bars):
                tf_row_idx = base_to_tf_idx[bar_idx]
                if tf_row_idx >= 2 and tf_row_idx < len(tf_buy):
                    buy_votes_per_tf[tf_idx, bar_idx] = tf_buy[tf_row_idx]
                    sell_votes_per_tf[tf_idx, bar_idx] = tf_sell[tf_row_idx]

    # === PRE-COMPUTE OR USE PROVIDED RSI GATES (v12.0) ===
    if precomputed_rsi_gates is not None:
        # Use strategy-provided gates
        rsi_gate_signals = precomputed_rsi_gates
    else:
        # Compute gates inline (legacy behavior)
        rsi_gate_signals = np.zeros((4, n_bars), dtype=np.bool_)

        gate_configs = [
            ("1d", rsi_gate_24h),
            ("12h", rsi_gate_12h),
            ("8h", rsi_gate_8h),
            ("6h", rsi_gate_6h),
        ]

        for gate_idx, (tf, threshold) in enumerate(gate_configs):
            if tf not in data or len(data[tf]) == 0:
                continue

            df_tf = data[tf]
            rsi_vals = rsi(df_tf["close"], RSI_LENGTH).values.astype(np.float64)
            tf_gate = precompute_rsi_gate(rsi_vals, threshold)

            # Map to base timeframe
            tf_ts = df_tf.index.values.astype(np.int64)
            base_to_tf_idx = precompute_timeframe_indices(base_ts, tf_ts)

            for bar_idx in range(n_bars):
                tf_row_idx = base_to_tf_idx[bar_idx]
                if tf_row_idx >= 1 and tf_row_idx < len(tf_gate):
                    rsi_gate_signals[gate_idx, bar_idx] = tf_gate[tf_row_idx]

    # === v6.0: PLUGIN ARRAYS (defaults for backward compatibility) ===
    if filter_mask is None:
        filter_mask = np.ones(n_bars, dtype=np.bool_)
    if gate_mask is None:
        gate_mask = np.ones(n_bars, dtype=np.bool_)
    if position_mult is None:
        position_mult = np.ones(n_bars, dtype=np.float64)

    # === RUN NUMBA TRADING LOOP ===
    final_equity, trades_arr, n_trades = trading_loop_numba(
        close=close,
        high=high_arr,
        low=low_arr,
        buy_votes_per_tf=buy_votes_per_tf,
        sell_votes_per_tf=sell_votes_per_tf,
        rsi_gate_signals=rsi_gate_signals,
        required_buy_votes=required_buy_votes,
        min_sell_votes=min_sell_votes,
        min_hold=min_hold,
        position_pct=position_pct,
        trailing_stop_pct=trailing_stop_pct,
        slippage=slippage,
        fee=FEE,
        start_idx=actual_start,
        end_idx=actual_end,
        filter_mask=filter_mask,
        gate_mask=gate_mask,
        position_mult=position_mult,
        max_hold_hours=max_hold_hours,
        catastrophic_stop_pct=CATASTROPHIC_STOP_PCT,
    )

    # === CONVERT TRADES TO LIST OF DICTS ===
    reason_map = {0: "signal@open", 1: "trailing_stop", 2: "close@eod", 3: "catastrophic_stop", 4: "max_hold"}
    trades = []

    for i in range(n_trades):
        entry_idx = int(trades_arr[i, 0])
        exit_idx = int(trades_arr[i, 1])
        entry_price = trades_arr[i, 2]
        exit_price = trades_arr[i, 3]
        pnl_abs = trades_arr[i, 4]
        pnl_pct = trades_arr[i, 5]
        reason_code = int(trades_arr[i, 6])
        size = trades_arr[i, 7]
        capital_at_entry = trades_arr[i, 8]

        trades.append(
            {
                "entry_ts": base.index[entry_idx].isoformat(),
                "entry_price": round(entry_price, 6),
                "exit_ts": base.index[exit_idx].isoformat(),
                "exit_price": round(exit_price, 6),
                "hold_bars": exit_idx - entry_idx,
                "size": round(size, 8),
                "capital_at_entry": round(capital_at_entry, 2),
                "pnl_abs": round(pnl_abs, 2),
                "pnl_pct": pnl_pct,
                "symbol": symbol,
                "reason": reason_map.get(reason_code, "unknown"),
            }
        )

    return BacktestResult(equity=float(final_equity), trades=trades, backtest_days=int(backtest_days))
