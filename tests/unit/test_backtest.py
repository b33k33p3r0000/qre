"""Unit tests for QRE backtest engine."""

import numpy as np
import pandas as pd
import pytest

from qre.core.backtest import (
    simulate_trades_fast,
    precompute_crossover_signals,
    precompute_rsi_gate,
    precompute_timeframe_indices,
    BacktestResult,
)


class TestPrecomputeFunctions:
    def test_crossover_signals_shape(self):
        """precompute_crossover_signals returns two arrays of same length."""
        k = np.random.rand(100)
        d = np.random.rand(100)
        buy, sell = precompute_crossover_signals(k, d, 0.2, 0.8)
        assert len(buy) == 100
        assert len(sell) == 100

    def test_crossover_signals_boolean(self):
        """Crossover signals should be boolean-like."""
        k = np.random.rand(100)
        d = np.random.rand(100)
        buy, sell = precompute_crossover_signals(k, d, 0.2, 0.8)
        assert set(np.unique(buy)).issubset({0, 1, True, False})

    def test_rsi_gate_shape(self):
        """precompute_rsi_gate returns array of same length."""
        rsi_vals = np.random.rand(100) * 100
        gate = precompute_rsi_gate(rsi_vals, 50.0)
        assert len(gate) == 100

    def test_rsi_gate_logic(self):
        """RSI gate is True when RSI > threshold."""
        rsi_vals = np.array([30.0, 60.0, np.nan, 80.0])
        gate = precompute_rsi_gate(rsi_vals, 50.0)
        assert gate[0] == False  # 30 < 50
        assert gate[1] == True   # 60 > 50
        assert gate[2] == False  # NaN
        assert gate[3] == True   # 80 > 50

    def test_timeframe_indices_shape(self):
        """precompute_timeframe_indices returns array matching base length."""
        base_ts = np.arange(0, 100 * 3600000, 3600000, dtype=np.int64)  # 1h bars
        tf_ts = np.arange(0, 100 * 3600000, 4 * 3600000, dtype=np.int64)  # 4h bars
        idx = precompute_timeframe_indices(base_ts, tf_ts)
        assert len(idx) == len(base_ts)


def _make_ohlcv_data(n_bars=500):
    """Helper: create multi-timeframe OHLCV DataFrames for backtest.

    Index must be DatetimeIndex (simulate_trades_fast uses .total_seconds() and .isoformat()).
    """
    np.random.seed(42)
    # Base timeframe (1h)
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="1h")
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.3)
    high = close + np.abs(np.random.randn(n_bars) * 0.5)
    low = close - np.abs(np.random.randn(n_bars) * 0.5)
    open_ = close + np.random.randn(n_bars) * 0.1

    base_df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=dates,
    )

    data = {"1h": base_df}

    # Higher timeframes via subsampling
    tf_configs = [("2h", 2), ("4h", 4), ("6h", 6), ("8h", 8), ("12h", 12), ("1d", 24)]
    for tf, factor in tf_configs:
        tf_len = n_bars // factor
        if tf_len < 10:
            continue
        tf_idx = dates[::factor][:tf_len]
        data[tf] = pd.DataFrame(
            {
                "open": open_[::factor][:tf_len],
                "high": high[::factor][:tf_len],
                "low": low[::factor][:tf_len],
                "close": close[::factor][:tf_len],
            },
            index=tf_idx,
        )

    return data


class TestSimulateTradesFast:
    def test_returns_backtest_result(self):
        """simulate_trades_fast returns BacktestResult."""
        data = _make_ohlcv_data(n_bars=500)
        params = {
            "kB": 3, "dB": 2, "k_sell": 1, "min_hold": 8,
            "p_buy": 0.2,
            "low_2h": 0.2, "high_2h": 0.8,
            "low_4h": 0.2, "high_4h": 0.8,
            "low_6h": 0.2, "high_6h": 0.8,
            "low_8h": 0.2, "high_8h": 0.8,
            "low_12h": 0.2, "high_12h": 0.8,
            "low_24h": 0.2, "high_24h": 0.8,
            "rsi_gate_24h": 50, "rsi_gate_12h": 50,
            "rsi_gate_8h": 50, "rsi_gate_6h": 50,
        }
        result = simulate_trades_fast("BTC/USDC", data, params)
        assert isinstance(result, BacktestResult)
        assert isinstance(result.trades, list)
        assert result.equity >= 0

    def test_with_precomputed_signals(self):
        """simulate_trades_fast works with pre-computed signals."""
        data = _make_ohlcv_data(n_bars=500)
        n_bars = len(data["1h"])
        params = {
            "kB": 3, "dB": 2, "k_sell": 1, "min_hold": 8,
            "p_buy": 0.2,
        }
        # Pre-compute simple signals
        buy_signals = np.zeros((6, n_bars), dtype=np.bool_)
        buy_signals[:, ::50] = True  # Buy every 50 bars on all TFs
        sell_signals = np.ones((6, n_bars), dtype=np.bool_)
        rsi_gates = np.ones((4, n_bars), dtype=np.bool_)

        result = simulate_trades_fast(
            "BTC/USDC", data, params,
            precomputed_buy_signals=buy_signals,
            precomputed_sell_signals=sell_signals,
            precomputed_rsi_gates=rsi_gates,
        )
        assert isinstance(result, BacktestResult)

    def test_pnl_finite(self):
        """All trade PnLs should be finite."""
        data = _make_ohlcv_data(n_bars=500)
        n_bars = len(data["1h"])
        params = {"kB": 3, "dB": 2, "k_sell": 1, "min_hold": 3, "p_buy": 0.15}

        buy_signals = np.zeros((6, n_bars), dtype=np.bool_)
        buy_signals[:, 210::80] = True
        sell_signals = np.ones((6, n_bars), dtype=np.bool_)
        rsi_gates = np.ones((4, n_bars), dtype=np.bool_)

        result = simulate_trades_fast(
            "BTC/USDC", data, params,
            precomputed_buy_signals=buy_signals,
            precomputed_sell_signals=sell_signals,
            precomputed_rsi_gates=rsi_gates,
        )
        for trade in result.trades:
            assert np.isfinite(trade["pnl_abs"]), f"Non-finite PnL: {trade}"
