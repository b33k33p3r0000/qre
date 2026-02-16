"""Unit tests for QRE backtest engine (post-redesign)."""

import inspect

import numpy as np
import pandas as pd
import pytest

from qre.core.backtest import (
    simulate_trades_fast,
    precompute_timeframe_indices,
    BacktestResult,
)


def _make_1h_data(n_bars=500, seed=42):
    """Helper: create 1H OHLCV data."""
    np.random.seed(seed)
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="1h")
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.3)
    high = close + np.abs(np.random.randn(n_bars) * 0.5)
    low = close - np.abs(np.random.randn(n_bars) * 0.5)
    open_ = close + np.random.randn(n_bars) * 0.1
    return {
        "1h": pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close},
            index=dates,
        )
    }


def _make_signals(n_bars, buy_bars=None, sell_bars=None):
    """Helper: create 1D buy/sell signal arrays."""
    buy = np.zeros(n_bars, dtype=np.bool_)
    sell = np.zeros(n_bars, dtype=np.bool_)
    if buy_bars:
        for b in buy_bars:
            buy[b] = True
    if sell_bars:
        for s in sell_bars:
            sell[s] = True
    return buy, sell


class TestPrecomputeTimeframeIndices:
    def test_shape(self):
        """Maps base timestamps to TF indices correctly."""
        base_ts = np.arange(0, 100 * 3600000, 3600000, dtype=np.int64)
        tf_ts = np.arange(0, 100 * 3600000, 4 * 3600000, dtype=np.int64)
        idx = precompute_timeframe_indices(base_ts, tf_ts)
        assert len(idx) == len(base_ts)


class TestNoLegacyFunctions:
    def test_no_crossover_signals(self):
        """precompute_crossover_signals removed — StochRSI eliminated."""
        import qre.core.backtest as mod
        assert not hasattr(mod, "precompute_crossover_signals")

    def test_no_rsi_gate(self):
        """precompute_rsi_gate removed — RSI gates eliminated."""
        import qre.core.backtest as mod
        assert not hasattr(mod, "precompute_rsi_gate")

    def test_simplified_signature(self):
        """No legacy params: no plugin arrays, no precomputed_rsi_gates."""
        sig = inspect.signature(simulate_trades_fast)
        param_names = set(sig.parameters.keys())
        for legacy in ["filter_mask", "gate_mask", "position_mult",
                        "precomputed_buy_signals", "precomputed_sell_signals",
                        "precomputed_rsi_gates", "params"]:
            assert legacy not in param_names, f"Legacy param {legacy} found"
        assert "buy_signal" in param_names
        assert "sell_signal" in param_names


class TestSimulateTradesFast:
    def test_returns_backtest_result(self):
        """simulate_trades_fast returns BacktestResult."""
        data = _make_1h_data()
        n = len(data["1h"])
        buy, sell = _make_signals(n, buy_bars=[250], sell_bars=[300])
        result = simulate_trades_fast("BTC/USDC", data, buy, sell)
        assert isinstance(result, BacktestResult)
        assert isinstance(result.trades, list)
        assert result.equity >= 0

    def test_direction_in_trades(self):
        """Trades have direction field (long/short)."""
        data = _make_1h_data()
        n = len(data["1h"])
        buy, sell = _make_signals(n, buy_bars=[250], sell_bars=[300])
        result = simulate_trades_fast("BTC/USDC", data, buy, sell)
        for trade in result.trades:
            assert "direction" in trade
            assert trade["direction"] in ("long", "short")

    def test_long_only_no_shorts(self):
        """With long_only=True, no short trades."""
        data = _make_1h_data()
        n = len(data["1h"])
        buy, sell = _make_signals(n, buy_bars=[250, 350], sell_bars=[300, 400])
        result = simulate_trades_fast("BTC/USDC", data, buy, sell, long_only=True)
        for trade in result.trades:
            assert trade["direction"] == "long"

    def test_short_trade_exists(self):
        """With long_only=False, sell signal when flat opens a short."""
        data = _make_1h_data()
        n = len(data["1h"])
        buy, sell = _make_signals(n, sell_bars=[250])
        result = simulate_trades_fast("BTC/USDC", data, buy, sell, long_only=False)
        assert len(result.trades) >= 1
        assert any(t["direction"] == "short" for t in result.trades)

    def test_pnl_finite(self):
        """All trade PnLs should be finite."""
        data = _make_1h_data()
        n = len(data["1h"])
        buy, sell = _make_signals(n, buy_bars=[250, 350], sell_bars=[300, 400])
        result = simulate_trades_fast("BTC/USDC", data, buy, sell)
        for trade in result.trades:
            assert np.isfinite(trade["pnl_abs"]), f"Non-finite PnL: {trade}"
            assert np.isfinite(trade["pnl_pct"]), f"Non-finite PnL%: {trade}"

    def test_catastrophic_stop(self):
        """Huge price drop triggers catastrophic stop for long."""
        n = 500
        dates = pd.date_range("2025-01-01", periods=n, freq="1h")
        close = np.full(n, 100.0)
        close[260:] = 50.0  # 50% crash after entry
        high = close + 1.0
        low = close - 1.0
        data = {"1h": pd.DataFrame(
            {"open": close.copy(), "high": high, "low": low, "close": close},
            index=dates,
        )}
        buy, sell = _make_signals(n, buy_bars=[250])
        result = simulate_trades_fast("BTC/USDC", data, buy, sell)
        assert len(result.trades) >= 1
        assert any(t["reason"] == "catastrophic_stop" for t in result.trades)

    def test_force_close_at_end(self):
        """Open position is closed at end of data."""
        data = _make_1h_data()
        n = len(data["1h"])
        buy, sell = _make_signals(n, buy_bars=[490])
        result = simulate_trades_fast("BTC/USDC", data, buy, sell)
        assert len(result.trades) >= 1
        assert any(t["reason"] == "force_close" for t in result.trades)

    def test_min_hold_respected(self):
        """Cannot exit before min_hold bars."""
        data = _make_1h_data()
        n = len(data["1h"])
        # Buy at 250, sell at 251 (too early), sell at 253 (ok)
        buy, sell = _make_signals(n, buy_bars=[250], sell_bars=[251, 253])
        result = simulate_trades_fast("BTC/USDC", data, buy, sell)
        for trade in result.trades:
            if trade["reason"] != "force_close":
                assert trade["hold_bars"] >= 2, f"Exited too early: {trade['hold_bars']} bars"

    def test_position_flip(self):
        """Sell signal closes long and opens short (when not long_only)."""
        data = _make_1h_data()
        n = len(data["1h"])
        buy, sell = _make_signals(n, buy_bars=[250], sell_bars=[260])
        result = simulate_trades_fast("BTC/USDC", data, buy, sell, long_only=False)
        # Should have 2 trades: long closed by signal, short force-closed at end
        directions = [t["direction"] for t in result.trades]
        assert "long" in directions
        assert "short" in directions
