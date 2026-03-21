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
from tests.conftest import make_1h_ohlcv


def _make_1h_data(n_bars=500, seed=42):
    """Helper: create 1H OHLCV data."""
    return {"1h": make_1h_ohlcv(n_bars=n_bars, seed=seed)}


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

    def test_catastrophic_stop_custom_pct(self):
        """Custom catastrophic_stop_pct is used when provided."""
        n = 500
        dates = pd.date_range("2025-01-01", periods=n, freq="1h")
        close = np.full(n, 100.0)
        close[260:] = 92.0  # 8% drop — triggers 5% stop but NOT 10%
        high = close + 1.0
        low = close - 1.0
        data = {"1h": pd.DataFrame(
            {"open": close.copy(), "high": high, "low": low, "close": close},
            index=dates,
        )}
        buy, sell = _make_signals(n, buy_bars=[250])

        # With default (10%) — no catastrophic stop
        result_default = simulate_trades_fast("BTC/USDC", data, buy, sell)
        assert not any(t["reason"] == "catastrophic_stop" for t in result_default.trades)

        # With custom 5% — should trigger catastrophic stop
        result_custom = simulate_trades_fast("BTC/USDC", data, buy, sell, catastrophic_stop_pct=0.05)
        assert any(t["reason"] == "catastrophic_stop" for t in result_custom.trades)

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


def _make_controlled_data(n_bars, prices, spread=2.0):
    """Helper: create 1H OHLCV with controlled close prices.

    Args:
        n_bars: Number of bars.
        prices: Array of close prices (length n_bars).
        spread: High-low spread around close (default 2.0).

    Returns:
        Dict with "1h" key containing DataFrame.
    """
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="1h")
    close = np.array(prices, dtype=np.float64)
    high = close + spread / 2.0
    low = close - spread / 2.0
    df = pd.DataFrame(
        {"open": close.copy(), "high": high, "low": low, "close": close},
        index=dates,
    )
    return {"1h": df}


class TestTrailingStop:
    """Tests for trailing stop in Numba trading loop."""

    def test_trailing_stop_long(self):
        """Long trade: price rises above activation, then drops through trail.

        Setup: Entry at bar 250 (price 100). Price rises to 120 over next bars
        (ATR ~2 with spread=2). activation_mult=2.0 → need 2*ATR ~4 profit.
        Price 120 means profit = 20, well above activation. Then price drops
        to 110 → trail_level = 120 - 2*ATR = ~116, low of 109 < 116 → exit.
        """
        n = 500
        close = np.full(n, 100.0)
        # Warmup: constant price → ATR = spread (high-low = 2.0)
        # After entry at 250: price rises to 120
        close[255:280] = 120.0
        # Then drops sharply — triggers trailing stop
        close[280:] = 110.0

        data = _make_controlled_data(n, close, spread=2.0)
        buy, sell = _make_signals(n, buy_bars=[250])
        result = simulate_trades_fast(
            "BTC/USDC", data, buy, sell,
            trail_activation_mult=2.0,
            trail_mult=2.0,
        )
        assert len(result.trades) >= 1
        trail_trades = [t for t in result.trades if t["reason"] == "trailing_stop"]
        assert len(trail_trades) >= 1, (
            f"Expected trailing_stop exit, got reasons: "
            f"{[t['reason'] for t in result.trades]}"
        )
        assert trail_trades[0]["direction"] == "long"

    def test_trailing_stop_short(self):
        """Short trade: price drops (profit), then rises through trail.

        Setup: Short entry at bar 250 (price 100). Price drops to 80 →
        profit = 20, activation at 2*ATR ~4. Then price rises to 90 →
        trail_level = 80 + 2*ATR = ~84, high of 91 > 84 → exit.
        """
        n = 500
        close = np.full(n, 100.0)
        # Price drops = profit for short
        close[255:280] = 80.0
        # Then rises = trailing stop triggers
        close[280:] = 90.0

        data = _make_controlled_data(n, close, spread=2.0)
        buy, sell = _make_signals(n, sell_bars=[250])
        result = simulate_trades_fast(
            "BTC/USDC", data, buy, sell,
            long_only=False,
            trail_activation_mult=2.0,
            trail_mult=2.0,
        )
        assert len(result.trades) >= 1
        trail_trades = [t for t in result.trades if t["reason"] == "trailing_stop"]
        assert len(trail_trades) >= 1, (
            f"Expected trailing_stop exit, got reasons: "
            f"{[t['reason'] for t in result.trades]}"
        )
        assert trail_trades[0]["direction"] == "short"

    def test_trailing_stop_not_activated_when_no_profit(self):
        """Trade never reaches activation threshold → no trailing stop exit.

        Price stays flat (no profit above activation threshold).
        With activation_mult=5.0 and ATR~2, need 10 points profit — never reached.
        """
        n = 500
        close = np.full(n, 100.0)
        # Small move up — not enough for activation (need 5*2=10)
        close[255:] = 103.0

        data = _make_controlled_data(n, close, spread=2.0)
        buy, sell = _make_signals(n, buy_bars=[250])
        result = simulate_trades_fast(
            "BTC/USDC", data, buy, sell,
            trail_activation_mult=5.0,
            trail_mult=2.0,
        )
        # Should NOT have any trailing stop exits
        trail_trades = [t for t in result.trades if t["reason"] == "trailing_stop"]
        assert len(trail_trades) == 0, (
            f"Unexpected trailing_stop exit: {trail_trades}"
        )

    def test_catastrophic_stop_beats_trailing(self):
        """When both could trigger, catastrophic stop takes priority.

        Catastrophic stop is checked first in the loop. Price crashes 50% →
        catastrophic fires before trailing stop can activate.
        """
        n = 500
        close = np.full(n, 100.0)
        # Brief up move to activate trailing, then catastrophic crash
        close[255:260] = 110.0
        close[260:] = 40.0  # 60% crash — catastrophic at 10%

        data = _make_controlled_data(n, close, spread=2.0)
        buy, sell = _make_signals(n, buy_bars=[250])
        result = simulate_trades_fast(
            "BTC/USDC", data, buy, sell,
            trail_activation_mult=2.0,
            trail_mult=2.0,
        )
        assert len(result.trades) >= 1
        # The first exit should be catastrophic_stop (checked first)
        first_trade = result.trades[0]
        assert first_trade["reason"] == "catastrophic_stop", (
            f"Expected catastrophic_stop, got {first_trade['reason']}"
        )

    def test_trailing_stop_disabled_by_default(self):
        """Without trail params, trailing stop never triggers (backward compat).

        Same price data as test_trailing_stop_long, but without trail params.
        """
        n = 500
        close = np.full(n, 100.0)
        close[255:280] = 120.0
        close[280:] = 110.0

        data = _make_controlled_data(n, close, spread=2.0)
        buy, sell = _make_signals(n, buy_bars=[250])

        # No trail params → defaults to 0.0 = disabled
        result = simulate_trades_fast("BTC/USDC", data, buy, sell)
        trail_trades = [t for t in result.trades if t["reason"] == "trailing_stop"]
        assert len(trail_trades) == 0, (
            f"Trailing stop should be disabled by default, got: {trail_trades}"
        )

    def test_trailing_stop_bypasses_min_hold(self):
        """Trailing stop can fire before min_hold (same as catastrophic stop).

        Entry at bar 250. Min hold = 2 bars. Price spikes up at bar 251
        (activates trail), drops at bar 251 same bar (trail fires).
        Since trailing stop bypasses min_hold, exit should happen at bar 251.
        """
        n = 500
        close = np.full(n, 100.0)
        # Bar 251: high goes way up (activates trail), but close/low drops
        # We need the trail to activate AND trigger in the same bar or at bar 251
        close[251] = 100.0  # close is normal

        data = _make_controlled_data(n, close, spread=2.0)
        # Manually set high at 251 very high (activates trail)
        # and then at 252 low drops below trail level
        df = data["1h"]
        # At bar 251: huge spike up → peak_price updates, trail activates
        df.iloc[251, df.columns.get_loc("high")] = 130.0
        # At bar 252: price crashes, low drops below trail
        df.iloc[252, df.columns.get_loc("close")] = 100.0
        df.iloc[252, df.columns.get_loc("high")] = 105.0
        df.iloc[252, df.columns.get_loc("low")] = 90.0

        buy, sell = _make_signals(n, buy_bars=[250])
        result = simulate_trades_fast(
            "BTC/USDC", data, buy, sell,
            trail_activation_mult=2.0,
            trail_mult=2.0,
        )
        # Should have trailing stop exit — bars_held = 2 (entry 250, exit 252)
        # MIN_HOLD_HOURS = 2 — signal exit can't fire at bar 252 (exactly min_hold)
        # but trailing stop bypasses min_hold
        trail_trades = [t for t in result.trades if t["reason"] == "trailing_stop"]
        assert len(trail_trades) >= 1, (
            f"Trailing stop should bypass min_hold, got reasons: "
            f"{[t['reason'] for t in result.trades]}"
        )
