"""Unit tests for QRE MACD+RSI strategy."""

import numpy as np
import pandas as pd
import pytest

from qre.core.strategy import MACDRSIStrategy


@pytest.fixture
def strategy():
    return MACDRSIStrategy()


@pytest.fixture
def sample_data():
    """Create minimal multi-TF OHLCV data structure for testing."""
    np.random.seed(42)
    n_bars = 500
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="1h")
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.abs(np.random.randn(n_bars))
    low = close - np.abs(np.random.randn(n_bars))
    open_ = close + np.random.randn(n_bars) * 0.2

    data = {
        "1h": pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=dates),
    }

    for tf, factor in [("2h", 2), ("4h", 4), ("6h", 6), ("8h", 8), ("12h", 12), ("1d", 24)]:
        tf_len = n_bars // factor
        if tf_len < 10:
            continue
        tf_idx = dates[::factor][:tf_len]
        data[tf] = pd.DataFrame(
            {"open": open_[::factor][:tf_len], "high": high[::factor][:tf_len],
             "low": low[::factor][:tf_len], "close": close[::factor][:tf_len]},
            index=tf_idx,
        )

    return data


class TestMACDRSIStrategy:
    def test_name(self, strategy):
        assert strategy.name == "macd_rsi"

    def test_version(self, strategy):
        assert strategy.version == "2.0.0"

    def test_required_indicators(self, strategy):
        indicators = strategy.get_required_indicators()
        assert "macd" in indicators
        assert "rsi" in indicators or "stochrsi" in indicators

    def test_default_params(self, strategy):
        """Default params should include MACD and RSI settings."""
        params = strategy.get_default_params()
        assert "macd_fast" in params
        assert "macd_slow" in params
        assert "macd_mode" in params
        assert "rsi_mode" in params

    def test_precompute_signals_shape(self, strategy, sample_data):
        """precompute_signals returns arrays of correct shape."""
        params = strategy.get_default_params()
        buy_votes, sell_votes, rsi_gates = strategy.precompute_signals(sample_data, params)
        n_bars = len(sample_data["1h"])
        assert buy_votes.shape[1] == n_bars
        assert sell_votes.shape[1] == n_bars
        assert rsi_gates.shape == (4, n_bars)

    def test_precompute_signals_boolean(self, strategy, sample_data):
        """Signal arrays should contain boolean-like values (0 or 1)."""
        params = strategy.get_default_params()
        buy_votes, sell_votes, _ = strategy.precompute_signals(sample_data, params)
        assert set(np.unique(buy_votes)).issubset({0, 1, True, False})
        assert set(np.unique(sell_votes)).issubset({0, 1, True, False})

    def test_no_adx_filter(self, strategy, sample_data):
        """ADX filter removed in QRE — no import errors even with use_adx_filter=True."""
        params = strategy.get_default_params()
        # Even if someone passes use_adx_filter, it should be ignored
        params["use_adx_filter"] = True
        # Should NOT raise ImportError (adx_rsi module doesn't exist in QRE)
        buy_votes, sell_votes, rsi_gates = strategy.precompute_signals(sample_data, params)
        assert buy_votes.shape[1] == len(sample_data["1h"])
