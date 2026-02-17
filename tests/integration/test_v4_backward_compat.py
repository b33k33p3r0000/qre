"""Integration test: v4.0 with lookback=0 and trend_strict=0 matches v3.0 behavior."""

import numpy as np
import pandas as pd
import pytest

from qre.core.strategy import MACDRSIStrategy


@pytest.fixture
def strategy():
    return MACDRSIStrategy()


@pytest.fixture
def realistic_data():
    """Longer dataset with trends for meaningful signal count."""
    np.random.seed(123)
    n_bars = 2000
    dates_1h = pd.date_range("2024-01-01", periods=n_bars, freq="1h")

    # Generate price with trends
    trend = np.sin(np.linspace(0, 8 * np.pi, n_bars)) * 20
    noise = np.cumsum(np.random.randn(n_bars) * 0.3)
    close = 100 + trend + noise
    close = np.maximum(close, 10)  # floor

    high = close + np.abs(np.random.randn(n_bars))
    low = close - np.abs(np.random.randn(n_bars))
    open_ = close + np.random.randn(n_bars) * 0.2

    data = {
        "1h": pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close},
            index=dates_1h,
        ),
    }

    # Add higher TFs via resample
    for tf, rule in [("4h", "4h"), ("8h", "8h"), ("1d", "1D")]:
        resampled = data["1h"].resample(rule).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        data[tf] = resampled

    return data


class TestV4BackwardCompat:
    def test_v3_params_produce_same_signals(self, strategy, realistic_data):
        """v3.0 default params (no lookback, no trend) = identical signals."""
        v3_params = {
            "macd_fast": 12, "macd_slow": 30, "macd_signal": 8,
            "rsi_period": 16, "rsi_lower": 32, "rsi_upper": 68,
        }

        # v3.0 behavior: only 1H data, no new params
        data_1h_only = {"1h": realistic_data["1h"]}
        buy_v3, sell_v3 = strategy.precompute_signals(data_1h_only, v3_params)

        # v4.0 with legacy params
        v4_params = {**v3_params, "rsi_lookback": 0, "trend_strict": 0, "trend_tf": "4h"}
        buy_v4, sell_v4 = strategy.precompute_signals(realistic_data, v4_params)

        np.testing.assert_array_equal(buy_v3, buy_v4)
        np.testing.assert_array_equal(sell_v3, sell_v4)

    def test_lookback_generates_more_trades(self, strategy, realistic_data):
        """rsi_lookback=6 generates strictly more signals than lookback=0."""
        params = {
            "macd_fast": 12, "macd_slow": 30, "macd_signal": 8,
            "rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70,
            "rsi_lookback": 0, "trend_strict": 0, "trend_tf": "4h",
        }
        buy_0, sell_0 = strategy.precompute_signals(realistic_data, params)

        params["rsi_lookback"] = 6
        buy_6, sell_6 = strategy.precompute_signals(realistic_data, params)

        # Lookback should generate more or equal signals
        assert buy_6.sum() >= buy_0.sum()
        assert (buy_6.sum() + sell_6.sum()) > (buy_0.sum() + sell_0.sum()), \
            "Lookback=6 should generate more total signals than lookback=0"

    def test_trend_filter_reduces_signals(self, strategy, realistic_data):
        """trend_strict=1 should reduce or equal signal count."""
        params = {
            "macd_fast": 12, "macd_slow": 30, "macd_signal": 8,
            "rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70,
            "rsi_lookback": 3, "trend_strict": 0, "trend_tf": "4h",
        }
        buy_off, sell_off = strategy.precompute_signals(realistic_data, params)

        params["trend_strict"] = 1
        buy_on, sell_on = strategy.precompute_signals(realistic_data, params)

        total_off = buy_off.sum() + sell_off.sum()
        total_on = buy_on.sum() + sell_on.sum()
        assert total_on <= total_off

    def test_no_simultaneous_buy_sell(self, strategy, realistic_data):
        """Even with lookback + trend, buy and sell never overlap."""
        params = {
            "macd_fast": 8, "macd_slow": 21, "macd_signal": 5,
            "rsi_period": 10, "rsi_lower": 40, "rsi_upper": 60,
            "rsi_lookback": 12, "trend_strict": 1, "trend_tf": "4h",
        }
        buy, sell = strategy.precompute_signals(realistic_data, params)
        overlap = buy & sell
        assert not overlap.any(), "Buy and sell signals overlap with lookback + trend"
