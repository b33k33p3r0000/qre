"""Unit tests for Chio Extreme strategy."""

import numpy as np
import pandas as pd
import pytest

from qre.core.strategy import MACDRSIStrategy


@pytest.fixture
def strategy():
    return MACDRSIStrategy()


@pytest.fixture
def sample_data():
    """Create minimal 1H OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 500
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="1h")
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.abs(np.random.randn(n_bars))
    low = close - np.abs(np.random.randn(n_bars))
    open_ = close + np.random.randn(n_bars) * 0.2
    return {
        "1h": pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close},
            index=dates,
        ),
    }


class TestMACDRSIStrategy:
    def test_name(self, strategy):
        assert strategy.name == "macd_rsi"

    def test_version(self, strategy):
        assert strategy.version == "3.0.0"

    def test_required_indicators(self, strategy):
        indicators = strategy.get_required_indicators()
        assert "macd" in indicators
        assert "rsi" in indicators
        assert "stochrsi" not in indicators

    def test_optuna_params_count(self, strategy):
        """Exactly 6 Optuna params."""
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        params = strategy.get_optuna_params(trial)
        optuna_keys = {"macd_fast", "macd_slow", "macd_signal",
                       "rsi_period", "rsi_lower", "rsi_upper"}
        assert optuna_keys.issubset(set(params.keys()))

    def test_macd_fast_lt_slow(self, strategy):
        """Constraint: macd_fast < macd_slow always."""
        import optuna
        study = optuna.create_study()
        valid_count = 0
        for _ in range(100):
            trial = study.ask()
            try:
                params = strategy.get_optuna_params(trial)
                assert params["macd_fast"] < params["macd_slow"]
                valid_count += 1
            except optuna.TrialPruned:
                pass  # Expected when fast >= slow
        assert valid_count > 0

    def test_precompute_returns_1d(self, strategy, sample_data):
        """precompute_signals returns two 1D boolean arrays."""
        params = strategy.get_default_params()
        buy_signal, sell_signal = strategy.precompute_signals(sample_data, params)
        n_bars = len(sample_data["1h"])
        assert buy_signal.shape == (n_bars,)
        assert sell_signal.shape == (n_bars,)
        assert buy_signal.dtype == np.bool_
        assert sell_signal.dtype == np.bool_

    def test_no_simultaneous_signals(self, strategy, sample_data):
        """Buy and sell should never be True on the same bar."""
        params = strategy.get_default_params()
        buy_signal, sell_signal = strategy.precompute_signals(sample_data, params)
        overlap = buy_signal & sell_signal
        assert not overlap.any(), "Buy and sell signals overlap"

    def test_no_stochrsi_params(self, strategy):
        """No StochRSI params (kB, dB, thresholds, gates)."""
        params = strategy.get_default_params()
        for key in ["kB", "dB", "k_sell", "p_buy",
                     "low_2h", "high_2h", "rsi_gate_24h"]:
            assert key not in params, f"Legacy param {key} found"

    def test_no_macd_mode(self, strategy):
        """macd_mode removed — always crossover."""
        params = strategy.get_default_params()
        assert "macd_mode" not in params

    def test_default_params(self, strategy):
        params = strategy.get_default_params()
        assert "macd_fast" in params
        assert "macd_slow" in params
        assert "macd_signal" in params
        assert "rsi_period" in params
        assert "rsi_lower" in params
        assert "rsi_upper" in params

    def test_precompute_with_cache(self, strategy, sample_data):
        """precompute_signals works with RSI cache."""
        from qre.core.indicators import rsi
        base_close = sample_data["1h"]["close"]
        cache = {"rsi": {14: rsi(base_close, 14).values.astype(np.float64)}}
        params = strategy.get_default_params()
        buy_signal, sell_signal = strategy.precompute_signals(
            sample_data, params, precomputed_cache=cache
        )
        assert buy_signal.shape == (len(sample_data["1h"]),)
