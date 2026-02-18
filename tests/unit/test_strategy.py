"""Unit tests for Quant Whale Strategy strategy."""

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


@pytest.fixture
def sample_data_multi_tf():
    """Create 1H + 4H + 8H + 1D OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 500
    dates_1h = pd.date_range("2025-01-01", periods=n_bars, freq="1h")
    close_1h = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high_1h = close_1h + np.abs(np.random.randn(n_bars))
    low_1h = close_1h - np.abs(np.random.randn(n_bars))
    open_1h = close_1h + np.random.randn(n_bars) * 0.2

    data = {
        "1h": pd.DataFrame(
            {"open": open_1h, "high": high_1h, "low": low_1h, "close": close_1h},
            index=dates_1h,
        ),
    }

    # Resample to higher TFs
    for tf, rule in [("4h", "4h"), ("8h", "8h"), ("1d", "1D")]:
        resampled = data["1h"].resample(rule).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        data[tf] = resampled

    return data


class TestMACDRSIStrategy:
    def test_name(self, strategy):
        assert strategy.name == "macd_rsi"

    def test_version(self, strategy):
        assert strategy.version == "4.1.0"

    def test_required_indicators(self, strategy):
        indicators = strategy.get_required_indicators()
        assert "macd" in indicators
        assert "rsi" in indicators
        assert "stochrsi" not in indicators

    def test_optuna_params_count(self, strategy):
        """10 Optuna params (6 original + rsi_lookback + trend_tf + trend_strict + allow_flip)."""
        import optuna
        study = optuna.create_study()
        optuna_keys = {"macd_fast", "macd_slow", "macd_signal",
                       "rsi_period", "rsi_lower", "rsi_upper",
                       "rsi_lookback", "trend_tf", "trend_strict",
                       "allow_flip"}
        # Retry on prune (MACD spread constraint may reject random combos)
        for _ in range(50):
            trial = study.ask()
            try:
                params = strategy.get_optuna_params(trial)
                assert optuna_keys.issubset(set(params.keys()))
                return
            except optuna.TrialPruned:
                continue
        pytest.fail("All 50 trials pruned — check MACD constraint")

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

    def test_macd_fast_is_float(self, strategy):
        """macd_fast should be float in Optuna params."""
        import optuna
        study = optuna.create_study()
        for _ in range(50):
            trial = study.ask()
            try:
                params = strategy.get_optuna_params(trial)
            except optuna.TrialPruned:
                continue
            assert isinstance(params["macd_fast"], float), f"macd_fast is {type(params['macd_fast'])}"
            assert 1.0 <= params["macd_fast"] <= 20.0
            return
        pytest.fail("All 50 trials pruned")

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


class TestRSILookback:
    def test_lookback_zero_is_deterministic(self, strategy, sample_data):
        """rsi_lookback=0 produces identical signals across two runs."""
        params = strategy.get_default_params()
        params["rsi_lookback"] = 0
        buy_a, sell_a = strategy.precompute_signals(sample_data, params)

        buy_b, sell_b = strategy.precompute_signals(sample_data, params)

        np.testing.assert_array_equal(buy_a, buy_b)
        np.testing.assert_array_equal(sell_a, sell_b)

    def test_lookback_increases_signals(self, strategy, sample_data):
        """rsi_lookback > 0 should produce >= signals than lookback=0."""
        params = strategy.get_default_params()
        params["rsi_lookback"] = 0
        buy_0, sell_0 = strategy.precompute_signals(sample_data, params)

        params["rsi_lookback"] = 6
        buy_6, sell_6 = strategy.precompute_signals(sample_data, params)

        # With lookback, every v3.0 signal should still be present
        assert (buy_0 & buy_6).sum() == buy_0.sum(), "Lookback lost original buy signals"
        assert (sell_0 & sell_6).sum() == sell_0.sum(), "Lookback lost original sell signals"
        # And there should be at least as many
        assert buy_6.sum() >= buy_0.sum()
        assert sell_6.sum() >= sell_0.sum()

    def test_lookback_in_optuna_params(self, strategy):
        """rsi_lookback is in Optuna search space (1-3)."""
        import optuna
        study = optuna.create_study()
        for _ in range(50):
            trial = study.ask()
            try:
                params = strategy.get_optuna_params(trial)
            except optuna.TrialPruned:
                continue
            assert "rsi_lookback" in params
            assert 1 <= params["rsi_lookback"] <= 3
            return
        pytest.fail("All 50 trials pruned")

    def test_lookback_in_default_params(self, strategy):
        """Default rsi_lookback is midpoint of range (2)."""
        params = strategy.get_default_params()
        assert "rsi_lookback" in params
        assert params["rsi_lookback"] == 2


class TestTrendFilter:
    def test_trend_strict_zero_is_passthrough(self, strategy, sample_data_multi_tf):
        """trend_strict=0 produces identical signals to no trend filter."""
        params = strategy.get_default_params()
        buy_no_tf, sell_no_tf = strategy.precompute_signals(sample_data_multi_tf, params)

        params["trend_strict"] = 0
        params["trend_tf"] = "4h"
        buy_off, sell_off = strategy.precompute_signals(sample_data_multi_tf, params)

        np.testing.assert_array_equal(buy_no_tf, buy_off)
        np.testing.assert_array_equal(sell_no_tf, sell_off)

    def test_trend_strict_filters_signals(self, strategy, sample_data_multi_tf):
        """trend_strict=1 should produce <= signals than trend_strict=0."""
        params = strategy.get_default_params()
        params["trend_strict"] = 0
        buy_off, sell_off = strategy.precompute_signals(sample_data_multi_tf, params)

        params["trend_strict"] = 1
        params["trend_tf"] = "4h"
        buy_on, sell_on = strategy.precompute_signals(sample_data_multi_tf, params)

        assert buy_on.sum() <= buy_off.sum()
        assert sell_on.sum() <= sell_off.sum()

    def test_trend_filter_missing_tf_passthrough(self, strategy, sample_data):
        """If trend_tf data is missing, trend filter passes through."""
        params_base = strategy.get_default_params()
        buy_expected, sell_expected = strategy.precompute_signals(sample_data, params_base)

        params = strategy.get_default_params()
        params["trend_strict"] = 1
        params["trend_tf"] = "4h"  # Not in sample_data (only has 1h)

        buy_base, sell_base = strategy.precompute_signals(sample_data, params)
        # Should not crash, should produce signals identical to no trend filter
        np.testing.assert_array_equal(buy_base, buy_expected)
        np.testing.assert_array_equal(sell_base, sell_expected)

    def test_trend_params_in_optuna(self, strategy):
        """trend_tf and trend_strict are in Optuna search space."""
        import optuna
        study = optuna.create_study()
        for _ in range(20):
            trial = study.ask()
            try:
                params = strategy.get_optuna_params(trial)
            except optuna.TrialPruned:
                continue
            assert "trend_tf" in params
            assert params["trend_tf"] in ("4h", "8h", "1d")
            assert "trend_strict" in params
            assert params["trend_strict"] in (0, 1)
            return
        pytest.fail("All 20 trials pruned")

    def test_trend_default_params(self, strategy):
        """Default trend params disable trend filter."""
        params = strategy.get_default_params()
        assert params.get("trend_strict", 0) == 0
