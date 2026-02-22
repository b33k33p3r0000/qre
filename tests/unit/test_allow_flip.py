"""Tests for allow_flip parameter (v4.2.0)."""

import numpy as np
import pandas as pd
import pytest

from qre.core.backtest import simulate_trades_fast
from qre.core.strategy import MACDRSIStrategy
from tests.conftest import resample_to_multi_tf


# strategy fixture is inherited from tests/conftest.py


@pytest.fixture
def sample_data_multi_tf():
    """Create 1H + higher TF data with sinusoidal trends for signal generation."""
    np.random.seed(42)
    n_bars = 2000
    dates_1h = pd.date_range("2024-01-01", periods=n_bars, freq="1h")

    trend = np.sin(np.linspace(0, 10 * np.pi, n_bars)) * 15
    noise = np.cumsum(np.random.randn(n_bars) * 0.2)
    close = 100 + trend + noise
    close = np.maximum(close, 10)
    high = close + np.abs(np.random.randn(n_bars))
    low = close - np.abs(np.random.randn(n_bars))
    open_ = close + np.random.randn(n_bars) * 0.1

    df_1h = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=dates_1h,
    )
    return resample_to_multi_tf(df_1h)


class TestAllowFlipParam:
    def test_allow_flip_in_optuna_params(self, strategy):
        """allow_flip is in Optuna search space as int 0-1."""
        import optuna
        study = optuna.create_study()
        for _ in range(50):
            trial = study.ask()
            try:
                params = strategy.get_optuna_params(trial)
            except optuna.TrialPruned:
                continue
            assert "allow_flip" in params
            assert params["allow_flip"] in (0, 1)
            return
        pytest.fail("All 50 trials pruned")

    def test_allow_flip_in_default_params(self, strategy):
        """Default allow_flip is 0 (selective mode, v4.2.1+)."""
        params = strategy.get_default_params()
        assert params["allow_flip"] == 0

    def test_param_count_is_10(self, strategy):
        """v4.2.1 has 10 Optuna params (9 original + allow_flip)."""
        import optuna
        study = optuna.create_study()
        expected_keys = {
            "macd_fast", "macd_slow", "macd_signal",
            "rsi_period", "rsi_lower", "rsi_upper",
            "rsi_lookback", "trend_tf", "trend_strict",
            "allow_flip",
        }
        for _ in range(50):
            trial = study.ask()
            try:
                params = strategy.get_optuna_params(trial)
                assert expected_keys.issubset(set(params.keys()))
                return
            except optuna.TrialPruned:
                continue
        pytest.fail("All 50 trials pruned")


class TestAllowFlipBacktest:
    def test_flip_on_produces_no_flat_periods(self, strategy, sample_data_multi_tf):
        """allow_flip=1: strategy flips on signal exits (always-in behavior)."""
        params = strategy.get_default_params()
        params["allow_flip"] = 1
        params["trend_strict"] = 0
        params["rsi_lookback"] = 8

        buy, sell = strategy.precompute_signals(sample_data_multi_tf, params)
        result = simulate_trades_fast(
            "BTC/USDC", sample_data_multi_tf, buy, sell, allow_flip=True,
        )

        assert len(result.trades) >= 5, \
            f"Need >= 5 trades for meaningful flip test, got {len(result.trades)}"
        consecutive_flips = 0
        for i in range(len(result.trades) - 1):
            if result.trades[i]["reason"] == "signal":
                if result.trades[i]["exit_ts"] == result.trades[i + 1]["entry_ts"]:
                    consecutive_flips += 1
        signal_exits = sum(1 for t in result.trades if t["reason"] == "signal")
        assert signal_exits > 3, \
            f"Need > 3 signal exits for meaningful test, got {signal_exits}"
        assert consecutive_flips >= signal_exits * 0.8, \
            f"Only {consecutive_flips}/{signal_exits} signal exits flipped"

    def test_flip_off_produces_flat_periods(self, strategy, sample_data_multi_tf):
        """allow_flip=0: strategy has flat periods between positions."""
        params = strategy.get_default_params()
        params["allow_flip"] = 0
        params["trend_strict"] = 0
        params["rsi_lookback"] = 8

        buy, sell = strategy.precompute_signals(sample_data_multi_tf, params)
        result = simulate_trades_fast(
            "BTC/USDC", sample_data_multi_tf, buy, sell, allow_flip=False,
        )

        assert len(result.trades) >= 5, \
            f"Need >= 5 trades for meaningful flip test, got {len(result.trades)}"
        consecutive_flips = 0
        for i in range(len(result.trades) - 1):
            if result.trades[i]["reason"] == "signal":
                if result.trades[i]["exit_ts"] == result.trades[i + 1]["entry_ts"]:
                    consecutive_flips += 1
        signal_exits = sum(1 for t in result.trades if t["reason"] == "signal")
        assert signal_exits > 3, \
            f"Need > 3 signal exits for meaningful test, got {signal_exits}"
        assert consecutive_flips < signal_exits * 0.5, \
            f"Too many flips ({consecutive_flips}/{signal_exits}) with allow_flip=0"

    def test_flip_on_backward_compat(self, strategy, sample_data_multi_tf):
        """allow_flip=True produces identical results to default (no allow_flip param)."""
        params = strategy.get_default_params()
        params["trend_strict"] = 0
        params["rsi_lookback"] = 6

        buy, sell = strategy.precompute_signals(sample_data_multi_tf, params)

        result_flip = simulate_trades_fast(
            "BTC/USDC", sample_data_multi_tf, buy, sell, allow_flip=True,
        )
        result_default = simulate_trades_fast(
            "BTC/USDC", sample_data_multi_tf, buy, sell,
        )

        assert len(result_flip.trades) == len(result_default.trades)
        assert abs(result_flip.equity - result_default.equity) < 0.01

    def test_flip_off_fewer_trades(self, strategy, sample_data_multi_tf):
        """allow_flip=0 produces <= trades than allow_flip=1."""
        params = strategy.get_default_params()
        params["trend_strict"] = 0
        params["rsi_lookback"] = 6

        buy, sell = strategy.precompute_signals(sample_data_multi_tf, params)

        result_flip = simulate_trades_fast(
            "BTC/USDC", sample_data_multi_tf, buy, sell, allow_flip=True,
        )
        result_no_flip = simulate_trades_fast(
            "BTC/USDC", sample_data_multi_tf, buy, sell, allow_flip=False,
        )

        assert len(result_no_flip.trades) <= len(result_flip.trades), \
            f"No-flip ({len(result_no_flip.trades)}) should have <= trades than flip ({len(result_flip.trades)})"
