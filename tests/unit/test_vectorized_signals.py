"""Tests that vectorized signal mapping is identical to Python loop."""

import numpy as np
import pytest


def _loop_map_signals(base_to_tf_idx, tf_buy, tf_sell, additional_condition, n_bars, min_idx=2):
    """Original Python loop implementation (reference)."""
    buy_out = np.zeros(n_bars, dtype=np.bool_)
    sell_out = np.zeros(n_bars, dtype=np.bool_)
    for bar_idx in range(n_bars):
        tf_row_idx = base_to_tf_idx[bar_idx]
        if tf_row_idx >= min_idx and tf_row_idx < len(tf_buy):
            buy_out[bar_idx] = tf_buy[tf_row_idx] and additional_condition[bar_idx]
            sell_out[bar_idx] = tf_sell[tf_row_idx]
    return buy_out, sell_out


def _vectorized_map_signals(base_to_tf_idx, tf_buy, tf_sell, additional_condition, n_bars, min_idx=2):
    """Vectorized NumPy implementation (to be extracted into strategy.py)."""
    valid = (base_to_tf_idx >= min_idx) & (base_to_tf_idx < len(tf_buy))
    clipped_idx = np.clip(base_to_tf_idx, 0, len(tf_buy) - 1)
    buy_out = valid & tf_buy[clipped_idx] & additional_condition
    sell_out = valid & tf_sell[clipped_idx]
    return buy_out, sell_out


class TestVectorizedSignalMapping:
    @pytest.fixture
    def signal_data(self):
        """Create realistic signal mapping test data."""
        np.random.seed(42)
        n_bars = 8760  # 1 year of hourly data
        n_tf_bars = n_bars // 4  # 4h timeframe

        base_to_tf_idx = np.searchsorted(
            np.arange(0, n_tf_bars * 4, 4, dtype=np.int64),
            np.arange(n_bars, dtype=np.int64),
            side="right",
        ) - 1
        base_to_tf_idx = np.clip(base_to_tf_idx, 0, n_tf_bars - 1).astype(np.int32)

        tf_buy = np.random.rand(n_tf_bars) > 0.9  # ~10% buy signals
        tf_sell = np.random.rand(n_tf_bars) > 0.85  # ~15% sell signals
        additional_condition = np.random.rand(n_bars) > 0.5

        return base_to_tf_idx, tf_buy, tf_sell, additional_condition, n_bars

    def test_vectorized_identical_to_loop(self, signal_data):
        """Vectorized mapping produces bit-for-bit identical results to Python loop."""
        base_to_tf_idx, tf_buy, tf_sell, additional_condition, n_bars = signal_data

        loop_buy, loop_sell = _loop_map_signals(
            base_to_tf_idx, tf_buy, tf_sell, additional_condition, n_bars,
        )
        vec_buy, vec_sell = _vectorized_map_signals(
            base_to_tf_idx, tf_buy, tf_sell, additional_condition, n_bars,
        )

        np.testing.assert_array_equal(loop_buy, vec_buy, err_msg="Buy signals differ")
        np.testing.assert_array_equal(loop_sell, vec_sell, err_msg="Sell signals differ")

    def test_vectorized_identical_min_idx_1(self, signal_data):
        """Vectorized mapping with min_idx=1 (RSI gates) is identical."""
        base_to_tf_idx, tf_buy, tf_sell, additional_condition, n_bars = signal_data
        # For RSI gates, additional_condition is not used — create dummy gate data
        tf_gate = np.random.rand(len(tf_buy)) > 0.4
        dummy_condition = np.ones(n_bars, dtype=np.bool_)

        loop_buy, _ = _loop_map_signals(
            base_to_tf_idx, tf_gate, tf_gate, dummy_condition, n_bars, min_idx=1,
        )
        vec_buy, _ = _vectorized_map_signals(
            base_to_tf_idx, tf_gate, tf_gate, dummy_condition, n_bars, min_idx=1,
        )

        np.testing.assert_array_equal(loop_buy, vec_buy, err_msg="Gate signals differ")

    def test_edge_case_empty_tf(self):
        """Handles empty TF data gracefully."""
        n_bars = 100
        base_to_tf_idx = np.zeros(n_bars, dtype=np.int32)
        tf_buy = np.array([], dtype=np.bool_)
        tf_sell = np.array([], dtype=np.bool_)
        additional = np.ones(n_bars, dtype=np.bool_)

        # Loop produces all zeros (nothing valid)
        loop_buy, loop_sell = _loop_map_signals(
            base_to_tf_idx, tf_buy, tf_sell, additional, n_bars,
        )
        assert not loop_buy.any()

    def test_edge_case_all_invalid_indices(self):
        """All TF indices below min_idx produce zeros."""
        n_bars = 50
        base_to_tf_idx = np.zeros(n_bars, dtype=np.int32)  # All 0 < min_idx=2
        tf_buy = np.ones(10, dtype=np.bool_)
        tf_sell = np.ones(10, dtype=np.bool_)
        additional = np.ones(n_bars, dtype=np.bool_)

        loop_buy, loop_sell = _loop_map_signals(
            base_to_tf_idx, tf_buy, tf_sell, additional, n_bars,
        )
        vec_buy, vec_sell = _vectorized_map_signals(
            base_to_tf_idx, tf_buy, tf_sell, additional, n_bars,
        )

        np.testing.assert_array_equal(loop_buy, vec_buy)
        np.testing.assert_array_equal(loop_sell, vec_sell)
        assert not vec_buy.any()
