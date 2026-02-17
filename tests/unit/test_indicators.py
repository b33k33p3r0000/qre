"""Unit tests for QRE indicators."""

import numpy as np
import pandas as pd
import pytest

from qre.core.indicators import rsi, macd


class TestRSI:
    def test_rsi_basic(self):
        """RSI on known data produces expected range."""
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(100)))
        result = rsi(prices, length=14)
        assert len(result) == len(prices)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_uptrend(self):
        """RSI in strong uptrend should be > 50."""
        np.random.seed(42)
        # Realistický uptrend — cumsum s pozitivním driftem
        prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5 + 0.5))
        result = rsi(prices, length=14)
        valid = result.dropna()
        assert valid.iloc[-1] > 50

    def test_rsi_downtrend(self):
        """RSI in strong downtrend should be < 50."""
        prices = pd.Series(np.arange(200, 100, -1, dtype=float))
        result = rsi(prices, length=14)
        assert result.iloc[-1] < 40

    def test_rsi_length(self):
        """RSI respects length parameter."""
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(100)))
        r14 = rsi(prices, length=14)
        r21 = rsi(prices, length=21)
        assert not np.allclose(r14.values[30:], r21.values[30:], equal_nan=True)


class TestNoRemovedIndicators:
    def test_no_stochrsi(self):
        """stochrsi removed — not needed for Quant Whale Strategy."""
        import qre.core.indicators as mod
        assert not hasattr(mod, "stochrsi")

    def test_no_macd_rising(self):
        """macd_rising removed — only crossover mode."""
        import qre.core.indicators as mod
        assert not hasattr(mod, "macd_rising")

    def test_no_ema(self):
        """standalone ema removed — MACD computes EMA internally."""
        import qre.core.indicators as mod
        assert not hasattr(mod, "ema")


class TestMACD:
    def test_macd_output_shape(self):
        """MACD returns three Series of same length as input."""
        prices = pd.Series(np.random.randn(100) + 100)
        macd_line, signal_line, histogram = macd(prices, fast_period=12, slow_period=26, signal_period=9)
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)

    def test_macd_histogram_is_diff(self):
        """MACD histogram = MACD line - signal line."""
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(100)))
        macd_line, signal_line, histogram = macd(prices, fast_period=12, slow_period=26, signal_period=9)
        valid_mask = macd_line.notna() & signal_line.notna() & histogram.notna()
        np.testing.assert_allclose(
            histogram[valid_mask].values,
            (macd_line[valid_mask] - signal_line[valid_mask]).values,
            atol=1e-10,
        )

    def test_macd_uptrend(self):
        """MACD line should be positive in uptrend."""
        prices = pd.Series(np.linspace(100, 200, 100))
        macd_line, _, _ = macd(prices, fast_period=12, slow_period=26, signal_period=9)
        assert macd_line.iloc[-1] > 0
