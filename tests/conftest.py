"""Shared test fixtures and factory functions for QRE tests."""

import numpy as np
import pandas as pd
import pytest

from qre.core.strategy import MACDRSIStrategy


# =============================================================================
# Shared fixtures
# =============================================================================


@pytest.fixture
def strategy():
    return MACDRSIStrategy()


# =============================================================================
# Factory functions — call directly in tests or fixtures
# =============================================================================


def make_1h_ohlcv(n_bars: int = 500, seed: int = 42, start: str = "2025-01-01") -> pd.DataFrame:
    """Create random-walk 1H OHLCV DataFrame."""
    np.random.seed(seed)
    dates = pd.date_range(start, periods=n_bars, freq="1h")
    close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.abs(np.random.randn(n_bars))
    low = close - np.abs(np.random.randn(n_bars))
    open_ = close + np.random.randn(n_bars) * 0.2
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=dates,
    )


def resample_to_multi_tf(df_1h: pd.DataFrame) -> dict:
    """Resample 1H DataFrame to dict with 1h/4h/8h/1d keys."""
    data = {"1h": df_1h}
    for tf, rule in [("4h", "4h"), ("8h", "8h"), ("1d", "1D")]:
        resampled = df_1h.resample(rule).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        data[tf] = resampled
    return data
