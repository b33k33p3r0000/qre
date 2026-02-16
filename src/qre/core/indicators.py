"""
Technical Indicators
====================
RSI, MACD — core indicators for Chio Extreme (MACD crossover + RSI extreme zones).
"""

from typing import Tuple

import numpy as np
import pandas as pd

from qre.config import RSI_LENGTH


def rsi(series: pd.Series, length: int = RSI_LENGTH) -> pd.Series:
    """
    RSI (Relative Strength Index).

    Args:
        series: Close prices.
        length: RSI period.

    Returns:
        pd.Series with RSI values (0-100).
    """
    delta = series.diff()

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = pd.Series(gain, index=series.index).rolling(length).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(length).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))

    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence).

    Args:
        series: Close prices.
        fast_period: Fast EMA period (default 12).
        slow_period: Slow EMA period (default 26).
        signal_period: Signal line period (default 9).

    Returns:
        Tuple[macd_line, signal_line, histogram].
    """
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram
