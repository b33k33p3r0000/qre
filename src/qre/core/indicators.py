"""
Technical Indicators
====================
RSI, StochRSI, MACD — čisté matematické funkce pro MACD+RSI strategii.
Přeneseno z optimizer s auditem. Odstraněno: adx, bollinger_bands, stochastic, atr.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from qre.config import RSI_LENGTH, STOCH_LENGTH


def rsi(series: pd.Series, length: int = RSI_LENGTH) -> pd.Series:
    """
    Výpočet RSI (Relative Strength Index).

    RSI měří momentum - zda je coin "overbought" nebo "oversold".

    Args:
        series: Cenová data (close prices)
        length: Perioda pro výpočet

    Returns:
        pd.Series s RSI hodnotami (0-100)
    """
    delta = series.diff()

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = pd.Series(gain, index=series.index).rolling(length).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(length).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))

    return 100 - (100 / (1 + rs))


def stochrsi(
    series: pd.Series,
    rsi_len: int = STOCH_LENGTH,
    stoch_len: int = STOCH_LENGTH,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Výpočet StochRSI - stochastický oscilátor aplikovaný na RSI.

    StochRSI kombinuje RSI a Stochastic - dává rychlejší signály.
    %K je hlavní linka, %D je signal line.

    Args:
        series: Cenová data
        rsi_len: Perioda pro RSI
        stoch_len: Perioda pro stochastic
        k_smooth: Vyhlazení %K
        d_smooth: Vyhlazení %D

    Returns:
        Tuple[%K, %D] - obě Series s hodnotami 0-1
    """
    r = rsi(series, rsi_len)

    lowest_low = r.rolling(stoch_len).min()
    highest_high = r.rolling(stoch_len).max()

    raw_k = (r - lowest_low) / (highest_high - lowest_low)

    k_line = raw_k.rolling(k_smooth).mean()
    d_line = k_line.rolling(d_smooth).mean()

    return k_line, d_line


def macd(
    series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Výpočet MACD (Moving Average Convergence Divergence).

    MACD měří trend momentum pomocí rozdílu dvou EMA.

    Args:
        series: Cenová data (close prices)
        fast_period: Perioda rychlé EMA (default 12)
        slow_period: Perioda pomalé EMA (default 26)
        signal_period: Perioda signal line (default 9)

    Returns:
        Tuple[macd_line, signal_line, histogram]
    """
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def macd_rising(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.Series:
    """
    Vrátí True kde MACD linka roste (current > previous).

    Použití jako filter pro buy signály - kupovat pouze když MACD roste.

    Args:
        series: Cenová data (close prices)
        fast_period: Perioda rychlé EMA
        slow_period: Perioda pomalé EMA
        signal_period: Perioda signal line

    Returns:
        pd.Series s boolean hodnotami (True = MACD rising)
    """
    macd_line, _, _ = macd(series, fast_period, slow_period, signal_period)
    return macd_line > macd_line.shift(1)


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        series: Cenová data
        period: Perioda EMA

    Returns:
        pd.Series s EMA hodnotami
    """
    return series.ewm(span=period, adjust=False).mean()
