"""
Quant Whale Strategy v4.1
==========================

MACD signal-line crossover + RSI extreme zones + multi-TF trend filter.

Evidence:
- Chio (2022): MACD + RSI achieved win rates of 84%, 86%, 78% on US equities
- Entry: MACD crossover AND RSI in extreme zone (with lookback) AND higher-TF trend
- Exit: Opposite signal (flip or flat, controlled by allow_flip)

10 Optuna parameters. Base TF 1H + trend filter from 4H/8H/1D.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import optuna
import pandas as pd

from qre.config import BASE_TF
from qre.core.backtest import precompute_timeframe_indices
from qre.core.indicators import macd, rsi


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    name: str = "base"
    version: str = "1.0.0"
    description: str = ""

    @abstractmethod
    def get_optuna_params(self, trial: optuna.trial.Trial, symbol: str | None = None) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_required_indicators(self) -> list[str]:
        pass

    @abstractmethod
    def precompute_signals(
        self,
        data: dict[str, Any],
        params: dict[str, Any],
        precomputed_cache: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    def get_default_params(self) -> dict[str, Any]:
        return {}

    def validate_params(self, params: dict[str, Any]) -> bool:
        return True


class MACDRSIStrategy(BaseStrategy):
    """
    Quant Whale Strategy: MACD crossover + RSI extreme zones.

    Buy:  MACD bullish crossover AND RSI < rsi_lower (oversold)
    Sell: MACD bearish crossover AND RSI > rsi_upper (overbought)
    """

    name = "macd_rsi"
    version = "4.1.0"
    description = "Quant Whale Strategy: MACD crossover + RSI extreme zones"

    def get_optuna_params(self, trial: optuna.trial.Trial, symbol: str | None = None) -> dict[str, Any]:
        """10 Optuna parameters: 6 original + rsi_lookback + trend_tf + trend_strict + allow_flip."""
        params = {}

        params["macd_fast"] = trial.suggest_float("macd_fast", 1.0, 20.0)
        params["macd_slow"] = trial.suggest_int("macd_slow", 10, 45)

        # Constraint: macd_fast must be < macd_slow with minimum spread of 5
        if params["macd_slow"] - params["macd_fast"] < 5:
            raise optuna.TrialPruned("macd_slow - macd_fast < 5")

        params["macd_signal"] = trial.suggest_int("macd_signal", 1, 15)
        params["rsi_period"] = trial.suggest_int("rsi_period", 3, 30)
        params["rsi_lower"] = trial.suggest_int("rsi_lower", 25, 35)
        params["rsi_upper"] = trial.suggest_int("rsi_upper", 65, 75)
        params["rsi_lookback"] = trial.suggest_int("rsi_lookback", 4, 8)
        params["trend_tf"] = trial.suggest_categorical("trend_tf", ["4h", "8h", "1d"])
        params["trend_strict"] = trial.suggest_int("trend_strict", 0, 1)
        params["allow_flip"] = trial.suggest_int("allow_flip", 0, 1)

        return params

    def get_required_indicators(self) -> list[str]:
        return ["macd", "rsi"]

    def precompute_signals(
        self,
        data: dict[str, Any],
        params: dict[str, Any],
        precomputed_cache: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Precompute 1D buy/sell signal arrays.

        Buy:  MACD bullish crossover AND RSI < rsi_lower
        Sell: MACD bearish crossover AND RSI > rsi_upper
        """
        base = data[BASE_TF]
        n_bars = len(base)

        # MACD params
        macd_fast = float(params.get("macd_fast", 8))
        macd_slow = int(params.get("macd_slow", 21))
        macd_signal_period = int(params.get("macd_signal", 9))

        # RSI params
        rsi_period = int(params.get("rsi_period", 14))
        rsi_lower = int(params.get("rsi_lower", 30))
        rsi_upper = int(params.get("rsi_upper", 70))

        # Compute MACD
        macd_line, signal_line, _ = macd(base["close"], macd_fast, macd_slow, macd_signal_period)
        macd_vals = macd_line.values.astype(np.float64)
        signal_vals = signal_line.values.astype(np.float64)

        # Previous values for crossover detection
        macd_prev = np.roll(macd_vals, 1)
        signal_prev = np.roll(signal_vals, 1)
        macd_prev[0] = np.nan
        signal_prev[0] = np.nan

        # MACD crossover signals
        macd_bullish_cross = (macd_prev <= signal_prev) & (macd_vals > signal_vals)
        macd_bearish_cross = (macd_prev >= signal_prev) & (macd_vals < signal_vals)

        # Compute RSI
        if precomputed_cache and "rsi" in precomputed_cache and rsi_period in precomputed_cache["rsi"]:
            rsi_vals = precomputed_cache["rsi"][rsi_period]
        else:
            rsi_vals = rsi(base["close"], rsi_period).values.astype(np.float64)

        # RSI extreme zone conditions
        rsi_oversold = rsi_vals < rsi_lower
        rsi_overbought = rsi_vals > rsi_upper

        # Layer 2: RSI lookback window (v4.0)
        rsi_lookback = int(params.get("rsi_lookback", 0))
        if rsi_lookback > 0:
            rsi_oversold = (
                pd.Series(rsi_oversold)
                .rolling(rsi_lookback + 1, min_periods=1)
                .max()
                .astype(bool)
                .values
            )
            rsi_overbought = (
                pd.Series(rsi_overbought)
                .rolling(rsi_lookback + 1, min_periods=1)
                .max()
                .astype(bool)
                .values
            )

        # Handle NaN — no signal where any indicator is NaN
        has_nan = np.isnan(macd_vals) | np.isnan(signal_vals) | np.isnan(rsi_vals)

        # Layer 3: Multi-TF trend filter (v4.0)
        trend_strict = int(params.get("trend_strict", 0))
        trend_tf = params.get("trend_tf", "4h")

        if trend_strict and trend_tf in data:
            htf = data[trend_tf]
            htf_macd, htf_signal, _ = macd(
                htf["close"], macd_fast, macd_slow, macd_signal_period,
            )
            htf_bullish = (htf_macd > htf_signal).values

            # Align higher TF to 1H bars
            base_ts = base.index.astype(np.int64) // 10**6
            htf_ts = htf.index.astype(np.int64) // 10**6
            tf_indices = precompute_timeframe_indices(base_ts, htf_ts)

            # Handle NaN in higher TF MACD
            htf_has_nan = np.isnan(htf_macd.values) | np.isnan(htf_signal.values)
            htf_valid = ~htf_has_nan
            htf_bullish_raw = htf_bullish
            htf_bearish_raw = ~htf_bullish_raw

            htf_bullish_aligned = (htf_bullish_raw & htf_valid)[tf_indices]
            htf_bearish_aligned = (htf_bearish_raw & htf_valid)[tf_indices]
        else:
            htf_bullish_aligned = np.ones(n_bars, dtype=bool)
            htf_bearish_aligned = np.ones(n_bars, dtype=bool)

        # Combined signals
        buy_signal = macd_bullish_cross & rsi_oversold & htf_bullish_aligned & ~has_nan
        sell_signal = macd_bearish_cross & rsi_overbought & htf_bearish_aligned & ~has_nan

        return buy_signal.astype(np.bool_), sell_signal.astype(np.bool_)

    def get_default_params(self) -> dict[str, Any]:
        """Default params (midpoint of Optuna ranges)."""
        return {
            "macd_fast": 10.0,
            "macd_slow": 27,
            "macd_signal": 8,
            "rsi_period": 16,
            "rsi_lower": 30,
            "rsi_upper": 70,
            "rsi_lookback": 6,
            "trend_tf": "8h",
            "trend_strict": 0,
            "allow_flip": 1,  # backward compat: flip enabled by default
        }
