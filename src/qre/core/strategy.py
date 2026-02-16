"""
Chio Extreme Strategy v3.0
==========================

MACD signal-line crossover + RSI extreme zones.

Evidence:
- Chio (2022): MACD + RSI achieved win rates of 84%, 86%, 78% on US equities
- Entry: MACD crossover AND RSI in extreme zone (oversold/overbought)
- Exit: Opposite signal (symmetric flip)

6 Optuna parameters only. Single timeframe (1H).
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import optuna

from qre.config import BASE_TF
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
    Chio Extreme: MACD crossover + RSI extreme zones.

    Buy:  MACD bullish crossover AND RSI < rsi_lower (oversold)
    Sell: MACD bearish crossover AND RSI > rsi_upper (overbought)
    """

    name = "macd_rsi"
    version = "3.0.0"
    description = "Chio Extreme: MACD crossover + RSI extreme zones"

    def get_optuna_params(self, trial: optuna.trial.Trial, symbol: str | None = None) -> dict[str, Any]:
        """6 Optuna parameters with evidence-based ranges."""
        params = {}

        params["macd_fast"] = trial.suggest_int("macd_fast", 5, 15)
        params["macd_slow"] = trial.suggest_int("macd_slow", 17, 30)

        # Constraint: macd_fast must be < macd_slow
        if params["macd_fast"] >= params["macd_slow"]:
            raise optuna.TrialPruned("macd_fast >= macd_slow")

        params["macd_signal"] = trial.suggest_int("macd_signal", 3, 12)
        params["rsi_period"] = trial.suggest_int("rsi_period", 3, 25)
        params["rsi_lower"] = trial.suggest_int("rsi_lower", 20, 40)
        params["rsi_upper"] = trial.suggest_int("rsi_upper", 60, 80)

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
        macd_fast = int(params.get("macd_fast", 8))
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

        # Handle NaN — no signal where any indicator is NaN
        has_nan = np.isnan(macd_vals) | np.isnan(signal_vals) | np.isnan(rsi_vals)

        # Combined signals
        buy_signal = macd_bullish_cross & rsi_oversold & ~has_nan
        sell_signal = macd_bearish_cross & rsi_overbought & ~has_nan

        return buy_signal.astype(np.bool_), sell_signal.astype(np.bool_)

    def get_default_params(self) -> dict[str, Any]:
        """Default params (midpoint of Optuna ranges)."""
        return {
            "macd_fast": 10,
            "macd_slow": 23,
            "macd_signal": 7,
            "rsi_period": 14,
            "rsi_lower": 30,
            "rsi_upper": 70,
        }
