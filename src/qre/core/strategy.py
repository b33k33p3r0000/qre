"""
MACD + RSI Strategy v2.0 (Evidence-Based)
==========================================

Kombinuje baseline StochRSI s MACD momentum a RSI filtrem.

Evidence:
- Chio (2022): MACD + RSI achieved win rates of 84%, 86%, 78% on US equities
- Kang (2021): Default MACD (12,26,9) has ~32% win rate - "almost a contra-indicator"
- Zatwarnicki (2023): RSI as momentum indicator (50 midline) outperforms mean-reversion on crypto

Two RSI Modes:
- EXTREME: RSI < 40 for buy (highest win rate 84-86%, fewer trades)
- TREND_FILTER: RSI > 50 for buy (more trades, 55-65% win rate)

Transferred from optimizer with audit. Removed: strategy registry, ADX filter,
multi-strategy support. Inlined suggest_all_params from phases.py.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import optuna

from qre.config import (
    BASE_TF,
    DEFAULT_SYMBOL_CONFIG,
    RSI_LENGTH,
    STOCH_LENGTH,
    TF_LIST,
    get_symbol_config,
)
from qre.core.backtest import (
    precompute_crossover_signals,
    precompute_rsi_gate,
    precompute_timeframe_indices,
)
from qre.core.indicators import macd, rsi, stochrsi


# MACD mode options
MACD_MODES = ["crossover", "rising", "positive"]

# RSI mode options (v2.0)
RSI_MODES = ["extreme", "trend_filter"]


class BaseStrategy(ABC):
    """Abstraktní base třída pro trading strategie."""

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
        tf_index_maps: dict[str, np.ndarray] | None = None,
        precomputed_cache: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def get_default_params(self) -> dict[str, Any]:
        return {}

    def validate_params(self, params: dict[str, Any]) -> bool:
        return True


def _suggest_all_params(trial: optuna.trial.Trial, symbol: str = None) -> dict[str, Any]:
    """
    Suggest all baseline Optuna parameters (inlined from optimizer phases.py).

    Core params: kB, dB, k_sell, min_hold, p_buy
    Threshold params: low_X, high_X for each timeframe
    Gate params: rsi_gate_24h, 12h, 8h, 6h
    """
    cfg = get_symbol_config(symbol) if symbol else DEFAULT_SYMBOL_CONFIG
    params = {}

    # Core params
    params["kB"] = trial.suggest_int("kB", 2, 4)
    params["dB"] = trial.suggest_int("dB", 2, 4)
    params["k_sell"] = trial.suggest_int("k_sell", 1, 3)
    params["min_hold"] = trial.suggest_int("min_hold", cfg["min_hold_min"], cfg["min_hold_max"])
    params["p_buy"] = trial.suggest_float("p_buy", cfg["p_buy_min"], cfg["p_buy_max"])

    # Threshold params (per timeframe)
    tightness = cfg.get("threshold_tightness", 1.0)
    low_min = 0.05
    low_max = min(0.40 * tightness, 0.50)
    high_min = max(0.60 / tightness, 0.50)
    high_max = 0.95

    for tf in TF_LIST:
        key = "24h" if tf == "1d" else tf
        params[f"low_{key}"] = trial.suggest_float(f"low_{key}", low_min, low_max)
        params[f"high_{key}"] = trial.suggest_float(f"high_{key}", high_min, high_max)

    # Gate params
    params["rsi_gate_24h"] = trial.suggest_float("rsi_gate_24h", 40.0, 60.0)
    params["rsi_gate_12h"] = trial.suggest_float("rsi_gate_12h", 40.0, 60.0)
    params["rsi_gate_8h"] = trial.suggest_float("rsi_gate_8h", 40.0, 60.0)
    params["rsi_gate_6h"] = trial.suggest_float("rsi_gate_6h", 40.0, 60.0)

    # Metadata
    params["tf"] = "1h"
    params["range"] = "FULL"
    params["n_votes"] = len(TF_LIST)

    return params


class MACDRSIStrategy(BaseStrategy):
    """
    MACD + RSI evidence-based strategie.

    Rank #2 podle evidence-based analýzy:
    - Nejsilnější empirická evidence (Chio 2022: 84-86% WR)
    - Dva RSI módy: extreme (highest WR) vs trend_filter (more trades)
    """

    name = "macd_rsi"
    version = "2.0.0"
    description = "Evidence-based: MACD momentum + RSI (extreme/trend_filter modes)"

    def get_optuna_params(self, trial: optuna.trial.Trial, symbol: str | None = None) -> dict[str, Any]:
        """Vrátí Optuna parametry."""
        params = _suggest_all_params(trial, symbol)

        # MACD parameters (v2.0: evidence-based ranges)
        params["macd_fast"] = trial.suggest_int("macd_fast", 5, 15)
        params["macd_slow"] = trial.suggest_int("macd_slow", 17, 35)
        params["macd_signal"] = trial.suggest_int("macd_signal", 3, 12)
        params["macd_mode"] = trial.suggest_categorical("macd_mode", MACD_MODES)

        # RSI mode (v2.0)
        params["rsi_mode"] = trial.suggest_categorical("rsi_mode", RSI_MODES)
        params["rsi_upper"] = trial.suggest_int("rsi_upper", 60, 80)
        params["rsi_lower"] = trial.suggest_int("rsi_lower", 20, 40)
        params["rsi_momentum_level"] = trial.suggest_int("rsi_momentum_level", 45, 55)

        return params

    def get_required_indicators(self) -> list[str]:
        return ["stochrsi", "rsi", "macd"]

    def precompute_signals(
        self,
        data: dict[str, Any],
        params: dict[str, Any],
        tf_index_maps: dict[str, np.ndarray] | None = None,
        precomputed_cache: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Precompute buy/sell signály.

        v2.0 Logic:
        - EXTREME MODE: MACD bullish AND RSI < rsi_lower (value entry)
        - TREND_FILTER MODE: MACD bullish AND RSI > rsi_momentum_level (momentum confirmation)
        """
        base = data[BASE_TF]
        n_bars = len(base)
        total_timeframes = len(TF_LIST)

        # Baseline parameters
        k_smooth = int(params.get("kB", 3))
        d_smooth = int(params.get("dB", 3))

        low_thresholds = {tf: float(params.get(f"low_{'24h' if tf == '1d' else tf}", 0.2)) for tf in TF_LIST}
        high_thresholds = {tf: float(params.get(f"high_{'24h' if tf == '1d' else tf}", 0.8)) for tf in TF_LIST}

        rsi_gate_24h = float(params.get("rsi_gate_24h", 50))
        rsi_gate_12h = float(params.get("rsi_gate_12h", 50))
        rsi_gate_8h = float(params.get("rsi_gate_8h", 50))
        rsi_gate_6h = float(params.get("rsi_gate_6h", 50))

        # MACD parameters
        macd_fast = int(params.get("macd_fast", 8))
        macd_slow = int(params.get("macd_slow", 21))
        macd_signal_period = int(params.get("macd_signal", 9))
        macd_mode = params.get("macd_mode", "rising")

        # RSI parameters (v2.0)
        rsi_mode = params.get("rsi_mode", "trend_filter")
        rsi_upper = int(params.get("rsi_upper", 65))
        rsi_lower = int(params.get("rsi_lower", 35))
        rsi_momentum_level = int(params.get("rsi_momentum_level", 50))

        base_ts = base.index.values.astype(np.int64)

        # Compute MACD (on base TF)
        macd_line, signal_line, histogram = macd(base["close"], macd_fast, macd_slow, macd_signal_period)
        macd_vals = macd_line.values.astype(np.float64)
        signal_vals = signal_line.values.astype(np.float64)

        # Previous values for crossover detection
        macd_prev = np.roll(macd_vals, 1)
        signal_prev = np.roll(signal_vals, 1)
        macd_prev[0] = np.nan
        signal_prev[0] = np.nan

        # MACD conditions based on mode
        if macd_mode == "crossover":
            macd_bullish = (macd_prev < signal_prev) & (macd_vals > signal_vals)
        elif macd_mode == "rising":
            macd_bullish = macd_vals > macd_prev
        elif macd_mode == "positive":
            macd_bullish = macd_vals > 0
        else:  # "any"
            crossover = (macd_prev < signal_prev) & (macd_vals > signal_vals)
            rising = macd_vals > macd_prev
            positive = macd_vals > 0
            macd_bullish = crossover | rising | positive

        # Compute RSI (on base TF)
        if precomputed_cache and "base_rsi" in precomputed_cache:
            base_rsi_vals = precomputed_cache["base_rsi"]
        else:
            base_rsi_vals = rsi(base["close"], RSI_LENGTH).values.astype(np.float64)

        # RSI condition based on mode (v2.0)
        if rsi_mode == "extreme":
            rsi_entry_condition = base_rsi_vals < rsi_lower
            rsi_not_overbought = base_rsi_vals < rsi_upper
            rsi_condition = rsi_entry_condition & rsi_not_overbought
        else:  # trend_filter
            rsi_condition = base_rsi_vals > rsi_momentum_level

        # Additional entry condition (no ADX filter in QRE)
        additional_condition = macd_bullish & rsi_condition

        # Handle NaN
        has_nan = np.isnan(macd_vals) | np.isnan(base_rsi_vals)
        additional_condition = np.where(has_nan, False, additional_condition)

        # Pre-compute StochRSI & crossovers
        buy_votes_per_tf = np.zeros((total_timeframes, n_bars), dtype=np.bool_)
        sell_votes_per_tf = np.zeros((total_timeframes, n_bars), dtype=np.bool_)

        for tf_idx, tf in enumerate(TF_LIST):
            if tf not in data or len(data[tf]) == 0:
                continue

            df_tf = data[tf]
            cache_key = (k_smooth, d_smooth, tf)
            if precomputed_cache and "stochrsi" in precomputed_cache and cache_key in precomputed_cache["stochrsi"]:
                k_vals, d_vals = precomputed_cache["stochrsi"][cache_key]
            else:
                k_line, d_line = stochrsi(df_tf["close"], STOCH_LENGTH, STOCH_LENGTH, k_smooth, d_smooth)
                k_vals = k_line.values.astype(np.float64)
                d_vals = d_line.values.astype(np.float64)

            tf_buy, tf_sell = precompute_crossover_signals(k_vals, d_vals, low_thresholds[tf], high_thresholds[tf])

            if tf_index_maps is not None and tf in tf_index_maps:
                base_to_tf_idx = tf_index_maps[tf]
            else:
                tf_ts = df_tf.index.values.astype(np.int64)
                base_to_tf_idx = precompute_timeframe_indices(base_ts, tf_ts)

            valid = (base_to_tf_idx >= 2) & (base_to_tf_idx < len(tf_buy))
            clipped_idx = np.clip(base_to_tf_idx, 0, max(len(tf_buy) - 1, 0))
            buy_votes_per_tf[tf_idx] = valid & tf_buy[clipped_idx] & additional_condition
            sell_votes_per_tf[tf_idx] = valid & tf_sell[clipped_idx]

        # Pre-compute RSI gates
        rsi_gate_signals = np.zeros((4, n_bars), dtype=np.bool_)
        gate_configs = [
            ("1d", rsi_gate_24h),
            ("12h", rsi_gate_12h),
            ("8h", rsi_gate_8h),
            ("6h", rsi_gate_6h),
        ]

        for gate_idx, (tf, threshold) in enumerate(gate_configs):
            if tf not in data or len(data[tf]) == 0:
                continue

            df_tf = data[tf]
            if precomputed_cache and "gate_rsi" in precomputed_cache and tf in precomputed_cache["gate_rsi"]:
                rsi_vals = precomputed_cache["gate_rsi"][tf]
            else:
                rsi_vals = rsi(df_tf["close"], RSI_LENGTH).values.astype(np.float64)
            tf_gate = precompute_rsi_gate(rsi_vals, threshold)

            if tf_index_maps is not None and tf in tf_index_maps:
                base_to_tf_idx = tf_index_maps[tf]
            else:
                tf_ts = df_tf.index.values.astype(np.int64)
                base_to_tf_idx = precompute_timeframe_indices(base_ts, tf_ts)

            valid = (base_to_tf_idx >= 1) & (base_to_tf_idx < len(tf_gate))
            clipped_idx = np.clip(base_to_tf_idx, 0, max(len(tf_gate) - 1, 0))
            rsi_gate_signals[gate_idx] = valid & tf_gate[clipped_idx]

        return buy_votes_per_tf, sell_votes_per_tf, rsi_gate_signals

    def get_default_params(self) -> dict[str, Any]:
        """Vrátí defaultní hodnoty (v12.9: z impruvment-v1 runu)."""
        return {
            "kB": 3, "dB": 2, "k_sell": 1, "min_hold": 8, "p_buy": 0.14,
            "low_2h": 0.2, "high_2h": 0.8,
            "low_4h": 0.2, "high_4h": 0.8,
            "low_6h": 0.2, "high_6h": 0.8,
            "low_8h": 0.2, "high_8h": 0.8,
            "low_12h": 0.2, "high_12h": 0.8,
            "low_24h": 0.2, "high_24h": 0.8,
            "rsi_gate_24h": 50, "rsi_gate_12h": 50,
            "rsi_gate_8h": 50, "rsi_gate_6h": 50,
            "macd_fast": 7, "macd_slow": 28, "macd_signal": 8,
            "macd_mode": "any",
            "rsi_mode": "trend_filter",
            "rsi_upper": 73, "rsi_lower": 32,
            "rsi_momentum_level": 47,
        }
