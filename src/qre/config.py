"""
QRE Configuration
=================
Centralized config for Quantitative Research Engine.
Only BTC/USDC and SOL/USDC. Only MACD+RSI strategy.
"""

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# =============================================================================
# SYMBOLS & TIMEFRAMES
# =============================================================================

SYMBOLS = ["BTC/USDC", "SOL/USDC"]

TF_LIST = ["2h", "4h", "6h", "8h", "12h", "1d"]
BASE_TF = "1h"

TF_MS: Dict[str, int] = {
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "8h": 8 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

# =============================================================================
# TRADING COSTS
# =============================================================================

FEE = float(os.environ.get("FEE", "0.00075"))

SLIPPAGE_MAP: Dict[str, float] = {
    "BTC/USDC": 0.0008,
    "SOL/USDC": 0.0018,
}

DEFAULT_SLIPPAGE = float(os.environ.get("SLIPPAGE", "0.0015"))


def get_slippage(symbol: str) -> float:
    env_slippage = os.environ.get("SLIPPAGE")
    if env_slippage:
        return float(env_slippage)
    return SLIPPAGE_MAP.get(symbol, DEFAULT_SLIPPAGE)


# =============================================================================
# PER-PAIR CONFIGURATION
# =============================================================================

SYMBOL_CONFIGS: Dict[str, Dict] = {
    "BTC/USDC": {
        "min_trades_year": 80,
        "min_trades_test": 8,
        "max_drawdown_limit": 0.15,
        "min_hold_min": 8,
        "min_hold_max": 12,
        "p_buy_min": 0.10,
        "p_buy_max": 0.22,
        "volatility_class": "low",
        "threshold_tightness": 0.95,
    },
    "SOL/USDC": {
        "min_trades_year": 80,
        "min_trades_test": 8,
        "max_drawdown_limit": 0.25,
        "min_hold_min": 8,
        "min_hold_max": 12,
        "p_buy_min": 0.10,
        "p_buy_max": 0.25,
        "volatility_class": "high",
        "threshold_tightness": 1.0,
    },
}

DEFAULT_SYMBOL_CONFIG: Dict = {
    "min_trades_year": 80,
    "min_trades_test": 8,
    "max_drawdown_limit": 0.18,
    "min_hold_min": 8,
    "min_hold_max": 12,
    "p_buy_min": 0.12,
    "p_buy_max": 0.50,
    "volatility_class": "medium",
    "threshold_tightness": 0.95,
}


def get_symbol_config(symbol: str) -> Dict:
    return SYMBOL_CONFIGS.get(symbol, DEFAULT_SYMBOL_CONFIG)


# =============================================================================
# TRADE COUNT CONSTRAINTS
# =============================================================================

MIN_TRADES_YEAR_HARD = 80
MIN_TRADES_TEST_HARD = 8
MAX_TRADES_YEAR = 500
OVERTRADING_PENALTY = 0.15

# =============================================================================
# PENALTIES
# =============================================================================

MIN_HOLD_BARS = 3
SHORT_HOLD_PENALTY = 0.25
MAX_SHORT_HOLD_RATIO = 0.15

MAX_ACCEPTABLE_DRAWDOWN = 0.10
DRAWDOWN_PENALTY_FACTOR = 1.5

MIN_PROFITABLE_MONTHS_RATIO = 0.50

MIN_TEST_TRAIN_RATIO = 0.50
DIVERGENCE_PENALTY_FACTOR = 1.5
MIN_TEST_TRADES = 5
MIN_TEST_SHARPE = 0.0

MIN_TEST_SHARPE_TIME_BASED = 0.3
MIN_TEST_TRAIN_RATIO_STRICT = 0.25

# =============================================================================
# OPTIMIZATION
# =============================================================================

DEFAULT_TRIALS = 10000
DEFAULT_TIMEOUT = 0  # 0 = no timeout

# Sampler (TPE)
SAMPLER_STRATEGY = "tpe"
MIN_STARTUP_TRIALS = 50
STARTUP_TRIALS_RATIO = 0.10
TPE_N_EI_CANDIDATES = 24
TPE_CONSIDER_ENDPOINTS = True

# Pruning (SuccessiveHalving)
ENABLE_PRUNING = True
MIN_PRUNING_WARMUP = 30
PRUNING_WARMUP_RATIO = 0.10

# Monte Carlo
MONTE_CARLO_SIMULATIONS = 1000
MONTE_CARLO_MIN_TRADES = 30

# =============================================================================
# ANCHORED WALK-FORWARD
# =============================================================================

ANCHORED_WF_MIN_DATA_HOURS = 4000  # ~167 days minimum
ANCHORED_WF_SHORT_THRESHOLD_HOURS = 13140  # 1.5 years

ANCHORED_WF_SPLITS = [
    {"train_end": 0.60, "test_end": 0.70},
    {"train_end": 0.70, "test_end": 0.80},
    {"train_end": 0.80, "test_end": 0.90},
]

ANCHORED_WF_SPLITS_SHORT = [
    {"train_end": 0.70, "test_end": 0.85},
    {"train_end": 0.85, "test_end": 1.00},
]

N_SPLITS_DEFAULT = 3

# =============================================================================
# DATA FETCHING
# =============================================================================

OHLCV_LIMIT_PER_CALL = 1500
SAFETY_MAX_ROWS = 200000
MAX_API_RETRIES = 5

# =============================================================================
# BACKTESTING
# =============================================================================

MIN_WARMUP_BARS = 200
RSI_LENGTH = 21
STOCH_LENGTH = 14

# =============================================================================
# FUNDED ACCOUNT
# =============================================================================

ACCOUNT_SIZE = 100_000.0
TOTAL_PAIRS = 2  # QRE: only BTC + SOL
PAIR_ALLOCATION = ACCOUNT_SIZE / TOTAL_PAIRS  # $50,000 per pair
STARTING_EQUITY = PAIR_ALLOCATION
DAILY_DRAWDOWN_LIMIT = 0.05
TOTAL_DRAWDOWN_LIMIT = 0.10
BACKTEST_POSITION_PCT = 0.25  # Statických 25% pro všechny backtesty i EE

# =============================================================================
# CATASTROPHIC STOP
# =============================================================================

CATASTROPHIC_STOP_PCT = 0.15  # -15% emergency exit

# =============================================================================
# DISCORD (optional)
# =============================================================================

DISCORD_WEBHOOK_RUNS = os.environ.get("DISCORD_WEBHOOK_RUNS", "")
DISCORD_WEBHOOK_ALERTS = os.environ.get("DISCORD_WEBHOOK_ALERTS", "")
