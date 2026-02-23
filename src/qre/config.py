"""
QRE Configuration
=================
Centralized config for Quantitative Research Engine.
Only BTC/USDC and SOL/USDC. Only MACD+RSI strategy.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# =============================================================================
# SYMBOLS & TIMEFRAMES
# =============================================================================

SYMBOLS = ["BTC/USDC", "SOL/USDC"]

TREND_TFS = ["4h", "8h", "1d"]
BASE_TF = "1h"

TF_MS: dict[str, int] = {
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

SLIPPAGE_MAP: dict[str, float] = {
    "BTC/USDC": 0.0008,
    "SOL/USDC": 0.0018,
}

DEFAULT_SLIPPAGE = float(os.environ.get("SLIPPAGE", "0.0015"))


def get_slippage(symbol: str) -> float:
    """Get slippage for symbol. Falls back to SLIPPAGE env var or 0.0015."""
    return SLIPPAGE_MAP.get(symbol, DEFAULT_SLIPPAGE)


# =============================================================================
# TRADE COUNT CONSTRAINTS
# =============================================================================

MIN_TRADES_YEAR_HARD = 30
MIN_TRADES_TEST_HARD = 5

# =============================================================================
# OPTIMIZATION
# =============================================================================

# --- Objective: Log Calmar + anti-gaming guards ---
SHARPE_SUSPECT_THRESHOLD = 3.0   # OOS Sharpe above this triggers decay penalty
SHARPE_DECAY_RATE = 0.3          # Decay rate: penalty = 1/(1 + rate*(sharpe - threshold))
MIN_DRAWDOWN_FLOOR = 0.05        # DD floor 5% — prevents Calmar gaming via tiny drawdowns
TARGET_TRADES_YEAR = 100          # Trade count ramp: full score at 100+ trades/year

# --- Walk-forward purge gap ---
PURGE_GAP_BARS = 50              # Bars skipped between train end and test start
                                 # = max(macd_slow_max=45, rsi_period_max=30) + 5
DEFAULT_TRIALS = 10000
DEFAULT_TIMEOUT = 0  # 0 = no timeout

# Sampler (TPE)
MIN_STARTUP_TRIALS = 50
STARTUP_TRIALS_RATIO = 0.10
TPE_N_EI_CANDIDATES = 24
TPE_CONSIDER_ENDPOINTS = True

# Pruning (SuccessiveHalving)
ENABLE_PRUNING = True

# Monte Carlo
MONTE_CARLO_SIMULATIONS = 1000
MONTE_CARLO_MIN_TRADES = 20

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

# =============================================================================
# FUNDED ACCOUNT
# =============================================================================

ACCOUNT_SIZE = 100_000.0
TOTAL_PAIRS = 2  # QRE: only BTC + SOL
PAIR_ALLOCATION = ACCOUNT_SIZE / TOTAL_PAIRS  # $50,000 per pair
STARTING_EQUITY = PAIR_ALLOCATION
BACKTEST_POSITION_PCT = 0.25  # Statických 25% pro všechny backtesty i EE

# =============================================================================
# CATASTROPHIC STOP
# =============================================================================

CATASTROPHIC_STOP_PCT = {"BTC": 0.08, "SOL": 0.12}  # per-symbol emergency exit
CATASTROPHIC_STOP_PCT_DEFAULT = 0.10  # fallback for unknown symbols

# =============================================================================
# STRATEGY
# =============================================================================

LONG_ONLY = False  # True = long only, False = long + short
MIN_HOLD_HOURS = 2  # Minimum bars before exit signal can trigger

# =============================================================================
# DISCORD (optional)
# =============================================================================

DISCORD_WEBHOOK_RUNS = os.environ.get("DISCORD_WEBHOOK_RUNS", "")
