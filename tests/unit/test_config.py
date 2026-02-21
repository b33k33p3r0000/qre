"""Unit tests for QRE config after strategy redesign."""

import pytest


def test_no_symbol_configs():
    """SYMBOL_CONFIGS removed — strategy uses global params only."""
    from qre import config
    assert not hasattr(config, "SYMBOL_CONFIGS")
    assert not hasattr(config, "get_symbol_config")
    assert not hasattr(config, "DEFAULT_SYMBOL_CONFIG")


def test_no_stoch_length():
    """STOCH_LENGTH removed — StochRSI eliminated."""
    from qre import config
    assert not hasattr(config, "STOCH_LENGTH")


def test_no_legacy_penalties():
    """Legacy penalty constants removed."""
    from qre import config
    for name in [
        "SHORT_HOLD_PENALTY", "MAX_SHORT_HOLD_RATIO",
        "MAX_ACCEPTABLE_DRAWDOWN", "DRAWDOWN_PENALTY_FACTOR",
        "MIN_PROFITABLE_MONTHS_RATIO", "MIN_TEST_TRAIN_RATIO",
        "DIVERGENCE_PENALTY_FACTOR", "MIN_TEST_TRADES",
        "MIN_TEST_SHARPE", "MIN_TEST_SHARPE_TIME_BASED",
        "MIN_TEST_TRAIN_RATIO_STRICT", "SPLIT_FAIL_PENALTY",
        "MIN_HOLD_BARS",
    ]:
        assert not hasattr(config, name), f"{name} should be removed"


def test_long_only_flag():
    """LONG_ONLY config flag exists."""
    from qre.config import LONG_ONLY
    assert isinstance(LONG_ONLY, bool)


def test_min_hold_constant():
    """MIN_HOLD_HOURS fixed constant exists."""
    from qre.config import MIN_HOLD_HOURS
    assert isinstance(MIN_HOLD_HOURS, int)
    assert MIN_HOLD_HOURS >= 1


def test_catastrophic_stop_10pct():
    """Catastrophic stop is 10% (0.10) per Quant Whale Strategy spec."""
    from qre.config import CATASTROPHIC_STOP_PCT
    assert CATASTROPHIC_STOP_PCT == 0.10


def test_trend_tfs_constant():
    """TREND_TFS list for multi-TF trend filter."""
    from qre.config import TREND_TFS
    assert isinstance(TREND_TFS, list)
    assert "4h" in TREND_TFS
    assert "8h" in TREND_TFS
    assert "1d" in TREND_TFS
    assert "1h" not in TREND_TFS


def test_calmar_objective_constants_exist():
    """Calmar objective constants for Sharpe sanity penalty."""
    from qre.config import (
        SHARPE_SUSPECT_THRESHOLD,
        SHARPE_DECAY_RATE,
        MIN_DRAWDOWN_FLOOR,
    )
    assert isinstance(SHARPE_SUSPECT_THRESHOLD, float)
    assert SHARPE_SUSPECT_THRESHOLD > 0

    assert isinstance(SHARPE_DECAY_RATE, float)
    assert SHARPE_DECAY_RATE > 0

    assert isinstance(MIN_DRAWDOWN_FLOOR, float)
    assert MIN_DRAWDOWN_FLOOR >= 0.05  # DD floor must be at least 5%


def test_target_trades_year_exists():
    """TARGET_TRADES_YEAR for trade count ramp in objective."""
    from qre.config import TARGET_TRADES_YEAR
    assert isinstance(TARGET_TRADES_YEAR, (int, float))
    assert TARGET_TRADES_YEAR >= 50


def test_purge_gap_bars_exists():
    """PURGE_GAP_BARS for walk-forward purge gap."""
    from qre.config import PURGE_GAP_BARS
    assert isinstance(PURGE_GAP_BARS, int)
    assert PURGE_GAP_BARS >= 30


def test_sharpe_penalty_tiers_removed():
    """SHARPE_PENALTY_TIERS replaced by Calmar constants."""
    from qre import config
    assert not hasattr(config, "SHARPE_PENALTY_TIERS"), \
        "SHARPE_PENALTY_TIERS should be removed"


def test_soft_penalty_constants_removed():
    """All soft penalty constants removed from config."""
    from qre import config
    for name in [
        "MIN_TRADES_TEST_SOFT", "LOW_TEST_TRADES_PENALTY",
        "RSI_ASYMMETRY_THRESHOLD", "RSI_ASYMMETRY_PENALTY",
        "SOL_MIN_TRADES_YEAR", "SOL_LOW_TRADE_PENALTY",
        "OVERTRADING_PENALTY", "MAX_TRADES_YEAR",
    ]:
        assert not hasattr(config, name), f"{name} should be removed"


def test_kept_constants():
    """Critical constants still present."""
    from qre import config
    assert hasattr(config, "BASE_TF")
    assert hasattr(config, "SYMBOLS")
    assert hasattr(config, "FEE")
    assert hasattr(config, "STARTING_EQUITY")
    assert hasattr(config, "CATASTROPHIC_STOP_PCT")
    assert hasattr(config, "BACKTEST_POSITION_PCT")
    assert hasattr(config, "RSI_LENGTH")
    assert hasattr(config, "MIN_WARMUP_BARS")
    assert hasattr(config, "MIN_TRADES_YEAR_HARD")
    assert hasattr(config, "SHARPE_SUSPECT_THRESHOLD")
    assert hasattr(config, "SHARPE_DECAY_RATE")
    assert hasattr(config, "MIN_DRAWDOWN_FLOOR")
    assert hasattr(config, "PURGE_GAP_BARS")
