"""Unit tests for autonomous optimizer evaluation logic and state management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qre.autonomous import (
    SymbolVerdict,
    append_changelog,
    check_top_verdict,
    compare_symbol,
    is_red,
    is_top_tier,
    load_config,
    load_iteration_log,
    overall_verdict,
    save_config,
    save_iteration_log,
)


# =============================================================================
# Helpers — metric fixtures
# =============================================================================


def _healthy_metrics() -> dict:
    """Return metrics dict that passes all checks (not RED, not TOP)."""
    return {
        "log_calmar": 2.5,
        "sharpe_equity": 2.0,
        "max_drawdown": -4.0,
        "trades_per_year": 100,
        "mc_confidence": "HIGH",
        "test_sharpe_equity": 1.5,
        "total_pnl_pct": 120.0,
        "train_sharpe_equity": 2.0,
    }


def _top_metrics() -> dict:
    """Return metrics dict that passes ALL TOP tier checks."""
    return {
        "log_calmar": 3.0,
        "sharpe_equity": 3.0,
        "max_drawdown": -3.0,
        "trades_per_year": 100,
        "mc_confidence": "HIGH",
        "test_sharpe_equity": 2.5,
        "train_sharpe_equity": 2.8,
        "total_pnl_pct": 200.0,
    }


def _red_metrics() -> dict:
    """Return metrics dict with RED flags."""
    return {
        "log_calmar": 0.5,
        "sharpe_equity": 1.0,
        "max_drawdown": -15.0,
        "trades_per_year": 20,
        "mc_confidence": "LOW",
        "test_sharpe_equity": -0.5,
        "total_pnl_pct": 30.0,
    }


# =============================================================================
# TestCompareSymbol
# =============================================================================


class TestCompareSymbol:
    """Per-symbol comparison logic."""

    def test_better_when_calmar_improves(self):
        """BETTER when calmar improves >1.5% and no RED."""
        curr = _healthy_metrics()
        prev = _healthy_metrics()
        curr["log_calmar"] = 3.0
        prev["log_calmar"] = 2.0  # +50% improvement
        assert compare_symbol(curr, prev) == SymbolVerdict.BETTER

    def test_worse_when_calmar_degrades(self):
        """WORSE when calmar drops >3%."""
        curr = _healthy_metrics()
        prev = _healthy_metrics()
        curr["log_calmar"] = 2.0
        prev["log_calmar"] = 3.0  # -33% drop
        assert compare_symbol(curr, prev) == SymbolVerdict.WORSE

    def test_worse_when_pnl_drops_over_20pct(self):
        """WORSE when PnL drops >20%."""
        curr = _healthy_metrics()
        prev = _healthy_metrics()
        prev["total_pnl_pct"] = 200.0
        curr["total_pnl_pct"] = 150.0  # -25% drop
        assert compare_symbol(curr, prev) == SymbolVerdict.WORSE

    def test_worse_when_new_red_metric(self):
        """WORSE when a new RED metric appears."""
        curr = _healthy_metrics()
        prev = _healthy_metrics()
        # Introduce a RED metric in current
        curr["mc_confidence"] = "LOW"
        assert compare_symbol(curr, prev) == SymbolVerdict.WORSE

    def test_neutral_when_small_change(self):
        """NEUTRAL when changes are within thresholds."""
        curr = _healthy_metrics()
        prev = _healthy_metrics()
        # Tiny calmar improvement — not enough for BETTER
        curr["log_calmar"] = 2.51
        prev["log_calmar"] = 2.50  # +0.4% change
        assert compare_symbol(curr, prev) == SymbolVerdict.NEUTRAL


# =============================================================================
# TestOverallVerdict
# =============================================================================


class TestOverallVerdict:
    """Cross-symbol verdict aggregation."""

    def test_better_when_majority_better(self):
        """BETTER when all symbols are BETTER."""
        verdicts = {
            "BTC": SymbolVerdict.BETTER,
            "SOL": SymbolVerdict.BETTER,
            "BNB": SymbolVerdict.BETTER,
        }
        assert overall_verdict(verdicts) == "BETTER"

    def test_worse_when_majority_worse(self):
        """WORSE when majority of symbols are WORSE."""
        verdicts = {
            "BTC": SymbolVerdict.WORSE,
            "SOL": SymbolVerdict.WORSE,
            "BNB": SymbolVerdict.NEUTRAL,
        }
        assert overall_verdict(verdicts) == "WORSE"

    def test_one_worse_two_better_is_neutral(self):
        """NEUTRAL when one WORSE blocks majority BETTER."""
        verdicts = {
            "BTC": SymbolVerdict.BETTER,
            "SOL": SymbolVerdict.BETTER,
            "BNB": SymbolVerdict.WORSE,
        }
        # Has WORSE → cannot be BETTER. worse_count (1) < better_count (2) → not WORSE majority
        assert overall_verdict(verdicts) == "NEUTRAL"

    def test_neutral_when_all_neutral(self):
        """NEUTRAL when all symbols are NEUTRAL."""
        verdicts = {
            "BTC": SymbolVerdict.NEUTRAL,
            "SOL": SymbolVerdict.NEUTRAL,
            "BNB": SymbolVerdict.NEUTRAL,
        }
        assert overall_verdict(verdicts) == "NEUTRAL"


# =============================================================================
# TestTopTier
# =============================================================================


class TestTopTier:
    """TOP tier detection."""

    def test_top_tier_all_green(self):
        """is_top_tier returns True when all metrics in TOP zone."""
        assert is_top_tier(_top_metrics()) is True

    def test_not_top_when_calmar_low(self):
        """Not TOP when log_calmar below threshold."""
        m = _top_metrics()
        m["log_calmar"] = 1.5  # Below 2.0
        assert is_top_tier(m) is False

    def test_check_top_verdict_needs_neutral_history(self):
        """check_top_verdict requires all TOP + last 2 iterations NEUTRAL."""
        all_metrics = {
            "BTC": _top_metrics(),
            "SOL": _top_metrics(),
            "BNB": _top_metrics(),
        }
        # Only 1 NEUTRAL in history — should fail
        history_short = [{"overall_verdict": "NEUTRAL"}]
        assert check_top_verdict(all_metrics, history_short) is False

        # 2 NEUTRAL iterations — should pass
        history_ok = [
            {"overall_verdict": "NEUTRAL"},
            {"overall_verdict": "NEUTRAL"},
        ]
        assert check_top_verdict(all_metrics, history_ok) is True

    def test_check_top_verdict_fails_if_not_all_top(self):
        """check_top_verdict fails if any symbol is not TOP."""
        all_metrics = {
            "BTC": _top_metrics(),
            "SOL": _top_metrics(),
            "BNB": _healthy_metrics(),  # healthy but NOT top tier
        }
        history_ok = [
            {"overall_verdict": "NEUTRAL"},
            {"overall_verdict": "NEUTRAL"},
        ]
        assert check_top_verdict(all_metrics, history_ok) is False


# =============================================================================
# TestIterationLog
# =============================================================================


class TestIterationLog:
    """Iteration log persistence."""

    def test_save_and_load(self, tmp_path: Path):
        """Round-trip save/load preserves data."""
        log_path = tmp_path / "iteration_log.json"
        entries = [
            {"iteration": 1, "verdict": "BETTER"},
            {"iteration": 2, "verdict": "NEUTRAL"},
        ]
        save_iteration_log(entries, log_path)
        loaded = load_iteration_log(log_path)
        assert loaded == entries

    def test_load_empty_returns_list(self, tmp_path: Path):
        """Loading from non-existent file returns empty list."""
        log_path = tmp_path / "nonexistent.json"
        assert load_iteration_log(log_path) == []


# =============================================================================
# TestChangelog
# =============================================================================


class TestChangelog:
    """Changelog append functionality."""

    def test_append_creates_file(self, tmp_path: Path):
        """append_changelog creates file and writes entry."""
        changelog_path = tmp_path / "CHANGELOG.md"
        append_changelog(
            path=changelog_path,
            iteration=1,
            branch="main",
            change="initial params",
            reason="baseline",
            run_dir="/runs/run_001",
            preset="default",
            pairs=["BTC", "SOL", "BNB"],
            metrics={"BTC": {"log_calmar": 2.5}},
            prev_metrics=None,
            verdict="BETTER",
        )
        assert changelog_path.exists()
        content = changelog_path.read_text()
        assert "iteration: 1" in content.lower() or "Iteration: 1" in content
        assert "BETTER" in content

    def test_append_adds_to_existing(self, tmp_path: Path):
        """append_changelog appends to existing file."""
        changelog_path = tmp_path / "CHANGELOG.md"
        for i in range(1, 3):
            append_changelog(
                path=changelog_path,
                iteration=i,
                branch="main",
                change=f"change {i}",
                reason=f"reason {i}",
                run_dir=f"/runs/run_{i:03d}",
                preset="default",
                pairs=["BTC"],
                metrics={"BTC": {"log_calmar": 2.5}},
                prev_metrics=None,
                verdict="NEUTRAL",
            )
        content = changelog_path.read_text()
        assert "change 1" in content
        assert "change 2" in content


# =============================================================================
# TestConfig
# =============================================================================


class TestConfig:
    """Config persistence."""

    def test_save_and_load(self, tmp_path: Path):
        """Round-trip save/load preserves config."""
        config_path = tmp_path / "auto_config.json"
        config = {
            "max_iterations": 10,
            "symbols": ["BTC", "SOL", "BNB"],
            "preset": "default",
        }
        save_config(config, config_path)
        loaded = load_config(config_path)
        assert loaded == config


# =============================================================================
# TestIsRed
# =============================================================================


class TestIsRed:
    """RED zone detection."""

    def test_healthy_is_not_red(self):
        """Healthy metrics are not RED."""
        assert is_red(_healthy_metrics()) is False

    def test_low_calmar_is_red(self):
        m = _healthy_metrics()
        m["log_calmar"] = 0.5
        assert is_red(m) is True

    def test_low_sharpe_is_red(self):
        m = _healthy_metrics()
        m["sharpe_equity"] = 1.0
        assert is_red(m) is True

    def test_high_drawdown_is_red(self):
        m = _healthy_metrics()
        m["max_drawdown"] = -15.0
        assert is_red(m) is True

    def test_low_trades_is_red(self):
        m = _healthy_metrics()
        m["trades_per_year"] = 20
        assert is_red(m) is True

    def test_high_trades_is_red(self):
        m = _healthy_metrics()
        m["trades_per_year"] = 350
        assert is_red(m) is True

    def test_low_mc_is_red(self):
        m = _healthy_metrics()
        m["mc_confidence"] = "LOW"
        assert is_red(m) is True

    def test_negative_test_sharpe_is_red(self):
        m = _healthy_metrics()
        m["test_sharpe_equity"] = -0.1
        assert is_red(m) is True

    def test_low_pnl_is_red(self):
        m = _healthy_metrics()
        m["total_pnl_pct"] = 30.0
        assert is_red(m) is True
