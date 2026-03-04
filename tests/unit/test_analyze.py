"""Unit tests for qre.analyze — rules-based diagnostics pipeline."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from qre.analyze import (
    _build_findings,
    _classify,
    _find_symbol_dir,
    analyze_thresholds,
    analyze_trades,
    build_discord_embed,
    check_robustness,
    compute_verdict,
    generate_suggestions,
    health_check,
    save_analysis,
)


# =============================================================================
# Helpers
# =============================================================================


def _good_params() -> Dict[str, Any]:
    """Return healthy optimizer result params (all-green health check)."""
    return {
        "sharpe_equity": 2.0,
        "max_drawdown": -3.0,
        "trades_per_year": 100,
        "win_rate": 0.55,
        "profit_factor": 2.0,
        "expectancy": 200.0,
        "train_sharpe_equity": 2.0,
        "test_sharpe_equity": 1.8,
        "split_results": [
            {"test_sharpe": 1.5},
            {"test_sharpe": 2.0},
            {"test_sharpe": 1.0},
        ],
    }


def _bad_params() -> Dict[str, Any]:
    """Return poor optimizer result params (multiple reds)."""
    return {
        "sharpe_equity": 0.2,
        "max_drawdown": -15.0,
        "trades_per_year": 10,
        "win_rate": 0.30,
        "profit_factor": 0.8,
        "expectancy": -50.0,
        "train_sharpe_equity": 4.0,
        "test_sharpe_equity": 1.0,
        "split_results": [
            {"test_sharpe": -0.5},
            {"test_sharpe": -1.0},
            {"test_sharpe": 0.5},
        ],
    }


def _write_trades_csv(path: Path, trades: list) -> None:
    """Write trades to CSV file."""
    fieldnames = [
        "entry_ts", "entry_price", "exit_ts", "exit_price",
        "hold_bars", "size", "capital_at_entry",
        "pnl_abs", "pnl_pct", "symbol", "reason", "direction",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            row = {k: t.get(k, "") for k in fieldnames}
            writer.writerow(row)


def _sample_trades() -> list:
    """Return sample trade dicts for CSV."""
    return [
        {
            "entry_ts": "2025-01-01 00:00:00",
            "entry_price": "100.0",
            "exit_ts": "2025-01-01 05:00:00",
            "exit_price": "105.0",
            "hold_bars": "5",
            "size": "1.0",
            "capital_at_entry": "100000",
            "pnl_abs": "500.0",
            "pnl_pct": "0.5",
            "symbol": "BTC/USDT",
            "reason": "trailing_stop",
            "direction": "long",
        },
        {
            "entry_ts": "2025-01-02 00:00:00",
            "entry_price": "105.0",
            "exit_ts": "2025-01-02 03:00:00",
            "exit_price": "103.0",
            "hold_bars": "3",
            "size": "1.0",
            "capital_at_entry": "100500",
            "pnl_abs": "-200.0",
            "pnl_pct": "-0.2",
            "symbol": "BTC/USDT",
            "reason": "hard_stop",
            "direction": "long",
        },
        {
            "entry_ts": "2025-01-03 00:00:00",
            "entry_price": "103.0",
            "exit_ts": "2025-01-03 02:00:00",
            "exit_price": "100.0",
            "hold_bars": "2",
            "size": "1.0",
            "capital_at_entry": "100300",
            "pnl_abs": "300.0",
            "pnl_pct": "0.3",
            "symbol": "BTC/USDT",
            "reason": "signal",
            "direction": "short",
        },
        {
            "entry_ts": "2025-01-04 00:00:00",
            "entry_price": "100.0",
            "exit_ts": "2025-01-04 02:00:00",
            "exit_price": "92.0",
            "hold_bars": "2",
            "size": "1.0",
            "capital_at_entry": "100600",
            "pnl_abs": "-800.0",
            "pnl_pct": "-0.8",
            "symbol": "BTC/USDT",
            "reason": "catastrophic_stop",
            "direction": "long",
        },
    ]


# =============================================================================
# TestClassify
# =============================================================================


class TestClassify:
    def test_value_in_green_range(self):
        assert _classify(2.0, (1.0, 3.5), (0.5, 5.0)) == "green"

    def test_value_in_yellow_range(self):
        assert _classify(0.7, (1.0, 3.5), (0.5, 5.0)) == "yellow"
        assert _classify(4.5, (1.0, 3.5), (0.5, 5.0)) == "yellow"

    def test_value_in_red_range(self):
        assert _classify(0.1, (1.0, 3.5), (0.5, 5.0)) == "red"
        assert _classify(6.0, (1.0, 3.5), (0.5, 5.0)) == "red"

    def test_boundary_values(self):
        # Exactly on green boundary = green (inclusive)
        assert _classify(1.0, (1.0, 3.5), (0.5, 5.0)) == "green"
        assert _classify(3.5, (1.0, 3.5), (0.5, 5.0)) == "green"
        # Exactly on yellow boundary = yellow (inclusive)
        assert _classify(0.5, (1.0, 3.5), (0.5, 5.0)) == "yellow"
        assert _classify(5.0, (1.0, 3.5), (0.5, 5.0)) == "yellow"


# =============================================================================
# TestHealthCheck
# =============================================================================


class TestHealthCheck:
    def test_all_green(self):
        health = health_check(_good_params())
        for metric, info in health.items():
            assert info["status"] == "green", f"{metric} expected green, got {info['status']}"

    def test_all_red(self):
        health = health_check(_bad_params())
        red_metrics = [m for m, i in health.items() if i["status"] == "red"]
        assert len(red_metrics) >= 4, f"Expected 4+ reds, got {red_metrics}"

    def test_sharpe_fallback_chain(self):
        # sharpe_equity preferred over sharpe_time over sharpe
        h1 = health_check({"sharpe_equity": 2.0})
        assert h1["sharpe"]["value"] == 2.0

        h2 = health_check({"sharpe_time": 1.5})
        assert h2["sharpe"]["value"] == 1.5

        h3 = health_check({"sharpe": 1.0})
        assert h3["sharpe"]["value"] == 1.0

    def test_missing_keys_default_to_zero(self):
        health = health_check({})
        assert "sharpe" in health
        assert health["sharpe"]["value"] == 0

    def test_split_consistency_negative_splits(self):
        params = _good_params()
        params["split_results"] = [
            {"test_sharpe": -0.5},
            {"test_sharpe": -1.0},
        ]
        health = health_check(params)
        assert health["split_consistency"]["status"] == "red"
        assert health["split_consistency"]["value"] == 2

    def test_empty_split_results(self):
        params = _good_params()
        params["split_results"] = []
        health = health_check(params)
        assert health["split_consistency"]["status"] == "green"
        assert health["split_consistency"]["value"] == 0


# =============================================================================
# TestComputeVerdict
# =============================================================================


class TestComputeVerdict:
    def test_all_green_is_pass(self):
        health = {
            "a": {"status": "green"},
            "b": {"status": "green"},
            "c": {"status": "green"},
        }
        assert compute_verdict(health) == "PASS"

    def test_two_reds_is_fail(self):
        health = {
            "a": {"status": "red"},
            "b": {"status": "red"},
            "c": {"status": "green"},
        }
        assert compute_verdict(health) == "FAIL"

    def test_one_red_is_review(self):
        health = {
            "a": {"status": "red"},
            "b": {"status": "green"},
            "c": {"status": "green"},
        }
        assert compute_verdict(health) == "REVIEW"

    def test_three_yellows_is_review(self):
        health = {
            "a": {"status": "yellow"},
            "b": {"status": "yellow"},
            "c": {"status": "yellow"},
        }
        assert compute_verdict(health) == "REVIEW"

    def test_two_yellows_is_pass(self):
        health = {
            "a": {"status": "yellow"},
            "b": {"status": "yellow"},
            "c": {"status": "green"},
        }
        assert compute_verdict(health) == "PASS"


# =============================================================================
# TestAnalyzeThresholds
# =============================================================================


class TestAnalyzeThresholds:
    def test_normal_params(self):
        result = analyze_thresholds({
            "macd_fast": 5.0,
            "macd_slow": 20,
            "macd_signal": 9,
            "rsi_period": 14,
            "rsi_lower": 25,
            "rsi_upper": 70,
        })
        assert result["macd_spread"] == 15
        assert result["macd_spread_status"] == "green"
        assert result["rsi_zone_width"] == 45
        assert result["rsi_zone_status"] == "green"

    def test_narrow_macd_spread(self):
        result = analyze_thresholds({"macd_fast": 10.0, "macd_slow": 15})
        assert result["macd_spread"] == 5
        assert result["macd_spread_status"] == "yellow"

    def test_narrow_rsi_zone(self):
        result = analyze_thresholds({"rsi_lower": 40, "rsi_upper": 60})
        assert result["rsi_zone_width"] == 20
        assert result["rsi_zone_status"] == "red"

    def test_defaults_when_missing_keys(self):
        result = analyze_thresholds({})
        # Defaults: macd_fast=12, macd_slow=26, rsi_lower=30, rsi_upper=70
        assert result["macd_spread"] == 14
        assert result["rsi_zone_width"] == 40


# =============================================================================
# TestCheckRobustness
# =============================================================================


class TestCheckRobustness:
    def test_normal_robustness(self):
        result = check_robustness(_good_params())
        assert result["overfit_risk"] == "low"
        assert result["splits_positive"] == 3
        assert result["splits_total"] == 3
        assert result["splits_pct_positive"] == 1.0

    def test_high_overfit(self):
        params = {
            "train_sharpe_equity": 4.0,
            "test_sharpe_equity": 1.0,
        }
        result = check_robustness(params)
        assert result["overfit_score"] == 0.75
        assert result["overfit_risk"] == "high"

    def test_train_sharpe_zero_no_division_error(self):
        result = check_robustness({"train_sharpe_equity": 0, "test_sharpe_equity": 1.0})
        assert result["overfit_score"] == 0

    def test_empty_splits(self):
        result = check_robustness({"split_results": []})
        assert result["splits_total"] == 0
        assert result["splits_pct_positive"] == 0

    def test_mc_fields_passthrough(self):
        params = {"mc_sharpe_mean": 1.5, "mc_confidence": "HIGH"}
        result = check_robustness(params)
        assert result["mc_sharpe_mean"] == 1.5
        assert result["mc_confidence"] == "HIGH"


# =============================================================================
# TestAnalyzeTrades
# =============================================================================


class TestAnalyzeTrades:
    def test_normal_trades(self, tmp_path):
        csv_path = tmp_path / "trades.csv"
        _write_trades_csv(csv_path, _sample_trades())

        result = analyze_trades(csv_path)
        assert result["total_trades"] == 4
        assert "trailing_stop" in result["exit_reasons"]
        assert "catastrophic_stop" in result["exit_reasons"]
        assert result["catastrophic_pct"] == 0.25
        assert len(result["top_winners"]) == 2  # 2 positive trades
        assert len(result["top_losers"]) == 2   # 2 negative trades

    def test_direction_stats(self, tmp_path):
        csv_path = tmp_path / "trades.csv"
        _write_trades_csv(csv_path, _sample_trades())

        result = analyze_trades(csv_path)
        assert "long" in result["direction_stats"]
        assert "short" in result["direction_stats"]
        assert result["direction_stats"]["long"]["count"] == 3
        assert result["direction_stats"]["short"]["count"] == 1

    def test_min_hold_pct(self, tmp_path):
        csv_path = tmp_path / "trades.csv"
        _write_trades_csv(csv_path, _sample_trades())

        result = analyze_trades(csv_path)
        # 2 trades with hold_bars=2 out of 4 total
        assert result["min_hold_pct"] == 0.5

    def test_missing_direction_column(self, tmp_path):
        """Gracefully handles CSV without direction column."""
        csv_path = tmp_path / "trades.csv"
        trades = _sample_trades()
        # Remove direction from all trades
        for t in trades:
            del t["direction"]
        fieldnames = [
            "entry_ts", "entry_price", "exit_ts", "exit_price",
            "hold_bars", "size", "capital_at_entry",
            "pnl_abs", "pnl_pct", "symbol", "reason",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in trades:
                writer.writerow(t)

        result = analyze_trades(csv_path)
        assert result["total_trades"] == 4
        assert result["direction_stats"] == {}


# =============================================================================
# TestGenerateSuggestions
# =============================================================================


class TestGenerateSuggestions:
    def test_no_issues_no_suggestions(self):
        health = {k: {"status": "green"} for k in [
            "sharpe", "max_drawdown", "trades_per_year",
            "win_rate", "profit_factor", "expectancy",
        ]}
        thresholds = {"rsi_zone_status": "green", "macd_spread_status": "green"}
        trades = {"catastrophic_pct": 0.1, "direction_stats": {}}
        robustness = {"overfit_risk": "low"}

        suggestions = generate_suggestions(health, thresholds, trades, robustness)
        assert len(suggestions) == 0

    def test_max_five_suggestions(self):
        health = {
            "trades_per_year": {"status": "red"},
            "sharpe": {"status": "red"},
        }
        thresholds = {"rsi_zone_status": "red", "macd_spread_status": "yellow"}
        trades = {
            "catastrophic_pct": 0.5,
            "direction_stats": {
                "long": {"total_pnl": -1000},
                "short": {"total_pnl": -500},
            },
        }
        robustness = {"overfit_risk": "high", "overfit_score": 0.8}

        suggestions = generate_suggestions(health, thresholds, trades, robustness)
        assert len(suggestions) <= 5

    def test_high_catastrophic_triggers_suggestion(self):
        health = {k: {"status": "green"} for k in ["trades_per_year", "sharpe"]}
        thresholds = {"rsi_zone_status": "green", "macd_spread_status": "green"}
        trades = {"catastrophic_pct": 0.5, "direction_stats": {}}
        robustness = {"overfit_risk": "low"}

        suggestions = generate_suggestions(health, thresholds, trades, robustness)
        assert len(suggestions) == 1
        assert "stop" in suggestions[0]["action"].lower()


# =============================================================================
# TestBuildDiscordEmbed
# =============================================================================


class TestBuildDiscordEmbed:
    def test_normal_embed(self):
        analysis = {
            "run_name": "2025-01-01_test",
            "symbol": "BTC/USDT",
            "n_trials": 10000,
            "n_splits": 3,
            "verdict": "PASS",
            "health": {
                "sharpe": {"status": "green", "value": 2.0},
                "max_drawdown": {"status": "green", "value": -3.0},
            },
            "suggestions": [],
        }
        embed = build_discord_embed(analysis)
        assert embed.startswith("```")
        assert embed.endswith("```")
        assert "PASS" in embed
        assert "BTC/USDT" in embed
        assert "10,000" in embed

    def test_truncation_at_1900_chars(self):
        # Create analysis with many health metrics to exceed 1900 chars
        health = {}
        for i in range(50):
            health[f"metric_{i:03d}_with_long_name_here"] = {
                "status": "yellow",
                "value": f"some_long_value_{i}",
            }
        analysis = {
            "run_name": "test",
            "symbol": "BTC",
            "n_trials": 1000,
            "n_splits": 3,
            "verdict": "REVIEW",
            "health": health,
            "suggestions": [{"action": f"suggestion {i}"} for i in range(5)],
        }
        embed = build_discord_embed(analysis)
        assert len(embed) <= 1900
        assert embed.endswith("```")


# =============================================================================
# TestSaveAnalysis
# =============================================================================


class TestSaveAnalysis:
    def test_saves_json_with_timestamp(self, tmp_path):
        analysis = {"verdict": "PASS", "health": {}}
        out_path = tmp_path / "analysis.json"

        save_analysis(analysis, out_path)

        with open(out_path) as f:
            data = json.load(f)
        assert data["verdict"] == "PASS"
        assert "timestamp" in data

    def test_creates_parent_directory(self, tmp_path):
        out_path = tmp_path / "nested" / "dir" / "analysis.json"

        save_analysis({"verdict": "FAIL"}, out_path)

        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert data["verdict"] == "FAIL"


# =============================================================================
# TestFindSymbolDir
# =============================================================================


class TestFindSymbolDir:
    def test_finds_symbol_dir(self, tmp_path):
        btc_dir = tmp_path / "BTC"
        btc_dir.mkdir()
        (btc_dir / "best_params.json").write_text("{}")

        result = _find_symbol_dir(tmp_path)
        assert result == btc_dir

    def test_skips_checkpoints(self, tmp_path):
        ckpt = tmp_path / "checkpoints"
        ckpt.mkdir()
        (ckpt / "best_params.json").write_text("{}")

        btc_dir = tmp_path / "SOL"
        btc_dir.mkdir()
        (btc_dir / "best_params.json").write_text("{}")

        result = _find_symbol_dir(tmp_path)
        assert result.name == "SOL"

    def test_raises_when_no_symbol_dir(self, tmp_path):
        (tmp_path / "empty_dir").mkdir()
        with pytest.raises(FileNotFoundError):
            _find_symbol_dir(tmp_path)


# =============================================================================
# TestBuildFindings
# =============================================================================


class TestBuildFindings:
    def test_collects_reds_and_yellows(self):
        health = {
            "sharpe": {"status": "red", "value": 0.3},
            "win_rate": {"status": "yellow", "value": 0.45},
            "max_drawdown": {"status": "green", "value": -2.0},
        }
        trades = {"catastrophic_pct": 0.1, "total_trades": 100,
                  "exit_reasons": {"signal": {"pct": 0.6}},
                  "min_hold_pct": 0.1, "direction_stats": {}}
        thresholds = {}
        robustness = {"overfit_risk": "low"}

        findings = _build_findings(health, trades, thresholds, robustness)
        metrics = [f["metric"] for f in findings]
        assert "sharpe" in metrics
        assert "win_rate" in metrics
        assert "max_drawdown" not in metrics

    def test_high_catastrophic_is_red_finding(self):
        health = {}
        trades = {"catastrophic_pct": 0.5, "total_trades": 100,
                  "exit_reasons": {"signal": {"pct": 0.6}},
                  "min_hold_pct": 0.1, "direction_stats": {}}
        findings = _build_findings(health, trades, {}, {"overfit_risk": "low"})
        cat_finding = [f for f in findings if f["metric"] == "catastrophic_pct"]
        assert len(cat_finding) == 1
        assert cat_finding[0]["severity"] == "red"

    def test_sorted_red_before_yellow(self):
        health = {
            "a": {"status": "yellow", "value": 1},
            "b": {"status": "red", "value": 2},
        }
        trades = {"catastrophic_pct": 0, "total_trades": 0,
                  "exit_reasons": {}, "min_hold_pct": 0, "direction_stats": {}}
        findings = _build_findings(health, trades, {}, {"overfit_risk": "low"})
        assert findings[0]["severity"] == "red"
        assert findings[1]["severity"] == "yellow"

    def test_overfit_risk_high(self):
        health = {}
        trades = {"catastrophic_pct": 0, "total_trades": 0,
                  "exit_reasons": {}, "min_hold_pct": 0, "direction_stats": {}}
        findings = _build_findings(health, trades, {}, {
            "overfit_risk": "high", "overfit_score": 0.7,
        })
        assert any(f["metric"] == "overfit_risk" for f in findings)
