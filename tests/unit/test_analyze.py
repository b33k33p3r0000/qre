# tests/unit/test_analyze.py
"""Unit tests for QRE post-run analysis pipeline."""

import csv
import json
import math

import pytest

from unittest.mock import patch

from qre.analyze import (
    analyze_run,
    analyze_thresholds,
    analyze_trades,
    build_discord_embed,
    check_robustness,
    compute_verdict,
    generate_suggestions,
    health_check,
    save_analysis,
)


@pytest.fixture
def good_params():
    return {
        "sharpe": 2.0,
        "max_drawdown": -4.0,
        "trades_per_year": 120,
        "win_rate": 0.55,
        "profit_factor": 1.8,
        "expectancy": 150.0,
        "train_sharpe": 2.5,
        "test_sharpe": 2.0,
        "split_results": [
            {"test_equity": 11000, "test_sharpe": 1.5},
            {"test_equity": 11500, "test_sharpe": 1.8},
            {"test_equity": 10800, "test_sharpe": 1.2},
            {"test_equity": 11200, "test_sharpe": 1.6},
        ],
    }


@pytest.fixture
def bad_params():
    return {
        "sharpe": 0.3,
        "max_drawdown": -15.0,
        "trades_per_year": 5,
        "win_rate": 0.35,
        "profit_factor": 0.9,
        "expectancy": -10.0,
        "train_sharpe": 4.0,
        "test_sharpe": 0.5,
        "split_results": [
            {"test_equity": 9000, "test_sharpe": -0.5},
            {"test_equity": 8500, "test_sharpe": -0.8},
            {"test_equity": 10200, "test_sharpe": 0.3},
            {"test_equity": 9800, "test_sharpe": -0.1},
        ],
    }


class TestHealthCheck:
    def test_all_green(self, good_params):
        """Good params produce all-green health check."""
        result = health_check(good_params)
        for metric, info in result.items():
            assert info["status"] == "green", (
                f"{metric} should be green, got {info['status']}"
            )

    def test_all_red(self, bad_params):
        """Bad params produce all-red health check."""
        result = health_check(bad_params)
        for metric, info in result.items():
            assert info["status"] == "red", (
                f"{metric} should be red, got {info['status']}"
            )

    def test_sharpe_too_high_is_red(self, good_params):
        """Sharpe > 5.0 is red (bidirectional — not just too low)."""
        good_params["sharpe"] = 6.0
        result = health_check(good_params)
        assert result["sharpe"]["status"] == "red"

    def test_yellow_zones(self, good_params):
        """Values in yellow ranges produce yellow status."""
        good_params["sharpe"] = 0.8          # yellow: 0.5–1.0
        good_params["max_drawdown"] = -7.0   # yellow: -5% to -10%
        good_params["trades_per_year"] = 50  # yellow: 30–80
        good_params["win_rate"] = 0.45       # yellow: 40%–50%
        good_params["profit_factor"] = 1.2   # yellow: 1.0–1.5
        good_params["expectancy"] = 50.0     # yellow: $0–$100
        good_params["train_sharpe"] = 3.0    # diff = 1.0 → yellow: 1.0–2.0
        good_params["test_sharpe"] = 2.0
        # split_results: 1 negative → yellow
        good_params["split_results"] = [
            {"test_equity": 9800, "test_sharpe": -0.2},
            {"test_equity": 11000, "test_sharpe": 1.5},
            {"test_equity": 10800, "test_sharpe": 1.2},
            {"test_equity": 11200, "test_sharpe": 1.6},
        ]
        result = health_check(good_params)
        for metric, info in result.items():
            assert info["status"] == "yellow", (
                f"{metric} should be yellow, got {info['status']}"
            )

    def test_train_test_divergence(self, good_params):
        """Train/test sharpe diff > 2.0 is red."""
        good_params["train_sharpe"] = 4.0
        good_params["test_sharpe"] = 1.0
        result = health_check(good_params)
        assert result["train_test_sharpe"]["status"] == "red"

    def test_split_consistency(self, good_params):
        """3 negative splits → red (2+ negative = red)."""
        good_params["split_results"] = [
            {"test_equity": 9000, "test_sharpe": -0.5},
            {"test_equity": 8500, "test_sharpe": -0.8},
            {"test_equity": 10200, "test_sharpe": -0.3},
            {"test_equity": 11200, "test_sharpe": 1.6},
        ]
        result = health_check(good_params)
        assert result["split_consistency"]["status"] == "red"


# --- analyze_thresholds tests ---


@pytest.fixture
def threshold_params():
    params = {
        "p_buy": 0.25, "k_sell": 2,
        "kB": 5, "dB": 3,
        "macd_fast": 8, "macd_slow": 22, "macd_signal": 6,
        "macd_mode": "rising",
        "rsi_mode": "trend_filter",
        "rsi_lower": 32, "rsi_upper": 60, "rsi_momentum_level": 46,
        "rsi_gate_6h": 48, "rsi_gate_8h": 50,
        "rsi_gate_12h": 52, "rsi_gate_24h": 45,
    }
    for tf in ["2h", "4h", "6h", "8h", "12h", "24h"]:
        params[f"low_{tf}"] = 0.20
        params[f"high_{tf}"] = 0.80
    params["low_24h"] = 0.05
    params["high_24h"] = 0.92
    return params


class TestAnalyzeThresholds:
    def test_required_votes(self, threshold_params):
        """ceil(0.25 * 6) = 2 required buy votes."""
        result = analyze_thresholds(threshold_params)
        assert result["required_buy_votes"] == math.ceil(0.25 * 6)
        assert result["required_buy_votes"] == 2

    def test_threshold_width(self, threshold_params):
        """2h width = high - low = 0.80 - 0.20 = 0.60."""
        result = analyze_thresholds(threshold_params)
        tf_2h = result["tf_analysis"]["2h"]
        assert tf_2h["width"] == pytest.approx(0.60)

    def test_dead_tf_detection(self, threshold_params):
        """24h width = 0.92 - 0.05 = 0.87 > 0.8 → dead."""
        result = analyze_thresholds(threshold_params)
        tf_24h = result["tf_analysis"]["24h"]
        assert tf_24h["width"] == pytest.approx(0.87)
        assert tf_24h["dead"] is True

    def test_cap_collision_buy(self, threshold_params):
        """low_2h = 0.65 > BUY_CAP(0.6) → buy_cap_collision."""
        threshold_params["low_2h"] = 0.65
        result = analyze_thresholds(threshold_params)
        tf_2h = result["tf_analysis"]["2h"]
        assert tf_2h["buy_cap_collision"] is True

    def test_macd_spread(self, threshold_params):
        """MACD spread = slow - fast = 22 - 8 = 14."""
        result = analyze_thresholds(threshold_params)
        assert result["macd_spread"] == 14

    def test_rsi_gates(self, threshold_params):
        """4 RSI gates with correct values."""
        result = analyze_thresholds(threshold_params)
        gates = result["rsi_gates"]
        assert len(gates) == 4
        assert gates["6h"] == 48
        assert gates["8h"] == 50
        assert gates["12h"] == 52
        assert gates["24h"] == 45


# --- check_robustness tests ---


class TestCheckRobustness:
    def test_overfit_score(self, good_params):
        """good_params → (2.5 - 2.0) / 2.5 = 0.2."""
        result = check_robustness(good_params)
        assert result["overfit_score"] == pytest.approx(0.2)

    def test_high_overfit(self, bad_params):
        """bad_params → (4.0 - 0.5) / 4.0 = 0.875, risk = 'high'."""
        result = check_robustness(bad_params)
        assert result["overfit_score"] == pytest.approx(0.875)
        assert result["overfit_risk"] == "high"

    def test_split_stats(self, good_params):
        """good_params → 4/4 positive, pct = 1.0."""
        result = check_robustness(good_params)
        assert result["splits_positive"] == 4
        assert result["splits_total"] == 4
        assert result["splits_pct_positive"] == pytest.approx(1.0)

    def test_monte_carlo_from_params(self, good_params):
        """MC fields passed through from params."""
        good_params["mc_sharpe_mean"] = 1.8
        good_params["mc_confidence"] = 0.95
        result = check_robustness(good_params)
        assert result["mc_sharpe_mean"] == 1.8
        assert result["mc_confidence"] == 0.95


# --- analyze_trades tests ---


@pytest.fixture
def trades_csv(tmp_path):
    """Create a synthetic trades CSV with 5 trades, mixed exit reasons."""
    path = tmp_path / "trades.csv"
    header = [
        "entry_ts",
        "entry_price",
        "exit_ts",
        "exit_price",
        "hold_bars",
        "size",
        "capital_at_entry",
        "pnl_abs",
        "pnl_pct",
        "symbol",
        "reason",
    ]
    rows = [
        # Trade 1: catastrophic_stop, big loss
        [
            "2025-06-01 09:00",
            "67500.00",
            "2025-06-01 09:45",
            "66200.00",
            3,
            0.10,
            10000.00,
            -130.00,
            -0.013,
            "BTC/USD",
            "catastrophic_stop",
        ],
        # Trade 2: signal@open, small win
        [
            "2025-06-02 14:00",
            "67800.00",
            "2025-06-02 18:00",
            "68100.00",
            16,
            0.10,
            9870.00,
            30.00,
            0.00304,
            "BTC/USD",
            "signal@open",
        ],
        # Trade 3: catastrophic_stop, medium loss
        [
            "2025-06-03 10:00",
            "68000.00",
            "2025-06-03 10:30",
            "67400.00",
            2,
            0.10,
            9900.00,
            -60.00,
            -0.00606,
            "BTC/USD",
            "catastrophic_stop",
        ],
        # Trade 4: signal@open, big win
        [
            "2025-06-04 08:00",
            "67000.00",
            "2025-06-04 20:00",
            "68500.00",
            48,
            0.10,
            9840.00,
            150.00,
            0.01524,
            "BTC/USD",
            "signal@open",
        ],
        # Trade 5: trailing_stop, small win
        [
            "2025-06-05 12:00",
            "68200.00",
            "2025-06-05 16:00",
            "68600.00",
            16,
            0.10,
            9990.00,
            40.00,
            0.004,
            "BTC/USD",
            "trailing_stop",
        ],
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


class TestAnalyzeTrades:
    def test_exit_reason_breakdown(self, trades_csv):
        """Verify count per exit reason."""
        result = analyze_trades(trades_csv)
        reasons = result["exit_reasons"]
        assert reasons["catastrophic_stop"]["count"] == 2
        assert reasons["signal@open"]["count"] == 2
        assert reasons["trailing_stop"]["count"] == 1

    def test_catastrophic_percentage(self, trades_csv):
        """Catastrophic stop percentage: 2 out of 5 = 0.4."""
        result = analyze_trades(trades_csv)
        assert result["catastrophic_pct"] == pytest.approx(0.4)

    def test_hold_time_stats(self, trades_csv):
        """Min and max hold_bars from the 5 trades."""
        result = analyze_trades(trades_csv)
        hold = result["hold_bars"]
        assert hold["min"] == 2
        assert hold["max"] == 48

    def test_top_winners_losers(self, trades_csv):
        """Winners sorted desc by pnl_abs (max 3), losers are only negative."""
        result = analyze_trades(trades_csv)
        winners = result["top_winners"]
        losers = result["top_losers"]
        # 3 winners: 150, 40, 30 (sorted desc)
        assert len(winners) == 3
        assert winners[0]["pnl_abs"] == pytest.approx(150.0)
        assert winners[1]["pnl_abs"] == pytest.approx(40.0)
        assert winners[2]["pnl_abs"] == pytest.approx(30.0)
        # 2 losers: -60, -130 (sorted asc = most negative first)
        assert len(losers) == 2
        assert losers[0]["pnl_abs"] == pytest.approx(-130.0)
        assert losers[1]["pnl_abs"] == pytest.approx(-60.0)
        # All losers must have negative pnl
        for loser in losers:
            assert loser["pnl_abs"] < 0

    def test_trade_count(self, trades_csv):
        """Total trade count = 5."""
        result = analyze_trades(trades_csv)
        assert result["total_trades"] == 5


# --- compute_verdict tests ---


class TestComputeVerdict:
    def test_pass_all_green(self):
        """All green metrics → PASS."""
        health = {
            "sharpe": {"status": "green", "value": 2.0},
            "max_drawdown": {"status": "green", "value": -0.03},
            "trades_per_year": {"status": "green", "value": 120},
            "win_rate": {"status": "green", "value": 0.55},
            "profit_factor": {"status": "green", "value": 1.8},
        }
        assert compute_verdict(health) == "PASS"

    def test_review_on_red(self):
        """1 red metric → REVIEW."""
        health = {
            "sharpe": {"status": "red", "value": 0.3},
            "max_drawdown": {"status": "green", "value": -0.03},
            "trades_per_year": {"status": "green", "value": 120},
        }
        assert compute_verdict(health) == "REVIEW"

    def test_fail_on_multiple_red(self):
        """2+ red metrics → FAIL."""
        health = {
            "sharpe": {"status": "red", "value": 0.3},
            "max_drawdown": {"status": "red", "value": -0.15},
            "trades_per_year": {"status": "green", "value": 120},
        }
        assert compute_verdict(health) == "FAIL"

    def test_pass_with_few_yellows(self):
        """2 yellows → PASS."""
        health = {
            "sharpe": {"status": "yellow", "value": 0.8},
            "max_drawdown": {"status": "yellow", "value": -0.07},
            "trades_per_year": {"status": "green", "value": 120},
        }
        assert compute_verdict(health) == "PASS"

    def test_review_with_many_yellows(self):
        """3+ yellows → REVIEW."""
        health = {
            "sharpe": {"status": "yellow", "value": 0.8},
            "max_drawdown": {"status": "yellow", "value": -0.07},
            "trades_per_year": {"status": "yellow", "value": 50},
            "win_rate": {"status": "green", "value": 0.55},
        }
        assert compute_verdict(health) == "REVIEW"


# --- generate_suggestions tests ---


class TestGenerateSuggestions:
    def test_low_trades_suggests_lower_p_buy(self):
        """Low trades + high required_buy_votes → suggest lowering p_buy."""
        health = {
            "trades_per_year": {"status": "red", "value": 5},
            "sharpe": {"status": "green", "value": 2.0},
        }
        thresholds = {"required_buy_votes": 3, "macd_mode": "rising"}
        trades = {"catastrophic_pct": 0.1}
        robustness = {"overfit_risk": "low"}

        suggestions = generate_suggestions(health, thresholds, trades, robustness)
        actions = [s["action"] for s in suggestions]
        assert any("p_buy" in a.lower() for a in actions)

    def test_high_catastrophic_suggests_change(self):
        """High catastrophic_pct → high priority suggestion."""
        health = {
            "trades_per_year": {"status": "green", "value": 120},
            "sharpe": {"status": "green", "value": 2.0},
        }
        thresholds = {"required_buy_votes": 2, "macd_mode": "rising"}
        trades = {"catastrophic_pct": 0.65}
        robustness = {"overfit_risk": "low"}

        suggestions = generate_suggestions(health, thresholds, trades, robustness)
        assert any(s["priority"] == "high" for s in suggestions)

    def test_max_5_suggestions(self):
        """Many problems → max 5 suggestions returned."""
        health = {
            "trades_per_year": {"status": "red", "value": 5},
            "sharpe": {"status": "red", "value": 0.3},
            "max_drawdown": {"status": "red", "value": -0.15},
            "win_rate": {"status": "red", "value": 0.35},
            "profit_factor": {"status": "red", "value": 0.8},
        }
        thresholds = {"required_buy_votes": 4, "macd_mode": "crossover"}
        trades = {"catastrophic_pct": 0.65}
        robustness = {"overfit_risk": "high"}

        suggestions = generate_suggestions(health, thresholds, trades, robustness)
        assert len(suggestions) <= 5


# --- build_discord_embed tests ---


@pytest.fixture
def full_analysis():
    """A complete analysis dict for testing embed and save."""
    return {
        "run_name": "btc_run_042",
        "symbol": "BTC/USD",
        "n_trials": 200,
        "n_splits": 4,
        "verdict": "FAIL",
        "health": {
            "sharpe": {"status": "green", "value": 2.0},
            "max_drawdown": {"status": "red", "value": -0.12},
            "trades_per_year": {"status": "red", "value": 15},
            "win_rate": {"status": "yellow", "value": 0.45},
            "profit_factor": {"status": "green", "value": 1.6},
            "expectancy": {"status": "green", "value": 120.0},
            "train_test_sharpe": {"status": "yellow", "value": 1.5},
            "split_consistency": {"status": "green", "value": 0},
        },
        "suggestions": [
            {
                "priority": "high",
                "action": "Lower p_buy to reduce required buy votes",
                "reason": "Too few trades",
                "impact": "More trades",
            },
            {
                "priority": "high",
                "action": "Tighten stop-loss",
                "reason": "High catastrophic rate",
                "impact": "Better risk management",
            },
        ],
    }


class TestBuildDiscordEmbed:
    def test_embed_under_6000_chars(self, full_analysis):
        """Full analysis → embed length <= 6000 chars (Discord limit)."""
        embed = build_discord_embed(full_analysis)
        assert len(embed) <= 6000

    def test_embed_contains_verdict(self, full_analysis):
        """Verdict string appears in embed."""
        embed = build_discord_embed(full_analysis)
        assert "FAIL" in embed


# --- save_analysis tests ---


class TestSaveAnalysis:
    def test_saves_json(self, full_analysis, tmp_path):
        """Saves to path, loadable as JSON, verdict preserved."""
        path = tmp_path / "analysis.json"
        save_analysis(full_analysis, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["verdict"] == "FAIL"

    def test_adds_timestamp(self, full_analysis, tmp_path):
        """Saved JSON has 'timestamp' key."""
        path = tmp_path / "analysis.json"
        save_analysis(full_analysis, path)
        with open(path) as f:
            loaded = json.load(f)
        assert "timestamp" in loaded


# --- analyze_run orchestrator tests ---


class TestAnalyzeRun:
    def _make_full_params(self, good_params):
        """Merge good_params with threshold + strategy params for full pipeline."""
        full = dict(good_params)
        full.update({
            # threshold params
            "p_buy": 0.25, "k_sell": 2,
            "macd_fast": 8, "macd_slow": 22, "macd_signal": 6,
            "macd_mode": "rising",
            "rsi_mode": "trend_filter",
            "rsi_gate_6h": 48, "rsi_gate_8h": 50,
            "rsi_gate_12h": 52, "rsi_gate_24h": 45,
            # strategy meta
            "symbol": "BTC/USDC", "n_trials": 200, "n_splits": 4,
            "run_timestamp": "2026-02-14_test-run",
        })
        for tf in ["2h", "4h", "6h", "8h", "12h", "24h"]:
            full[f"low_{tf}"] = 0.20
            full[f"high_{tf}"] = 0.80
        return full

    def test_orchestrator_produces_complete_output(self, good_params, trades_csv, tmp_path):
        """Full pipeline: creates analysis.json, returns verdict/health/suggestions."""
        full_params = self._make_full_params(good_params)

        # Build run dir structure: tmp_path/2026-02-14_test-run/BTC/
        run_dir = tmp_path / "2026-02-14_test-run"
        symbol_dir = run_dir / "BTC"
        symbol_dir.mkdir(parents=True)

        # Write best_params.json
        with open(symbol_dir / "best_params.json", "w") as f:
            json.dump(full_params, f)

        # Copy trades CSV into symbol dir
        import shutil
        shutil.copy(trades_csv, symbol_dir / "trades_BTC_USDC_1h_FULL.csv")

        # No DISCORD_WEBHOOK_RUNS env var → no discord call
        result = analyze_run(str(run_dir))

        # Verify result structure
        assert result["verdict"] == "PASS"  # good_params → all green → PASS
        assert "health" in result
        assert "trades" in result
        assert "thresholds" in result
        assert "robustness" in result
        assert "suggestions" in result
        assert "findings" in result

        # Health has expected metrics
        health = result["health"]
        assert "sharpe" in health
        assert "max_drawdown" in health

        # analysis.json was saved
        analysis_path = symbol_dir / "analysis.json"
        assert analysis_path.exists()
        with open(analysis_path) as f:
            saved = json.load(f)
        assert saved["verdict"] == "PASS"
        assert "timestamp" in saved
