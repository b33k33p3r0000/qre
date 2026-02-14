# tests/unit/test_analyze.py
"""Unit tests for QRE post-run analysis pipeline."""

import csv
import math

import pytest

from qre.analyze import (
    analyze_thresholds,
    analyze_trades,
    health_check,
)


@pytest.fixture
def good_params():
    return {
        "sharpe": 2.0,
        "max_drawdown": -0.04,
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
        "max_drawdown": -0.15,
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
        good_params["max_drawdown"] = -0.07  # yellow: -5% to -10%
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
