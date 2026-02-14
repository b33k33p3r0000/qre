# tests/unit/test_analyze.py
"""Unit tests for QRE post-run analysis pipeline."""

import pytest

from qre.analyze import health_check


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
