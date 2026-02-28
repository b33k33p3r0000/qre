# tests/unit/test_notify.py
"""Unit tests for QRE Discord notifications."""

from unittest.mock import patch, MagicMock

import pytest

from qre.notify import format_start_message, format_complete_message, discord_notify


class TestFormatStartMessage:
    def test_contains_symbol(self):
        msg = format_start_message(symbol="BTC/USDT", n_trials=10000, hours=8760, n_splits=3, run_tag="test-v1")
        assert "BTC/USDT" in msg

    def test_contains_trials(self):
        msg = format_start_message(symbol="BTC/USDT", n_trials=10000, hours=8760, n_splits=3)
        assert "10,000" in msg or "10000" in msg

    def test_contains_tag_when_provided(self):
        msg = format_start_message(symbol="BTC/USDT", n_trials=10000, hours=8760, n_splits=3, run_tag="my-tag")
        assert "my-tag" in msg

    def test_no_tag_when_none(self):
        msg = format_start_message(symbol="BTC/USDT", n_trials=10000, hours=8760, n_splits=3, run_tag=None)
        assert "None" not in msg


class TestFormatCompleteMessage:
    def test_contains_equity(self):
        params = {"symbol": "BTC/USDT", "equity": 51234.56, "sharpe": 2.5, "trades": 200,
                  "max_drawdown": -3.0, "win_rate": 0.48, "total_pnl_pct": 2.47,
                  "mc_confidence": "HIGH", "train_sharpe": 3.0, "test_sharpe": -1.5}
        msg = format_complete_message(params)
        assert "51,234" in msg or "51234" in msg

    def test_contains_sharpe(self):
        params = {"symbol": "BTC/USDT", "equity": 51000, "sharpe": 2.5, "trades": 200,
                  "max_drawdown": -3.0, "win_rate": 0.48, "total_pnl_pct": 2.47}
        msg = format_complete_message(params)
        assert "2.5" in msg or "2.50" in msg

    def test_overfit_warning_when_negative_test_sharpe(self):
        params = {"symbol": "BTC/USDT", "equity": 51000, "sharpe": 2.5, "trades": 200,
                  "max_drawdown": -3.0, "win_rate": 0.48, "total_pnl_pct": 2.47,
                  "train_sharpe": 3.0, "test_sharpe": -1.5}
        msg = format_complete_message(params)
        assert "overfit" in msg.lower() or "warning" in msg.lower() or "\u26a0" in msg


class TestDiscordNotify:
    @patch("qre.notify.requests.post")
    def test_sends_when_webhook_set(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        result = discord_notify("test message", "https://discord.com/api/webhooks/test")
        assert result is True
        mock_post.assert_called_once()

    def test_returns_false_when_no_webhook(self):
        result = discord_notify("test message", "")
        assert result is False

    @patch("qre.notify.requests.post")
    def test_returns_false_on_error(self, mock_post):
        mock_post.side_effect = Exception("connection error")
        result = discord_notify("test message", "https://discord.com/api/webhooks/test")
        assert result is False
