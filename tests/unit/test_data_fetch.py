# tests/unit/test_data_fetch.py
"""Unit tests for QRE data fetching."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from qre.data.fetch import fetch_ohlcv_paginated, load_all_data, utcnow_ms


class TestUtcnowMs:
    def test_returns_int(self):
        result = utcnow_ms()
        assert isinstance(result, int)

    def test_reasonable_value(self):
        """Should be after 2025-01-01 in milliseconds."""
        result = utcnow_ms()
        assert result > 1735689600000  # 2025-01-01


class TestLoadAllData:
    @patch("qre.data.fetch.time.sleep")
    def test_returns_base_and_trend_timeframes(self, mock_sleep):
        """load_all_data returns dict with 1h + trend TFs (4h, 8h, 1d)."""
        mock_exchange = MagicMock()
        call_count = {"n": 0}

        def mock_fetch(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] % 2 == 1:
                return [[utcnow_ms() - 60_000, 100.0, 101.0, 99.0, 100.5, 1000.0]]
            return []

        mock_exchange.fetch_ohlcv.side_effect = mock_fetch
        mock_exchange.rateLimit = 100

        data = load_all_data(mock_exchange, "BTC/USDC", 100)

        assert "1h" in data
        assert "4h" in data
        assert "8h" in data
        assert "1d" in data
        assert len(data) == 4

    @patch("qre.data.fetch.time.sleep")
    def test_returns_base_timeframe(self, mock_sleep):
        """Backward compat: 1h is always present."""
        mock_exchange = MagicMock()
        call_count = {"n": 0}

        def mock_fetch(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] % 2 == 1:
                return [[utcnow_ms() - 60_000, 100.0, 101.0, 99.0, 100.5, 1000.0]]
            return []

        mock_exchange.fetch_ohlcv.side_effect = mock_fetch
        mock_exchange.rateLimit = 100

        data = load_all_data(mock_exchange, "BTC/USDC", 100)
        assert "1h" in data
