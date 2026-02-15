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
    def test_returns_all_timeframes(self, mock_sleep):
        """load_all_data should return dict with all configured timeframes."""
        mock_exchange = MagicMock()
        # First call returns data, subsequent calls return empty (end of data)
        mock_exchange.fetch_ohlcv.side_effect = lambda *a, **kw: [
            [utcnow_ms() - 60_000, 100.0, 101.0, 99.0, 100.5, 1000.0],
        ] if mock_exchange.fetch_ohlcv.call_count <= 7 else []
        mock_exchange.rateLimit = 100

        data = load_all_data(mock_exchange, "BTC/USDC", 100)

        assert "1h" in data
        assert "2h" in data
        assert "4h" in data
        assert "1d" in data
