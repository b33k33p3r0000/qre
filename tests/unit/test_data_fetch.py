# tests/unit/test_data_fetch.py
"""Unit tests for QRE data fetching & caching."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from qre.data.fetch import DataCache, fetch_ohlcv_paginated, load_all_data, utcnow_ms


class TestUtcnowMs:
    def test_returns_int(self):
        result = utcnow_ms()
        assert isinstance(result, int)

    def test_reasonable_value(self):
        """Should be after 2025-01-01 in milliseconds."""
        result = utcnow_ms()
        assert result > 1735689600000  # 2025-01-01


class TestDataCache:
    def test_init_creates_dir(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = DataCache(str(cache_dir))
        assert cache_dir.exists()

    def test_save_and_load(self, tmp_path):
        cache = DataCache(str(tmp_path))
        df = pd.DataFrame(
            {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2025-01-01", tz="UTC")]),
        )
        cache.save("BTC/USDC", "1h", df)
        loaded = cache.load("BTC/USDC", "1h")
        assert loaded is not None
        assert len(loaded) == 1

    def test_load_nonexistent_returns_none(self, tmp_path):
        cache = DataCache(str(tmp_path))
        assert cache.load("BTC/USDC", "1h") is None

    def test_get_last_timestamp(self, tmp_path):
        cache = DataCache(str(tmp_path))
        ts = pd.Timestamp("2025-06-15 12:00", tz="UTC")
        df = pd.DataFrame(
            {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100.0]},
            index=pd.DatetimeIndex([ts]),
        )
        cache.save("BTC/USDC", "1h", df)
        last = cache.get_last_timestamp("BTC/USDC", "1h")
        assert last == int(ts.timestamp() * 1000)

    def test_update_merges_data(self, tmp_path):
        cache = DataCache(str(tmp_path))
        ts1 = pd.Timestamp("2025-01-01", tz="UTC")
        ts2 = pd.Timestamp("2025-01-02", tz="UTC")
        df1 = pd.DataFrame(
            {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100.0]},
            index=pd.DatetimeIndex([ts1]),
        )
        df2 = pd.DataFrame(
            {"open": [2.0], "high": [3.0], "low": [1.0], "close": [2.5], "volume": [200.0]},
            index=pd.DatetimeIndex([ts2]),
        )
        cache.save("BTC/USDC", "1h", df1)
        combined = cache.update("BTC/USDC", "1h", df2)
        assert len(combined) == 2

    def test_clear_symbol(self, tmp_path):
        cache = DataCache(str(tmp_path))
        df = pd.DataFrame(
            {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2025-01-01", tz="UTC")]),
        )
        cache.save("BTC/USDC", "1h", df)
        cache.clear("BTC/USDC")
        assert cache.load("BTC/USDC", "1h") is None


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

        data = load_all_data(mock_exchange, "BTC/USDC", 100, cache=None)

        assert "1h" in data
        assert "2h" in data
        assert "4h" in data
        assert "1d" in data
