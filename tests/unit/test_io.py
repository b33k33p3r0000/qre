# tests/unit/test_io.py
"""Unit tests for QRE IO utilities."""

import json
from pathlib import Path

import pytest

from qre.io import save_json, save_trades_csv


class TestSaveJson:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "output" / "test.json"
        save_json(path, {"key": "value"})
        assert path.exists()

    def test_valid_json(self, tmp_path):
        path = tmp_path / "test.json"
        data = {"equity": 51000.0, "trades": 201, "symbol": "BTC/USDC"}
        save_json(path, data)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "test.json"
        save_json(path, {"x": 1})
        assert path.exists()


class TestSaveTradesCsv:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "trades.csv"
        trades = [{"entry_ts": "2025-01-01", "pnl_abs": 100.0}]
        save_trades_csv(path, trades)
        assert path.exists()

    def test_has_header(self, tmp_path):
        path = tmp_path / "trades.csv"
        trades = [{"entry_ts": "2025-01-01", "pnl_abs": 100.0}]
        save_trades_csv(path, trades)
        with open(path) as f:
            header = f.readline()
        assert "entry_ts" in header
        assert "pnl_abs" in header

    def test_correct_row_count(self, tmp_path):
        path = tmp_path / "trades.csv"
        trades = [
            {"entry_ts": "2025-01-01", "pnl_abs": 100.0},
            {"entry_ts": "2025-01-02", "pnl_abs": -50.0},
        ]
        save_trades_csv(path, trades)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 trades
