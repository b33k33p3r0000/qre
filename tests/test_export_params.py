"""Tests for export_params script."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from export_params import find_runs, format_run_row  # noqa: E402


@pytest.fixture
def mock_results(tmp_path: Path) -> Path:
    """Create mock QRE results directory structure."""
    for run_name, symbol, equity, sharpe in [
        ("2026-02-17_12-30-29_v2-1-btc", "BTC", 279950.80, 8.88),
        ("2026-02-17_07-30-01_v1", "BTC", 284549.08, 1.03),
        ("2026-02-17_14-09-35_v2-2-sol", "SOL", 999584.58, 8.78),
    ]:
        run_dir = tmp_path / run_name / symbol
        run_dir.mkdir(parents=True)
        params = {
            "run_timestamp": run_name,
            "symbol": f"{symbol}/USDC",
            "equity": equity,
            "sharpe_time": sharpe,
            "trades": 182,
            "win_rate": 0.874,
            "mc_confidence": "HIGH",
            "macd_fast": 3,
            "macd_slow": 17,
        }
        (run_dir / "best_params.json").write_text(json.dumps(params))
    return tmp_path


class TestFindRuns:
    def test_finds_btc_runs(self, mock_results):
        runs = find_runs("BTC", mock_results)
        assert len(runs) == 2

    def test_finds_sol_runs(self, mock_results):
        runs = find_runs("SOL", mock_results)
        assert len(runs) == 1

    def test_no_runs_for_missing_symbol(self, mock_results):
        runs = find_runs("ETH", mock_results)
        assert len(runs) == 0

    def test_runs_sorted_newest_first(self, mock_results):
        runs = find_runs("BTC", mock_results)
        # v2-1 (12:30) should come after v1 (07:30) alphabetically reversed
        assert "v2-1" in runs[0]["timestamp"]

    def test_run_has_required_fields(self, mock_results):
        runs = find_runs("BTC", mock_results)
        run = runs[0]
        assert "path" in run
        assert "timestamp" in run
        assert "equity" in run
        assert "sharpe" in run
        assert "trades" in run
        assert "win_rate" in run
        assert "mc_confidence" in run


class TestFormatRunRow:
    def test_formats_correctly(self):
        run = {
            "timestamp": "2026-02-17_12-30-29_v2-1",
            "equity": 279950.80,
            "sharpe": 8.88,
            "trades": 182,
            "win_rate": 0.874,
            "mc_confidence": "HIGH",
        }
        row = format_run_row(1, run)
        assert "279,951" in row or "279951" in row
        assert "8.88" in row
        assert "182" in row
        assert "87.4%" in row
        assert "HIGH" in row
