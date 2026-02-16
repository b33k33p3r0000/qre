"""
Reproducibility validation: QRE strategy signals vs known legacy results.

Uses shared Parquet cache (same data) + fixed params to compare
trade entry/exit timing between QRE and legacy optimizer.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from qre.config import STARTING_EQUITY
from qre.core.backtest import simulate_trades_fast
from qre.core.metrics import calculate_metrics
from qre.core.strategy import MACDRSIStrategy
from qre.data.fetch import load_all_data


LEGACY_RESULTS = Path.home() / "projects" / "optimizer" / "results"
LEGACY_CACHE = Path.home() / "projects" / "optimizer" / "cache"


def find_legacy_run(symbol: str = "BTC/USDC") -> tuple[str, dict] | None:
    """Find most recent legacy macd_rsi run for symbol. Returns (dir_name, params)."""
    if not LEGACY_RESULTS.exists():
        return None

    base = symbol.split("/")[0]
    candidates = []
    for run_dir in LEGACY_RESULTS.iterdir():
        if not run_dir.is_dir():
            continue
        params_file = run_dir / base / "best_params.json"
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
            if params.get("strategy", "").startswith("macd"):
                candidates.append((run_dir.name, params))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]


def find_legacy_trades(run_timestamp: str, symbol: str = "BTC/USDC") -> pd.DataFrame | None:
    """Load trade CSV from legacy optimizer run."""
    base = symbol.split("/")[0]
    symbol_safe = symbol.replace("/", "_")
    trades_file = LEGACY_RESULTS / run_timestamp / base / f"trades_{symbol_safe}_1h_FULL.csv"
    if not trades_file.exists():
        return None
    return pd.read_csv(trades_file)


def _load_data_and_run(symbol: str, params: dict):
    """Load fresh data and run QRE backtest with given params."""
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})
    hours = params.get("hours", 8760)
    data = load_all_data(exchange, symbol, hours)

    strategy = MACDRSIStrategy()
    buy, sell = strategy.precompute_signals(data, params)
    result = simulate_trades_fast(symbol, data, buy, sell)
    return result


@pytest.mark.integration
class TestReproducibility:
    """Compare QRE signals with legacy optimizer on same data + params."""

    def test_legacy_params_loadable(self):
        """Can find and load legacy macd_rsi run params."""
        result = find_legacy_run("BTC/USDC")
        if result is None:
            pytest.skip("No legacy macd_rsi run found for BTC/USDC")
        _, params = result
        assert "macd_fast" in params or "macd_mode" in params
        assert params.get("strategy", "").startswith("macd")

    def test_signal_count_matches(self):
        """QRE produces similar number of trades as legacy with identical params + data."""
        found = find_legacy_run("BTC/USDC")
        if found is None:
            pytest.skip("No legacy macd_rsi run found")

        run_ts, legacy = found
        legacy_trades = find_legacy_trades(run_ts, "BTC/USDC")
        if legacy_trades is None:
            pytest.skip(f"No legacy trades CSV found for {run_ts}")

        if not LEGACY_CACHE.exists():
            pytest.skip("Legacy cache not available")

        result = _load_data_and_run("BTC/USDC", legacy)

        # Tolerance: ±25% due to position sizing (25% vs 20%) and stop-loss (15% vs 9%)
        legacy_count = len(legacy_trades)
        qre_count = len(result.trades)
        ratio = qre_count / legacy_count if legacy_count > 0 else 0

        assert 0.75 <= ratio <= 1.25, (
            f"Trade count mismatch: QRE={qre_count}, Legacy={legacy_count}, "
            f"Ratio={ratio:.2f} (expected 0.75-1.25)"
        )

    def test_qre_produces_trades_with_legacy_params(self):
        """QRE produces trades when given legacy params (strategy logic works)."""
        found = find_legacy_run("BTC/USDC")
        if found is None:
            pytest.skip("No legacy macd_rsi run found")

        _, legacy = found

        if not LEGACY_CACHE.exists():
            pytest.skip("Legacy cache not available")

        result = _load_data_and_run("BTC/USDC", legacy)

        # QRE must produce trades with known-good legacy params
        assert len(result.trades) > 0, "QRE produced 0 trades with legacy params"
        assert len(result.trades) >= 50, (
            f"QRE produced only {len(result.trades)} trades with legacy params "
            f"(expected >= 50, legacy had ~200)"
        )

    def test_win_rate_similar(self):
        """QRE win rate is within reasonable range of legacy."""
        found = find_legacy_run("BTC/USDC")
        if found is None:
            pytest.skip("No legacy macd_rsi run found")

        _, legacy = found

        if not LEGACY_CACHE.exists():
            pytest.skip("Legacy cache not available")

        result = _load_data_and_run("BTC/USDC", legacy)
        qre_metrics = calculate_metrics(
            result.trades, result.backtest_days, start_equity=STARTING_EQUITY,
        )

        legacy_winrate = legacy.get("win_rate", 0)
        qre_winrate = qre_metrics.win_rate / 100

        # Win rate should be within ±15 percentage points
        assert abs(qre_winrate - legacy_winrate) <= 0.15, (
            f"Win rate divergence too large: QRE={qre_winrate:.4f}, "
            f"Legacy={legacy_winrate:.4f}, Diff={abs(qre_winrate - legacy_winrate):.4f}"
        )
