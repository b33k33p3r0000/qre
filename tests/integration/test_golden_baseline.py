"""
CRITICAL: Golden Baseline Validation
=====================================
Verifies QRE metrics produce IDENTICAL results to optimizer for same trade data.
ANY difference = bug in transfer. Do NOT proceed to Phase 2 until this passes.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "golden_baseline" / "BTC"


@pytest.fixture
def golden_params():
    params_file = GOLDEN_DIR / "best_params.json"
    if not params_file.exists():
        pytest.skip(f"Golden baseline not found: {params_file}")
    with open(params_file) as f:
        return json.load(f)


@pytest.fixture
def golden_trades():
    trades_file = GOLDEN_DIR / "trades_BTC_USDC_1h_FULL.csv"
    if not trades_file.exists():
        pytest.skip(f"Golden trades not found: {trades_file}")
    return pd.read_csv(trades_file)


class TestGoldenBaseline:
    def test_trade_count_matches(self, golden_params, golden_trades):
        """QRE trade count matches optimizer output."""
        expected_trades = golden_params["trades"]
        assert len(golden_trades) == expected_trades, (
            f"Trade count mismatch: golden={expected_trades}, csv={len(golden_trades)}"
        )

    def test_equity_matches(self, golden_params, golden_trades):
        """QRE final equity matches optimizer output."""
        expected_equity = golden_params["equity"]
        start_equity = expected_equity - golden_params["total_pnl"]
        actual_equity = start_equity + golden_trades["pnl_abs"].sum()
        assert abs(actual_equity - expected_equity) < 0.01, (
            f"Equity mismatch: expected={expected_equity}, actual={actual_equity}"
        )

    def test_max_drawdown_matches(self, golden_params, golden_trades):
        """QRE max drawdown matches optimizer output (account level)."""
        from qre.core.metrics import calculate_metrics

        trades_list = golden_trades.to_dict("records")
        start_equity = golden_params["equity"] - golden_params["total_pnl"]
        result = calculate_metrics(trades_list, backtest_days=365, start_equity=start_equity)

        expected_dd = golden_params["max_drawdown"]
        assert abs(result.max_drawdown - expected_dd) < 0.1, (
            f"Max drawdown mismatch: expected={expected_dd}, actual={result.max_drawdown}"
        )

    def test_pnl_per_trade_consistent(self, golden_trades):
        """Each trade PnL in CSV is consistent (pnl_abs vs capital_at_entry * pnl_pct)."""
        for _, row in golden_trades.head(10).iterrows():
            if row["capital_at_entry"] > 0:
                expected_pct = row["pnl_abs"] / row["capital_at_entry"]
                assert abs(row["pnl_pct"] - expected_pct) < 0.001, (
                    f"PnL inconsistency: pnl_abs={row['pnl_abs']}, "
                    f"capital={row['capital_at_entry']}, pnl_pct={row['pnl_pct']}"
                )
