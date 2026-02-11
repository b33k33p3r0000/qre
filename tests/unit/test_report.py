# tests/unit/test_report.py
"""Unit tests for QRE HTML report generator."""

import json
from pathlib import Path

import pytest

from qre.report import generate_report, build_equity_curve, build_drawdown_curve


class TestBuildEquityCurve:
    def test_starts_at_start_equity(self):
        trades = [{"pnl_abs": 100.0}, {"pnl_abs": -50.0}]
        curve = build_equity_curve(trades, start_equity=50000.0)
        assert curve[0] == 50000.0

    def test_length_is_trades_plus_one(self):
        trades = [{"pnl_abs": 100.0}, {"pnl_abs": -50.0}]
        curve = build_equity_curve(trades, start_equity=50000.0)
        assert len(curve) == 3

    def test_pnl_applied_correctly(self):
        trades = [{"pnl_abs": 100.0}, {"pnl_abs": -50.0}]
        curve = build_equity_curve(trades, start_equity=50000.0)
        assert curve[1] == 50100.0
        assert curve[2] == 50050.0


class TestBuildDrawdownCurve:
    def test_starts_at_zero(self):
        equity_curve = [50000.0, 51000.0, 50500.0]
        dd = build_drawdown_curve(equity_curve)
        assert dd[0] == 0.0

    def test_drawdown_is_negative(self):
        equity_curve = [50000.0, 51000.0, 50500.0]
        dd = build_drawdown_curve(equity_curve)
        assert dd[2] < 0  # 50500 < peak 51000

    def test_no_drawdown_on_new_high(self):
        equity_curve = [50000.0, 51000.0, 52000.0]
        dd = build_drawdown_curve(equity_curve)
        assert dd[1] == 0.0
        assert dd[2] == 0.0


class TestGenerateReport:
    def test_returns_html_string(self):
        params = {"symbol": "BTC/USDC", "equity": 51000, "start_equity": 50000,
                  "sharpe": 2.5, "trades": 100, "max_drawdown": -3.0,
                  "win_rate": 0.48, "total_pnl_pct": 2.0, "sortino": 1.5,
                  "calmar": 2.0, "recovery_factor": 1.8, "profit_factor": 1.3,
                  "trades_per_year": 200, "expectancy": 10.0,
                  "profitable_months_ratio": 0.75}
        trades = [{"pnl_abs": 100.0}, {"pnl_abs": -50.0}]
        html = generate_report(params, trades)
        assert "<!DOCTYPE html>" in html
        assert "BTC/USDC" in html

    def test_contains_plotly_cdn(self):
        params = {"symbol": "BTC/USDC", "equity": 51000, "start_equity": 50000,
                  "sharpe": 2.5, "trades": 100, "max_drawdown": -3.0,
                  "win_rate": 0.48, "total_pnl_pct": 2.0, "sortino": 1.5,
                  "calmar": 2.0, "recovery_factor": 1.8, "profit_factor": 1.3,
                  "trades_per_year": 200, "expectancy": 10.0,
                  "profitable_months_ratio": 0.75}
        trades = [{"pnl_abs": 100.0}]
        html = generate_report(params, trades)
        assert "plotly" in html.lower()

    def test_uses_start_equity_not_hardcoded(self):
        params = {"symbol": "BTC/USDC", "equity": 51000, "start_equity": 50000,
                  "sharpe": 2.5, "trades": 100, "max_drawdown": -3.0,
                  "win_rate": 0.48, "total_pnl_pct": 2.0, "sortino": 1.5,
                  "calmar": 2.0, "recovery_factor": 1.8, "profit_factor": 1.3,
                  "trades_per_year": 200, "expectancy": 10.0,
                  "profitable_months_ratio": 0.75}
        trades = [{"pnl_abs": 1000.0}]
        html = generate_report(params, trades)
        # Equity curve should start at 50000, not 10000 (old hardcoded value)
        assert "50000" in html

    def test_saves_to_file(self, tmp_path):
        params = {"symbol": "BTC/USDC", "equity": 51000, "start_equity": 50000,
                  "sharpe": 2.5, "trades": 100, "max_drawdown": -3.0,
                  "win_rate": 0.48, "total_pnl_pct": 2.0, "sortino": 1.5,
                  "calmar": 2.0, "recovery_factor": 1.8, "profit_factor": 1.3,
                  "trades_per_year": 200, "expectancy": 10.0,
                  "profitable_months_ratio": 0.75}
        trades = [{"pnl_abs": 100.0}]
        path = tmp_path / "report.html"
        html = generate_report(params, trades)
        path.write_text(html)
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_split_results_shown_when_present(self):
        params = {"symbol": "BTC/USDC", "equity": 51000, "start_equity": 50000,
                  "sharpe": 2.5, "trades": 100, "max_drawdown": -3.0,
                  "win_rate": 0.48, "total_pnl_pct": 2.0, "sortino": 1.5,
                  "calmar": 2.0, "recovery_factor": 1.8, "profit_factor": 1.3,
                  "trades_per_year": 200, "expectancy": 10.0,
                  "profitable_months_ratio": 0.75,
                  "split_results": [
                      {"split": 1, "test_equity": 50500, "test_trades": 20, "test_sharpe": 2.5},
                      {"split": 2, "test_equity": 49800, "test_trades": 15, "test_sharpe": -0.5},
                  ]}
        trades = [{"pnl_abs": 100.0}]
        html = generate_report(params, trades)
        assert "Split" in html or "split" in html

    def test_mc_section_shown_when_present(self):
        params = {"symbol": "BTC/USDC", "equity": 51000, "start_equity": 50000,
                  "sharpe": 2.5, "trades": 100, "max_drawdown": -3.0,
                  "win_rate": 0.48, "total_pnl_pct": 2.0, "sortino": 1.5,
                  "calmar": 2.0, "recovery_factor": 1.8, "profit_factor": 1.3,
                  "trades_per_year": 200, "expectancy": 10.0,
                  "profitable_months_ratio": 0.75,
                  "mc_confidence": "HIGH", "mc_sharpe_mean": 1.3,
                  "mc_sharpe_ci_low": 1.1, "mc_sharpe_ci_high": 1.5}
        trades = [{"pnl_abs": 100.0}]
        html = generate_report(params, trades)
        assert "Monte Carlo" in html or "MC" in html
