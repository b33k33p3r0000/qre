# tests/unit/test_report.py
"""Unit tests for QRE HTML report generator."""

import json
from pathlib import Path

import pytest

from qre.report import (
    generate_report, build_equity_curve, build_drawdown_curve,
    _compute_direction_stats, _render_long_short_metrics,
)


# --- Helpers ---

def _make_trade(pnl_abs=100.0, direction="long", entry_ts="2025-01-01T00:00:00",
                exit_ts="2025-01-02T00:00:00", pnl_pct=0.002, hold_bars=24,
                reason="signal"):
    return {
        "pnl_abs": pnl_abs, "direction": direction,
        "entry_ts": entry_ts, "exit_ts": exit_ts,
        "pnl_pct": pnl_pct, "hold_bars": hold_bars,
        "reason": reason,
    }


SAMPLE_PARAMS = {
    "symbol": "BTC/USDC", "equity": 51000, "start_equity": 50000,
    "sharpe": 2.5, "trades": 100, "max_drawdown": -3.0,
    "win_rate": 0.48, "total_pnl_pct": 2.0, "sortino": 1.5,
    "calmar": 2.0, "recovery_factor": 1.8, "profit_factor": 1.3,
    "trades_per_year": 200, "expectancy": 10.0,
    "profitable_months_ratio": 0.75,
}


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


class TestComputeDirectionStats:
    def test_splits_long_short(self):
        trades = [
            _make_trade(100, "long"), _make_trade(-50, "long"),
            _make_trade(200, "short"), _make_trade(-30, "short"),
        ]
        ds = _compute_direction_stats(trades)
        assert ds["long"]["count"] == 2
        assert ds["short"]["count"] == 2

    def test_pnl_calculated(self):
        trades = [_make_trade(100, "long"), _make_trade(-50, "short")]
        ds = _compute_direction_stats(trades)
        assert ds["long"]["pnl"] == 100.0
        assert ds["short"]["pnl"] == -50.0

    def test_win_rate(self):
        trades = [
            _make_trade(100, "long"), _make_trade(50, "long"), _make_trade(-20, "long"),
        ]
        ds = _compute_direction_stats(trades)
        assert abs(ds["long"]["win_rate"] - 66.7) < 0.1

    def test_empty_direction(self):
        trades = [_make_trade(100, "long")]
        ds = _compute_direction_stats(trades)
        assert ds["short"]["count"] == 0
        assert ds["short"]["pnl"] == 0.0

    def test_no_direction_field(self):
        trades = [{"pnl_abs": 100.0}]
        ds = _compute_direction_stats(trades)
        assert ds["long"]["count"] == 0
        assert ds["short"]["count"] == 0


class TestRenderLongShortMetrics:
    def test_contains_long_short_labels(self):
        trades = [_make_trade(100, "long"), _make_trade(-50, "short")]
        html = _render_long_short_metrics(trades)
        assert "LONG" in html
        assert "SHORT" in html

    def test_contains_pnl_values(self):
        trades = [_make_trade(100, "long"), _make_trade(-50, "short")]
        html = _render_long_short_metrics(trades)
        assert "$100.00" in html
        assert "-$50.00" in html


class TestGenerateReport:
    def test_returns_html_string(self):
        trades = [_make_trade(100), _make_trade(-50, "short")]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "<!DOCTYPE html>" in html
        assert "BTC/USDC" in html

    def test_contains_plotly_cdn(self):
        trades = [_make_trade(100)]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "plotly" in html.lower()

    def test_uses_start_equity_not_hardcoded(self):
        trades = [_make_trade(1000)]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "50000" in html

    def test_saves_to_file(self, tmp_path):
        trades = [_make_trade(100)]
        path = tmp_path / "report.html"
        html = generate_report(SAMPLE_PARAMS, trades)
        path.write_text(html)
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_split_results_shown_when_present(self):
        params = {**SAMPLE_PARAMS, "split_results": [
            {"split": 1, "test_equity": 50500, "test_trades": 20, "test_sharpe": 2.5},
            {"split": 2, "test_equity": 49800, "test_trades": 15, "test_sharpe": -0.5},
        ]}
        trades = [_make_trade(100)]
        html = generate_report(params, trades)
        assert "Split" in html or "split" in html

    def test_mc_section_shown_when_present(self):
        params = {**SAMPLE_PARAMS,
                  "mc_confidence": "HIGH", "mc_sharpe_mean": 1.3,
                  "mc_sharpe_ci_low": 1.1, "mc_sharpe_ci_high": 1.5}
        trades = [_make_trade(100)]
        html = generate_report(params, trades)
        assert "Monte Carlo" in html or "MC" in html

    def test_long_short_section_present(self):
        trades = [_make_trade(100, "long"), _make_trade(-50, "short")]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "Long / Short Breakdown" in html
        assert "LONG" in html
        assert "SHORT" in html

    def test_equity_chart_has_dates(self):
        trades = [
            _make_trade(100, "long", "2025-01-01T00:00:00", "2025-01-05T00:00:00"),
            _make_trade(-50, "short", "2025-01-06T00:00:00", "2025-01-10T00:00:00"),
        ]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "2025-01-05" in html
        assert "2025-01-10" in html
        # Equity chart xaxis should use 'Date', not 'Trade #'
        assert "equity-chart" in html

    def test_long_short_chart_present(self):
        trades = [_make_trade(100, "long"), _make_trade(-50, "short")]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "long-short-chart" in html
        assert "Long vs Short" in html

    def test_no_trades_still_works(self):
        html = generate_report(SAMPLE_PARAMS, [])
        assert "<!DOCTYPE html>" in html

    def test_trades_without_timestamps_fallback(self):
        trades = [{"pnl_abs": 100.0}, {"pnl_abs": -50.0}]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "<!DOCTYPE html>" in html


class TestCatastrophicStopKeyFix:
    def test_counts_catastrophic_stops_with_reason_key(self):
        """Bug: report used 'exit_reason' but backtest stores 'reason'."""
        trades = [
            _make_trade(100, reason="signal"),
            _make_trade(-500, reason="catastrophic_stop"),
            _make_trade(50, reason="signal"),
        ]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "1 / 3" in html


class TestColorConsistency:
    def test_long_short_chart_uses_green_not_yellow(self):
        """Long bars should use green (#c3e88d) not yellow (#ffc777)."""
        trades = [_make_trade(100, "long"), _make_trade(-50, "short")]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "rgba(255, 199, 119" not in html
        assert "rgba(195, 232, 141" in html


class TestCumulativePnlChart:
    def test_cumulative_pnl_chart_present(self):
        trades = [_make_trade(100), _make_trade(-50), _make_trade(200)]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "cumulative-pnl-chart" in html
        assert "Cumulative P&amp;L" in html or "Cumulative P&L" in html


class TestRollingMetrics:
    def test_rolling_metrics_present(self):
        trades = [_make_trade(100 if i % 3 else -50) for i in range(35)]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "rolling-metrics-chart" in html

    def test_rolling_metrics_skipped_when_few_trades(self):
        trades = [_make_trade(100) for _ in range(5)]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "rolling-metrics-chart" not in html


class TestStreakTimeline:
    def test_streak_timeline_present(self):
        trades = [_make_trade(100), _make_trade(-50), _make_trade(80), _make_trade(60)]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "streak-timeline-chart" in html


class TestHoldDurationChart:
    def test_hold_duration_chart_present(self):
        trades = [_make_trade(100, hold_bars=6), _make_trade(-50, hold_bars=24)]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "hold-duration-chart" in html
        assert "Hold Duration" in html


class TestPnlHeatmap:
    def test_pnl_heatmap_present(self):
        trades = [
            _make_trade(100, entry_ts="2025-01-06T09:00:00"),
            _make_trade(-50, entry_ts="2025-01-07T14:00:00"),
        ]
        html = generate_report(SAMPLE_PARAMS, trades)
        assert "pnl-heatmap-chart" in html
        assert "Heatmap" in html or "heatmap" in html


class TestBulletChart:
    def test_bullet_chart_replaces_table(self):
        params = {
            **SAMPLE_PARAMS,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70,
            "rsi_lookback": 3, "trend_tf": "4h", "trend_strict": 1,
        }
        trades = [_make_trade(100)]
        html = generate_report(params, trades)
        assert "param-bullet" in html
        assert "MACD fast" in html
        assert "RSI lookback" in html
        assert "Trend TF" in html


class TestTrendTfScale:
    def test_trend_tf_scale_in_strategy_flow(self):
        params = {
            **SAMPLE_PARAMS,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70,
            "rsi_lookback": 3, "trend_tf": "4h", "trend_strict": 1,
        }
        trades = [_make_trade(100)]
        html = generate_report(params, trades)
        assert "tf-scale" in html
        assert "tf-active" in html

    def test_lookback_shown_in_flow(self):
        params = {
            **SAMPLE_PARAMS,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70,
            "rsi_lookback": 5, "trend_tf": "8h", "trend_strict": 0,
        }
        trades = [_make_trade(100)]
        html = generate_report(params, trades)
        assert "lookback" in html.lower() or "Lookback" in html


class TestStrategyParamsV4:
    def test_v4_params_shown(self):
        """v4.0 params (rsi_lookback, trend_tf, trend_strict) should appear in report."""
        params = {
            **SAMPLE_PARAMS,
            "rsi_period": 14, "rsi_lookback": 3,
            "trend_tf": "4h", "trend_strict": 1,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "rsi_upper": 70, "rsi_lower": 30,
        }
        trades = [_make_trade(100)]
        html = generate_report(params, trades)
        assert "RSI lookback" in html or "rsi_lookback" in html
        assert "Trend TF" in html or "trend_tf" in html
        assert "Trend strict" in html or "trend_strict" in html
