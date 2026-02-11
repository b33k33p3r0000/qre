# src/qre/report.py
"""
QRE HTML Report Generator
==========================
Self-contained HTML report with Plotly charts.

IMPORTANT: Uses start_equity from params (account level $50k),
NOT hardcoded $10k. This fixes the known drawdown bug.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger("qre.report")

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"


def build_equity_curve(trades: List[Dict], start_equity: float) -> List[float]:
    """Build equity curve from trades. Starts at start_equity."""
    curve = [start_equity]
    for t in trades:
        curve.append(curve[-1] + t.get("pnl_abs", 0))
    return curve


def build_drawdown_curve(equity_curve: List[float]) -> List[float]:
    """Build drawdown curve (percentage) from equity curve."""
    peak = equity_curve[0]
    dd = []
    for eq in equity_curve:
        peak = max(peak, eq)
        dd.append((eq - peak) / peak * 100 if peak > 0 else 0.0)
    return dd


def _render_split_results(params: Dict[str, Any]) -> str:
    """Render AWF split results table if present."""
    splits = params.get("split_results")
    if not splits:
        return ""

    rows = ""
    for s in splits:
        sharpe = s.get("test_sharpe", 0)
        css_class = "positive" if sharpe > 0 else "negative"
        rows += (
            f'<tr>'
            f'<td>Split {s.get("split", "?")}</td>'
            f'<td>${s.get("test_equity", 0):,.2f}</td>'
            f'<td>{s.get("test_trades", 0)}</td>'
            f'<td class="{css_class}">{sharpe:.4f}</td>'
            f'</tr>\n'
        )

    return f"""
    <h2>Walk-Forward Splits</h2>
    <div class="chart-container">
        <table class="params-table">
            <tr><th>Split</th><th>Test Equity</th><th>Test Trades</th><th>Test Sharpe</th></tr>
            {rows}
        </table>
    </div>
    """


def _render_mc_section(params: Dict[str, Any]) -> str:
    """Render Monte Carlo validation section if present."""
    mc_conf = params.get("mc_confidence")
    if not mc_conf:
        return ""

    return f"""
    <h2>Monte Carlo Validation</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">MC Confidence</div>
            <div class="metric-value {'positive' if mc_conf == 'HIGH' else 'negative'}">{mc_conf}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">MC Sharpe Mean</div>
            <div class="metric-value">{params.get('mc_sharpe_mean', 0):.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">MC Sharpe CI</div>
            <div class="metric-value">[{params.get('mc_sharpe_ci_low', 0):.2f}, {params.get('mc_sharpe_ci_high', 0):.2f}]</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">MC Max DD Mean</div>
            <div class="metric-value negative">{params.get('mc_max_dd_mean', 0):.2f}%</div>
        </div>
    </div>
    """


def _build_performance_data(trades: List[Dict]) -> Dict[str, Any]:
    """Extract data needed for performance charts from trades."""
    winners_pnl = [t.get("pnl_pct", 0) * 100 for t in trades if t.get("pnl_pct", 0) > 0]
    losers_pnl = [t.get("pnl_pct", 0) * 100 for t in trades if t.get("pnl_pct", 0) <= 0]

    # Monthly returns: group by YYYY-MM
    monthly: Dict[str, float] = {}
    for t in trades:
        exit_ts = t.get("exit_ts", "")
        if exit_ts:
            month_key = exit_ts[:7]  # "2025-03"
            monthly[month_key] = monthly.get(month_key, 0) + t.get("pnl_abs", 0)
    month_labels = sorted(monthly.keys())
    month_values = [round(monthly[m], 2) for m in month_labels]

    # Trade duration vs P&L scatter
    win_hold = [t.get("hold_bars", 0) for t in trades if t.get("pnl_pct", 0) > 0]
    win_pnl = [t.get("pnl_pct", 0) * 100 for t in trades if t.get("pnl_pct", 0) > 0]
    lose_hold = [t.get("hold_bars", 0) for t in trades if t.get("pnl_pct", 0) <= 0]
    lose_pnl = [t.get("pnl_pct", 0) * 100 for t in trades if t.get("pnl_pct", 0) <= 0]

    return {
        "winners_pnl": winners_pnl,
        "losers_pnl": losers_pnl,
        "month_labels": month_labels,
        "month_values": month_values,
        "win_hold": win_hold,
        "win_pnl": win_pnl,
        "lose_hold": lose_hold,
        "lose_pnl": lose_pnl,
    }


def _render_performance_charts(trades: List[Dict]) -> tuple[str, str]:
    """Render performance analysis section (HTML + JS)."""
    if not trades:
        return "", ""

    perf = _build_performance_data(trades)

    html = """
    <h2>Performance Analysis</h2>
    <div class="chart-container">
        <div id="pnl-dist-chart"></div>
    </div>
    <div class="chart-container">
        <div id="monthly-returns-chart"></div>
    </div>
    <div class="chart-container">
        <div id="duration-pnl-chart"></div>
    </div>
    """

    month_colors = [
        "'#50fa7b'" if v > 0 else "'#ff5555'" for v in perf["month_values"]
    ]

    js = f"""
        // P&L Distribution
        Plotly.newPlot('pnl-dist-chart', [{{
            x: {json.dumps(perf['winners_pnl'])},
            type: 'histogram',
            name: 'Winners',
            marker: {{ color: 'rgba(80, 250, 123, 0.7)' }},
            xbins: {{ size: 1 }}
        }}, {{
            x: {json.dumps(perf['losers_pnl'])},
            type: 'histogram',
            name: 'Losers',
            marker: {{ color: 'rgba(255, 85, 85, 0.7)' }},
            xbins: {{ size: 1 }}
        }}], {{
            paper_bgcolor: '#44475a',
            plot_bgcolor: '#44475a',
            font: {{ color: '#f8f8f2', size: 10 }},
            margin: {{ t: 30, b: 40, l: 50, r: 20 }},
            title: {{ text: 'P&L Distribution', font: {{ size: 12, color: '#6272a4' }} }},
            xaxis: {{ gridcolor: '#282a36', title: 'P&L (%)' }},
            yaxis: {{ gridcolor: '#282a36', title: 'Count' }},
            barmode: 'overlay',
            legend: {{ font: {{ size: 10 }} }}
        }});

        // Monthly Returns
        Plotly.newPlot('monthly-returns-chart', [{{
            x: {json.dumps(perf['month_labels'])},
            y: {json.dumps(perf['month_values'])},
            type: 'bar',
            marker: {{ color: [{', '.join(month_colors)}] }},
            text: {json.dumps([f'${v:,.0f}' for v in perf['month_values']])},
            textposition: 'outside',
            textfont: {{ size: 9, color: '#f8f8f2' }}
        }}], {{
            paper_bgcolor: '#44475a',
            plot_bgcolor: '#44475a',
            font: {{ color: '#f8f8f2', size: 10 }},
            margin: {{ t: 30, b: 60, l: 60, r: 20 }},
            title: {{ text: 'Monthly Returns', font: {{ size: 12, color: '#6272a4' }} }},
            xaxis: {{ gridcolor: '#282a36', title: 'Month' }},
            yaxis: {{ gridcolor: '#282a36', title: 'P&L ($)' }},
            showlegend: false
        }});

        // Trade Duration vs P&L
        Plotly.newPlot('duration-pnl-chart', [{{
            x: {json.dumps(perf['win_hold'])},
            y: {json.dumps(perf['win_pnl'])},
            type: 'scatter',
            mode: 'markers',
            name: 'Winners',
            marker: {{ color: '#50fa7b', size: 6, opacity: 0.7 }}
        }}, {{
            x: {json.dumps(perf['lose_hold'])},
            y: {json.dumps(perf['lose_pnl'])},
            type: 'scatter',
            mode: 'markers',
            name: 'Losers',
            marker: {{ color: '#ff5555', size: 6, opacity: 0.7 }}
        }}], {{
            paper_bgcolor: '#44475a',
            plot_bgcolor: '#44475a',
            font: {{ color: '#f8f8f2', size: 10 }},
            margin: {{ t: 30, b: 40, l: 50, r: 20 }},
            title: {{ text: 'Trade Duration vs P&L', font: {{ size: 12, color: '#6272a4' }} }},
            xaxis: {{ gridcolor: '#282a36', title: 'Hold Duration (bars)' }},
            yaxis: {{ gridcolor: '#282a36', title: 'P&L (%)' }},
            legend: {{ font: {{ size: 10 }} }}
        }});
    """

    return html, js


def _render_strategy_flow(params: Dict[str, Any]) -> str:
    """Render MACD+RSI strategy flow with actual parameter values."""
    macd_fast = params.get("macd_fast", "?")
    macd_slow = params.get("macd_slow", "?")
    macd_signal = params.get("macd_signal", "?")
    macd_mode = params.get("macd_mode", "?")
    rsi_mode = params.get("rsi_mode", "?")
    rsi_upper = params.get("rsi_upper", "?")
    rsi_lower = params.get("rsi_lower", "?")
    rsi_momentum = params.get("rsi_momentum_level", "?")
    kB = params.get("kB", "?")
    dB = params.get("dB", "?")
    k_sell = params.get("k_sell", "?")
    min_hold = params.get("min_hold", "?")
    p_buy = params.get("p_buy", 0)
    n_votes = params.get("n_votes", 6)

    # Required votes calculation
    import math
    required_votes = math.ceil(p_buy * n_votes) if isinstance(p_buy, (int, float)) else "?"

    # RSI mode description
    rsi_desc = {
        "extreme": f"RSI &lt; {rsi_lower} (oversold zone)",
        "trend_filter": f"RSI &gt; {rsi_momentum} (momentum filter)",
        "momentum": f"RSI &gt; {rsi_momentum} (momentum)",
    }.get(rsi_mode, str(rsi_mode))

    # MACD mode description
    macd_desc = {
        "rising": "MACD histogram rising (h[i] &gt; h[i-1])",
        "any": "MACD histogram positive (h &gt; 0)",
        "crossover": "MACD line crosses above signal",
    }.get(macd_mode, str(macd_mode))

    # Threshold bars per TF
    tfs = ["2h", "4h", "6h", "8h", "12h", "24h"]
    tf_labels = ["2h", "4h", "6h", "8h", "12h", "1d"]
    threshold_rows = ""
    for tf, label in zip(tfs, tf_labels):
        low = params.get(f"low_{tf}", 0)
        high = params.get(f"high_{tf}", 1)
        low_pct = low * 100
        high_pct = high * 100
        gap_pct = (high - low) * 100
        threshold_rows += f"""
            <div class="flow-threshold-row">
                <span class="flow-tf-label">{label}</span>
                <span class="flow-val flow-val-low">{low:.3f}</span>
                <div class="flow-bar-track">
                    <div class="flow-bar-fill" style="left:{low_pct:.1f}%;width:{gap_pct:.1f}%"></div>
                </div>
                <span class="flow-val flow-val-high">{high:.3f}</span>
            </div>"""

    # RSI gate rows (only for TFs that have gates)
    gate_tfs = [("6h", "rsi_gate_6h"), ("8h", "rsi_gate_8h"),
                ("12h", "rsi_gate_12h"), ("1d", "rsi_gate_24h")]
    gate_rows = ""
    for label, key in gate_tfs:
        val = params.get(key)
        if val is not None:
            gate_pct = val
            gate_rows += f"""
            <div class="flow-threshold-row">
                <span class="flow-tf-label">{label}</span>
                <span class="flow-val flow-val-low">{val:.1f}</span>
                <div class="flow-bar-track">
                    <div class="flow-bar-gate" style="left:{gate_pct:.1f}%;width:2px"></div>
                </div>
                <span class="flow-val flow-val-high">100</span>
            </div>"""

    return f"""
    <h2>Strategy Flow</h2>

    <div class="flow-phase">
        <div class="flow-phase-header">
            <span class="flow-phase-num">1</span>
            <span class="flow-phase-title">DATA INPUT</span>
        </div>
        <div class="flow-phase-body">
            <div class="flow-io">
                <div class="flow-io-label">Input</div>
                <div>OHLCV 1h bars ({params.get('symbol', '?')}) via Binance API + Parquet cache</div>
            </div>
            <div class="flow-io">
                <div class="flow-io-label">Process</div>
                <div>Resample 1h &rarr; 2h, 4h, 6h, 8h, 12h, 1d ({n_votes} timeframes)</div>
            </div>
            <div class="flow-io">
                <div class="flow-io-label">Output</div>
                <div>Multi-TF OHLCV dict &middot; RSI(21) &middot; StochRSI(14, K={kB}, D={dB}) &middot; MACD({macd_fast},{macd_slow},{macd_signal})</div>
            </div>
        </div>
    </div>

    <div class="flow-arrow">&darr;</div>

    <div class="flow-phase">
        <div class="flow-phase-header">
            <span class="flow-phase-num">2</span>
            <span class="flow-phase-title">BUY SIGNAL</span>
            <span class="flow-phase-meta">3 conditions &times; {n_votes} TFs</span>
        </div>
        <div class="flow-phase-body">
            <div class="flow-condition">
                <div class="flow-cond-label">MACD ({macd_mode})</div>
                <div class="flow-cond-desc">{macd_desc}</div>
            </div>
            <div class="flow-condition">
                <div class="flow-cond-label">RSI ({rsi_mode})</div>
                <div class="flow-cond-desc">{rsi_desc}</div>
            </div>
            <div class="flow-condition">
                <div class="flow-cond-label">StochRSI Crossover</div>
                <div class="flow-cond-desc">K({kB}) crosses above D({dB}) within threshold range</div>
            </div>
            <div class="flow-io" style="margin-top:10px">
                <div class="flow-io-label">Voting</div>
                <div>p_buy = {p_buy:.4f} &rarr; required votes = {required_votes}/{n_votes} TFs must agree</div>
            </div>
        </div>
    </div>

    <div class="flow-arrow">&darr;</div>

    <div class="flow-phase">
        <div class="flow-phase-header">
            <span class="flow-phase-num">3</span>
            <span class="flow-phase-title">THRESHOLDS</span>
            <span class="flow-phase-meta">12 params</span>
        </div>
        <div class="flow-phase-body">
            <div class="flow-desc">LOW = oversold (buy) &middot; HIGH = overbought (sell)</div>
            <div class="flow-thresholds">{threshold_rows}
            </div>
            <div class="flow-desc" style="margin-top:12px">RSI Gates (min RSI for buy signal)</div>
            <div class="flow-thresholds">{gate_rows}
            </div>
        </div>
    </div>

    <div class="flow-arrow">&darr;</div>

    <div class="flow-phase">
        <div class="flow-phase-header">
            <span class="flow-phase-num">4</span>
            <span class="flow-phase-title">EXECUTION</span>
        </div>
        <div class="flow-phase-body">
            <div class="flow-condition">
                <div class="flow-cond-label">Sell Signal</div>
                <div class="flow-cond-desc">StochRSI K({k_sell}) crosses below D &middot; min hold = {min_hold} bars</div>
            </div>
            <div class="flow-condition">
                <div class="flow-cond-label">Position Sizing</div>
                <div class="flow-cond-desc">25% of equity per trade</div>
            </div>
            <div class="flow-condition">
                <div class="flow-cond-label">Catastrophic Stop</div>
                <div class="flow-cond-desc">-15% emergency exit</div>
            </div>
            <div class="flow-io" style="margin-top:10px">
                <div class="flow-io-label">Output</div>
                <div>Trade list (entry/exit timestamps, P&amp;L, hold duration)</div>
            </div>
        </div>
    </div>
    """


def _render_strategy_params(params: Dict[str, Any]) -> str:
    """Render strategy parameters table."""
    strategy_keys = [
        ("kB", "K smoothing"), ("dB", "D smoothing"), ("k_sell", "Sell K"),
        ("min_hold", "Min hold (bars)"), ("p_buy", "Buy probability"),
        ("macd_fast", "MACD fast"), ("macd_slow", "MACD slow"), ("macd_signal", "MACD signal"),
        ("macd_mode", "MACD mode"), ("rsi_mode", "RSI mode"),
        ("rsi_upper", "RSI upper"), ("rsi_lower", "RSI lower"),
    ]

    rows = ""
    for key, label in strategy_keys:
        if key in params:
            val = params[key]
            if isinstance(val, float):
                val = f"{val:.4f}"
            rows += f'<tr><td>{label}</td><td>{val}</td></tr>\n'

    return f"""
    <h2>Strategy Parameters</h2>
    <div class="chart-container">
        <table class="params-table">
            <tr><th>Parameter</th><th>Value</th></tr>
            {rows}
        </table>
    </div>
    """


def _render_optuna_history(optuna_history: List[Dict]) -> tuple[str, str]:
    """Render Optuna optimization history chart."""
    if not optuna_history:
        return "", ""

    numbers = [t["number"] for t in optuna_history]
    values = [t["value"] for t in optuna_history]

    # Running best
    best_so_far = []
    current_best = 0
    for v in values:
        current_best = max(current_best, v)
        best_so_far.append(current_best)

    # Stats
    non_zero = [v for v in values if v > 0]
    zero_count = len(values) - len(non_zero)
    zero_pct = zero_count / len(values) * 100 if values else 0

    html = f"""
    <h2>Optimization History</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Total Trials</div>
            <div class="metric-value">{len(values)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Viable Trials</div>
            <div class="metric-value {'positive' if len(non_zero) > 10 else 'negative'}">{len(non_zero)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Zero Trials</div>
            <div class="metric-value {'negative' if zero_pct > 80 else ''}">{zero_pct:.0f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Best Trial</div>
            <div class="metric-value positive">#{numbers[values.index(max(values))] if values else '?'}</div>
        </div>
    </div>
    <div class="chart-container">
        <div id="optuna-history-chart"></div>
    </div>
    """

    js = f"""
        // Optuna History
        Plotly.newPlot('optuna-history-chart', [{{
            x: {json.dumps(numbers)},
            y: {json.dumps(values)},
            type: 'scatter',
            mode: 'markers',
            name: 'Trial Value',
            marker: {{
                color: {json.dumps(values)},
                colorscale: [[0, '#ff5555'], [0.01, '#6272a4'], [0.5, '#8be9fd'], [1, '#50fa7b']],
                size: 4,
                opacity: 0.6
            }}
        }}, {{
            x: {json.dumps(numbers)},
            y: {json.dumps(best_so_far)},
            type: 'scatter',
            mode: 'lines',
            name: 'Best So Far',
            line: {{ color: '#50fa7b', width: 2 }}
        }}], {{
            paper_bgcolor: '#44475a',
            plot_bgcolor: '#44475a',
            font: {{ color: '#f8f8f2', size: 10 }},
            margin: {{ t: 30, b: 40, l: 60, r: 20 }},
            title: {{ text: 'Optimization Progress', font: {{ size: 12, color: '#6272a4' }} }},
            xaxis: {{ gridcolor: '#282a36', title: 'Trial #' }},
            yaxis: {{ gridcolor: '#282a36', title: 'Objective Value (Equity)' }},
            legend: {{ font: {{ size: 10 }} }}
        }});
    """

    return html, js


def generate_report(params: Dict[str, Any], trades: List[Dict],
                    optuna_history: List[Dict] | None = None) -> str:
    """
    Generate self-contained HTML report.

    CRITICAL: Uses params["start_equity"] for equity curve, NOT hardcoded value.
    This fixes the drawdown bug from the old optimizer dashboard.
    """
    symbol = params.get("symbol", "?")
    start_equity = params.get("start_equity", 50000.0)

    equity_curve = build_equity_curve(trades, start_equity)
    drawdown_curve = build_drawdown_curve(equity_curve)

    split_html = _render_split_results(params)
    mc_html = _render_mc_section(params)
    flow_html = _render_strategy_flow(params)
    strategy_html = _render_strategy_params(params)
    perf_html, perf_js = _render_performance_charts(trades)
    optuna_html, optuna_js = _render_optuna_history(optuna_history or [])

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QRE Report - {symbol}</title>
    <script src="{PLOTLY_CDN}"></script>
    <style>
        :root {{
            --bg-primary: #282a36;
            --bg-secondary: #44475a;
            --text-primary: #f8f8f2;
            --text-secondary: #6272a4;
            --accent-green: #50fa7b;
            --accent-red: #ff5555;
            --accent-purple: #bd93f9;
            --accent-cyan: #8be9fd;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            padding: 20px;
            font-size: 12px;
        }}
        h1 {{
            color: var(--accent-purple);
            margin-bottom: 20px;
            font-size: 24px;
        }}
        h2 {{
            color: var(--text-secondary);
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin: 20px 0 10px 0;
            border-bottom: 1px solid var(--bg-secondary);
            padding-bottom: 5px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: var(--bg-secondary);
            border-radius: 6px;
            padding: 12px;
            text-align: center;
        }}
        .metric-label {{
            font-size: 10px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 18px;
            font-weight: bold;
            margin-top: 4px;
        }}
        .positive {{ color: var(--accent-green); }}
        .negative {{ color: var(--accent-red); }}
        .chart-container {{
            background: var(--bg-secondary);
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        .params-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }}
        .params-table th, .params-table td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid var(--bg-primary);
        }}
        .params-table th {{
            color: var(--text-secondary);
            font-weight: normal;
            text-transform: uppercase;
        }}
        footer {{
            margin-top: 30px;
            text-align: center;
            color: var(--text-secondary);
            font-size: 10px;
        }}
        .flow-phase {{
            background: var(--bg-secondary);
            border-radius: 6px;
            margin-bottom: 4px;
            overflow: hidden;
        }}
        .flow-phase-header {{
            display: flex;
            align-items: center;
            padding: 10px 15px;
            border-bottom: 1px solid var(--bg-primary);
        }}
        .flow-phase-num {{
            background: var(--accent-purple);
            color: var(--bg-primary);
            width: 22px; height: 22px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; font-size: 11px;
            margin-right: 10px; flex-shrink: 0;
        }}
        .flow-phase-title {{
            font-size: 12px;
            font-weight: bold;
            letter-spacing: 1px;
            text-transform: uppercase;
        }}
        .flow-phase-meta {{
            margin-left: auto;
            color: var(--text-secondary);
            font-size: 10px;
        }}
        .flow-phase-body {{
            padding: 12px 15px;
        }}
        .flow-arrow {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 16px;
            margin: 2px 0;
        }}
        .flow-io {{
            display: flex;
            gap: 10px;
            margin-bottom: 6px;
            font-size: 11px;
        }}
        .flow-io-label {{
            color: var(--accent-cyan);
            font-size: 10px;
            text-transform: uppercase;
            min-width: 55px;
            flex-shrink: 0;
        }}
        .flow-condition {{
            margin-bottom: 8px;
        }}
        .flow-cond-label {{
            color: var(--accent-purple);
            font-size: 11px;
            font-weight: bold;
            margin-bottom: 2px;
        }}
        .flow-cond-desc {{
            color: var(--text-primary);
            font-size: 11px;
            opacity: 0.85;
        }}
        .flow-desc {{
            color: var(--text-secondary);
            font-size: 10px;
            font-style: italic;
            margin-bottom: 8px;
        }}
        .flow-thresholds {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .flow-threshold-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
        }}
        .flow-tf-label {{
            width: 24px;
            color: var(--text-secondary);
            text-align: right;
            flex-shrink: 0;
        }}
        .flow-val {{
            width: 42px;
            font-size: 10px;
            flex-shrink: 0;
        }}
        .flow-val-low {{ color: var(--accent-green); text-align: right; }}
        .flow-val-high {{ color: var(--accent-red); text-align: left; }}
        .flow-bar-track {{
            flex: 1;
            height: 8px;
            background: var(--bg-primary);
            border-radius: 4px;
            position: relative;
            min-width: 80px;
        }}
        .flow-bar-fill {{
            position: absolute;
            top: 0; height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, var(--accent-green), var(--accent-red));
            opacity: 0.6;
        }}
        .flow-bar-gate {{
            position: absolute;
            top: -2px; height: 12px;
            background: var(--accent-cyan);
            border-radius: 1px;
        }}
    </style>
</head>
<body>
    <h1>QRE Report: {symbol}</h1>
    <p style="color: var(--text-secondary); margin-bottom: 20px;">
        Generated: {now} | Start equity: ${start_equity:,.0f}
    </p>

    <h2>Key Metrics</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Final Equity</div>
            <div class="metric-value {'positive' if params.get('equity', 0) > start_equity else 'negative'}">
                ${params.get('equity', 0):,.2f}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total P&L</div>
            <div class="metric-value {'positive' if params.get('total_pnl_pct', 0) > 0 else 'negative'}">
                {params.get('total_pnl_pct', 0):+.1f}%
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Trades</div>
            <div class="metric-value">{params.get('trades', 0)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Trades/Year</div>
            <div class="metric-value">{params.get('trades_per_year', 0):.1f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value {'positive' if params.get('win_rate', 0) > 0.5 else 'negative'}">
                {params.get('win_rate', 0) * 100:.1f}%
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe</div>
            <div class="metric-value {'positive' if params.get('sharpe', 0) > 1 else 'negative'}">
                {params.get('sharpe', 0):.2f}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sortino</div>
            <div class="metric-value">{params.get('sortino', 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Calmar</div>
            <div class="metric-value">{params.get('calmar', 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">{params.get('max_drawdown', 0):.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Recovery Factor</div>
            <div class="metric-value">{params.get('recovery_factor', 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value">{params.get('profit_factor', 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Expectancy</div>
            <div class="metric-value">${params.get('expectancy', 0):.2f}</div>
        </div>
    </div>

    <h2>Equity Curve</h2>
    <div class="chart-container">
        <div id="equity-chart"></div>
    </div>

    <h2>Drawdown</h2>
    <div class="chart-container">
        <div id="drawdown-chart"></div>
    </div>

    {split_html}
    {mc_html}
    {optuna_html}
    {flow_html}
    {strategy_html}
    {perf_html}

    <footer>
        QRE v0.3.0 | MACD+RSI | Anchored Walk-Forward
    </footer>

    <script>
        Plotly.newPlot('equity-chart', [{{
            y: {json.dumps(equity_curve)},
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#8be9fd', width: 2 }},
            fill: 'tozeroy',
            fillcolor: 'rgba(139, 233, 253, 0.1)'
        }}], {{
            paper_bgcolor: '#44475a',
            plot_bgcolor: '#44475a',
            font: {{ color: '#f8f8f2', size: 10 }},
            margin: {{ t: 20, b: 40, l: 60, r: 20 }},
            xaxis: {{ gridcolor: '#282a36', title: 'Trade #' }},
            yaxis: {{ gridcolor: '#282a36', title: 'Equity ($)' }}
        }});

        Plotly.newPlot('drawdown-chart', [{{
            y: {json.dumps(drawdown_curve)},
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#ff5555', width: 2 }},
            fill: 'tozeroy',
            fillcolor: 'rgba(255, 85, 85, 0.2)'
        }}], {{
            paper_bgcolor: '#44475a',
            plot_bgcolor: '#44475a',
            font: {{ color: '#f8f8f2', size: 10 }},
            margin: {{ t: 20, b: 40, l: 60, r: 20 }},
            xaxis: {{ gridcolor: '#282a36', title: 'Trade #' }},
            yaxis: {{ gridcolor: '#282a36', title: 'Drawdown (%)' }}
        }});
        {perf_js}
        {optuna_js}
    </script>
</body>
</html>"""

    return html


def save_report(path, params: Dict[str, Any], trades: List[Dict],
                optuna_history: List[Dict] | None = None) -> None:
    """Generate and save HTML report to file."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = generate_report(params, trades, optuna_history=optuna_history)
    path.write_text(html, encoding="utf-8")
    logger.info(f"Generated HTML report: {path}")
