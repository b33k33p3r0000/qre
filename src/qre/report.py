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


def generate_report(params: Dict[str, Any], trades: List[Dict]) -> str:
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
    strategy_html = _render_strategy_params(params)

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
    {strategy_html}

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
    </script>
</body>
</html>"""

    return html


def save_report(path, params: Dict[str, Any], trades: List[Dict]) -> None:
    """Generate and save HTML report to file."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = generate_report(params, trades)
    path.write_text(html, encoding="utf-8")
    logger.info(f"Generated HTML report: {path}")
