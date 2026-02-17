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

    mc_source = params.get("mc_source", "full_data")
    mc_splits = params.get("mc_splits_evaluated", "?")
    if mc_source == "oos_per_split":
        mc_title = f"OOS Monte Carlo ({mc_splits} splits)"
    else:
        mc_title = "Monte Carlo Validation (full data)"

    conf_class = "positive" if mc_conf == "HIGH" else "negative"

    return f"""
    <h2>{mc_title}</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">MC Confidence</div>
            <div class="metric-value {conf_class}">{mc_conf}</div>
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


def _compute_direction_stats(trades: List[Dict]) -> Dict[str, Any]:
    """Compute per-direction (long/short) statistics from trades."""
    longs = [t for t in trades if t.get("direction") == "long"]
    shorts = [t for t in trades if t.get("direction") == "short"]

    def _stats(subset: List[Dict]) -> Dict[str, Any]:
        if not subset:
            return {"count": 0, "pnl": 0.0, "win_rate": 0.0,
                    "avg_win": 0.0, "avg_loss": 0.0, "winners": 0, "losers": 0}
        winners = [t for t in subset if t.get("pnl_abs", 0) > 0]
        losers = [t for t in subset if t.get("pnl_abs", 0) <= 0]
        total_pnl = sum(t.get("pnl_abs", 0) for t in subset)
        avg_w = sum(t.get("pnl_abs", 0) for t in winners) / len(winners) if winners else 0.0
        avg_l = sum(t.get("pnl_abs", 0) for t in losers) / len(losers) if losers else 0.0
        return {
            "count": len(subset),
            "pnl": total_pnl,
            "win_rate": len(winners) / len(subset) * 100,
            "avg_win": avg_w,
            "avg_loss": avg_l,
            "winners": len(winners),
            "losers": len(losers),
        }

    return {"long": _stats(longs), "short": _stats(shorts)}


def _fmt_usd(value: float) -> str:
    """Format dollar value with sign before $: -$50.00 instead of $-50.00."""
    if value < 0:
        return f"-${abs(value):,.2f}"
    return f"${value:,.2f}"


def _render_long_short_metrics(trades: List[Dict]) -> str:
    """Render long/short breakdown section."""
    ds = _compute_direction_stats(trades)
    lo, sh = ds["long"], ds["short"]
    total = len(trades) or 1

    def _card(label: str, s: Dict[str, Any], css: str) -> str:
        return f"""
        <div class="ls-card {css}">
            <div class="ls-header">{label}</div>
            <div class="ls-count">{s['count']} <span class="ls-pct">({s['count']/total*100:.0f}%)</span></div>
            <div class="ls-row">
                <span class="ls-label">P&amp;L</span>
                <span class="ls-val {'positive' if s['pnl'] >= 0 else 'negative'}">{_fmt_usd(s['pnl'])}</span>
            </div>
            <div class="ls-row">
                <span class="ls-label">Win Rate</span>
                <span class="ls-val">{s['win_rate']:.1f}%</span>
            </div>
            <div class="ls-row">
                <span class="ls-label">Winners / Losers</span>
                <span class="ls-val">{s['winners']} / {s['losers']}</span>
            </div>
            <div class="ls-row">
                <span class="ls-label">Avg Win</span>
                <span class="ls-val positive">{_fmt_usd(s['avg_win'])}</span>
            </div>
            <div class="ls-row">
                <span class="ls-label">Avg Loss</span>
                <span class="ls-val negative">{_fmt_usd(s['avg_loss'])}</span>
            </div>
        </div>"""

    return f"""
    <h2>Long / Short Breakdown</h2>
    <div class="ls-grid">
        {_card("LONG", lo, "ls-long")}
        {_card("SHORT", sh, "ls-short")}
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

    # Long vs Short comparison data
    ds = _compute_direction_stats(trades)

    return {
        "winners_pnl": winners_pnl,
        "losers_pnl": losers_pnl,
        "month_labels": month_labels,
        "month_values": month_values,
        "win_hold": win_hold,
        "win_pnl": win_pnl,
        "lose_hold": lose_hold,
        "lose_pnl": lose_pnl,
        "direction_stats": ds,
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
    <div class="chart-container">
        <div id="long-short-chart"></div>
    </div>
    """

    month_colors = [
        "'#c3e88d'" if v > 0 else "'#ff757f'" for v in perf["month_values"]
    ]

    js = f"""
        // P&L Distribution
        Plotly.newPlot('pnl-dist-chart', [{{
            x: {json.dumps(perf['winners_pnl'])},
            type: 'histogram',
            name: 'Winners',
            marker: {{ color: 'rgba(195, 232, 141, 0.7)' }},
            xbins: {{ size: 1 }}
        }}, {{
            x: {json.dumps(perf['losers_pnl'])},
            type: 'histogram',
            name: 'Losers',
            marker: {{ color: 'rgba(255, 117, 127, 0.7)' }},
            xbins: {{ size: 1 }}
        }}], {{
            paper_bgcolor: '#2f334d',
            plot_bgcolor: '#2f334d',
            font: {{ color: '#c8d3f5', size: 10 }},
            margin: {{ t: 30, b: 40, l: 50, r: 20 }},
            title: {{ text: 'P&L Distribution', font: {{ size: 12, color: '#636da6' }} }},
            xaxis: {{ gridcolor: '#3b4261', title: 'P&L (%)' }},
            yaxis: {{ gridcolor: '#3b4261', title: 'Count' }},
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
            textfont: {{ size: 9, color: '#c8d3f5' }}
        }}], {{
            paper_bgcolor: '#2f334d',
            plot_bgcolor: '#2f334d',
            font: {{ color: '#c8d3f5', size: 10 }},
            margin: {{ t: 30, b: 60, l: 60, r: 20 }},
            title: {{ text: 'Monthly Returns', font: {{ size: 12, color: '#636da6' }} }},
            xaxis: {{ gridcolor: '#3b4261', title: 'Month' }},
            yaxis: {{ gridcolor: '#3b4261', title: 'P&L ($)' }},
            showlegend: false
        }});

        // Trade Duration vs P&L
        Plotly.newPlot('duration-pnl-chart', [{{
            x: {json.dumps(perf['win_hold'])},
            y: {json.dumps(perf['win_pnl'])},
            type: 'scatter',
            mode: 'markers',
            name: 'Winners',
            marker: {{ color: '#c3e88d', size: 6, opacity: 0.7 }}
        }}, {{
            x: {json.dumps(perf['lose_hold'])},
            y: {json.dumps(perf['lose_pnl'])},
            type: 'scatter',
            mode: 'markers',
            name: 'Losers',
            marker: {{ color: '#ff757f', size: 6, opacity: 0.7 }}
        }}], {{
            paper_bgcolor: '#2f334d',
            plot_bgcolor: '#2f334d',
            font: {{ color: '#c8d3f5', size: 10 }},
            margin: {{ t: 30, b: 40, l: 50, r: 20 }},
            title: {{ text: 'Trade Duration vs P&L', font: {{ size: 12, color: '#636da6' }} }},
            xaxis: {{ gridcolor: '#3b4261', title: 'Hold Duration (bars)' }},
            yaxis: {{ gridcolor: '#3b4261', title: 'P&L (%)' }},
            legend: {{ font: {{ size: 10 }} }}
        }});

        // Long vs Short Comparison
        var lsCategories = ['Trades', 'Winners', 'Losers', 'Win Rate (%)'];
        var lsLong = [{perf['direction_stats']['long']['count']}, {perf['direction_stats']['long']['winners']}, {perf['direction_stats']['long']['losers']}, {perf['direction_stats']['long']['win_rate']:.1f}];
        var lsShort = [{perf['direction_stats']['short']['count']}, {perf['direction_stats']['short']['winners']}, {perf['direction_stats']['short']['losers']}, {perf['direction_stats']['short']['win_rate']:.1f}];
        Plotly.newPlot('long-short-chart', [{{
            x: lsCategories,
            y: lsLong,
            type: 'bar',
            name: 'Long',
            marker: {{ color: 'rgba(255, 199, 119, 0.8)' }},
            text: lsLong.map(function(v, i) {{ return i === 3 ? v.toFixed(1) + '%' : v; }}),
            textposition: 'outside',
            textfont: {{ size: 10, color: '#c8d3f5' }}
        }}, {{
            x: lsCategories,
            y: lsShort,
            type: 'bar',
            name: 'Short',
            marker: {{ color: 'rgba(255, 117, 127, 0.8)' }},
            text: lsShort.map(function(v, i) {{ return i === 3 ? v.toFixed(1) + '%' : v; }}),
            textposition: 'outside',
            textfont: {{ size: 10, color: '#c8d3f5' }}
        }}], {{
            paper_bgcolor: '#2f334d',
            plot_bgcolor: '#2f334d',
            font: {{ color: '#c8d3f5', size: 10 }},
            margin: {{ t: 30, b: 40, l: 50, r: 20 }},
            title: {{ text: 'Long vs Short', font: {{ size: 12, color: '#636da6' }} }},
            xaxis: {{ gridcolor: '#3b4261' }},
            yaxis: {{ gridcolor: '#3b4261' }},
            barmode: 'group',
            legend: {{ font: {{ size: 10 }} }}
        }});
    """

    return html, js


def _render_strategy_flow(params: Dict[str, Any], trades: List[Dict] | None = None) -> str:
    """Render Quant Whale Strategy v3.0 strategy flow with actual parameter values."""
    macd_fast = params.get("macd_fast", "?")
    macd_slow = params.get("macd_slow", "?")
    macd_signal = params.get("macd_signal", "?")
    rsi_period = params.get("rsi_period", "?")
    rsi_upper = params.get("rsi_upper", "?")
    rsi_lower = params.get("rsi_lower", "?")
    min_hold = params.get("min_hold", "?")
    long_only = params.get("long_only", False)
    position_pct = params.get("position_pct", 0.25)
    catastrophic_stop_pct = params.get("catastrophic_stop_pct", 0.15)

    # Catastrophic stop stats from trades
    cat_stop_count = 0
    total_trades = 0
    if trades:
        total_trades = len(trades)
        cat_stop_count = sum(1 for t in trades if t.get("exit_reason") == "catastrophic_stop")

    mode_label = "Long only" if long_only else "Long + Short"

    return f"""
    <h2>Strategy Flow</h2>

    <div class="flow-grid">

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
                <div class="flow-io-label">Compute</div>
                <div>MACD({macd_fast}, {macd_slow}, {macd_signal}) &middot; RSI({rsi_period})</div>
            </div>
            <div class="flow-io">
                <div class="flow-io-label">Output</div>
                <div>1H OHLCV + MACD line/signal + RSI values</div>
            </div>
        </div>
    </div>

    <div class="flow-phase">
        <div class="flow-phase-header">
            <span class="flow-phase-num">2</span>
            <span class="flow-phase-title">BUY SIGNAL</span>
            <span class="flow-phase-meta">2 conditions (AND)</span>
        </div>
        <div class="flow-phase-body">
            <div class="flow-condition">
                <div class="flow-cond-label">MACD Bullish Crossover</div>
                <div class="flow-cond-desc">MACD line crosses above signal line</div>
            </div>
            <div class="flow-condition">
                <div class="flow-cond-label">RSI Oversold</div>
                <div class="flow-cond-desc">RSI({rsi_period}) &lt; {rsi_lower}</div>
            </div>
        </div>
    </div>

    <div class="flow-phase">
        <div class="flow-phase-header">
            <span class="flow-phase-num">3</span>
            <span class="flow-phase-title">SELL SIGNAL</span>
            <span class="flow-phase-meta">2 conditions (AND)</span>
        </div>
        <div class="flow-phase-body">
            <div class="flow-condition">
                <div class="flow-cond-label">MACD Bearish Crossover</div>
                <div class="flow-cond-desc">MACD line crosses below signal line</div>
            </div>
            <div class="flow-condition">
                <div class="flow-cond-label">RSI Overbought</div>
                <div class="flow-cond-desc">RSI({rsi_period}) &gt; {rsi_upper}</div>
            </div>
        </div>
    </div>

    <div class="flow-phase">
        <div class="flow-phase-header">
            <span class="flow-phase-num">4</span>
            <span class="flow-phase-title">EXECUTION</span>
        </div>
        <div class="flow-phase-body">
            <div class="flow-condition">
                <div class="flow-cond-label">Mode</div>
                <div class="flow-cond-desc">{mode_label} &middot; min hold = {min_hold} bars</div>
            </div>
            <div class="flow-condition">
                <div class="flow-cond-label">Position Sizing</div>
                <div class="flow-cond-desc">{position_pct*100:.0f}% of equity per trade</div>
            </div>
            <div class="flow-condition">
                <div class="flow-cond-label">Catastrophic Stop</div>
                <div class="flow-cond-desc">-{catastrophic_stop_pct*100:.0f}% emergency exit{f' &middot; <span class="negative">triggered {cat_stop_count}&times;</span>' if cat_stop_count > 0 else ' &middot; <span class="positive">0 triggers</span>'}</div>
            </div>
            <div class="flow-io" style="margin-top:10px">
                <div class="flow-io-label">Output</div>
                <div>Trade list (entry/exit timestamps, P&amp;L, hold duration)</div>
            </div>
        </div>
    </div>

    </div><!-- /flow-grid -->
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
                colorscale: [[0, '#ff757f'], [0.01, '#3b4261'], [0.3, '#82aaff'], [0.7, '#4fd6be'], [1, '#c3e88d']],
                size: 4,
                opacity: 0.6,
                colorbar: {{
                    thickness: 12,
                    outlinewidth: 0,
                    tickfont: {{ color: '#636da6', size: 9 }}
                }}
            }}
        }}, {{
            x: {json.dumps(numbers)},
            y: {json.dumps(best_so_far)},
            type: 'scatter',
            mode: 'lines',
            name: 'Best So Far',
            line: {{ color: '#c3e88d', width: 2 }}
        }}], {{
            paper_bgcolor: '#2f334d',
            plot_bgcolor: '#2f334d',
            font: {{ color: '#c8d3f5', size: 10 }},
            margin: {{ t: 30, b: 40, l: 60, r: 20 }},
            title: {{ text: 'Optimization Progress', font: {{ size: 12, color: '#636da6' }} }},
            xaxis: {{ gridcolor: '#3b4261', title: 'Trial #' }},
            yaxis: {{ gridcolor: '#3b4261', title: 'Objective Value (Equity)' }},
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

    # Catastrophic stop stats
    cat_stops = [t for t in trades if t.get("exit_reason") == "catastrophic_stop"]
    cat_stop_count = len(cat_stops)
    cat_stop_active = cat_stop_count > 0

    equity_curve = build_equity_curve(trades, start_equity)
    drawdown_curve = build_drawdown_curve(equity_curve)

    # Build date labels for equity/drawdown X-axis
    equity_dates: List[str] = []
    if trades and trades[0].get("entry_ts"):
        equity_dates.append(trades[0]["entry_ts"][:10])  # start = first entry date
        for t in trades:
            equity_dates.append(t.get("exit_ts", "")[:10] if t.get("exit_ts") else "")
    else:
        equity_dates = [str(i) for i in range(len(equity_curve))]

    split_html = _render_split_results(params)
    mc_html = _render_mc_section(params)
    flow_html = _render_strategy_flow(params, trades)
    strategy_html = _render_strategy_params(params)
    ls_html = _render_long_short_metrics(trades)
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
            --bg-primary: #222436;
            --bg-deep: #1e2030;
            --bg-secondary: #2f334d;
            --text-primary: #c8d3f5;
            --text-secondary: #636da6;
            --text-muted: #3b4261;
            --accent-green: #c3e88d;
            --accent-red: #ff757f;
            --accent-purple: #c099ff;
            --accent-cyan: #86e1fc;
            --accent-teal: #4fd6be;
            --accent-yellow: #ffc777;
            --accent-orange: #ff966c;
            --accent-blue: #82aaff;
            --border: #545c7e;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            background: var(--bg-deep);
            color: var(--text-primary);
            padding: 20px;
            font-size: 12px;
            max-width: 1200px;
            margin: 0 auto;
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
            border-bottom: 1px solid var(--border);
            padding-bottom: 5px;
        }}
        .hero-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }}
        .hero-card {{
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border: 1px solid var(--text-muted);
        }}
        .hero-label {{
            font-size: 10px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }}
        .hero-value {{
            font-size: 28px;
            font-weight: bold;
        }}
        .detail-list {{
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 16px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }}
        .detail-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 12px 6px 0;
            border-bottom: 1px solid var(--text-muted);
            font-size: 12px;
        }}
        .detail-row:nth-last-child(-n+2) {{ border-bottom: none; }}
        .detail-label {{
            color: var(--text-secondary);
        }}
        .detail-value {{
            font-weight: bold;
            font-size: 13px;
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
            border-bottom: 1px solid var(--text-muted);
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
        .flow-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 12px;
        }}
        .flow-grid .flow-arrow {{ display: none; }}
        @media (max-width: 800px) {{
            .flow-grid {{ grid-template-columns: 1fr; }}
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
        .ls-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 16px;
        }}
        .ls-card {{
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 16px;
            border: 1px solid var(--text-muted);
        }}
        .ls-long {{ border-top: 3px solid var(--accent-yellow); }}
        .ls-short {{ border-top: 3px solid var(--accent-red); }}
        .ls-header {{
            font-size: 13px;
            font-weight: bold;
            letter-spacing: 1px;
            margin-bottom: 4px;
        }}
        .ls-long .ls-header {{ color: var(--accent-yellow); }}
        .ls-short .ls-header {{ color: var(--accent-red); }}
        .ls-count {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .ls-pct {{
            font-size: 12px;
            color: var(--text-secondary);
            font-weight: normal;
        }}
        .ls-row {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            border-bottom: 1px solid var(--text-muted);
            font-size: 11px;
        }}
        .ls-row:last-child {{ border-bottom: none; }}
        .ls-label {{ color: var(--text-secondary); }}
        .ls-val {{ font-weight: bold; }}
    </style>
</head>
<body>
    <h1>QRE Report: {symbol}</h1>
    <p style="color: var(--text-secondary); margin-bottom: 20px;">
        Generated: {now} | Start equity: ${start_equity:,.0f}
    </p>

    <h2>Key Metrics</h2>

    <div class="hero-grid">
        <div class="hero-card">
            <div class="hero-label">Final Equity</div>
            <div class="hero-value {'positive' if params.get('equity', 0) > start_equity else 'negative'}">
                ${params.get('equity', 0):,.0f}
            </div>
        </div>
        <div class="hero-card">
            <div class="hero-label">Total P&L</div>
            <div class="hero-value {'positive' if params.get('total_pnl_pct', 0) > 0 else 'negative'}">
                {params.get('total_pnl_pct', 0):+.1f}%
            </div>
        </div>
        <div class="hero-card">
            <div class="hero-label">Sharpe Ratio</div>
            <div class="hero-value {'positive' if params.get('sharpe', 0) > 1 else 'negative'}">
                {params.get('sharpe', 0):.2f}
            </div>
        </div>
    </div>

    <div class="detail-list">
        <div class="detail-row">
            <span class="detail-label">Max Drawdown</span>
            <span class="detail-value negative">{params.get('max_drawdown', 0):.1f}%</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Win Rate</span>
            <span class="detail-value {'positive' if params.get('win_rate', 0) > 0.5 else 'negative'}">{params.get('win_rate', 0) * 100:.1f}%</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Trades</span>
            <span class="detail-value">{params.get('trades', 0)}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Trades/Year</span>
            <span class="detail-value">{params.get('trades_per_year', 0):.1f}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Sortino</span>
            <span class="detail-value">{params.get('sortino', 0):.2f}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Calmar</span>
            <span class="detail-value">{params.get('calmar', 0):.2f}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Recovery Factor</span>
            <span class="detail-value">{params.get('recovery_factor', 0):.2f}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Profit Factor</span>
            <span class="detail-value">{params.get('profit_factor', 0):.2f}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Expectancy</span>
            <span class="detail-value">${params.get('expectancy', 0):.2f}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Catastrophic Stops</span>
            <span class="detail-value {'negative' if cat_stop_count > 0 else 'positive'}">{cat_stop_count} / {len(trades)}{f' ({cat_stop_count/len(trades)*100:.0f}%)' if trades else ''}</span>
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

    {ls_html}

    {split_html}
    {mc_html}
    {optuna_html}
    {flow_html}
    {strategy_html}
    {perf_html}

    <footer>
        QRE v0.4.0 | MACD+RSI | Anchored Walk-Forward
    </footer>

    <script>
        Plotly.newPlot('equity-chart', [{{
            x: {json.dumps(equity_dates)},
            y: {json.dumps(equity_curve)},
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#86e1fc', width: 2 }},
            fill: 'tozeroy',
            fillcolor: 'rgba(134, 225, 252, 0.08)'
        }}], {{
            paper_bgcolor: '#2f334d',
            plot_bgcolor: '#2f334d',
            font: {{ color: '#c8d3f5', size: 10 }},
            margin: {{ t: 20, b: 60, l: 60, r: 20 }},
            xaxis: {{ gridcolor: '#3b4261', title: 'Date', type: 'category', tickangle: -45 }},
            yaxis: {{ gridcolor: '#3b4261', title: 'Equity ($)' }}
        }});

        Plotly.newPlot('drawdown-chart', [{{
            x: {json.dumps(equity_dates)},
            y: {json.dumps(drawdown_curve)},
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#ff757f', width: 2 }},
            fill: 'tozeroy',
            fillcolor: 'rgba(255, 117, 127, 0.15)'
        }}], {{
            paper_bgcolor: '#2f334d',
            plot_bgcolor: '#2f334d',
            font: {{ color: '#c8d3f5', size: 10 }},
            margin: {{ t: 20, b: 60, l: 60, r: 20 }},
            xaxis: {{ gridcolor: '#3b4261', title: 'Date', type: 'category', tickangle: -45 }},
            yaxis: {{ gridcolor: '#3b4261', title: 'Drawdown (%)' }}
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
