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

SHARPE_CAP = 5.0  # Sharpe values above this get warning CSS class


def _fmt_sharpe(value: float) -> str:
    """Format Sharpe for display: always show exact value."""
    return f"{value:.2f}"


def _sharpe_css(value: float) -> str:
    """CSS class for Sharpe value."""
    if value > SHARPE_CAP:
        return "warning"
    if value > 1:
        return "positive"
    return "negative"


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
        sharpe_time = s.get("test_sharpe_time", s.get("test_sharpe", 0))
        sharpe_equity = s.get("test_sharpe_equity", 0)
        rows += (
            f'<tr>'
            f'<td>Split {s.get("split", "?")}</td>'
            f'<td>${s.get("test_equity", 0):,.2f}</td>'
            f'<td>{s.get("test_trades", 0)}</td>'
            f'<td class="{_sharpe_css(sharpe_time)}">{_fmt_sharpe(sharpe_time)}</td>'
            f'<td class="{_sharpe_css(sharpe_equity)}">{_fmt_sharpe(sharpe_equity)}</td>'
            f'</tr>\n'
        )

    return f"""
    <h2>Walk-Forward Splits</h2>
    <div class="chart-container">
        <table class="params-table">
            <tr><th>Split</th><th>Test Equity</th><th>Test Trades</th><th>Sharpe (time)</th><th>Sharpe (equity)</th></tr>
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


def _compute_yearly_breakdown(
    trades: List[Dict], start_equity: float
) -> List[Dict[str, Any]]:
    """Compute per-calendar-year metrics from trades.

    Groups trades by exit_ts year. For each year computes:
    PnL ($), PnL (%), gross profit, gross loss, trade count,
    win rate, max drawdown, and partial year info.
    """
    if not trades:
        return []

    # Group trades by year (using exit_ts)
    yearly_trades: Dict[int, List[Dict]] = {}
    for t in trades:
        exit_ts = t.get("exit_ts", "")
        if not exit_ts:
            continue
        year = int(exit_ts[:4])
        yearly_trades.setdefault(year, []).append(t)

    if not yearly_trades:
        return []

    # Detect first entry and last exit for partial year detection
    all_entry_months: Dict[int, List[int]] = {}
    all_exit_months: Dict[int, List[int]] = {}
    for t in trades:
        entry_ts = t.get("entry_ts", "")
        exit_ts = t.get("exit_ts", "")
        if entry_ts:
            yr = int(entry_ts[:4])
            mo = int(entry_ts[5:7])
            all_entry_months.setdefault(yr, []).append(mo)
        if exit_ts:
            yr = int(exit_ts[:4])
            mo = int(exit_ts[5:7])
            all_exit_months.setdefault(yr, []).append(mo)

    month_names = [
        "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    results = []
    running_equity = start_equity

    for year in sorted(yearly_trades.keys()):
        year_trades = yearly_trades[year]
        year_start_equity = running_equity

        # PnL
        pnl_dollar = sum(t.get("pnl_abs", 0) for t in year_trades)
        pnl_pct = (pnl_dollar / year_start_equity * 100) if year_start_equity > 0 else 0.0

        # Gross profit / loss
        gross_profit = sum(t.get("pnl_abs", 0) for t in year_trades if t.get("pnl_abs", 0) > 0)
        gross_loss = sum(t.get("pnl_abs", 0) for t in year_trades if t.get("pnl_abs", 0) <= 0)

        # Trade count and win rate
        trade_count = len(year_trades)
        winners = sum(1 for t in year_trades if t.get("pnl_abs", 0) > 0)
        win_rate = (winners / trade_count * 100) if trade_count > 0 else 0.0

        # Max drawdown for this year (from equity curve within the year)
        equity = year_start_equity
        peak = equity
        max_dd = 0.0
        for t in year_trades:
            equity += t.get("pnl_abs", 0)
            peak = max(peak, equity)
            dd = (equity - peak) / peak * 100 if peak > 0 else 0.0
            max_dd = min(max_dd, dd)

        # Partial year detection
        entry_months = all_entry_months.get(year, [])
        exit_months = all_exit_months.get(year, [])
        all_months = sorted(set(entry_months + exit_months))
        first_month = min(all_months) if all_months else 1
        last_month = max(all_months) if all_months else 12
        partial = first_month > 1 or last_month < 12
        partial_label = ""
        if partial:
            partial_label = f"{month_names[first_month]}\u2013{month_names[last_month]}"

        results.append({
            "year": year,
            "pnl_dollar": pnl_dollar,
            "pnl_pct": pnl_pct,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "trade_count": trade_count,
            "win_rate": win_rate,
            "max_dd": max_dd,
            "partial": partial,
            "partial_label": partial_label,
        })

        running_equity += pnl_dollar

    return results


def _fmt_usd(value: float) -> str:
    """Format dollar value with sign before $: -$50.00 instead of $-50.00."""
    if value < 0:
        return f"-${abs(value):,.2f}"
    return f"${value:,.2f}"


def _render_exit_reason_breakdown(trades: List[Dict]) -> str:
    """Render exit reason breakdown table (signal / catastrophic_stop / force_close)."""
    if not trades:
        return ""

    reason_order = ["signal", "catastrophic_stop", "force_close"]
    reason_labels = {
        "signal": "Signal (planned exit)",
        "catastrophic_stop": "Catastrophic Stop",
        "force_close": "Force Close (end of period)",
    }
    reason_css = {
        "signal": "",
        "catastrophic_stop": "negative",
        "force_close": "warning",
    }

    total = len(trades)
    rows = ""
    for reason in reason_order:
        subset = [t for t in trades if t.get("reason") == reason]
        count = len(subset)
        pct = count / total * 100 if total else 0
        avg_pnl = sum(t.get("pnl_abs", 0) for t in subset) / count if count else 0
        avg_pnl_pct = sum(t.get("pnl_pct", 0) for t in subset) / count * 100 if count else 0
        css = reason_css.get(reason, "")
        pnl_css = "positive" if avg_pnl >= 0 else "negative"
        rows += (
            f'<tr>'
            f'<td class="{css}">{reason_labels.get(reason, reason)}</td>'
            f'<td>{count}</td>'
            f'<td>{pct:.0f}%</td>'
            f'<td class="{pnl_css}">{_fmt_usd(avg_pnl)}</td>'
            f'<td class="{pnl_css}">{avg_pnl_pct:+.2f}%</td>'
            f'</tr>\n'
        )

    return f"""
    <h2>Exit Reasons</h2>
    <div class="chart-container">
        <table class="params-table">
            <tr><th>Reason</th><th>Count</th><th>%</th><th>Avg P&amp;L ($)</th><th>Avg P&amp;L (%)</th></tr>
            {rows}
        </table>
    </div>
    """


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


def _render_monthly_returns(trades: List[Dict]) -> tuple[str, str]:
    """Render monthly returns bar chart."""
    if not trades:
        return "", ""

    perf = _build_performance_data(trades)
    if not perf["month_labels"]:
        return "", ""

    month_colors = [
        "'#c3e88d'" if v > 0 else "'#ff757f'" for v in perf["month_values"]
    ]

    html = """
    <div class="chart-container">
        <div id="monthly-returns-chart"></div>
    </div>
    """

    js = f"""
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
    """

    return html, js


def _render_trade_charts(trades: List[Dict]) -> tuple[str, str]:
    """Render P&L distribution and trade duration vs P&L scatter."""
    if not trades:
        return "", ""

    perf = _build_performance_data(trades)

    html = """
    <div class="chart-container">
        <div id="pnl-dist-chart"></div>
    </div>
    <div class="chart-container">
        <div id="duration-pnl-chart"></div>
    </div>
    """

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
    """

    return html, js


def _render_rolling_metrics(trades: List[Dict], window: int = 30) -> tuple[str, str]:
    """Render rolling win rate, avg P&L, and Sharpe over trade sequence."""
    if len(trades) < window:
        return "", ""

    trade_nums = list(range(window, len(trades) + 1))
    win_rates = []
    avg_pnls = []
    rolling_sharpes = []

    for i in range(window, len(trades) + 1):
        w = trades[i - window:i]
        wins = sum(1 for t in w if t.get("pnl_pct", 0) > 0)
        pnls = [t.get("pnl_pct", 0) * 100 for t in w]
        win_rates.append(round(wins / window * 100, 1))
        avg_pnls.append(round(sum(pnls) / window, 3))
        mean_pnl = sum(pnls) / window
        if window > 1:
            var = sum((p - mean_pnl) ** 2 for p in pnls) / (window - 1)
            std = var ** 0.5
            rolling_sharpes.append(round(mean_pnl / std, 3) if std > 0 else 0.0)
        else:
            rolling_sharpes.append(0.0)

    html = f"""
    <h2>Rolling Metrics ({window}-trade window)</h2>
    <div class="chart-container">
        <div id="rolling-metrics-chart"></div>
    </div>
    """

    js = f"""
        Plotly.newPlot('rolling-metrics-chart', [{{
            x: {json.dumps(trade_nums)},
            y: {json.dumps(win_rates)},
            type: 'scatter',
            mode: 'lines',
            name: 'Win Rate (%)',
            line: {{ color: '#c3e88d', width: 2 }},
            yaxis: 'y'
        }}, {{
            x: {json.dumps(trade_nums)},
            y: {json.dumps(avg_pnls)},
            type: 'scatter',
            mode: 'lines',
            name: 'Avg P&L (%)',
            line: {{ color: '#86e1fc', width: 2 }},
            yaxis: 'y2'
        }}, {{
            x: {json.dumps(trade_nums)},
            y: {json.dumps(rolling_sharpes)},
            type: 'scatter',
            mode: 'lines',
            name: 'Sharpe',
            line: {{ color: '#c099ff', width: 2, dash: 'dot' }},
            yaxis: 'y2'
        }}], {{
            paper_bgcolor: '#2f334d',
            plot_bgcolor: '#2f334d',
            font: {{ color: '#c8d3f5', size: 10 }},
            margin: {{ t: 30, b: 40, l: 50, r: 50 }},
            title: {{ text: 'Rolling {window}-Trade Metrics', font: {{ size: 12, color: '#636da6' }} }},
            xaxis: {{ gridcolor: '#3b4261', title: 'Trade #' }},
            yaxis: {{ gridcolor: '#3b4261', title: 'Win Rate (%)', side: 'left', range: [0, 100] }},
            yaxis2: {{ overlaying: 'y', side: 'right', title: 'Avg P&L (%) / Sharpe', gridcolor: 'rgba(59,66,97,0.3)' }},
            legend: {{ font: {{ size: 10 }}, orientation: 'h', y: -0.2 }}
        }});
    """

    return html, js


def _render_streak_timeline(trades: List[Dict]) -> tuple[str, str]:
    """Render win/loss streak timeline as horizontal bar chart."""
    if not trades:
        return "", ""

    pnls = [t.get("pnl_pct", 0) * 100 for t in trades]
    colors = ["#c3e88d" if p > 0 else "#ff757f" for p in pnls]
    trade_nums = list(range(1, len(trades) + 1))

    max_win_streak = max_loss_streak = 0
    cur_win = cur_loss = 0
    win_streak_end = loss_streak_end = 0
    for i, p in enumerate(pnls):
        if p > 0:
            cur_win += 1
            cur_loss = 0
            if cur_win > max_win_streak:
                max_win_streak = cur_win
                win_streak_end = i
        else:
            cur_loss += 1
            cur_win = 0
            if cur_loss > max_loss_streak:
                max_loss_streak = cur_loss
                loss_streak_end = i

    html = f"""
    <h2>Win/Loss Streak Timeline</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Max Win Streak</div>
            <div class="metric-value positive">{max_win_streak}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Loss Streak</div>
            <div class="metric-value negative">{max_loss_streak}</div>
        </div>
    </div>
    <div class="chart-container">
        <div id="streak-timeline-chart"></div>
    </div>
    """

    js = f"""
        Plotly.newPlot('streak-timeline-chart', [{{
            x: {json.dumps(trade_nums)},
            y: {json.dumps(pnls)},
            type: 'bar',
            marker: {{ color: {json.dumps(colors)} }},
            hovertemplate: 'Trade #%{{x}}<br>P&L: %{{y:.2f}}%<extra></extra>'
        }}], {{
            paper_bgcolor: '#2f334d',
            plot_bgcolor: '#2f334d',
            font: {{ color: '#c8d3f5', size: 10 }},
            margin: {{ t: 30, b: 40, l: 50, r: 20 }},
            title: {{ text: 'Trade Sequence (Win/Loss)', font: {{ size: 12, color: '#636da6' }} }},
            xaxis: {{ gridcolor: '#3b4261', title: 'Trade #' }},
            yaxis: {{ gridcolor: '#3b4261', title: 'P&L (%)' }},
            showlegend: false,
            shapes: [{{
                type: 'rect',
                x0: {win_streak_end - max_win_streak + 0.5 + 1},
                x1: {win_streak_end + 1.5},
                y0: 0, y1: 1, yref: 'paper',
                line: {{ color: '#c3e88d', width: 2, dash: 'dot' }},
                fillcolor: 'rgba(195, 232, 141, 0.05)'
            }}, {{
                type: 'rect',
                x0: {loss_streak_end - max_loss_streak + 0.5 + 1},
                x1: {loss_streak_end + 1.5},
                y0: 0, y1: 1, yref: 'paper',
                line: {{ color: '#ff757f', width: 2, dash: 'dot' }},
                fillcolor: 'rgba(255, 117, 127, 0.05)'
            }}]
        }});
    """

    return html, js


def _render_strategy_flow(params: Dict[str, Any], trades: List[Dict] | None = None) -> str:
    """Render Quant Whale Strategy v4.0 strategy flow with actual parameter values."""
    macd_fast = params.get("macd_fast", "?")
    macd_slow = params.get("macd_slow", "?")
    macd_signal = params.get("macd_signal", "?")
    rsi_period = params.get("rsi_period", "?")
    rsi_upper = params.get("rsi_upper", "?")
    rsi_lower = params.get("rsi_lower", "?")
    min_hold = params.get("min_hold", "?")
    long_only = params.get("long_only", False)
    position_pct = params.get("position_pct", 0.25)
    catastrophic_stop_pct = params.get("catastrophic_stop_pct", 0.10)
    rsi_lookback = params.get("rsi_lookback", 0)
    trend_tf = params.get("trend_tf", "?")
    trend_strict = params.get("trend_strict", 0)

    # Catastrophic stop stats from trades
    cat_stop_count = 0
    total_trades = 0
    if trades:
        total_trades = len(trades)
        cat_stop_count = sum(1 for t in trades if t.get("reason") == "catastrophic_stop")

    mode_label = "Long only" if long_only else "Long + Short"

    # Build TF scale HTML
    tf_options = ["1h", "4h", "8h", "1d"]
    tf_scale_html = ''.join(
        f'<span class="tf-opt{" tf-active" if tf == str(trend_tf) else ""}">{tf}</span>'
        for tf in tf_options
    )
    trend_strict_html = f' &middot; strict={trend_strict}' if trend_strict != '?' else ''

    # Condition counts for BUY/SELL phases
    buy_sell_cond_count = '3 conditions (AND)' if rsi_lookback > 0 else '2 conditions (AND)'

    # RSI Lookback condition HTML (shared by BUY and SELL phases)
    lookback_condition = f"""
            <div class="flow-condition">
                <div class="flow-cond-label">RSI Lookback Window</div>
                <div class="flow-cond-desc">Check RSI condition over last {rsi_lookback} bars (0 = current bar only)</div>
            </div>""" if rsi_lookback > 0 else ""

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
            <div class="flow-io">
                <div class="flow-io-label">Trend</div>
                <div>
                    <span class="tf-scale">
                        {tf_scale_html}
                    </span>
                    {trend_strict_html}
                </div>
            </div>
        </div>
    </div>

    <div class="flow-phase">
        <div class="flow-phase-header">
            <span class="flow-phase-num">2</span>
            <span class="flow-phase-title">BUY SIGNAL</span>
            <span class="flow-phase-meta">{buy_sell_cond_count}</span>
        </div>
        <div class="flow-phase-body">
            <div class="flow-condition">
                <div class="flow-cond-label">MACD Bullish Crossover</div>
                <div class="flow-cond-desc">MACD line crosses above signal line</div>
            </div>
            <div class="flow-condition">
                <div class="flow-cond-label">RSI Oversold</div>
                <div class="flow-cond-desc">RSI({rsi_period}) &lt; {rsi_lower}</div>
            </div>{lookback_condition}
        </div>
    </div>

    <div class="flow-phase">
        <div class="flow-phase-header">
            <span class="flow-phase-num">3</span>
            <span class="flow-phase-title">SELL SIGNAL</span>
            <span class="flow-phase-meta">{buy_sell_cond_count}</span>
        </div>
        <div class="flow-phase-body">
            <div class="flow-condition">
                <div class="flow-cond-label">MACD Bearish Crossover</div>
                <div class="flow-cond-desc">MACD line crosses below signal line</div>
            </div>
            <div class="flow-condition">
                <div class="flow-cond-label">RSI Overbought</div>
                <div class="flow-cond-desc">RSI({rsi_period}) &gt; {rsi_upper}</div>
            </div>{lookback_condition}
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
    """Render strategy parameters as bullet chart visualization."""
    # Ranges aligned with Optuna search space in strategy.py
    numeric_params = [
        ("macd_fast", "MACD fast", 1, 20),
        ("macd_slow", "MACD slow", 10, 45),
        ("macd_signal", "MACD signal", 3, 15),
        ("rsi_period", "RSI period", 3, 30),
        ("rsi_lower", "RSI lower", 20, 40),
        ("rsi_upper", "RSI upper", 60, 80),
        ("rsi_lookback", "RSI lookback", 1, 4),
    ]
    categorical_params = [
        ("trend_tf", "Trend TF", ["4h", "8h", "1d"]),
        ("trend_strict", "Strict", [0, 1]),
        ("allow_flip", "Allow Flip", [0, 1]),
    ]

    rows_html = ""
    for key, label, lo, hi in numeric_params:
        if key not in params:
            continue
        val = params[key]
        display_val = f"{val:.1f}" if isinstance(val, float) else val
        if hi == lo:
            pct = 50
        else:
            pct = max(0, min(100, (val - lo) / (hi - lo) * 100))
        rows_html += f"""
        <div class="param-bullet">
            <span class="param-label">{label}</span>
            <span class="param-range-lo">{lo}</span>
            <div class="param-bar-track">
                <div class="param-bar-fill" style="width:{pct:.0f}%"></div>
                <div class="param-bar-marker" style="left:{pct:.0f}%"></div>
            </div>
            <span class="param-range-hi">{hi}</span>
            <span class="param-val">{display_val}</span>
        </div>"""

    for key, label, options in categorical_params:
        if key not in params:
            continue
        val = params[key]
        opts_html = ""
        for opt in options:
            active = "param-cat-active" if str(opt) == str(val) else ""
            opts_html += f'<span class="param-cat-opt {active}">{opt}</span>'
        rows_html += f"""
        <div class="param-bullet">
            <span class="param-label">{label}</span>
            <span class="param-range-lo"></span>
            <div class="param-cat-row">{opts_html}</div>
            <span class="param-range-hi"></span>
            <span class="param-val">{val}</span>
        </div>"""

    return f"""
    <h2>Strategy Parameters</h2>
    <div class="chart-container">
        {rows_html}
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
            yaxis: {{ gridcolor: '#3b4261', title: 'Objective Value (Log Calmar)' }},
            legend: {{ font: {{ size: 10 }} }}
        }});
    """

    return html, js



def _render_hold_duration_chart(trades: List[Dict]) -> tuple[str, str]:
    """Render hold duration histogram with hour-based bins."""
    if not trades:
        return "", ""

    hours = [t.get("hold_bars", 0) for t in trades]
    bin_edges = [0, 6, 12, 24, 48, 96]
    bin_labels = ["0-6h", "6-12h", "12-24h", "24-48h", "48-96h", "96h+"]
    bin_colors = ["#c3e88d", "#86e1fc", "#82aaff", "#c099ff", "#ffc777", "#ff966c"]
    counts = []
    for i in range(len(bin_edges)):
        lo = bin_edges[i]
        hi = bin_edges[i + 1] if i + 1 < len(bin_edges) else float("inf")
        counts.append(sum(1 for h in hours if lo <= h < hi))

    html = """
    <div class="chart-container">
        <div id="hold-duration-chart"></div>
    </div>
    """

    js = f"""
        Plotly.newPlot('hold-duration-chart', [{{
            y: {json.dumps(bin_labels)},
            x: {json.dumps(counts)},
            type: 'bar',
            orientation: 'h',
            marker: {{ color: {json.dumps(bin_colors)} }},
            text: {json.dumps(counts)},
            textposition: 'outside',
            textfont: {{ size: 10, color: '#c8d3f5' }}
        }}], {{
            paper_bgcolor: '#2f334d',
            plot_bgcolor: '#2f334d',
            font: {{ color: '#c8d3f5', size: 10 }},
            margin: {{ t: 30, b: 40, l: 60, r: 40 }},
            title: {{ text: 'Hold Duration Distribution', font: {{ size: 12, color: '#636da6' }} }},
            xaxis: {{ gridcolor: '#3b4261', title: 'Number of Trades' }},
            yaxis: {{ gridcolor: '#3b4261', autorange: 'reversed' }},
            showlegend: false
        }});
    """

    return html, js


def _section_divider(title: str) -> str:
    """Render a category section divider."""
    return f'<div class="section-divider"><span>{title.upper()}</span></div>'


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
    cat_stops = [t for t in trades if t.get("reason") == "catastrophic_stop"]
    cat_stop_count = len(cat_stops)
    # Avg win / avg loss
    wins = [t["pnl_abs"] for t in trades if t.get("pnl_abs", 0) > 0]
    losses = [t["pnl_abs"] for t in trades if t.get("pnl_abs", 0) < 0]
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    equity_curve = build_equity_curve(trades, start_equity)
    drawdown_curve = build_drawdown_curve(equity_curve)

    # High-water mark
    hwm = []
    peak = equity_curve[0]
    for eq in equity_curve:
        peak = max(peak, eq)
        hwm.append(peak)

    # Build date labels for equity/drawdown X-axis
    equity_dates: List[str] = []
    if trades and trades[0].get("entry_ts"):
        equity_dates.append(trades[0]["entry_ts"][:10])  # start = first entry date
        for t in trades:
            equity_dates.append(t.get("exit_ts", "")[:10] if t.get("exit_ts") else "")
    else:
        equity_dates = [str(i) for i in range(len(equity_curve))]

    # Trade markers for equity chart (merged from Cumulative P&L)
    marker_colors_json = json.dumps(
        ["#c3e88d" if t.get("pnl_abs", 0) > 0 else "#ff757f" for t in trades]
    )
    marker_texts_json = json.dumps(
        [f"Trade: ${t.get('pnl_abs', 0):+,.0f}<br>Equity: ${eq:,.0f}"
         for t, eq in zip(trades, equity_curve[1:])]
    )

    split_html = _render_split_results(params)
    mc_html = _render_mc_section(params)
    flow_html = _render_strategy_flow(params, trades)
    strategy_html = _render_strategy_params(params)
    exit_reasons_html = _render_exit_reason_breakdown(trades)
    ls_html = _render_long_short_metrics(trades)
    monthly_html, monthly_js = _render_monthly_returns(trades)
    trade_charts_html, trade_charts_js = _render_trade_charts(trades)
    hold_dur_html, hold_dur_js = _render_hold_duration_chart(trades)
    rolling_html, rolling_js = _render_rolling_metrics(trades)
    streak_html, streak_js = _render_streak_timeline(trades)
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
            grid-template-columns: repeat(4, 1fr);
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
        .detail-group {{
            margin-bottom: 16px;
        }}
        .detail-group-title {{
            font-size: 11px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
            font-weight: normal;
        }}
        .detail-list {{
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 0;
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
        .warning {{ color: #ffc777; }}
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
        .section-divider {{
            margin: 32px 0 16px 0;
            border-bottom: 2px solid var(--accent-purple);
            padding-bottom: 6px;
        }}
        .section-divider span {{
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 2px;
            color: var(--accent-purple);
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
        .param-bullet {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 0;
            border-bottom: 1px solid var(--text-muted);
            font-size: 11px;
        }}
        .param-bullet:last-child {{ border-bottom: none; }}
        .param-label {{
            width: 100px;
            color: var(--text-secondary);
            flex-shrink: 0;
        }}
        .param-range-lo, .param-range-hi {{
            width: 28px;
            color: var(--text-muted);
            font-size: 10px;
            text-align: center;
            flex-shrink: 0;
        }}
        .param-bar-track {{
            flex: 1;
            height: 8px;
            background: var(--text-muted);
            border-radius: 4px;
            position: relative;
            min-width: 120px;
        }}
        .param-bar-fill {{
            position: absolute;
            top: 0; left: 0; height: 100%;
            background: var(--accent-blue);
            border-radius: 4px;
            opacity: 0.6;
        }}
        .param-bar-marker {{
            position: absolute;
            top: -3px;
            width: 6px; height: 14px;
            background: #fff;
            border-radius: 2px;
            transform: translateX(-3px);
        }}
        .param-val {{
            width: 36px;
            text-align: right;
            font-weight: bold;
            flex-shrink: 0;
        }}
        .param-cat-row {{
            flex: 1;
            display: flex;
            gap: 4px;
            min-width: 120px;
        }}
        .param-cat-opt {{
            padding: 2px 8px;
            border-radius: 4px;
            background: var(--text-muted);
            color: var(--text-secondary);
            font-size: 10px;
        }}
        .param-cat-active {{
            background: var(--accent-blue);
            color: var(--bg-primary);
            font-weight: bold;
        }}
        .tf-scale {{
            display: inline-flex;
            gap: 2px;
        }}
        .tf-opt {{
            padding: 1px 6px;
            border-radius: 3px;
            background: var(--text-muted);
            color: var(--text-secondary);
            font-size: 10px;
        }}
        .tf-active {{
            background: var(--accent-cyan);
            color: var(--bg-primary);
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>QRE Report: {symbol}</h1>
    <p style="color: var(--text-secondary); margin-bottom: 20px;">
        Generated: {now} | Start equity: ${start_equity:,.0f} | Objective: {params.get('objective_type', 'sharpe').upper()}
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
            <div class="hero-label">Calmar Ratio</div>
            <div class="hero-value {'positive' if params.get('calmar', 0) > 1 else 'negative'}">
                {params.get('calmar', 0):.2f}
            </div>
        </div>
        <div class="hero-card">
            <div class="hero-label">Sharpe (equity)</div>
            <div class="hero-value {_sharpe_css(params.get('sharpe_equity', 0))}">
                {_fmt_sharpe(params.get('sharpe_equity', 0))}
            </div>
        </div>
    </div>

    <div class="detail-group">
        <h3 class="detail-group-title">Risk &amp; Returns</h3>
        <div class="detail-list">
            <div class="detail-row">
                <span class="detail-label">Max Drawdown</span>
                <span class="detail-value negative">{params.get('max_drawdown', 0):.1f}%</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Sharpe (time)</span>
                <span class="detail-value {_sharpe_css(params.get('sharpe_time', params.get('sharpe', 0)))}">{_fmt_sharpe(params.get('sharpe_time', params.get('sharpe', 0)))}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Sortino</span>
                <span class="detail-value">{params.get('sortino', 0):.2f}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Log Calmar (objective)</span>
                <span class="detail-value">{params.get('log_calmar', 0):.4f}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Recovery Factor</span>
                <span class="detail-value">{params.get('recovery_factor', 0):.2f}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Profit Factor</span>
                <span class="detail-value">{params.get('profit_factor', 0):.2f}</span>
            </div>
        </div>
    </div>

    <div class="detail-group">
        <h3 class="detail-group-title">Trade Statistics</h3>
        <div class="detail-list">
            <div class="detail-row">
                <span class="detail-label">Trades</span>
                <span class="detail-value">{params.get('trades', 0)}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Trades/Year</span>
                <span class="detail-value">{params.get('trades_per_year', 0):.1f}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Win Rate</span>
                <span class="detail-value {'positive' if params.get('win_rate', 0) > 0.5 else 'negative'}">{params.get('win_rate', 0) * 100:.1f}%</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Expectancy</span>
                <span class="detail-value">${params.get('expectancy', 0):.2f}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Avg Win / Avg Loss</span>
                <span class="detail-value">${avg_win:,.0f} / ${avg_loss:,.0f} ({win_loss_ratio:.2f}x)</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Catastrophic Stops</span>
                <span class="detail-value {'negative' if cat_stop_count > 0 else 'positive'}">{cat_stop_count} / {len(trades)}{f' ({cat_stop_count/len(trades)*100:.0f}%)' if trades else ''}</span>
            </div>
        </div>
    </div>

    <div class="detail-group">
        <h3 class="detail-group-title">Consistency</h3>
        <div class="detail-list">
            <div class="detail-row">
                <span class="detail-label">Profitable Months</span>
                <span class="detail-value {'positive' if params.get('profitable_months_ratio', 0) > 0.6 else 'negative'}">{params.get('profitable_months_ratio', 0) * 100:.0f}%</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Time in Market</span>
                <span class="detail-value">{params.get('time_in_market', 0) * 100:.1f}%</span>
            </div>
        </div>
    </div>

    {_section_divider("Performance")}

    <div class="chart-container">
        <div id="equity-combo-chart" style="height:450px"></div>
    </div>

    {monthly_html}

    {_section_divider("Robustness")}

    {split_html}
    {mc_html}
    {optuna_html}

    {_section_divider("Trade Analysis")}

    {ls_html}
    {exit_reasons_html}
    {trade_charts_html}
    {rolling_html}
    {streak_html}
    {hold_dur_html}

    {_section_divider("Strategy")}

    {flow_html}
    {strategy_html}

    <footer>
        QRE v0.4.0 | MACD+RSI | Anchored Walk-Forward
    </footer>

    <script>
        Plotly.newPlot('equity-combo-chart', [{{
            x: {json.dumps(equity_dates)},
            y: {json.dumps(equity_curve)},
            type: 'scatter',
            mode: 'lines',
            name: 'Equity',
            line: {{ color: '#86e1fc', width: 2 }},
            fill: 'tozeroy',
            fillcolor: 'rgba(134, 225, 252, 0.08)',
            hoverinfo: 'skip',
            yaxis: 'y'
        }}, {{
            x: {json.dumps(equity_dates)},
            y: {json.dumps(hwm)},
            type: 'scatter',
            mode: 'lines',
            name: 'High-Water Mark',
            line: {{ color: '#636da6', width: 1, dash: 'dot' }},
            hoverinfo: 'skip',
            yaxis: 'y'
        }}, {{
            x: {json.dumps(equity_dates[1:])},
            y: {json.dumps(equity_curve[1:])},
            type: 'scatter',
            mode: 'markers',
            name: 'Trades',
            marker: {{
                color: {marker_colors_json},
                size: 4,
                opacity: 0.8,
                line: {{ color: '#1e2030', width: 0.5 }}
            }},
            text: {marker_texts_json},
            hoverinfo: 'text',
            yaxis: 'y'
        }}, {{
            x: {json.dumps(equity_dates)},
            y: {json.dumps(drawdown_curve)},
            type: 'scatter',
            mode: 'lines',
            name: 'Drawdown',
            line: {{ color: '#ff757f', width: 1.5 }},
            fill: 'tozeroy',
            fillcolor: 'rgba(255, 117, 127, 0.15)',
            yaxis: 'y2'
        }}], {{
            paper_bgcolor: '#2f334d',
            plot_bgcolor: '#2f334d',
            font: {{ color: '#c8d3f5', size: 10 }},
            margin: {{ t: 20, b: 40, l: 60, r: 60 }},
            xaxis: {{
                gridcolor: '#3b4261',
                type: 'category',
                nticks: 12,
                tickangle: -45,
                tickfont: {{ size: 9 }},
                anchor: 'y2'
            }},
            yaxis: {{
                gridcolor: '#3b4261',
                title: 'Equity ($)',
                domain: [0.28, 1]
            }},
            yaxis2: {{
                gridcolor: 'rgba(59,66,97,0.3)',
                title: 'DD (%)',
                domain: [0, 0.22],
                autorange: true
            }},
            legend: {{ font: {{ size: 10 }}, orientation: 'h', y: -0.12 }},
            showlegend: true
        }});
        {monthly_js}
        {trade_charts_js}
        {rolling_js}
        {streak_js}
        {hold_dur_js}
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
