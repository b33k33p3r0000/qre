"""Post-run analysis pipeline — rules-based diagnostics.

Produces analysis.json + Discord embed after each optimizer run.
Design: docs/plans/2026-02-12-analyze-pipeline-design.md
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median
from typing import Any


def _classify(value: float, green_range: tuple, yellow_range: tuple) -> str:
    """Classify a metric value as green/yellow/red.

    green_range and yellow_range are (low, high) inclusive bounds.
    Anything outside yellow is red.
    """
    lo_g, hi_g = green_range
    lo_y, hi_y = yellow_range
    if lo_g <= value <= hi_g:
        return "green"
    if lo_y <= value <= hi_y:
        return "yellow"
    return "red"


def health_check(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Run rules-based health check on optimizer result params.

    Returns dict mapping metric name to {"status": "green"|"yellow"|"red", "value": ...}.

    Thresholds:
        Sharpe:            green 1.0–3.5,  yellow 0.5–5.0,  red otherwise
        Max Drawdown:      green > -5%,    yellow -5% to -10%, red < -10%
        Trades/year:       green 80–500,   yellow 30–800,    red otherwise
        Win Rate:          green >= 50%,   yellow 40%–50%,   red < 40%
        Profit Factor:     green >= 1.5,   yellow 1.0–1.5,   red < 1.0
        Expectancy:        green >= $100,  yellow $0–$100,   red < $0
        Train/Test Sharpe: green diff < 1, yellow diff 1–2,  red diff > 2
        Split Consistency: green all pos,  yellow 1 neg,     red 2+ neg
    """
    result: dict[str, dict[str, Any]] = {}

    # Sharpe: green 1.0–3.5, yellow 0.5–5.0, red outside
    sharpe = params["sharpe"]
    result["sharpe"] = {
        "status": _classify(sharpe, (1.0, 3.5), (0.5, 5.0)),
        "value": sharpe,
    }

    # Max Drawdown (negative value, e.g. -0.04 = -4%)
    # green: > -0.05, yellow: -0.05 to -0.10, red: < -0.10
    dd = params["max_drawdown"]
    if dd > -0.05:
        dd_status = "green"
    elif dd >= -0.10:
        dd_status = "yellow"
    else:
        dd_status = "red"
    result["max_drawdown"] = {"status": dd_status, "value": dd}

    # Trades per year: green 80–500, yellow 30–800, red outside
    tpy = params["trades_per_year"]
    result["trades_per_year"] = {
        "status": _classify(tpy, (80, 500), (30, 800)),
        "value": tpy,
    }

    # Win rate: green >= 0.50, yellow 0.40–0.50, red < 0.40
    wr = params["win_rate"]
    if wr >= 0.50:
        wr_status = "green"
    elif wr >= 0.40:
        wr_status = "yellow"
    else:
        wr_status = "red"
    result["win_rate"] = {"status": wr_status, "value": wr}

    # Profit factor: green >= 1.5, yellow 1.0–1.5, red < 1.0
    pf = params["profit_factor"]
    if pf >= 1.5:
        pf_status = "green"
    elif pf >= 1.0:
        pf_status = "yellow"
    else:
        pf_status = "red"
    result["profit_factor"] = {"status": pf_status, "value": pf}

    # Expectancy: green >= 100, yellow 0–100, red < 0
    exp = params["expectancy"]
    if exp >= 100.0:
        exp_status = "green"
    elif exp >= 0.0:
        exp_status = "yellow"
    else:
        exp_status = "red"
    result["expectancy"] = {"status": exp_status, "value": exp}

    # Train/test sharpe divergence: green diff < 1.0, yellow 1.0–2.0, red > 2.0
    train_s = params["train_sharpe"]
    test_s = params["test_sharpe"]
    diff = abs(train_s - test_s)
    if diff < 1.0:
        tt_status = "green"
    elif diff <= 2.0:
        tt_status = "yellow"
    else:
        tt_status = "red"
    result["train_test_sharpe"] = {"status": tt_status, "value": diff}

    # Split consistency: count splits with negative test_sharpe
    splits = params.get("split_results", [])
    neg_count = sum(1 for s in splits if s.get("test_sharpe", 0) < 0)
    if neg_count == 0:
        sc_status = "green"
    elif neg_count == 1:
        sc_status = "yellow"
    else:
        sc_status = "red"
    result["split_consistency"] = {"status": sc_status, "value": neg_count}

    return result


def analyze_trades(trades_csv_path: str | Path) -> dict[str, Any]:
    """Analyze trades CSV — exit reasons, hold time stats, top winners/losers.

    Args:
        trades_csv_path: Path to trades CSV with columns:
            entry_ts, entry_price, exit_ts, exit_price, hold_bars,
            size, capital_at_entry, pnl_abs, pnl_pct, symbol, reason

    Returns:
        Dict with keys: total_trades, exit_reasons, catastrophic_pct,
        hold_bars, top_winners, top_losers.
    """
    trades: list[dict[str, Any]] = []
    with open(trades_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append({
                "entry_ts": row["entry_ts"],
                "exit_ts": row["exit_ts"],
                "hold_bars": int(row["hold_bars"]),
                "pnl_abs": float(row["pnl_abs"]),
                "pnl_pct": float(row["pnl_pct"]),
                "symbol": row["symbol"],
                "reason": row["reason"],
            })

    total = len(trades)

    # Exit reason breakdown: count, avg_pnl, pct per reason
    reason_groups: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        reason_groups.setdefault(t["reason"], []).append(t)

    exit_reasons: dict[str, dict[str, Any]] = {}
    for reason, group in reason_groups.items():
        count = len(group)
        avg_pnl = sum(t["pnl_abs"] for t in group) / count
        exit_reasons[reason] = {
            "count": count,
            "avg_pnl": avg_pnl,
            "pct": count / total if total > 0 else 0.0,
        }

    # Catastrophic stop percentage
    cat_count = exit_reasons.get("catastrophic_stop", {}).get("count", 0)
    catastrophic_pct = cat_count / total if total > 0 else 0.0

    # Hold bars stats
    hold_values = [t["hold_bars"] for t in trades]
    hold_stats = {
        "min": min(hold_values) if hold_values else 0,
        "max": max(hold_values) if hold_values else 0,
        "median": median(hold_values) if hold_values else 0,
    }

    # Top 3 winners (positive pnl, sorted desc) and all losers (negative pnl, sorted asc)
    winners = sorted(
        [t for t in trades if t["pnl_abs"] > 0],
        key=lambda t: t["pnl_abs"],
        reverse=True,
    )[:3]
    losers = sorted(
        [t for t in trades if t["pnl_abs"] < 0],
        key=lambda t: t["pnl_abs"],
    )

    return {
        "total_trades": total,
        "exit_reasons": exit_reasons,
        "catastrophic_pct": catastrophic_pct,
        "hold_bars": hold_stats,
        "top_winners": winners,
        "top_losers": losers,
    }
