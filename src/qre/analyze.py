"""Post-run analysis pipeline — rules-based diagnostics.

Produces analysis.json + Discord embed after each optimizer run.
Design: docs/plans/2026-02-12-analyze-pipeline-design.md
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import median
from typing import Any

# --- Threshold analysis constants ---
BUY_CAP = 0.6
SELL_CAP = 0.4
TF_LIST = ["2h", "4h", "6h", "8h", "12h", "24h"]
RSI_GATE_TFS = ["6h", "8h", "12h", "24h"]


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


def analyze_thresholds(params: dict[str, Any]) -> dict[str, Any]:
    """Analyze voting-system thresholds — dead TFs, cap collisions, MACD/RSI.

    Evaluates each timeframe's low/high thresholds for width, dead zones,
    aggressive zones, and cap collisions against BUY_CAP/SELL_CAP.

    Args:
        params: Optimizer result params containing p_buy, k_sell,
            low_<tf>, high_<tf> for each TF, MACD and RSI settings.

    Returns:
        Dict with p_buy, required_buy_votes, k_sell, tf_analysis,
        macd_spread, macd_mode, rsi_mode, rsi_gates.
    """
    p_buy = params["p_buy"]
    required_buy_votes = math.ceil(p_buy * len(TF_LIST))

    tf_analysis: dict[str, dict[str, Any]] = {}
    for tf in TF_LIST:
        low = params[f"low_{tf}"]
        high = params[f"high_{tf}"]
        width = high - low
        dead = width > 0.8
        aggressive = width < 0.3
        buy_cap_collision = low > BUY_CAP
        sell_cap_collision = high < SELL_CAP
        effective_low = min(low, BUY_CAP) if buy_cap_collision else low
        effective_high = max(high, SELL_CAP) if sell_cap_collision else high

        tf_analysis[tf] = {
            "low": low,
            "high": high,
            "width": width,
            "dead": dead,
            "aggressive": aggressive,
            "buy_cap_collision": buy_cap_collision,
            "sell_cap_collision": sell_cap_collision,
            "effective_low": effective_low,
            "effective_high": effective_high,
        }

    macd_fast = params["macd_fast"]
    macd_slow = params["macd_slow"]

    rsi_gates: dict[str, float] = {}
    for tf in RSI_GATE_TFS:
        key = f"rsi_gate_{tf}"
        if key in params:
            rsi_gates[tf] = params[key]

    return {
        "p_buy": p_buy,
        "required_buy_votes": required_buy_votes,
        "k_sell": params["k_sell"],
        "tf_analysis": tf_analysis,
        "macd_spread": macd_slow - macd_fast,
        "macd_mode": params["macd_mode"],
        "rsi_mode": params["rsi_mode"],
        "rsi_gates": rsi_gates,
    }


def check_robustness(params: dict[str, Any]) -> dict[str, Any]:
    """Evaluate optimization robustness -- overfit, splits, Monte Carlo.

    Computes overfit score from train/test sharpe divergence,
    counts positive splits, and passes through Monte Carlo results.

    Args:
        params: Optimizer result params containing train_sharpe, test_sharpe,
            split_results, and optionally mc_sharpe_mean, mc_confidence.

    Returns:
        Dict with train_sharpe, test_sharpe, sharpe_diff, overfit_score,
        overfit_risk, splits_positive, splits_total, splits_pct_positive,
        split_details, mc_sharpe_mean, mc_confidence.
    """
    train_s = params.get("train_sharpe", 0)
    test_s = params.get("test_sharpe", 0)
    overfit_score = (train_s - test_s) / train_s if train_s != 0 else 0
    overfit_risk = "high" if overfit_score > 0.5 else ("medium" if overfit_score > 0.3 else "low")

    splits = params.get("split_results", [])
    pos = sum(1 for s in splits if s.get("test_sharpe", 0) > 0)

    return {
        "train_sharpe": train_s,
        "test_sharpe": test_s,
        "sharpe_diff": abs(train_s - test_s),
        "overfit_score": overfit_score,
        "overfit_risk": overfit_risk,
        "splits_positive": pos,
        "splits_total": len(splits),
        "splits_pct_positive": pos / len(splits) if splits else 0,
        "split_details": splits,
        "mc_sharpe_mean": params.get("mc_sharpe_mean"),
        "mc_confidence": params.get("mc_confidence"),
    }
