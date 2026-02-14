"""Post-run analysis pipeline — rules-based diagnostics.

Produces analysis.json + Discord embed after each optimizer run.
Design: docs/plans/2026-02-12-analyze-pipeline-design.md
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

log = logging.getLogger(__name__)

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


def compute_verdict(health: dict[str, dict[str, Any]]) -> str:
    """Compute overall verdict from health check results.

    Rules:
        2+ reds  → FAIL
        1 red OR 3+ yellows → REVIEW
        else → PASS

    Args:
        health: Dict from health_check(), mapping metric name to
            {"status": "green"|"yellow"|"red", "value": ...}.

    Returns:
        "PASS", "REVIEW", or "FAIL".
    """
    reds = sum(1 for m in health.values() if m.get("status") == "red")
    yellows = sum(1 for m in health.values() if m.get("status") == "yellow")

    if reds >= 2:
        return "FAIL"
    if reds >= 1 or yellows >= 3:
        return "REVIEW"
    return "PASS"


def generate_suggestions(
    health: dict[str, dict[str, Any]],
    thresholds: dict[str, Any],
    trades: dict[str, Any],
    robustness: dict[str, Any],
) -> list[dict[str, str]]:
    """Generate actionable suggestions by cross-referencing analysis results.

    Returns max 5 suggestions. Each suggestion has keys:
        priority: "high" or "medium"
        action: what to do
        reason: why
        impact: expected effect

    Args:
        health: Dict from health_check().
        thresholds: Dict from analyze_thresholds().
        trades: Dict from analyze_trades().
        robustness: Dict from check_robustness().
    """
    suggestions: list[dict[str, str]] = []

    low_trades = health.get("trades_per_year", {}).get("status") == "red"
    low_sharpe = health.get("sharpe", {}).get("status") == "red"
    high_p_buy = thresholds.get("required_buy_votes", 0) >= 3
    high_catastrophic = trades.get("catastrophic_pct", 0) > 0.4
    macd_crossover = thresholds.get("macd_mode") == "crossover"
    high_overfit = robustness.get("overfit_risk") == "high"

    # Rule 1: Low trades + high p_buy → lower p_buy
    if low_trades and high_p_buy:
        suggestions.append({
            "priority": "high",
            "action": "Lower p_buy to reduce required buy votes",
            "reason": "Too few trades per year with strict voting threshold",
            "impact": "More signals will pass the vote, increasing trade frequency",
        })

    # Rule 2: High catastrophic_pct → adjust thresholds/stops
    if high_catastrophic:
        suggestions.append({
            "priority": "high",
            "action": "Tighten stop-loss or adjust entry thresholds",
            "reason": f"Catastrophic stop rate is {trades['catastrophic_pct']:.0%}, well above 40% threshold",
            "impact": "Fewer catastrophic exits, better risk management",
        })

    # Rule 3: Strict MACD mode + low trades → switch to rising
    if macd_crossover and low_trades:
        suggestions.append({
            "priority": "medium",
            "action": "Switch MACD mode from crossover to rising",
            "reason": "Crossover mode is restrictive and contributes to low trade count",
            "impact": "More MACD signals, potentially more trades",
        })

    # Rule 4: High overfit risk → broader search ranges
    if high_overfit:
        suggestions.append({
            "priority": "high",
            "action": "Broaden Optuna search ranges to reduce overfitting",
            "reason": f"Overfit risk is {robustness['overfit_risk']}",
            "impact": "Parameters less likely to be curve-fitted to training data",
        })

    # Rule 5: Low sharpe → suggest RSI/MACD changes
    if low_sharpe:
        suggestions.append({
            "priority": "medium",
            "action": "Review RSI and MACD parameter ranges",
            "reason": "Sharpe ratio is critically low",
            "impact": "Better signal quality may improve risk-adjusted returns",
        })

    return suggestions[:5]


# --- Status emoji mapping ---
_STATUS_EMOJI = {"green": "\U0001f7e2", "yellow": "\U0001f7e1", "red": "\U0001f534"}
_VERDICT_EMOJI = {"PASS": "\u2705", "REVIEW": "\U0001f7e1", "FAIL": "\u274c"}


def build_discord_embed(analysis: dict[str, Any]) -> str:
    """Build compact Discord embed string from analysis dict.

    Max 6000 chars for Discord webhook compatibility.

    Args:
        analysis: Full analysis dict with keys: run_name, symbol,
            n_trials, n_splits, verdict, health, suggestions.

    Returns:
        Formatted string for Discord embed.
    """
    lines: list[str] = []

    run_name = analysis.get("run_name", "unknown")
    symbol = analysis.get("symbol", "?")
    n_trials = analysis.get("n_trials", "?")
    n_splits = analysis.get("n_splits", "?")
    verdict = analysis.get("verdict", "?")

    lines.append(f"\U0001f4ca RUN ANALYSIS: {run_name}")
    lines.append(f"{symbol} \u00b7 {n_trials} trials \u00b7 AWF {n_splits} splits")
    lines.append("")

    verdict_emoji = _VERDICT_EMOJI.get(verdict, "?")
    lines.append(f"VERDICT: {verdict_emoji} {verdict}")
    lines.append("")

    # Health Check section
    health = analysis.get("health", {})
    if health:
        lines.append("Health Check")
        for metric, info in health.items():
            status = info.get("status", "?")
            emoji = _STATUS_EMOJI.get(status, "\u2753")
            lines.append(f"{emoji} {metric}: {status}")
        lines.append("")

    # Top Issues — red and yellow items
    issues = [
        (metric, info)
        for metric, info in health.items()
        if info.get("status") in ("red", "yellow")
    ]
    if issues:
        lines.append("Top Issues")
        for metric, info in issues:
            status = info.get("status", "?")
            emoji = _STATUS_EMOJI.get(status, "\u2753")
            value = info.get("value", "?")
            lines.append(f"{emoji} {metric} = {value}")
        lines.append("")

    # Suggestions section
    suggestions = analysis.get("suggestions", [])
    if suggestions:
        lines.append("Suggestions")
        for i, s in enumerate(suggestions, 1):
            lines.append(f"{i}. {s.get('action', '?')}")
        lines.append("")

    embed = "\n".join(lines)

    # Truncate to 6000 chars if needed
    if len(embed) > 6000:
        embed = embed[:5997] + "..."

    return embed


def save_analysis(analysis: dict[str, Any], path: str | Path) -> None:
    """Save analysis dict to JSON file with added timestamp.

    Args:
        analysis: Full analysis dict.
        path: Output file path.
    """
    output = dict(analysis)
    output["timestamp"] = datetime.now(UTC).isoformat()

    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# --- Orchestrator helpers ---


def _find_symbol_dir(run_dir: str | Path) -> Path:
    """Find the symbol subdirectory inside a run directory.

    Skips 'checkpoints' dir. Looks for a subdirectory containing
    best_params.json.

    Args:
        run_dir: Path to the run directory (e.g. results/2026-02-14_12-00-00/).

    Returns:
        Path to the symbol subdirectory.

    Raises:
        FileNotFoundError: If no symbol directory with best_params.json is found.
    """
    run_path = Path(run_dir)
    for child in sorted(run_path.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "checkpoints":
            continue
        if (child / "best_params.json").exists():
            return child
    raise FileNotFoundError(f"No symbol directory with best_params.json in {run_dir}")


def _build_findings(
    health: dict[str, dict[str, Any]],
    trades: dict[str, Any],
    thresholds: dict[str, Any],
    robustness: dict[str, Any],
) -> list[dict[str, str]]:
    """Collect red/yellow health items + high catastrophic + overfit into findings list.

    Each finding has keys: severity, metric, value, detail.

    Args:
        health: Dict from health_check().
        trades: Dict from analyze_trades().
        thresholds: Dict from analyze_thresholds().
        robustness: Dict from check_robustness().

    Returns:
        List of finding dicts sorted by severity (red first, then yellow).
    """
    findings: list[dict[str, str]] = []

    # Collect reds and yellows from health
    for metric, info in health.items():
        status = info.get("status", "green")
        if status in ("red", "yellow"):
            findings.append({
                "severity": status,
                "metric": metric,
                "value": str(info.get("value", "?")),
                "detail": f"{metric} is {status}",
            })

    # High catastrophic stop rate
    cat_pct = trades.get("catastrophic_pct", 0)
    if cat_pct > 0.4:
        findings.append({
            "severity": "red",
            "metric": "catastrophic_pct",
            "value": f"{cat_pct:.0%}",
            "detail": f"Catastrophic stop rate {cat_pct:.0%} exceeds 40% threshold",
        })

    # High overfit risk
    if robustness.get("overfit_risk") == "high":
        findings.append({
            "severity": "red",
            "metric": "overfit_risk",
            "value": f"{robustness.get('overfit_score', 0):.2f}",
            "detail": f"Overfit risk is high (score {robustness.get('overfit_score', 0):.2f})",
        })

    # Sort: red first, then yellow
    severity_order = {"red": 0, "yellow": 1}
    findings.sort(key=lambda f: severity_order.get(f["severity"], 2))

    return findings


def analyze_run(run_dir: str | Path) -> dict[str, Any]:
    """Main orchestrator — run full analysis pipeline on an optimizer run.

    Steps:
        1. Find symbol subdirectory (skip 'checkpoints', look for best_params.json)
        2. Load best_params.json
        3. Find trades CSV (glob trades_*.csv)
        4. Run: health_check, analyze_trades, analyze_thresholds, check_robustness
        5. Compute verdict from health
        6. Build findings list
        7. Generate suggestions
        8. Save analysis.json to symbol dir
        9. Send Discord embed if DISCORD_WEBHOOK_ALERTS env var exists
        10. Return full analysis dict

    Args:
        run_dir: Path to the run directory (e.g. results/2026-02-14_12-00-00/).

    Returns:
        Full analysis dict with all results.
    """
    run_path = Path(run_dir)
    log.info("analyze_run: starting analysis of %s", run_path)

    # 1. Find symbol dir
    symbol_dir = _find_symbol_dir(run_path)
    log.info("analyze_run: found symbol dir %s", symbol_dir.name)

    # 2. Load best_params.json
    params_path = symbol_dir / "best_params.json"
    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)

    # 3. Find trades CSV
    trades_csvs = list(symbol_dir.glob("trades_*.csv"))
    if not trades_csvs:
        raise FileNotFoundError(f"No trades CSV found in {symbol_dir}")
    trades_csv_path = trades_csvs[0]

    # 4. Run analysis functions
    health = health_check(params)
    trades = analyze_trades(trades_csv_path)
    thresholds = analyze_thresholds(params)
    robustness = check_robustness(params)

    # 5. Compute verdict
    verdict = compute_verdict(health)

    # 6. Build findings
    findings = _build_findings(health, trades, thresholds, robustness)

    # 7. Generate suggestions
    suggestions = generate_suggestions(health, thresholds, trades, robustness)

    # 8. Build full analysis dict
    analysis: dict[str, Any] = {
        "run_name": run_path.name,
        "symbol": params.get("symbol", symbol_dir.name),
        "n_trials": params.get("n_trials", "?"),
        "n_splits": params.get("n_splits", "?"),
        "verdict": verdict,
        "health": health,
        "trades": trades,
        "thresholds": thresholds,
        "robustness": robustness,
        "findings": findings,
        "suggestions": suggestions,
    }

    # 9. Save analysis.json
    save_analysis(analysis, symbol_dir / "analysis.json")
    log.info("analyze_run: saved analysis.json → %s", symbol_dir / "analysis.json")

    # 10. Discord notification
    webhook_url = os.environ.get("DISCORD_WEBHOOK_ALERTS", "")
    if webhook_url:
        from qre.notify import discord_notify

        embed = build_discord_embed(analysis)
        discord_notify(embed, webhook_url)
        log.info("analyze_run: sent Discord embed")

    log.info("analyze_run: done — verdict=%s", verdict)
    return analysis
