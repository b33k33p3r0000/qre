"""Post-run analysis pipeline — rules-based diagnostics.

Produces analysis.json + Discord embed after each optimizer run.
Design: docs/plans/2026-02-12-analyze-pipeline-design.md
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

log = logging.getLogger(__name__)

# --- Thresholds aligned with /diagnose skill (Quant Whale Strategy v3.0) ---
# MACD spread: <8 yellow, 8-18 green, >18 yellow
# RSI zone width: <30 red, 30-40 yellow, 40-55 green, >55 yellow


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
        Trades/year:       green 30–500,   yellow 30–800,    red otherwise
        Win Rate:          green >= 50%,   yellow 40%–50%,   red < 40%
        Profit Factor:     green >= 1.5,   yellow 1.0–1.5,   red < 1.0
        Expectancy:        green >= $100,  yellow $0–$100,   red < $0
        Train/Test Sharpe: green diff < 1, yellow diff 1–2,  red diff > 2
        Split Consistency: green all pos,  yellow 1 neg,     red 2+ neg
    """
    result: dict[str, dict[str, Any]] = {}

    # Sharpe: green 1.0–3.5, yellow 0.5–5.0, red outside
    # Use equity-based Sharpe if available, fallback to time-based, then legacy
    sharpe = params.get("sharpe_equity", params.get("sharpe_time", params.get("sharpe", 0)))
    result["sharpe"] = {
        "status": _classify(sharpe, (1.0, 3.5), (0.5, 5.0)),
        "value": sharpe,
    }

    # Max Drawdown (percentage, e.g. -1.43 = -1.43%)
    # green: > -5%, yellow: -5% to -10%, red: < -10%
    dd = params.get("max_drawdown", 0.0)
    if dd > -5.0:
        dd_status = "green"
    elif dd >= -10.0:
        dd_status = "yellow"
    else:
        dd_status = "red"
    result["max_drawdown"] = {"status": dd_status, "value": dd}

    # Trades per year: green 30–500, yellow 30–800, red outside
    tpy = params.get("trades_per_year", 0.0)
    result["trades_per_year"] = {
        "status": _classify(tpy, (30, 500), (30, 800)),
        "value": tpy,
    }

    # Win rate: green >= 0.50, yellow 0.40–0.50, red < 0.40
    wr = params.get("win_rate", 0.0)
    if wr >= 0.50:
        wr_status = "green"
    elif wr >= 0.40:
        wr_status = "yellow"
    else:
        wr_status = "red"
    result["win_rate"] = {"status": wr_status, "value": wr}

    # Profit factor: green >= 1.5, yellow 1.0–1.5, red < 1.0
    pf = params.get("profit_factor", 0.0)
    if pf >= 1.5:
        pf_status = "green"
    elif pf >= 1.0:
        pf_status = "yellow"
    else:
        pf_status = "red"
    result["profit_factor"] = {"status": pf_status, "value": pf}

    # Expectancy: green >= 100, yellow 0–100, red < 0
    exp = params.get("expectancy", 0.0)
    if exp >= 100.0:
        exp_status = "green"
    elif exp >= 0.0:
        exp_status = "yellow"
    else:
        exp_status = "red"
    result["expectancy"] = {"status": exp_status, "value": exp}

    # Train/test sharpe divergence: green diff < 1.0, yellow 1.0–2.0, red > 2.0
    train_s = params.get("train_sharpe_equity", params.get("train_sharpe", 0))
    test_s = params.get("test_sharpe_equity", params.get("test_sharpe", 0))
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
    """Analyze trades CSV — exit reasons, hold time, direction breakdown, top trades.

    Args:
        trades_csv_path: Path to trades CSV with columns:
            entry_ts, entry_price, exit_ts, exit_price, hold_bars,
            size, capital_at_entry, pnl_abs, pnl_pct, symbol, reason,
            direction (optional — graceful degradation if missing)

    Returns:
        Dict with keys: total_trades, exit_reasons, catastrophic_pct,
        hold_bars, min_hold_pct, top_winners, top_losers, direction_stats.
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
                "direction": row.get("direction", ""),
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

    # MIN_HOLD blocking: % trades at exactly 2 bars (MIN_HOLD_HOURS=2)
    min_hold_count = sum(1 for t in trades if t["hold_bars"] == 2)
    min_hold_pct = min_hold_count / total if total > 0 else 0.0

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

    # Direction breakdown (long/short)
    direction_stats: dict[str, dict[str, Any]] = {}
    dir_groups: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        d = t.get("direction", "")
        if d:
            dir_groups.setdefault(d, []).append(t)

    for direction, group in dir_groups.items():
        count = len(group)
        total_pnl = sum(t["pnl_abs"] for t in group)
        wins = sum(1 for t in group if t["pnl_abs"] > 0)
        direction_stats[direction] = {
            "count": count,
            "total_pnl": total_pnl,
            "win_rate": wins / count if count > 0 else 0.0,
            "avg_pnl": total_pnl / count if count > 0 else 0.0,
        }

    return {
        "total_trades": total,
        "exit_reasons": exit_reasons,
        "catastrophic_pct": catastrophic_pct,
        "hold_bars": hold_stats,
        "min_hold_pct": min_hold_pct,
        "top_winners": winners,
        "top_losers": losers,
        "direction_stats": direction_stats,
    }


def analyze_thresholds(params: dict[str, Any]) -> dict[str, Any]:
    """Analyze Quant Whale Strategy strategy params — MACD spread, RSI zones.

    Evaluates the 6 strategy parameters with green/yellow/red status
    aligned with /diagnose skill thresholds.

    Thresholds:
        MACD spread:    green 8-18, yellow <8 or >18
        RSI zone width: green 40-55, yellow 30-40 or >55, red <30

    Args:
        params: Optimizer result params containing macd_fast, macd_slow,
            macd_signal, rsi_period, rsi_lower, rsi_upper.

    Returns:
        Dict with macd_fast, macd_slow, macd_signal, macd_spread,
        macd_spread_status, rsi_period, rsi_lower, rsi_upper,
        rsi_zone_width, rsi_zone_status.
    """
    macd_fast = params.get("macd_fast", 12)
    macd_slow = params.get("macd_slow", 26)
    macd_signal = params.get("macd_signal", 9)
    macd_spread = macd_slow - macd_fast

    # MACD spread: <8 yellow, 8-18 green, >18 yellow (never red)
    macd_spread_status = _classify(macd_spread, (8, 18), (0, 9999))

    rsi_lower = params.get("rsi_lower", 30)
    rsi_upper = params.get("rsi_upper", 70)
    rsi_period = params.get("rsi_period", 14)
    rsi_zone_width = rsi_upper - rsi_lower

    # RSI zone: <30 red, 30-40 yellow, 40-55 green, >55 yellow
    rsi_zone_status = _classify(rsi_zone_width, (40, 55), (30, 9999))

    return {
        "macd_fast": macd_fast,
        "macd_slow": macd_slow,
        "macd_signal": macd_signal,
        "macd_spread": macd_spread,
        "macd_spread_status": macd_spread_status,
        "rsi_period": rsi_period,
        "rsi_lower": rsi_lower,
        "rsi_upper": rsi_upper,
        "rsi_zone_width": rsi_zone_width,
        "rsi_zone_status": rsi_zone_status,
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
    high_catastrophic = trades.get("catastrophic_pct", 0) > 0.4
    bad_rsi = thresholds.get("rsi_zone_status") in ("yellow", "red")
    bad_macd = thresholds.get("macd_spread_status") == "yellow"
    high_overfit = robustness.get("overfit_risk") == "high"

    # Rule 1: Low trades + bad RSI zones → widen RSI entry zones
    if low_trades and bad_rsi:
        suggestions.append({
            "priority": "high",
            "action": "Widen RSI entry zones (lower rsi_lower or raise rsi_upper)",
            "reason": "Too few trades per year with restrictive RSI extreme zones",
            "impact": "More signals will qualify, increasing trade frequency",
        })

    # Rule 2: Low trades + bad MACD → adjust MACD spread
    if low_trades and bad_macd:
        suggestions.append({
            "priority": "medium",
            "action": "Adjust MACD spread (target 8-18 range)",
            "reason": "MACD spread outside optimal range reduces signal quality",
            "impact": "Better crossover signals",
        })

    # Rule 3: High catastrophic_pct → adjust stops or entries
    if high_catastrophic:
        suggestions.append({
            "priority": "high",
            "action": "Tighten stop-loss or adjust entry thresholds",
            "reason": f"Catastrophic stop rate is {trades['catastrophic_pct']:.0%}, well above 40% threshold",
            "impact": "Fewer catastrophic exits, better risk management",
        })

    # Rule 4: High overfit risk → broader search ranges
    if high_overfit:
        suggestions.append({
            "priority": "high",
            "action": "Broaden Optuna search ranges to reduce overfitting",
            "reason": f"Overfit risk is {robustness['overfit_risk']}",
            "impact": "Parameters less likely to be curve-fitted to training data",
        })

    # Rule 5: Low sharpe → review MACD/RSI params
    if low_sharpe:
        suggestions.append({
            "priority": "medium",
            "action": "Review RSI and MACD parameter ranges",
            "reason": "Sharpe ratio is critically low",
            "impact": "Better signal quality may improve risk-adjusted returns",
        })

    # Rule 6: One direction losing → review RSI symmetry
    direction_stats = trades.get("direction_stats", {})
    for direction, stats in direction_stats.items():
        if stats.get("total_pnl", 0) < 0:
            suggestions.append({
                "priority": "medium",
                "action": f"Review RSI zone symmetry — {direction} trades are net negative",
                "reason": f"{direction.capitalize()} side is losing money overall",
                "impact": "Balanced long/short performance",
            })
            break

    return suggestions[:5]


# --- Status tag mapping ---
_STATUS_TAG = {"green": "[ok]", "yellow": "[!!]", "red": "[XX]"}
_STATUS_ORDER = {"green": 0, "yellow": 1, "red": 2}


def build_discord_embed(analysis: dict[str, Any]) -> str:
    """Build compact Discord embed string from analysis dict.

    Single code block format matching Optimization Completed style.
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
    sep = "=" * 30
    thin = "\u2500" * 30

    # Header
    lines.append("```")
    lines.append("RUN ANALYSIS")
    lines.append(sep)
    lines.append(f"Run:      {run_name}")
    lines.append(f"Symbol:   {symbol}")
    trials_str = f"{n_trials:,}" if isinstance(n_trials, int) else str(n_trials)
    lines.append(f"Trials:   {trials_str} \u00b7 AWF {n_splits} splits")
    lines.append(thin)
    lines.append(f"VERDICT:  {verdict}")
    lines.append(thin)

    # Health Check section — sorted: ok first, then !!, then XX
    health = analysis.get("health", {})
    if health:
        lines.append("")
        lines.append("Health Check")
        max_name = max(len(m) for m in health)
        sorted_health = sorted(
            health.items(),
            key=lambda x: _STATUS_ORDER.get(x[1].get("status", "green"), 0),
        )
        for metric, info in sorted_health:
            status = info.get("status", "green")
            tag = _STATUS_TAG.get(status, "[??]")
            value = info.get("value", "")
            lines.append(f"  {tag} {metric:<{max_name}}  {value}")

    # Top Issues — red and yellow items only
    issues = [
        (metric, info)
        for metric, info in health.items()
        if info.get("status") in ("red", "yellow")
    ]
    if issues:
        issues.sort(
            key=lambda x: _STATUS_ORDER.get(x[1].get("status", "green"), 0),
        )
        lines.append("")
        lines.append("Top Issues")
        for metric, info in issues:
            status = info.get("status", "yellow")
            tag = _STATUS_TAG.get(status, "[??]")
            value = info.get("value", "?")
            lines.append(f"  {tag} {metric} = {value}")

    # Suggestions section
    suggestions = analysis.get("suggestions", [])
    if suggestions:
        lines.append("")
        lines.append("Suggestions")
        for i, s in enumerate(suggestions, 1):
            lines.append(f"  {i}. {s.get('action', '?')}")

    lines.append("```")

    embed = "\n".join(lines)

    # Truncate to 6000 chars if needed
    if len(embed) > 6000:
        embed = embed[:5993] + "\n..."

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

    # Catastrophic stop rate: >40% red, 20-40% yellow
    cat_pct = trades.get("catastrophic_pct", 0)
    if cat_pct > 0.4:
        findings.append({
            "severity": "red",
            "metric": "catastrophic_pct",
            "value": f"{cat_pct:.0%}",
            "detail": f"Catastrophic stop rate {cat_pct:.0%} exceeds 40% threshold",
        })
    elif cat_pct > 0.2:
        findings.append({
            "severity": "yellow",
            "metric": "catastrophic_pct",
            "value": f"{cat_pct:.0%}",
            "detail": f"Catastrophic stop rate {cat_pct:.0%} is elevated (20-40%)",
        })

    # Signal exit rate: < 50% planned exits → yellow
    exit_reasons = trades.get("exit_reasons", {})
    signal_pct = exit_reasons.get("signal", {}).get("pct", 0)
    if trades.get("total_trades", 0) > 0 and signal_pct < 0.5:
        findings.append({
            "severity": "yellow",
            "metric": "signal_exit_pct",
            "value": f"{signal_pct:.0%}",
            "detail": f"Only {signal_pct:.0%} of exits are planned signals (< 50%)",
        })

    # MIN_HOLD blocking: >30% trades at exactly 2 bars
    min_hold_pct = trades.get("min_hold_pct", 0)
    if min_hold_pct > 0.3:
        findings.append({
            "severity": "yellow",
            "metric": "min_hold_blocking",
            "value": f"{min_hold_pct:.0%}",
            "detail": f"{min_hold_pct:.0%} of trades exit at MIN_HOLD_HOURS=2",
        })

    # Direction findings
    direction_stats = trades.get("direction_stats", {})
    for direction, stats in direction_stats.items():
        if stats.get("count", 0) > 0 and stats.get("win_rate", 0) < 0.3:
            findings.append({
                "severity": "red",
                "metric": f"{direction}_win_rate",
                "value": f"{stats['win_rate']:.0%}",
                "detail": f"{direction.capitalize()} win rate {stats['win_rate']:.0%} below 30%",
            })
        if stats.get("total_pnl", 0) < 0:
            findings.append({
                "severity": "yellow",
                "metric": f"{direction}_pnl",
                "value": f"${stats['total_pnl']:.2f}",
                "detail": f"{direction.capitalize()} trades net negative P&L",
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
        9. Send Discord embed if DISCORD_WEBHOOK_RUNS env var exists
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
    from qre.config import DISCORD_WEBHOOK_RUNS
    from qre.notify import discord_notify

    if DISCORD_WEBHOOK_RUNS:
        embed = build_discord_embed(analysis)
        discord_notify(embed, DISCORD_WEBHOOK_RUNS)
        log.info("analyze_run: sent Discord embed")

    log.info("analyze_run: done — verdict=%s", verdict)
    return analysis
