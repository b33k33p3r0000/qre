"""
Autonomous Optimizer — Evaluation Logic & State Management
==========================================================
Core module for metric comparison, verdict computation,
TOP tier detection, and state file I/O.
"""

from __future__ import annotations

import enum
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# SymbolVerdict enum
# =============================================================================


class SymbolVerdict(str, enum.Enum):
    """Per-symbol optimization verdict."""

    BETTER = "BETTER"
    WORSE = "WORSE"
    NEUTRAL = "NEUTRAL"


# =============================================================================
# RED zone detection
# =============================================================================


def is_red(metrics: dict) -> bool:
    """Check if ANY metric is in RED zone.

    RED thresholds:
        - log_calmar < 1.0
        - sharpe_equity < 1.5
        - abs(max_drawdown) > 12.0
        - trades_per_year < 30 or > 300
        - mc_confidence == "LOW"
        - test_sharpe_equity < 0
        - total_pnl_pct < 50.0
    """
    if metrics.get("log_calmar", 0) < 1.0:
        return True
    if metrics.get("sharpe_equity", 0) < 1.5:
        return True
    if abs(metrics.get("max_drawdown", 0)) > 12.0:
        return True
    tpy = metrics.get("trades_per_year", 0)
    if tpy < 30 or tpy > 300:
        return True
    if metrics.get("mc_confidence", "") == "LOW":
        return True
    if metrics.get("test_sharpe_equity", 0) < 0:
        return True
    if metrics.get("total_pnl_pct", 0) < 50.0:
        return True
    return False


# =============================================================================
# Per-symbol comparison
# =============================================================================


def compare_symbol(curr: dict, prev: dict) -> SymbolVerdict:
    """Compare current vs previous metrics for a single symbol.

    Returns:
        WORSE  = calmar_change < -3% OR new RED metric OR PnL drop > 20%
        BETTER = calmar_change > +1.5% AND no RED AND PnL drop < 10%
        NEUTRAL = neither
    """
    prev_calmar = prev.get("log_calmar", 0)
    curr_calmar = curr.get("log_calmar", 0)

    # Calmar relative change (avoid division by zero)
    if prev_calmar != 0:
        calmar_change = (curr_calmar - prev_calmar) / abs(prev_calmar)
    else:
        calmar_change = 0.0 if curr_calmar == 0 else 1.0

    # PnL relative drop
    prev_pnl = prev.get("total_pnl_pct", 0)
    curr_pnl = curr.get("total_pnl_pct", 0)
    if prev_pnl != 0:
        pnl_change = (curr_pnl - prev_pnl) / abs(prev_pnl)
    else:
        pnl_change = 0.0

    # New RED metric detection
    curr_is_red = is_red(curr)
    prev_is_red = is_red(prev)
    new_red = curr_is_red and not prev_is_red

    # WORSE conditions
    if calmar_change < -0.03:
        return SymbolVerdict.WORSE
    if new_red:
        return SymbolVerdict.WORSE
    if pnl_change < -0.20:
        return SymbolVerdict.WORSE

    # BETTER conditions
    if calmar_change > 0.015 and not curr_is_red and pnl_change >= -0.10:
        return SymbolVerdict.BETTER

    return SymbolVerdict.NEUTRAL


# =============================================================================
# Overall verdict (across all symbols)
# =============================================================================


def overall_verdict(verdicts: dict[str, SymbolVerdict]) -> str:
    """Aggregate per-symbol verdicts into overall verdict.

    Returns:
        "WORSE"   = majority WORSE, OR worse_count >= better_count when worse > 0
        "BETTER"  = majority BETTER and none WORSE
        "NEUTRAL" = everything else
    """
    total = len(verdicts)
    worse_count = sum(1 for v in verdicts.values() if v == SymbolVerdict.WORSE)
    better_count = sum(1 for v in verdicts.values() if v == SymbolVerdict.BETTER)

    # WORSE: majority worse, OR any worse that matches/exceeds better count
    if worse_count > total / 2:
        return "WORSE"
    if worse_count > 0 and worse_count >= better_count:
        return "WORSE"

    # BETTER: majority better and none worse
    if better_count > total / 2 and worse_count == 0:
        return "BETTER"

    return "NEUTRAL"


# =============================================================================
# TOP tier detection
# =============================================================================


def is_top_tier(metrics: dict) -> bool:
    """Check if ALL metrics are in TOP zone.

    TOP thresholds:
        - log_calmar > 2.0
        - sharpe_equity > 2.5
        - abs(max_drawdown) < 5.0
        - trades_per_year 80-150
        - mc_confidence == "HIGH"
        - test_sharpe > 0.8 * train_sharpe
        - total_pnl_pct > 150
    """
    if metrics.get("log_calmar", 0) <= 2.0:
        return False
    if metrics.get("sharpe_equity", 0) <= 2.5:
        return False
    if abs(metrics.get("max_drawdown", 0)) >= 5.0:
        return False
    tpy = metrics.get("trades_per_year", 0)
    if tpy < 80 or tpy > 150:
        return False
    if metrics.get("mc_confidence", "") != "HIGH":
        return False
    test_sharpe = metrics.get("test_sharpe_equity", 0)
    train_sharpe = metrics.get("train_sharpe_equity", 0)
    if test_sharpe < 0.8 * train_sharpe:
        return False
    if metrics.get("total_pnl_pct", 0) <= 150:
        return False
    return True


def check_top_verdict(
    all_metrics: dict[str, dict],
    iteration_history: list[dict],
) -> bool:
    """Check if we should stop: all symbols TOP AND last 2 iterations NEUTRAL."""
    # All symbols must be TOP tier
    for symbol_metrics in all_metrics.values():
        if not is_top_tier(symbol_metrics):
            return False

    # Need at least 2 iterations of NEUTRAL history
    if len(iteration_history) < 2:
        return False

    last_two = iteration_history[-2:]
    return all(
        entry.get("overall_verdict") == "NEUTRAL" for entry in last_two
    )


# =============================================================================
# State I/O
# =============================================================================


def load_iteration_log(path: Path) -> list[dict]:
    """Load iteration log from JSON file. Returns [] if file is missing."""
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_iteration_log(entries: list[dict], path: Path) -> None:
    """Save iteration log to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def load_config(path: Path) -> dict:
    """Load autonomous config from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_config(config: dict, path: Path) -> None:
    """Save autonomous config to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)


def append_changelog(
    path: Path,
    iteration: int,
    branch: str,
    change: str,
    reason: str,
    run_dir: str,
    preset: str,
    pairs: list[str],
    metrics: dict,
    prev_metrics: Optional[dict],
    verdict: str,
) -> None:
    """Append an entry to the changelog markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pairs_str = ", ".join(pairs)

    lines = [
        f"## Iteration: {iteration}",
        f"- **Timestamp:** {timestamp}",
        f"- **Branch:** {branch}",
        f"- **Preset:** {preset}",
        f"- **Pairs:** {pairs_str}",
        f"- **Change:** {change}",
        f"- **Reason:** {reason}",
        f"- **Run dir:** {run_dir}",
        f"- **Verdict:** {verdict}",
        "",
        "### Metrics",
        f"```json",
        json.dumps(metrics, indent=2, default=str),
        "```",
        "",
    ]

    if prev_metrics is not None:
        lines.extend([
            "### Previous Metrics",
            "```json",
            json.dumps(prev_metrics, indent=2, default=str),
            "```",
            "",
        ])

    lines.append("---\n\n")

    entry = "\n".join(lines)

    with open(path, "a") as f:
        f.write(entry)
