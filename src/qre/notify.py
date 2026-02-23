# src/qre/notify.py
"""
QRE Discord Notifications
==========================
Start + Complete notifications only. No progress, no heartbeat.
Master Plan rule: "Notifikace = akce nebo problem."

Channels:
  #qre-runs   — start, complete, run analysis (all in one channel)
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from qre.config import DISCORD_WEBHOOK_RUNS

logger = logging.getLogger("qre.notify")


def discord_notify(msg: str, webhook_url: str, timeout: int = 8) -> bool:
    """Send message to Discord webhook. Returns True on success."""
    if not webhook_url:
        return False
    try:
        response = requests.post(
            webhook_url,
            json={"content": msg},
            timeout=timeout,
        )
        return response.status_code < 300
    except Exception as e:
        logger.warning("Discord notify failed: %s", e)
        return False


def format_start_message(
    symbol: str,
    n_trials: int,
    hours: int,
    n_splits: int,
    run_tag: str | None = None,
) -> str:
    """Format optimization start notification."""
    tag_line = f"Tag:      {run_tag}\n" if run_tag else ""
    days = hours // 24
    return (
        f"```\n"
        f"OPTIMIZATION STARTED\n"
        f"{'=' * 30}\n"
        f"Symbol:   {symbol}\n"
        f"{tag_line}"
        f"Mode:     Anchored WF\n"
        f"Trials:   {n_trials:,}\n"
        f"History:  {hours}h (~{days} days)\n"
        f"Splits:   {n_splits}\n"
        f"```"
    )


def format_complete_message(params: dict[str, Any]) -> str:
    """Format optimization completion notification."""
    symbol = params.get("symbol", "?")
    equity = params.get("equity", 0)
    pnl_pct = params.get("total_pnl_pct", 0)
    trades = params.get("trades", 0)
    win_rate = params.get("win_rate", 0)
    sharpe_time = params.get("sharpe_time", params.get("sharpe", 0))
    sharpe_equity = params.get("sharpe_equity", 0)
    max_dd = params.get("max_drawdown", 0)
    mc_conf = params.get("mc_confidence", "N/A")

    train_sharpe = params.get("train_sharpe_time", params.get("train_sharpe"))
    test_sharpe = params.get("test_sharpe_time", params.get("test_sharpe"))

    # Overfit warning
    validation_line = ""
    if train_sharpe is not None and test_sharpe is not None:
        label = "overfit" if test_sharpe < 0 else "ok"
        validation_line = f"Train/Test: {train_sharpe:.2f} / {test_sharpe:.2f} [{label}]\n"
        if test_sharpe < 0:
            validation_line += "  \u26a0 WARNING: Negative test Sharpe\n"

    return (
        f"```\n"
        f"OPTIMIZATION COMPLETED\n"
        f"{'=' * 30}\n"
        f"Symbol:   {symbol}\n"
        f"Equity:   ${equity:,.2f} ({pnl_pct:+.1f}%)\n"
        f"Trades:   {trades}\n"
        f"Win Rate: {win_rate * 100:.1f}%\n"
        f"Max DD:   {max_dd:.1f}%\n"
        f"{'─' * 30}\n"
        f"Sharpe:   {sharpe_time:.2f} (time) / {sharpe_equity:.2f} (equity)\n"
        f"MC:       {mc_conf}\n"
        f"{validation_line}"
        f"```"
    )


def notify_start(**kwargs) -> bool:
    """Send start notification to #qre-runs."""
    msg = format_start_message(**kwargs)
    return discord_notify(msg, DISCORD_WEBHOOK_RUNS)


def notify_complete(params: dict[str, Any]) -> bool:
    """Send completion notification to #qre-runs."""
    msg = format_complete_message(params)
    return discord_notify(msg, DISCORD_WEBHOOK_RUNS)
