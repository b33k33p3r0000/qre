# src/qre/io.py
"""
QRE I/O Utilities
=================
Saving optimization results (JSON, CSV).
HTML report is in Phase 3 (report.py).
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("qre.io")


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    """Save dict as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logger.debug(f"Saved JSON: {path}")


def save_trades_csv(path: Path, trades: List[Dict[str, Any]]) -> None:
    """Save trades to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "entry_ts", "entry_price", "exit_ts", "exit_price",
            "hold_bars", "size", "capital_at_entry",
            "pnl_abs", "pnl_pct", "symbol", "reason", "direction",
        ])
        for trade in trades:
            writer.writerow([
                trade.get("entry_ts", ""),
                trade.get("entry_price", 0.0),
                trade.get("exit_ts", ""),
                trade.get("exit_price", 0.0),
                trade.get("hold_bars", 0),
                trade.get("size", 0.0),
                trade.get("capital_at_entry", 0.0),
                trade.get("pnl_abs", 0.0),
                trade.get("pnl_pct", 0.0),
                trade.get("symbol", ""),
                trade.get("reason", ""),
                trade.get("direction", ""),
            ])
    logger.debug(f"Saved trades CSV: {path}")
