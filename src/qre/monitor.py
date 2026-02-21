"""
QRE Live Monitor
================
Real-time TUI dashboard for running optimizer trials.

Usage:
    python -m qre.monitor                    # auto-detect active runs
    python -m qre.monitor calmar-btc-v3      # filter by run name
    python -m qre.monitor --results-dir /path/to/results

Reads Optuna SQLite checkpoint DBs in read-only mode.
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


ACTIVE_RUN_MAX_AGE = 300  # seconds — DB mtime threshold


@dataclass
class SymbolStats:
    symbol: str
    completed: int = 0
    pruned: int = 0
    failed: int = 0
    n_trials_requested: Optional[int] = None
    best_value: Optional[float] = None
    best_trial_number: Optional[int] = None
    best_params: Dict = field(default_factory=dict)
    user_attrs: Dict = field(default_factory=dict)
    start_time: Optional[str] = None
    trials_per_min: Optional[float] = None
    eta_minutes: Optional[float] = None


def find_active_runs(
    results_dir: Path,
    max_age_seconds: int = ACTIVE_RUN_MAX_AGE,
    name_filter: Optional[str] = None,
) -> List[dict]:
    """Find optimizer runs with recently modified checkpoint DBs.

    Returns list of dicts: {"run_name": str, "db_files": [Path, ...]}
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    now = time.time()
    runs = {}

    for db_path in sorted(results_dir.glob("*/checkpoints/optuna_*.db")):
        run_name = db_path.parent.parent.name

        if name_filter and name_filter.lower() not in run_name.lower():
            continue

        age = now - db_path.stat().st_mtime
        if age > max_age_seconds:
            continue

        if run_name not in runs:
            runs[run_name] = {"run_name": run_name, "db_files": []}
        runs[run_name]["db_files"].append(db_path)

    return list(runs.values())


def query_db_stats(db_path: Path) -> Optional[SymbolStats]:
    """Query Optuna SQLite DB for trial statistics.

    Opens DB in read-only mode to avoid interfering with running optimizer.
    Returns None if DB is corrupted or unreadable.
    """
    symbol = db_path.stem.replace("optuna_", "")  # "optuna_BTC.db" → "BTC"

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
    except Exception:
        return None

    try:
        stats = SymbolStats(symbol=symbol)

        # Trial counts by state
        cur.execute("SELECT state, COUNT(*) as cnt FROM trials GROUP BY state")
        for row in cur.fetchall():
            if row["state"] == "COMPLETE":
                stats.completed = row["cnt"]
            elif row["state"] == "PRUNED":
                stats.pruned = row["cnt"]
            elif row["state"] == "FAIL":
                stats.failed = row["cnt"]

        # n_trials_requested from study user_attrs
        cur.execute(
            "SELECT value_json FROM study_user_attributes WHERE key = 'n_trials_requested' LIMIT 1"
        )
        row = cur.fetchone()
        if row:
            val = row["value_json"]
            try:
                stats.n_trials_requested = int(json.loads(val))
            except (json.JSONDecodeError, TypeError, ValueError):
                try:
                    stats.n_trials_requested = int(val)
                except (TypeError, ValueError):
                    pass

        # Best trial
        cur.execute("""
            SELECT t.trial_id, t.number, tv.value
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE'
            ORDER BY tv.value DESC
            LIMIT 1
        """)
        best_row = cur.fetchone()
        if best_row:
            stats.best_value = best_row["value"]
            stats.best_trial_number = best_row["number"]
            best_trial_id = best_row["trial_id"]

            # Best trial params
            cur.execute(
                "SELECT param_name, param_value FROM trial_params WHERE trial_id = ?",
                (best_trial_id,),
            )
            stats.best_params = {row["param_name"]: row["param_value"] for row in cur.fetchall()}

            # Best trial user_attrs
            cur.execute(
                "SELECT key, value_json FROM trial_user_attributes WHERE trial_id = ?",
                (best_trial_id,),
            )
            for row in cur.fetchall():
                try:
                    stats.user_attrs[row["key"]] = json.loads(row["value_json"])
                except (json.JSONDecodeError, TypeError):
                    stats.user_attrs[row["key"]] = row["value_json"]

        # Start time and trials/min
        cur.execute("SELECT MIN(datetime_start) as first_start FROM trials")
        row = cur.fetchone()
        if row and row["first_start"]:
            stats.start_time = row["first_start"]
            try:
                start_dt = datetime.fromisoformat(row["first_start"])
                elapsed_min = (datetime.now() - start_dt).total_seconds() / 60.0
                if elapsed_min > 0 and stats.completed > 0:
                    stats.trials_per_min = round(stats.completed / elapsed_min, 1)
                    if stats.n_trials_requested:
                        remaining = stats.n_trials_requested - stats.completed
                        if remaining > 0 and stats.trials_per_min > 0:
                            stats.eta_minutes = round(remaining / stats.trials_per_min, 1)
            except (ValueError, TypeError):
                pass

        conn.close()
        return stats
    except Exception:
        conn.close()
        return None
