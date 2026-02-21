#!/usr/bin/env python3
"""Regenerate HTML reports for existing runs using the current report template.

Usage:
    # Regenerate ALL runs:
    python scripts/regenerate_reports.py

    # Regenerate specific run(s):
    python scripts/regenerate_reports.py results/2026-02-18_07-07-09_final-v1

    # Dry-run (show what would be regenerated):
    python scripts/regenerate_reports.py --dry-run
"""
import csv
import json
import sys
from pathlib import Path

# Allow running from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import optuna
from qre.report import save_report


def load_optuna_history(db_path: Path) -> list[dict]:
    """Extract trial history from Optuna SQLite checkpoint."""
    storage = optuna.storages.RDBStorage(f"sqlite:///{db_path}", skip_compatibility_check=True)
    summaries = storage.get_all_studies()
    if not summaries:
        return []
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)
    return [
        {"number": t.number, "value": t.value}
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]


def regenerate_run(run_dir: Path, dry_run: bool = False) -> int:
    """Regenerate reports for all symbols in a run directory. Returns count."""
    count = 0
    # Find symbol subdirectories (contain best_params.json)
    for params_file in sorted(run_dir.glob("*/best_params.json")):
        symbol_dir = params_file.parent
        symbol = symbol_dir.name  # e.g. "BTC", "SOL"

        # Find trades CSV
        trades_csvs = list(symbol_dir.glob("trades_*.csv"))
        if not trades_csvs:
            print(f"  SKIP {symbol} — no trades CSV found")
            continue

        report_path = symbol_dir / f"report_{symbol}.html"

        if dry_run:
            print(f"  Would regenerate: {report_path.relative_to(ROOT)}")
            count += 1
            continue

        # Load params
        params = json.loads(params_file.read_text())

        # Load trades from CSV
        trades = []
        with open(trades_csvs[0], newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for key in ("entry_price", "exit_price", "size", "capital_at_entry",
                            "pnl_abs", "pnl_pct"):
                    if key in row:
                        row[key] = float(row[key])
                for key in ("hold_bars",):
                    if key in row:
                        row[key] = int(row[key])
                trades.append(row)

        # Load Optuna history if checkpoint exists
        optuna_history = None
        db_files = list(run_dir.glob(f"checkpoints/optuna_{symbol}.db"))
        if db_files:
            try:
                optuna_history = load_optuna_history(db_files[0])
            except Exception as e:
                print(f"  WARN {symbol} — Optuna history failed: {e}")

        # Generate report
        save_report(report_path, params, trades, optuna_history=optuna_history)
        print(f"  OK {report_path.relative_to(ROOT)}")
        count += 1

    return count


def main():
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    results_dir = ROOT / "results"

    if args:
        # Specific run directories
        run_dirs = [Path(a).resolve() for a in args]
    else:
        # All runs
        run_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir())

    total = 0
    for run_dir in run_dirs:
        print(f"\n{'[DRY-RUN] ' if dry_run else ''}{run_dir.name}")
        total += regenerate_run(run_dir, dry_run=dry_run)

    action = "Would regenerate" if dry_run else "Regenerated"
    print(f"\n{action} {total} report(s) across {len(run_dirs)} run(s).")


if __name__ == "__main__":
    main()
