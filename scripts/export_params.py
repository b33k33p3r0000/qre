#!/usr/bin/env python3
"""
Export QRE best_params to EE bot on VPS.

Interactive script: lists runs per symbol, user selects by number,
then SCP uploads to VPS.

Usage:
    python scripts/export_params.py
    python scripts/export_params.py --results-dir ~/projects/qre/results
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_RESULTS_DIR = Path.home() / "projects" / "qre" / "results"
VPS_PARAMS_DIR = "C:/trading/ee/params"
SYMBOLS = ["BTC", "SOL"]


def find_runs(symbol: str, results_dir: Path = DEFAULT_RESULTS_DIR) -> list[dict]:
    """Find all completed runs for a symbol with best_params."""
    runs = []
    for bp_path in sorted(
        results_dir.glob(f"*/{symbol}/best_params.json"), reverse=True
    ):
        try:
            with open(bp_path) as f:
                params = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        runs.append(
            {
                "path": bp_path,
                "timestamp": params.get("run_timestamp", bp_path.parent.parent.name),
                "equity": params.get("equity", 0),
                "sharpe": params.get("sharpe_time", 0),
                "trades": params.get("trades", 0),
                "win_rate": params.get("win_rate", 0),
                "mc_confidence": params.get("mc_confidence", "?"),
            }
        )
    return runs


def format_run_row(idx: int, run: dict) -> str:
    """Format a single run as a table row."""
    wr = run["win_rate"]
    wr_str = f"{wr * 100:.1f}%" if wr <= 1 else f"{wr:.1f}%"
    return (
        f"  {idx:>3}  {run['timestamp']:<32} "
        f"${run['equity']:>11,.0f} {run['sharpe']:>8.2f} "
        f"{run['trades']:>7} {wr_str:>8} {run['mc_confidence']:<6}"
    )


def display_runs(symbol: str, runs: list[dict]) -> None:
    """Display runs in a table."""
    print(f"\n{symbol} runs:")
    header = (
        f"  {'#':>3}  {'Timestamp':<32} "
        f"{'Equity':>12} {'Sharpe':>8} {'Trades':>7} {'WinRate':>8} {'MC':<6}"
    )
    print(header)
    print(
        f"  {'---':>3}  {'─' * 32} {'─' * 12} {'─' * 8} {'─' * 7} {'─' * 8} {'─' * 6}"
    )
    for i, run in enumerate(runs, 1):
        print(format_run_row(i, run))
    print(f"  {'0':>3}  [skip {symbol}]")


def select_run(symbol: str, results_dir: Path) -> Path | None:
    """Let user select a run. Returns best_params.json path or None."""
    runs = find_runs(symbol, results_dir)
    if not runs:
        print(f"\nNo completed runs found for {symbol}")
        return None

    display_runs(symbol, runs)

    while True:
        try:
            choice = int(input(f"\nSelect {symbol} run #: "))
            if choice == 0:
                return None
            if 1 <= choice <= len(runs):
                selected = runs[choice - 1]
                print(f"  → Selected: {selected['timestamp']}")
                return selected["path"]
            print(f"  Invalid choice. Enter 0-{len(runs)}")
        except ValueError:
            print("  Enter a number")
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return None


def upload_params(local_path: Path, symbol: str) -> bool:
    """SCP best_params.json to VPS. Returns True on success."""
    remote_path = f"{VPS_PARAMS_DIR}/{symbol}/best_params.json"
    scp_target = f"VPS:{remote_path}"

    print(f"  Uploading → VPS:{remote_path}")
    result = subprocess.run(
        ["scp", str(local_path), scp_target],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"  ✓ {symbol} params uploaded")
        return True
    else:
        print(f"  ✗ Upload failed: {result.stderr.strip()}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Export QRE params to EE on VPS")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="QRE results directory",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  QRE → EE Parameter Export")
    print("=" * 65)

    selected: dict[str, Path] = {}
    for symbol in SYMBOLS:
        path = select_run(symbol, args.results_dir)
        if path:
            selected[symbol] = path

    if not selected:
        print("\nNo params selected. Exiting.")
        return

    print(f"\n{'=' * 65}")
    print("  Uploading selected params to VPS")
    print(f"{'=' * 65}")

    all_ok = True
    for symbol, path in selected.items():
        if not upload_params(path, symbol):
            all_ok = False

    if not all_ok:
        print("\nSome uploads failed. Check errors above.")
        sys.exit(1)

    print(f"\n{'=' * 65}")
    restart = input("Restart EE traders on VPS? [y/N]: ").strip().lower()
    if restart == "y":
        print("Restarting EE traders...")
        for symbol in selected:
            task_name = f"ee-{symbol.lower()}"
            r = subprocess.run(
                ["ssh", "VPS", f"schtasks /run /tn {task_name}"],
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                print(f"  ✓ {task_name} restart sent")
            else:
                print(f"  ✗ {task_name} restart failed: {r.stderr.strip()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
