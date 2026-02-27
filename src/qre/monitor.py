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

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


ACTIVE_RUN_MAX_AGE = 300  # seconds — DB mtime threshold


@dataclass
class SymbolStats:
    symbol: str
    completed: int = 0
    pruned: int = 0
    failed: int = 0
    n_trials_requested: int | None = None
    best_value: float | None = None
    best_trial_number: int | None = None
    best_params: dict = field(default_factory=dict)
    user_attrs: dict = field(default_factory=dict)
    start_time: str | None = None
    trials_per_min: float | None = None
    eta_minutes: float | None = None
    warm_start_source: str | None = None


def find_active_runs(
    results_dir: Path,
    max_age_seconds: int = ACTIVE_RUN_MAX_AGE,
    name_filter: str | None = None,
) -> list[dict]:
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


def query_db_stats(db_path: Path) -> SymbolStats | None:
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
        cur.execute("SELECT state, COUNT(*) as cnt FROM trials GROUP BY state")  # noqa: S608
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

        # Warm-start source from study user_attrs
        cur.execute(
            "SELECT value_json FROM study_user_attributes WHERE key = 'warm_start_source' LIMIT 1"
        )
        row = cur.fetchone()
        if row:
            try:
                stats.warm_start_source = json.loads(row["value_json"])
            except (json.JSONDecodeError, TypeError):
                stats.warm_start_source = row["value_json"]

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
                total_done = stats.completed + stats.pruned + stats.failed
                if elapsed_min > 0 and total_done > 0:
                    stats.trials_per_min = round(total_done / elapsed_min, 1)
                    if stats.n_trials_requested:
                        remaining = stats.n_trials_requested - total_done
                        if remaining > 0 and stats.trials_per_min > 0:
                            stats.eta_minutes = round(remaining / stats.trials_per_min, 1)
            except (ValueError, TypeError):
                pass

        return stats
    except Exception:
        return None
    finally:
        conn.close()


def format_params(params: dict) -> dict[str, str]:
    """Format strategy params into separate lines for MACD, RSI, Trend.

    Returns dict with keys: "macd", "rsi", "trend".
    """
    mf = params.get("macd_fast", "?")
    ms = params.get("macd_slow", "?")
    msig = params.get("macd_signal", "?")
    rp = params.get("rsi_period", "?")
    rl = params.get("rsi_lower", "?")
    ru = params.get("rsi_upper", "?")
    rlb = params.get("rsi_lookback", "?")
    ttf = params.get("trend_tf", "?")

    # Format macd_fast: float → 1 decimal, int → as-is
    if isinstance(mf, float):
        mf = f"{mf:.1f}"
    else:
        mf = str(int(mf)) if isinstance(mf, (int, float)) else str(mf)

    # trend_tf: Optuna stores categorical as index (0=4h, 1=8h, 2=1d)
    tf_map = {0: "4h", 1: "8h", 2: "1d"}
    if ttf in tf_map:
        ttf = tf_map[ttf]

    ms = int(ms) if isinstance(ms, float) else ms
    msig = int(msig) if isinstance(msig, float) else msig
    rp = int(rp) if isinstance(rp, float) else rp
    rl = int(rl) if isinstance(rl, float) else rl
    ru = int(ru) if isinstance(ru, float) else ru
    rlb = int(rlb) if isinstance(rlb, float) else rlb

    return {
        "macd": f"{mf} / {ms} / {msig}",
        "rsi": f"{rp} [{rl}-{ru}] lb={rlb}",
        "trend": str(ttf),
    }


def render_symbol_panel(stats: SymbolStats, prev_best: float | None = None) -> Panel:
    """Render a Rich Panel for one symbol's stats.

    Uses a borderless Rich Table for aligned label-value layout.
    """
    total = stats.completed + stats.pruned + stats.failed
    requested = stats.n_trials_requested or total

    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 1))
    table.add_column("label", style="dim", justify="right", min_width=12)
    table.add_column("value", justify="left")

    # — Progress section —
    table.add_row("Progress", f"{stats.completed:,} / {requested:,}")
    rate_str = f"{stats.trials_per_min} t/min" if stats.trials_per_min else "..."
    table.add_row("Speed", rate_str)
    if stats.eta_minutes:
        table.add_row("ETA", f"~{int(stats.eta_minutes)} min")
    if stats.warm_start_source:
        table.add_row("Warm start", f"[yellow]{stats.warm_start_source}[/yellow]")

    table.add_row("", "")  # section separator

    # — Best trial section —
    trial_num = f"#{stats.best_trial_number:,}" if stats.best_trial_number is not None else "?"
    table.add_row("Best trial", trial_num)

    is_new = prev_best is not None and stats.best_value is not None and stats.best_value > prev_best
    new_marker = "  [bold green]NEW[/bold green]" if is_new else ""
    val = f"{stats.best_value:.4f}" if stats.best_value is not None else "—"
    table.add_row("Log Calmar", f"{val}{new_marker}")

    table.add_row("", "")  # section separator

    # — Metrics section —
    ua = stats.user_attrs
    if ua:
        sharpe = ua.get("sharpe_equity", "—")
        dd = ua.get("max_drawdown", "—")
        pnl = ua.get("total_pnl_pct", "—")
        trades = ua.get("trades", "—")
        tpy = ua.get("trades_per_year", "—")

        sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else str(sharpe)
        dd_str = f"{dd:.1f}%" if isinstance(dd, (int, float)) else str(dd)
        if isinstance(pnl, (int, float)):
            pnl_str = f"+{pnl:.1f}%" if pnl > 0 else f"{pnl:.1f}%"
        else:
            pnl_str = str(pnl)

        table.add_row("Sharpe (eq)", sharpe_str)
        table.add_row("Max DD", dd_str)
        table.add_row("P&L", pnl_str)
        table.add_row("Trades", str(trades))
        table.add_row("Trades/yr", str(tpy))
    else:
        table.add_row("", "[dim](no metrics)[/dim]")

    # — Params section —
    if stats.best_params:
        table.add_row("", "")  # section separator
        p = format_params(stats.best_params)
        table.add_row("MACD", p["macd"])
        table.add_row("RSI", p["rsi"])
        table.add_row("Trend", p["trend"])

    return Panel(table, title=f"[bold]{stats.symbol}[/bold]", border_style="cyan")


def render_dashboard(
    console: Console,
    all_runs: list[dict],
    prev_bests: dict[str, float],
) -> dict[str, float]:
    """Render the full dashboard. Returns updated prev_bests dict."""
    console.clear()
    new_bests = {}

    if not all_runs:
        console.print("[yellow]No active runs detected. Waiting...[/yellow]\n")
        return prev_bests

    for run in all_runs:
        console.print(f"[bold white]{run['run_name']}[/bold white]")
        for db_path in run["db_files"]:
            stats = query_db_stats(db_path)
            if stats is None:
                console.print(f"  [red]Error reading {db_path.name}[/red]")
                continue

            key = f"{run['run_name']}:{stats.symbol}"
            prev = prev_bests.get(key)
            panel = render_symbol_panel(stats, prev_best=prev)
            console.print(panel)

            if stats.best_value is not None:
                new_bests[key] = stats.best_value

        console.print()

    now = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim]Last refresh: {now}   Auto-refresh: 30s   Ctrl+C to exit[/dim]")

    merged = {**prev_bests, **new_bests}
    return merged


def main():
    """CLI entry point for live monitor."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="QRE Live Monitor — real-time optimizer dashboard")
    parser.add_argument("filter", nargs="?", default=None, help="Partial run name filter")
    default_results = Path(__file__).resolve().parent.parent.parent / "results"
    parser.add_argument("--results-dir", type=str, default=str(default_results), help="Results directory path")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds")
    args = parser.parse_args()

    console = Console()
    results_dir = Path(args.results_dir)
    prev_bests: dict[str, float] = {}

    console.print(f"[bold]QRE Live Monitor[/bold] — watching {results_dir.resolve()}")
    if args.filter:
        console.print(f"Filter: [cyan]{args.filter}[/cyan]")
    console.print(f"Refresh: every {args.interval}s\n")

    try:
        while True:
            runs = find_active_runs(results_dir, name_filter=args.filter)
            prev_bests = render_dashboard(console, runs, prev_bests)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
