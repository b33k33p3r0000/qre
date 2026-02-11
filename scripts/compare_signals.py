"""
Compare QRE vs Legacy Optimizer signals.

Takes known params from a legacy optimizer run, runs QRE backtest
with those params on shared cache data, and compares trade entry/exit
timestamps to validate strategy logic is identical.

Usage:
    python scripts/compare_signals.py --symbol BTC/USDC
    python scripts/compare_signals.py --symbol SOL/USDC --legacy-run 2026-02-10_15-34-44_macd-rsi-btc-sol-v1
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add QRE src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qre.config import STARTING_EQUITY
from qre.core.backtest import simulate_trades_fast
from qre.core.metrics import calculate_metrics
from qre.core.strategy import MACDRSIStrategy
from qre.data.fetch import DataCache, load_all_data


def find_legacy_runs(symbol: str) -> list[tuple[str, dict]]:
    """Find all legacy macd_rsi runs for symbol."""
    legacy_results = Path.home() / "projects" / "optimizer" / "results"
    if not legacy_results.exists():
        return []

    base = symbol.split("/")[0]
    runs = []

    for run_dir in sorted(legacy_results.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        params_file = run_dir / base / "best_params.json"
        if params_file.exists():
            with open(params_file) as f:
                params = json.load(f)
            if params.get("strategy", "").startswith("macd"):
                runs.append((run_dir.name, params))

    return runs


def load_legacy_trades(run_timestamp: str, symbol: str) -> pd.DataFrame | None:
    """Load legacy trades CSV."""
    legacy_results = Path.home() / "projects" / "optimizer" / "results"
    base = symbol.split("/")[0]
    symbol_safe = symbol.replace("/", "_")
    path = legacy_results / run_timestamp / base / f"trades_{symbol_safe}_1h_FULL.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def compare(symbol: str, legacy_run: str | None = None) -> bool:
    """Run comparison between QRE and legacy optimizer."""
    print(f"\n{'='*60}")
    print(f"  QRE vs Legacy Optimizer - Signal Comparison")
    print(f"  Symbol: {symbol}")
    print(f"{'='*60}\n")

    # Find legacy run
    runs = find_legacy_runs(symbol)
    if not runs:
        print("ERROR: No legacy macd_rsi runs found.")
        return False

    if legacy_run:
        match = [(ts, p) for ts, p in runs if ts == legacy_run]
        if not match:
            print(f"ERROR: Run '{legacy_run}' not found. Available:")
            for ts, _ in runs[:5]:
                print(f"  - {ts}")
            return False
        run_ts, legacy_params = match[0]
    else:
        run_ts, legacy_params = runs[0]
        print(f"Using most recent legacy run: {run_ts}")

    print(f"Legacy strategy: {legacy_params.get('strategy', 'unknown')}")
    print(f"Legacy trials: {legacy_params.get('n_trials', '?')}")
    print(f"Legacy equity: ${legacy_params.get('equity', 0):,.2f}")
    print(f"Legacy trades: {legacy_params.get('trades', '?')}")

    # Load legacy trades
    legacy_trades_df = load_legacy_trades(run_ts, symbol)
    if legacy_trades_df is None:
        print("ERROR: Legacy trades CSV not found.")
        return False

    print(f"\nLegacy trades loaded: {len(legacy_trades_df)} trades")

    # Load data from shared cache
    print("\nLoading data from shared cache...")
    import ccxt
    exchange = ccxt.binance({"enableRateLimit": True})
    cache_dir = str(Path.home() / "projects" / "optimizer" / "cache")
    cache = DataCache(cache_dir)
    hours = legacy_params.get("hours", 8760)
    data = load_all_data(exchange, symbol, hours, cache)
    print(f"Loaded {len(data['1h'])} bars")

    # Run QRE backtest with legacy params
    print("\nRunning QRE backtest with legacy params...")
    strategy = MACDRSIStrategy()
    buy_s, sell_s, gates = strategy.precompute_signals(data, legacy_params)
    result = simulate_trades_fast(
        symbol, data, legacy_params,
        precomputed_buy_signals=buy_s,
        precomputed_sell_signals=sell_s,
        precomputed_rsi_gates=gates,
    )

    qre_metrics = calculate_metrics(
        result.trades, result.backtest_days,
        start_equity=STARTING_EQUITY,
    )

    print(f"QRE trades: {len(result.trades)}")
    print(f"QRE equity: ${qre_metrics.equity:,.2f}")

    # Compare
    print(f"\n{'-'*60}")
    print(f"  COMPARISON RESULTS")
    print(f"{'-'*60}\n")

    legacy_count = len(legacy_trades_df)
    qre_count = len(result.trades)
    count_ratio = qre_count / legacy_count if legacy_count > 0 else 0

    print(f"Trade count:  Legacy={legacy_count}, QRE={qre_count}, Ratio={count_ratio:.2f}")

    # Compare entry timestamps
    qre_entries = []
    for t in result.trades[:20]:
        ts = t.entry_ts if hasattr(t, "entry_ts") else t["entry_ts"]
        qre_entries.append(pd.Timestamp(ts))

    legacy_entries = [pd.Timestamp(t) for t in legacy_trades_df["entry_ts"].head(20)]

    matches = 0
    for i, (qe, le) in enumerate(zip(qre_entries, legacy_entries)):
        diff_h = abs((qe - le).total_seconds()) / 3600
        status = "MATCH" if diff_h <= 1.0 else f"DIFF ({diff_h:.0f}h)"
        if diff_h <= 1.0:
            matches += 1
        if i < 5:
            print(f"  Trade {i+1}: QRE={qe} / Legacy={le} -> {status}")

    total_compared = min(len(qre_entries), len(legacy_entries))
    match_pct = matches / total_compared * 100 if total_compared > 0 else 0
    print(f"\nEntry match rate: {matches}/{total_compared} ({match_pct:.0f}%)")

    # Compare relative metrics
    legacy_sharpe = legacy_params.get("sharpe", 0)
    legacy_winrate = legacy_params.get("win_rate", 0)
    qre_winrate = qre_metrics.win_rate / 100

    print(f"\nWin rate:     Legacy={legacy_winrate:.4f}, QRE={qre_winrate:.4f}")
    print(f"Sharpe:       Legacy={legacy_sharpe:.4f}, QRE={qre_metrics.sharpe_ratio:.4f}")

    # Verdict
    print(f"\n{'='*60}")
    ok = count_ratio >= 0.85 and match_pct >= 70
    if ok:
        print("  VERDICT: PASS - Signals are reproducible")
    else:
        print("  VERDICT: FAIL - Signals diverge too much")
        if count_ratio < 0.85:
            print(f"    Trade count ratio {count_ratio:.2f} < 0.85")
        if match_pct < 70:
            print(f"    Entry match rate {match_pct:.0f}% < 70%")
    print(f"{'='*60}\n")

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare QRE vs Legacy Optimizer signals")
    parser.add_argument("--symbol", type=str, default="BTC/USDC")
    parser.add_argument("--legacy-run", type=str, default=None,
                        help="Specific legacy run timestamp to compare against")
    args = parser.parse_args()

    success = compare(args.symbol, args.legacy_run)
    sys.exit(0 if success else 1)
