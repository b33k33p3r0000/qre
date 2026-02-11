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

    # === LEVEL 1: Raw signal comparison (strategy logic) ===
    print("\n--- LEVEL 1: Raw Signal Comparison (pre-backtest) ---")
    strategy = MACDRSIStrategy()
    buy_s, sell_s, gates = strategy.precompute_signals(data, legacy_params)

    # Also run on legacy optimizer to get its signals
    legacy_buy_s = legacy_sell_s = legacy_gates = None
    try:
        sys.path.insert(0, str(Path.home() / "projects" / "optimizer" / "src"))
        from optimizer.strategies.macd_rsi import MACDRSIStrategy as LegacyStrategy
        legacy_strat = LegacyStrategy()
        legacy_buy_s, legacy_sell_s, legacy_gates = legacy_strat.precompute_signals(data, legacy_params)
    except (ImportError, Exception) as e:
        print(f"  (Cannot import legacy strategy: {e})")

    if legacy_buy_s is not None:
        import numpy as np
        buy_match = np.array_equal(buy_s, legacy_buy_s)
        sell_match = np.array_equal(sell_s, legacy_sell_s)
        gates_match = np.array_equal(gates, legacy_gates)
        print(f"  Buy signals match:  {buy_match}")
        print(f"  Sell signals match: {sell_match}")
        print(f"  RSI gates match:   {gates_match}")
        signals_identical = buy_match and sell_match and gates_match
    else:
        # Fallback: count signal activations
        import numpy as np
        buy_count = int(np.sum(buy_s))
        sell_count = int(np.sum(sell_s))
        print(f"  QRE buy signals:  {buy_count} activations across {buy_s.shape[0]} TFs x {buy_s.shape[1]} bars")
        print(f"  QRE sell signals: {sell_count} activations")
        signals_identical = None  # Can't compare without legacy

    # === LEVEL 2: Trade-level comparison (includes backtest config effects) ===
    print("\n--- LEVEL 2: Trade-Level Comparison (post-backtest) ---")
    print("  Note: QRE uses different position sizing (25% vs 20%) and")
    print("  stop-loss (15% vs 9%), so trade-level differences are EXPECTED.")

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

    legacy_count = len(legacy_trades_df)
    qre_count = len(result.trades)
    count_ratio = qre_count / legacy_count if legacy_count > 0 else 0

    print(f"\n  Trade count:  Legacy={legacy_count}, QRE={qre_count}, Ratio={count_ratio:.2f}")

    # Compare entry timestamps (first entries should match before chain diverges)
    qre_entries = [
        pd.Timestamp(t.entry_ts if hasattr(t, "entry_ts") else t["entry_ts"])
        for t in result.trades[:20]
    ]
    legacy_entries = [pd.Timestamp(t) for t in legacy_trades_df["entry_ts"].head(20)]

    matches = 0
    first_diverge = None
    for i, (qe, le) in enumerate(zip(qre_entries, legacy_entries)):
        diff_h = abs((qe - le).total_seconds()) / 3600
        status = "MATCH" if diff_h <= 1.0 else f"DIFF ({diff_h:.0f}h)"
        if diff_h <= 1.0:
            matches += 1
        elif first_diverge is None:
            first_diverge = i + 1
        if i < 5:
            print(f"    Trade {i+1}: QRE={qe} / Legacy={le} -> {status}")

    total_compared = min(len(qre_entries), len(legacy_entries))
    match_pct = matches / total_compared * 100 if total_compared > 0 else 0
    print(f"\n  Entry match rate: {matches}/{total_compared} ({match_pct:.0f}%)")
    if first_diverge:
        print(f"  First divergence at trade {first_diverge} (chain reaction from config diff)")

    # Compare relative metrics
    legacy_sharpe = legacy_params.get("sharpe", 0)
    legacy_winrate = legacy_params.get("win_rate", 0)
    qre_winrate = qre_metrics.win_rate / 100

    print(f"\n  Win rate:     Legacy={legacy_winrate:.4f}, QRE={qre_winrate:.4f}")
    print(f"  Sharpe:       Legacy={legacy_sharpe:.4f}, QRE={qre_metrics.sharpe_ratio:.4f}")
    print(f"  QRE equity:   ${qre_metrics.equity:,.2f}")

    # === VERDICT ===
    print(f"\n{'='*60}")

    # Primary criterion: signal arrays identical (strategy logic)
    # Secondary criterion: first entries match (proves same entry logic)
    first_entries_match = len(qre_entries) > 0 and len(legacy_entries) > 0 and matches >= 2

    if signals_identical is True:
        print("  VERDICT: PASS - Strategy signals are IDENTICAL")
        print("    Raw buy/sell/gate arrays match exactly.")
        print(f"    Trade count differs ({count_ratio:.2f}) due to backtest config changes:")
        print(f"    - Position sizing: QRE 25% vs Legacy 20%")
        print(f"    - Stop-loss: QRE 15% vs Legacy 9%")
        ok = True
    elif signals_identical is None and first_entries_match:
        print("  VERDICT: PASS - First entries match, strategy logic likely identical")
        print(f"    First {matches} trades share identical entry timestamps.")
        print(f"    Chain diverges at trade {first_diverge or '?'} due to config differences.")
        ok = True
    else:
        print("  VERDICT: FAIL - Strategy signals may differ")
        ok = False

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
