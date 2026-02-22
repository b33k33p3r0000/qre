#!/usr/bin/env python3
"""
Compare catastrophic stop levels on existing best params.

Re-runs backtest with identical signals but different stop levels
to measure impact on PnL, drawdown, and trade outcomes.

Usage:
    python scripts/compare_stops.py
"""

import json
import logging
import sys
from pathlib import Path

import ccxt
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from qre.config import BASE_TF, STARTING_EQUITY
from qre.core.backtest import simulate_trades_fast
from qre.core.metrics import calculate_metrics
from qre.core.strategy import MACDRSIStrategy
from qre.data.fetch import load_all_data

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Runs to analyze: (results_dir, symbol_dir, hours_to_fetch)
RUNS = [
    ("2026-02-21_06-35-30_final-btc-v1-best", "BTC", 17520),    # ~2yr
    ("2026-02-21_06-35-19_final-sol-v1-best", "SOL", 17520),     # ~2yr
    ("2026-02-21_20-04-18_final-btc-3yr-v2", "BTC", 26280),      # ~3yr
    ("2026-02-21_15-15-41_final-btc-1yr", "BTC", 8760),          # ~1yr
]

STOP_LEVELS = [0.04, 0.05, 0.07, 0.10, 0.12]

RESULTS_ROOT = project_root / "results"


def load_best_params(run_dir: str, symbol_dir: str) -> dict:
    path = RESULTS_ROOT / run_dir / symbol_dir / "best_params.json"
    with open(path) as f:
        return json.load(f)


def run_backtest_with_stop(symbol, data, buy_signal, sell_signal, stop_pct, allow_flip):
    """Run backtest with a specific catastrophic stop level."""
    return simulate_trades_fast(
        symbol=symbol,
        data=data,
        buy_signal=buy_signal,
        sell_signal=sell_signal,
        allow_flip=allow_flip,
        catastrophic_stop_pct=stop_pct,
    )


def analyze_run(run_dir: str, symbol_dir: str, hours: int, exchange):
    """Analyze a single run across all stop levels."""
    params = load_best_params(run_dir, symbol_dir)
    symbol = params["symbol"]

    print(f"\n{'='*80}")
    print(f"  {run_dir} / {symbol_dir}")
    print(f"  Symbol: {symbol} | Original: {params['trades']} trades, "
          f"DD {params['max_drawdown']:.2f}%, PnL {params['total_pnl_pct']:.1f}%")
    print(f"{'='*80}")

    # Load data
    print(f"  Fetching {hours}h of data for {symbol}...", end=" ", flush=True)
    data = load_all_data(exchange, symbol, hours)
    n_bars = len(data[BASE_TF])
    print(f"{n_bars} bars loaded.")

    # Generate signals ONCE
    strategy = MACDRSIStrategy()
    buy_signal, sell_signal = strategy.precompute_signals(data, params)
    allow_flip = bool(params.get("allow_flip", 1))

    # Run backtests with different stop levels
    results = []
    for stop_pct in STOP_LEVELS:
        bt_result = run_backtest_with_stop(symbol, data, buy_signal, sell_signal, stop_pct, allow_flip)

        trades = bt_result.trades
        n_trades = len(trades)
        cat_stops = sum(1 for t in trades if t["reason"] == "catastrophic_stop")
        signal_exits = sum(1 for t in trades if t["reason"] == "signal")

        if n_trades > 0 and bt_result.backtest_days > 0:
            metrics = calculate_metrics(
                trades, bt_result.backtest_days,
                start_equity=STARTING_EQUITY,
                price_data=data[BASE_TF],
            )
            results.append({
                "stop_pct": stop_pct,
                "equity": bt_result.equity,
                "total_pnl": bt_result.equity - STARTING_EQUITY,
                "total_pnl_pct": metrics.total_pnl_pct,
                "trades": n_trades,
                "cat_stops": cat_stops,
                "cat_stop_pct": cat_stops / n_trades * 100 if n_trades > 0 else 0,
                "signal_exits": signal_exits,
                "max_dd": metrics.max_drawdown,
                "sharpe": metrics.sharpe_ratio_equity_based,
                "calmar": metrics.calmar_ratio,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "expectancy": metrics.expectancy,
            })
        else:
            results.append({
                "stop_pct": stop_pct,
                "equity": bt_result.equity,
                "total_pnl": bt_result.equity - STARTING_EQUITY,
                "total_pnl_pct": 0,
                "trades": n_trades,
                "cat_stops": cat_stops,
                "cat_stop_pct": 0,
                "signal_exits": signal_exits,
                "max_dd": 0,
                "sharpe": 0,
                "calmar": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "expectancy": 0,
            })

    # Print comparison table
    print()
    print(f"  {'Stop':>6} | {'PnL $':>10} | {'PnL %':>7} | {'Trades':>6} | "
          f"{'CatStop':>7} | {'CS %':>5} | {'Max DD':>7} | {'Sharpe':>7} | "
          f"{'Calmar':>8} | {'WinR':>5} | {'PF':>5} | {'Exp $':>6}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*6}-+-"
          f"{'-'*7}-+-{'-'*5}-+-{'-'*7}-+-{'-'*7}-+-"
          f"{'-'*8}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}")

    for r in results:
        marker = " <--" if r["stop_pct"] == 0.10 else ""
        print(f"  {r['stop_pct']:>5.0%}  | {r['total_pnl']:>10,.0f} | {r['total_pnl_pct']:>6.1f}% | "
              f"{r['trades']:>6} | {r['cat_stops']:>7} | {r['cat_stop_pct']:>4.1f}% | "
              f"{r['max_dd']:>6.2f}% | {r['sharpe']:>7.4f} | {r['calmar']:>8.2f} | "
              f"{r['win_rate']:>4.1f}% | {r['profit_factor']:>5.2f} | {r['expectancy']:>6.1f}{marker}")

    # Delta analysis vs 10%
    baseline = next(r for r in results if r["stop_pct"] == 0.10)
    print(f"\n  Delta vs 10% baseline:")
    for r in results:
        if r["stop_pct"] == 0.10:
            continue
        pnl_delta = r["total_pnl"] - baseline["total_pnl"]
        dd_delta = r["max_dd"] - baseline["max_dd"]
        trade_delta = r["trades"] - baseline["trades"]
        calmar_delta = r["calmar"] - baseline["calmar"]
        print(f"    {r['stop_pct']:>4.0%}: PnL {pnl_delta:>+8,.0f}$ | DD {dd_delta:>+5.2f}% | "
              f"Trades {trade_delta:>+4} | Calmar {calmar_delta:>+7.2f}")

    return results


def main():
    print("=" * 80)
    print("  CATASTROPHIC STOP LEVEL COMPARISON")
    print(f"  Testing stop levels: {', '.join(f'{s:.0%}' for s in STOP_LEVELS)}")
    print("=" * 80)

    exchange = ccxt.binance({"enableRateLimit": True})
    all_results = {}

    for run_dir, symbol_dir, hours in RUNS:
        try:
            results = analyze_run(run_dir, symbol_dir, hours, exchange)
            all_results[f"{run_dir}/{symbol_dir}"] = results
        except Exception as e:
            print(f"\n  ERROR: {run_dir}/{symbol_dir}: {e}")
            continue

    # Summary across all runs
    print(f"\n{'='*80}")
    print("  SUMMARY: Average impact across all runs")
    print(f"{'='*80}")

    for stop_pct in STOP_LEVELS:
        pnl_deltas = []
        dd_deltas = []
        calmar_deltas = []
        for run_results in all_results.values():
            baseline = next(r for r in run_results if r["stop_pct"] == 0.10)
            current = next(r for r in run_results if r["stop_pct"] == stop_pct)
            pnl_deltas.append(current["total_pnl_pct"] - baseline["total_pnl_pct"])
            dd_deltas.append(current["max_dd"] - baseline["max_dd"])
            calmar_deltas.append(current["calmar"] - baseline["calmar"])

        marker = " (baseline)" if stop_pct == 0.10 else ""
        avg_pnl = np.mean(pnl_deltas)
        avg_dd = np.mean(dd_deltas)
        avg_calmar = np.mean(calmar_deltas)
        print(f"  {stop_pct:>4.0%}: Avg PnL delta {avg_pnl:>+6.1f}% | "
              f"Avg DD delta {avg_dd:>+5.2f}% | "
              f"Avg Calmar delta {avg_calmar:>+7.2f}{marker}")


if __name__ == "__main__":
    main()
