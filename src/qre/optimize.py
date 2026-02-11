"""
QRE Optimizer
=============
Anchored Walk-Forward optimization with Optuna.

Single entry point: run_optimization(symbol, hours, n_trials, ...)
  1. Fetch data (OHLCV all timeframes)
  2. Compute AWF splits
  3. Run Optuna study (TPE sampler, SHA pruner)
  4. Final evaluation with best params
  5. Monte Carlo validation
  6. Save results (JSON + CSV)

Only AWF mode. Only MACD+RSI strategy.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd

from qre.config import (
    ANCHORED_WF_MIN_DATA_HOURS,
    ANCHORED_WF_SHORT_THRESHOLD_HOURS,
    ANCHORED_WF_SPLITS,
    ANCHORED_WF_SPLITS_SHORT,
    BASE_TF,
    DEFAULT_TRIALS,
    ENABLE_PRUNING,
    MIN_PRUNING_WARMUP,
    MIN_STARTUP_TRIALS,
    MIN_WARMUP_BARS,
    MONTE_CARLO_MIN_TRADES,
    MONTE_CARLO_SIMULATIONS,
    PRUNING_WARMUP_RATIO,
    STARTING_EQUITY,
    STARTUP_TRIALS_RATIO,
    TF_LIST,
    TPE_CONSIDER_ENDPOINTS,
    TPE_N_EI_CANDIDATES,
)
from qre.core.backtest import simulate_trades_fast
from qre.core.metrics import calculate_metrics, monte_carlo_validation
from qre.core.strategy import MACDRSIStrategy
from qre.data.fetch import DataCache, load_all_data
from qre.hooks import run_post_hooks, run_pre_hooks
from qre.io import save_json, save_trades_csv
from qre.notify import notify_complete, notify_start
from qre.penalties import apply_all_penalties
from qre.report import save_report

logger = logging.getLogger("qre.optimize")


def compute_awf_splits(
    total_hours: int, n_splits: Optional[int] = None
) -> Optional[List[Dict[str, float]]]:
    """Compute Anchored Walk-Forward splits based on data length."""
    if total_hours < ANCHORED_WF_MIN_DATA_HOURS:
        return None

    if n_splits is not None and n_splits >= 2:
        splits = []
        train_start = 0.50
        train_step = 0.40 / n_splits
        test_size = 0.10
        for i in range(n_splits):
            train_end = train_start + (i + 1) * train_step
            test_end = min(train_end + test_size, 1.0)
            splits.append({"train_end": train_end, "test_end": test_end})
        return splits

    if total_hours < ANCHORED_WF_SHORT_THRESHOLD_HOURS:
        return list(ANCHORED_WF_SPLITS_SHORT)
    return list(ANCHORED_WF_SPLITS)


def create_sampler(seed: int, n_trials: int) -> optuna.samplers.BaseSampler:
    """Create TPE sampler for Optuna study."""
    n_startup = max(MIN_STARTUP_TRIALS, int(n_trials * STARTUP_TRIALS_RATIO))
    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=n_startup,
        n_ei_candidates=TPE_N_EI_CANDIDATES,
        consider_endpoints=TPE_CONSIDER_ENDPOINTS,
    )


def create_pruner(n_trials: int) -> optuna.pruners.BasePruner:
    """Create SuccessiveHalving pruner."""
    if not ENABLE_PRUNING:
        return optuna.pruners.NopPruner()
    warmup = max(MIN_PRUNING_WARMUP, int(n_trials * PRUNING_WARMUP_RATIO))
    return optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=3,
        min_early_stopping_rate=0,
    )


def build_objective(
    symbol: str,
    data: Dict[str, pd.DataFrame],
    splits: List[Dict[str, float]],
) -> callable:
    """Build Optuna objective function for AWF optimization."""
    strategy = MACDRSIStrategy()
    base_df = data[BASE_TF]
    total_bars = len(base_df)

    def objective(trial: optuna.trial.Trial) -> float:
        params = strategy.get_optuna_params(trial, symbol)

        buy_signals, sell_signals, rsi_gates = strategy.precompute_signals(data, params)

        split_scores = []
        for split in splits:
            train_end = int(total_bars * split["train_end"])
            test_end = int(total_bars * split["test_end"])

            # TRAIN
            train_result = simulate_trades_fast(
                symbol, data, params,
                start_idx=MIN_WARMUP_BARS, end_idx=train_end,
                precomputed_buy_signals=buy_signals,
                precomputed_sell_signals=sell_signals,
                precomputed_rsi_gates=rsi_gates,
            )
            if not train_result.trades:
                split_scores.append(0.0)
                continue

            train_metrics = calculate_metrics(
                train_result.trades, train_result.backtest_days,
                start_equity=STARTING_EQUITY,
            )

            # TEST
            test_result = simulate_trades_fast(
                symbol, data, params,
                start_idx=train_end, end_idx=test_end,
                precomputed_buy_signals=buy_signals,
                precomputed_sell_signals=sell_signals,
                precomputed_rsi_gates=rsi_gates,
            )
            if not test_result.trades or len(test_result.trades) < 3:
                split_scores.append(0.0)
                continue

            test_metrics = calculate_metrics(
                test_result.trades, test_result.backtest_days,
                start_equity=STARTING_EQUITY,
            )

            penalized = apply_all_penalties(
                train_metrics.equity,
                train_metrics.trades_per_year,
                train_metrics.short_hold_ratio,
                train_metrics.max_drawdown,
                train_metrics.monthly_returns,
                train_equity=train_metrics.equity,
                test_equity=test_metrics.equity,
                test_sharpe=test_metrics.sharpe_ratio,
                test_trades=test_metrics.trades,
            )
            split_scores.append(penalized)

        if not split_scores or all(s == 0 for s in split_scores):
            return 0.0

        return float(np.mean([s for s in split_scores if s > 0]))

    return objective


def run_optimization(
    symbol: str,
    hours: int = 8760,
    n_trials: int = DEFAULT_TRIALS,
    n_splits: Optional[int] = None,
    seed: int = 42,
    timeout: int = 0,
    results_dir: str = "results",
    cache_dir: str = "cache",
    run_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full AWF optimization pipeline for a single symbol.

    Returns best_params dict with all metrics.
    """
    import ccxt

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    tag_suffix = f"_{run_tag}" if run_tag else ""
    run_timestamp = f"{run_timestamp}{tag_suffix}"

    run_pre_hooks({"symbol": symbol, "hours": hours, "n_trials": n_trials})
    notify_start(symbol=symbol, n_trials=n_trials, hours=hours, n_splits=n_splits or 3, run_tag=run_tag)

    exchange = ccxt.binance({"enableRateLimit": True})
    cache = DataCache(cache_dir)

    # 1. Fetch data
    logger.info(f"Loading {symbol} data ({hours}h history)...")
    data = load_all_data(exchange, symbol, hours, cache)
    total_bars = len(data[BASE_TF])
    logger.info(f"Loaded {total_bars} bars for {symbol}")

    # 2. Compute splits
    splits = compute_awf_splits(total_bars, n_splits)
    if splits is None:
        raise ValueError(f"Data too short for AWF: {total_bars} bars < {ANCHORED_WF_MIN_DATA_HOURS}")

    logger.info(f"AWF: {len(splits)} splits, {total_bars} total bars")
    for i, s in enumerate(splits):
        te = int(total_bars * s["train_end"])
        tse = int(total_bars * s["test_end"])
        logger.info(f"  Split {i+1}: Train 0-{te}, Test {te}-{tse}")

    # 3. Run Optuna
    sampler = create_sampler(seed, n_trials)
    pruner = create_pruner(n_trials)
    objective = build_objective(symbol, data, splits)

    base = symbol.split("/")[0]
    results_base = Path(results_dir)
    outdir = results_base / run_timestamp / base

    checkpoint_dir = results_base / run_timestamp / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"optuna_{base}.db"
    storage_url = f"sqlite:///{checkpoint_path}"
    storage = optuna.storages.RDBStorage(url=storage_url)

    study_name = f"{symbol.replace('/', '_')}_{run_timestamp}_awf"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    logger.info(f"Starting AWF optimization: {n_trials} trials")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout if timeout > 0 else None,
        n_jobs=1,
        show_progress_bar=True,
    )

    best_params = study.best_trial.params
    strategy = MACDRSIStrategy()

    # 4. Final evaluation
    logger.info("Running final evaluation...")
    best_params.update({
        "symbol": symbol, "tf": "1h", "range": "FULL",
        "n_votes": len(TF_LIST),
        "optimization_mode": "anchored_walk_forward",
        "n_splits": len(splits), "n_trials": n_trials,
        "strategy": strategy.name, "strategy_version": strategy.version,
    })

    buy_s, sell_s, gates = strategy.precompute_signals(data, best_params)
    full_result = simulate_trades_fast(
        symbol, data, best_params,
        precomputed_buy_signals=buy_s,
        precomputed_sell_signals=sell_s,
        precomputed_rsi_gates=gates,
    )
    full_metrics = calculate_metrics(
        full_result.trades, full_result.backtest_days,
        start_equity=STARTING_EQUITY,
    )

    # Per-split metrics
    split_metrics = []
    last_train_metrics = None
    last_test_metrics = None
    for i, split in enumerate(splits):
        train_end = int(total_bars * split["train_end"])
        test_end = int(total_bars * split["test_end"])

        if i == len(splits) - 1:
            tr = simulate_trades_fast(
                symbol, data, best_params,
                start_idx=MIN_WARMUP_BARS, end_idx=train_end,
                precomputed_buy_signals=buy_s, precomputed_sell_signals=sell_s,
                precomputed_rsi_gates=gates,
            )
            if tr.trades:
                last_train_metrics = calculate_metrics(
                    tr.trades, tr.backtest_days, start_equity=STARTING_EQUITY,
                )

        te_r = simulate_trades_fast(
            symbol, data, best_params,
            start_idx=train_end, end_idx=test_end,
            precomputed_buy_signals=buy_s, precomputed_sell_signals=sell_s,
            precomputed_rsi_gates=gates,
        )
        if te_r.trades:
            tm = calculate_metrics(te_r.trades, te_r.backtest_days, start_equity=STARTING_EQUITY)
            split_metrics.append({
                "split": i + 1,
                "test_equity": round(tm.equity, 2),
                "test_trades": tm.trades,
                "test_sharpe": round(tm.sharpe_ratio, 4),
            })
            if i == len(splits) - 1:
                last_test_metrics = tm

    best_params.update({
        "equity": full_metrics.equity,
        "total_pnl": full_metrics.total_pnl,
        "total_pnl_pct": round(full_metrics.total_pnl_pct, 2),
        "trades": full_metrics.trades,
        "trades_per_year": round(full_metrics.trades_per_year, 2),
        "win_rate": round(full_metrics.win_rate / 100, 4),
        "max_drawdown": round(full_metrics.max_drawdown, 2),
        "sharpe": round(full_metrics.sharpe_ratio, 4),
        "sortino": round(full_metrics.sortino_ratio, 4),
        "calmar": round(full_metrics.calmar_ratio, 4),
        "recovery_factor": round(full_metrics.recovery_factor, 4),
        "profit_factor": round(full_metrics.profit_factor, 4),
        "expectancy": round(full_metrics.expectancy, 2),
        "profitable_months_ratio": round(full_metrics.profitable_months_ratio, 4),
        "split_results": split_metrics,
        "run_timestamp": run_timestamp,
        "trades_file": f"trades_{symbol.replace('/', '_')}_1h_FULL.csv",
        "start_equity": STARTING_EQUITY,
    })

    if last_train_metrics:
        best_params.update({
            "train_equity": round(last_train_metrics.equity, 2),
            "train_trades": last_train_metrics.trades,
            "train_sharpe": round(last_train_metrics.sharpe_ratio, 4),
        })
    if last_test_metrics:
        best_params.update({
            "test_equity": round(last_test_metrics.equity, 2),
            "test_trades": last_test_metrics.trades,
            "test_sharpe": round(last_test_metrics.sharpe_ratio, 4),
        })
    if last_train_metrics and last_test_metrics:
        best_params["test_train_ratio"] = (
            round(last_test_metrics.equity / last_train_metrics.equity, 4)
            if last_train_metrics.equity > 0 else 0
        )

    # 5. Monte Carlo
    if len(full_result.trades) >= MONTE_CARLO_MIN_TRADES:
        logger.info(f"Running Monte Carlo ({MONTE_CARLO_SIMULATIONS} simulations)...")
        mc = monte_carlo_validation(full_result.trades, n_simulations=MONTE_CARLO_SIMULATIONS, seed=seed)
        best_params.update({
            "mc_sharpe_mean": mc.sharpe_mean,
            "mc_sharpe_ci_low": mc.sharpe_ci_low,
            "mc_sharpe_ci_high": mc.sharpe_ci_high,
            "mc_max_dd_mean": mc.max_dd_mean,
            "mc_max_dd_ci_low": mc.max_dd_ci_low,
            "mc_max_dd_ci_high": mc.max_dd_ci_high,
            "mc_confidence": mc.confidence_level,
            "mc_robustness": mc.robustness_score,
        })

    # 6. Save results
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "best_params.json", best_params)
    save_trades_csv(outdir / f"trades_{symbol.replace('/', '_')}_1h_FULL.csv", full_result.trades)

    # 7. HTML report
    trades_dicts = [t._asdict() if hasattr(t, '_asdict') else t for t in full_result.trades]
    save_report(outdir / f"report_{base}.html", best_params, trades_dicts)

    logger.info(f"Done {symbol}: Equity=${full_metrics.equity:,.2f}, "
                f"Sharpe={full_metrics.sharpe_ratio:.2f}, Trades={full_metrics.trades}")

    # 8. Notifications
    notify_complete(best_params)

    run_post_hooks(best_params)

    return best_params


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="QRE Optimizer — MACD+RSI AWF")
    parser.add_argument("--symbol", type=str, default="BTC/USDC", choices=["BTC/USDC", "SOL/USDC"])
    parser.add_argument("--hours", type=int, default=8760)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--cache-dir", type=str, default="cache")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    result = run_optimization(
        symbol=args.symbol,
        hours=args.hours,
        n_trials=args.trials,
        n_splits=args.splits,
        seed=args.seed,
        timeout=args.timeout,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        run_tag=args.tag,
    )

    print(f"\nResult: {result['symbol']}")
    print(f"  Equity: ${result['equity']:,.2f}")
    print(f"  Sharpe: {result['sharpe']:.4f}")
    print(f"  Trades: {result['trades']}")
    print(f"  Max DD: {result['max_drawdown']:.2f}%")


if __name__ == "__main__":
    main()
