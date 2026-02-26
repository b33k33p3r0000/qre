"""
QRE Optimizer
=============
Anchored Walk-Forward optimization with Optuna.

Single entry point: run_optimization(symbol, hours, n_trials, ...)
  1. Fetch data (OHLCV)
  2. Compute AWF splits
  3. Run Optuna study (TPE sampler, SHA pruner)
  4. Final evaluation with best params
  5. Monte Carlo validation
  6. Save results (JSON + CSV)

Only AWF mode. Only Quant Whale Strategy (MACD+RSI) strategy.
"""

from __future__ import annotations

import json
import logging
import math
import signal

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from qre.config import (
    ANCHORED_WF_MIN_DATA_HOURS,
    ANCHORED_WF_SHORT_THRESHOLD_HOURS,
    ANCHORED_WF_SPLITS,
    ANCHORED_WF_SPLITS_SHORT,
    BASE_TF,
    CATASTROPHIC_STOP_PCT,
    CATASTROPHIC_STOP_PCT_DEFAULT,
    DEFAULT_TRIALS,
    ENABLE_PRUNING,
    MIN_DRAWDOWN_FLOOR,
    MIN_STARTUP_TRIALS,
    MIN_TRADES_TEST_HARD,
    MIN_TRADES_YEAR_HARD,
    MIN_WARMUP_BARS,
    MONTE_CARLO_MIN_TRADES,
    MONTE_CARLO_SIMULATIONS,
    PURGE_GAP_BARS,
    SHARPE_DECAY_RATE,
    SHARPE_SUSPECT_THRESHOLD,
    STARTING_EQUITY,
    TARGET_TRADES_YEAR,
    STARTUP_TRIALS_RATIO,
    TF_MS,
    TPE_CONSIDER_ENDPOINTS,
    TPE_N_EI_CANDIDATES,
)
from qre.core.backtest import simulate_trades_fast
from qre.core.indicators import rsi as compute_rsi
from qre.core.metrics import aggregate_mc_results, calculate_metrics, monte_carlo_validation
from qre.core.strategy import MACDRSIStrategy
from qre.data.fetch import load_all_data
from qre.io import save_json, save_trades_csv
from qre.notify import notify_complete, notify_start
from qre.report import save_report

logger = logging.getLogger("qre.optimize")


def compute_awf_splits(
    total_hours: int, n_splits: int | None = None, test_size: float = 0.20
) -> list[dict[str, float]] | None:
    """Compute Anchored Walk-Forward splits based on data length."""
    if total_hours < ANCHORED_WF_MIN_DATA_HOURS:
        return None

    purge_frac = PURGE_GAP_BARS / total_hours  # gap as fraction of total data

    if n_splits is not None and n_splits >= 2:
        splits = []
        train_start = 0.50
        available = 1.0 - train_start - test_size - purge_frac
        train_step = available / n_splits
        for i in range(n_splits):
            train_end = train_start + (i + 1) * train_step
            test_start = train_end + purge_frac
            test_end = min(test_start + test_size, 1.0)
            splits.append({
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })
        return splits

    # Static splits — add purge gap
    base_splits = (
        ANCHORED_WF_SPLITS_SHORT
        if total_hours < ANCHORED_WF_SHORT_THRESHOLD_HOURS
        else ANCHORED_WF_SPLITS
    )
    return [
        {
            "train_end": s["train_end"],
            "test_start": s["train_end"] + purge_frac,
            "test_end": s["test_end"],
        }
        for s in base_splits
    ]


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
    return optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=3,
        min_early_stopping_rate=0,
    )


def compute_objective_score(
    raw_calmar: float,
    sharpe: float,
    trades_per_year: float,
) -> float:
    """Compute Log Calmar objective score with trade ramp and Sharpe decay.

    This is the core scoring function used by the Optuna objective.
    Extracted for testability.

    Args:
        raw_calmar: Raw Calmar ratio (annual_return / max_dd), already floored and clamped >= 0.
        sharpe: Equity-based Sharpe ratio (clamped >= 0).
        trades_per_year: Number of trades per year.

    Returns:
        Objective score (higher is better).
    """
    # Log dampening — compress extreme Calmar values
    log_calmar = math.log(1.0 + raw_calmar)

    # Trade count ramp — penalize low frequency
    trade_mult = min(1.0, max(0.0, trades_per_year / TARGET_TRADES_YEAR))

    # Smooth Sharpe decay — penalize suspiciously high Sharpe
    if sharpe > SHARPE_SUSPECT_THRESHOLD:
        penalty = 1.0 / (1.0 + SHARPE_DECAY_RATE * (sharpe - SHARPE_SUSPECT_THRESHOLD))
        log_calmar *= penalty

    return log_calmar * trade_mult


def build_objective(
    symbol: str,
    data: dict[str, pd.DataFrame],
    splits: list[dict[str, float]],
    allow_flip_setting: int = 1,
) -> Callable:
    """Build Optuna objective function for AWF optimization.

    Returns log(1+Calmar) with trade count ramp and smooth Sharpe decay penalty.
    Hard constraints: MIN_TRADES_YEAR_HARD on train, MIN_TRADES_TEST_HARD on test.
    """
    strategy = MACDRSIStrategy()
    base_df = data[BASE_TF]
    total_bars = len(base_df)

    # Pre-compute RSI for all possible Optuna periods (3-30)
    precomputed_cache = {"rsi": {}}
    for period in range(3, 31):
        precomputed_cache["rsi"][period] = compute_rsi(
            base_df["close"], period
        ).values.astype(np.float64)

    def objective(trial: optuna.trial.Trial) -> float:
        params = strategy.get_optuna_params(trial, symbol, allow_flip_override=allow_flip_setting)

        buy_signal, sell_signal = strategy.precompute_signals(
            data, params, precomputed_cache=precomputed_cache,
        )

        allow_flip = bool(params.get("allow_flip", 1))
        symbol_key = symbol.split("/")[0]
        catastrophic_stop_pct = CATASTROPHIC_STOP_PCT.get(symbol_key, CATASTROPHIC_STOP_PCT_DEFAULT)

        split_scores = []
        _sharpes = []
        _drawdowns = []
        _pnls = []
        _trade_counts = []
        _tpy = []
        for split in splits:
            train_end = int(total_bars * split["train_end"])
            test_start = int(total_bars * split.get("test_start", split["train_end"]))
            test_end = int(total_bars * split["test_end"])

            # TRAIN — only for hard constraint check
            train_result = simulate_trades_fast(
                symbol, data, buy_signal, sell_signal,
                start_idx=MIN_WARMUP_BARS, end_idx=train_end,
                allow_flip=allow_flip,
                catastrophic_stop_pct=catastrophic_stop_pct,
            )
            if not train_result.trades:
                split_scores.append(0.0)
                continue

            train_metrics = calculate_metrics(
                train_result.trades, train_result.backtest_days,
                start_equity=STARTING_EQUITY,
            )

            # Hard constraint: minimum trades per year (on train)
            if train_metrics.trades_per_year < MIN_TRADES_YEAR_HARD:
                split_scores.append(0.0)
                continue

            # TEST — this is what we optimize (start after purge gap)
            test_result = simulate_trades_fast(
                symbol, data, buy_signal, sell_signal,
                start_idx=test_start, end_idx=test_end,
                allow_flip=allow_flip,
                catastrophic_stop_pct=catastrophic_stop_pct,
            )

            # Hard constraint: minimum test trades
            if not test_result.trades or len(test_result.trades) < MIN_TRADES_TEST_HARD:
                split_scores.append(0.0)
                continue

            test_metrics = calculate_metrics(
                test_result.trades, test_result.backtest_days,
                start_equity=STARTING_EQUITY,
            )

            # Score = Log Calmar with trade ramp and Sharpe decay
            annual_return = test_metrics.total_pnl_pct / (test_result.backtest_days / 365.25)
            max_dd = abs(test_metrics.max_drawdown / 100.0)  # convert % to fraction
            raw_calmar = annual_return / max(max_dd, MIN_DRAWDOWN_FLOOR)
            raw_calmar = max(0.0, raw_calmar)

            trades_per_year = len(test_result.trades) / (test_result.backtest_days / 365.25)
            sharpe = max(0.0, test_metrics.sharpe_ratio_equity_based)

            score = compute_objective_score(raw_calmar, sharpe, trades_per_year)

            _sharpes.append(test_metrics.sharpe_ratio_equity_based)
            _drawdowns.append(test_metrics.max_drawdown)
            _pnls.append(test_metrics.total_pnl_pct)
            _trade_counts.append(len(test_result.trades))
            _tpy.append(trades_per_year)

            split_scores.append(score)

        if not split_scores or all(s == 0 for s in split_scores):
            return 0.0

        # Store extended metrics for live monitor
        if _sharpes:
            trial.set_user_attr("sharpe_equity", round(float(np.mean(_sharpes)), 4))
            trial.set_user_attr("max_drawdown", round(float(np.mean(_drawdowns)), 2))
            trial.set_user_attr("total_pnl_pct", round(float(np.mean(_pnls)), 2))
            trial.set_user_attr("trades", int(np.mean(_trade_counts)))
            trial.set_user_attr("trades_per_year", round(float(np.mean(_tpy)), 2))

        return float(np.mean(split_scores))

    return objective


WARM_START_KEYS = [
    "macd_fast", "macd_slow", "macd_signal",
    "rsi_period", "rsi_lower", "rsi_upper", "rsi_lookback",
    "trend_tf", "trend_strict", "allow_flip",
]


def load_warm_start_params(path: str | Path) -> dict[str, Any]:
    """Load and validate warm-start parameters from a best_params.json file.

    Returns dict with only the 10 strategy param keys needed for enqueue_trial.
    Raises ValueError if file is invalid or missing required keys.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Warm-start file not found: {path}")

    with open(path) as f:
        raw = json.load(f)

    missing = [k for k in WARM_START_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Warm-start file missing keys: {missing}")

    params = {k: raw[k] for k in WARM_START_KEYS}
    logger.info("Warm-start params loaded from %s: %s", path.name, params)
    return params


def run_optimization(
    symbol: str,
    hours: int = 8760,
    n_trials: int = DEFAULT_TRIALS,
    n_splits: int | None = None,
    seed: int = 42,
    timeout: int = 0,
    results_dir: str = "results",
    run_tag: str | None = None,
    skip_recent_hours: int = 0,
    test_size: float = 0.20,
    allow_flip: int = 0,
    warm_start_path: str | None = None,
) -> dict[str, Any]:
    """
    Run full AWF optimization pipeline for a single symbol.

    Args:
        skip_recent_hours: Drop the most recent N hours of 1H data before optimizing.

    Returns best_params dict with all metrics.
    """
    import ccxt

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    tag_suffix = f"_{run_tag}" if run_tag else ""
    run_timestamp = f"{run_timestamp}{tag_suffix}"

    notify_start(symbol=symbol, n_trials=n_trials, hours=hours, n_splits=n_splits or 5, run_tag=run_tag)

    exchange = ccxt.binance({"enableRateLimit": True})

    # 1. Fetch fresh data
    logger.info("Loading %s data (%dh history)...", symbol, hours)
    data = load_all_data(exchange, symbol, hours)

    # Trim recent data if requested
    if skip_recent_hours > 0:
        if skip_recent_hours < len(data[BASE_TF]):
            data[BASE_TF] = data[BASE_TF].iloc[:-skip_recent_hours]
        # Trim higher TFs proportionally
        for tf in list(data.keys()):
            if tf != BASE_TF and len(data[tf]) > 0:
                tf_hours = TF_MS[tf] // TF_MS["1h"]
                tf_bars_to_skip = max(1, skip_recent_hours // tf_hours)
                if tf_bars_to_skip < len(data[tf]):
                    data[tf] = data[tf].iloc[:-tf_bars_to_skip]
        logger.info("Trimmed %dh of recent data", skip_recent_hours)

    total_bars = len(data[BASE_TF])
    logger.info("Loaded %d bars for %s", total_bars, symbol)

    # 2. Compute splits
    splits = compute_awf_splits(total_bars, n_splits, test_size=test_size)
    if splits is None:
        raise ValueError(f"Data too short for AWF: {total_bars} bars < {ANCHORED_WF_MIN_DATA_HOURS}")

    logger.info("AWF: %d splits, %d total bars", len(splits), total_bars)
    for i, s in enumerate(splits):
        te = int(total_bars * s["train_end"])
        ts = int(total_bars * s.get("test_start", s["train_end"]))
        tse = int(total_bars * s["test_end"])
        logger.info("  Split %d: Train 0-%d, Purge %d-%d, Test %d-%d", i+1, te, te, ts, ts, tse)

    # 3. Run Optuna
    sampler = create_sampler(seed, n_trials)
    pruner = create_pruner(n_trials)
    objective = build_objective(symbol, data, splits, allow_flip_setting=allow_flip)

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
    study.set_user_attr("n_trials_requested", n_trials)
    study.set_user_attr("symbol", symbol)

    # Warm-start: enqueue known-good params as first trial
    if warm_start_path:
        ws_params = load_warm_start_params(warm_start_path)
        study.enqueue_trial(ws_params)
        ws_source = Path(warm_start_path).parent.parent.name
        study.set_user_attr("warm_start_source", ws_source)
        logger.info("Warm-start trial enqueued (will be evaluated as trial #0)")

    # Graceful shutdown: SIGTERM → study.stop()
    def _graceful_stop(signum, frame):
        logger.info("Received SIGTERM — stopping optimization gracefully...")
        study.stop()

    prev_handler = signal.signal(signal.SIGTERM, _graceful_stop)

    logger.info("Starting AWF optimization: %d trials", n_trials)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout if timeout > 0 else None,
        n_jobs=1,
        show_progress_bar=True,
    )

    signal.signal(signal.SIGTERM, prev_handler)

    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed_trials == 0:
        raise RuntimeError(f"No completed trials for {symbol}. Check data quality and constraints.")
    if study.best_value == 0.0:
        logger.warning("All trials scored 0.0 for %s — degenerate result", symbol)
    best_params = study.best_trial.params
    strategy_param_keys = list(best_params.keys())
    if completed_trials < n_trials:
        logger.info("Optimization stopped early: %d/%d trials completed", completed_trials, n_trials)
    strategy = MACDRSIStrategy()

    # 4. Final evaluation
    logger.info("Running final evaluation...")
    base_df = data[BASE_TF]
    best_params.update({
        "symbol": symbol, "tf": "1h", "range": "FULL",
        "optimization_mode": "anchored_walk_forward",
        "n_splits": len(splits), "n_trials": completed_trials,
        "n_trials_requested": n_trials,
        "strategy": strategy.name, "strategy_version": strategy.version,
    })

    allow_flip_final = bool(best_params.get("allow_flip", 1))
    symbol_key = symbol.split("/")[0]
    catastrophic_stop_pct_final = CATASTROPHIC_STOP_PCT.get(symbol_key, CATASTROPHIC_STOP_PCT_DEFAULT)
    best_params["catastrophic_stop_pct"] = catastrophic_stop_pct_final
    buy_s, sell_s = strategy.precompute_signals(data, best_params)
    full_result = simulate_trades_fast(symbol, data, buy_s, sell_s, allow_flip=allow_flip_final, catastrophic_stop_pct=catastrophic_stop_pct_final)
    full_metrics = calculate_metrics(
        full_result.trades, full_result.backtest_days,
        start_equity=STARTING_EQUITY,
        price_data=base_df,
    )

    # Per-split metrics + OOS Monte Carlo
    split_metrics = []
    split_mc_results = []
    last_train_metrics = None
    last_test_metrics = None
    for i, split in enumerate(splits):
        train_end = int(total_bars * split["train_end"])
        test_start = int(total_bars * split.get("test_start", split["train_end"]))
        test_end = int(total_bars * split["test_end"])

        if i == len(splits) - 1:
            tr = simulate_trades_fast(
                symbol, data, buy_s, sell_s,
                start_idx=MIN_WARMUP_BARS, end_idx=train_end,
                allow_flip=allow_flip_final,
                catastrophic_stop_pct=catastrophic_stop_pct_final,
            )
            if tr.trades:
                last_train_metrics = calculate_metrics(
                    tr.trades, tr.backtest_days, start_equity=STARTING_EQUITY,
                    price_data=base_df, start_idx=MIN_WARMUP_BARS, end_idx=train_end,
                )

        te_r = simulate_trades_fast(
            symbol, data, buy_s, sell_s,
            start_idx=test_start, end_idx=test_end,
            allow_flip=allow_flip_final,
            catastrophic_stop_pct=catastrophic_stop_pct_final,
        )
        if te_r.trades:
            tm = calculate_metrics(
                te_r.trades, te_r.backtest_days, start_equity=STARTING_EQUITY,
                price_data=base_df, start_idx=test_start, end_idx=test_end,
            )
            split_entry = {
                "split": i + 1,
                "test_equity": round(tm.equity, 2),
                "test_trades": tm.trades,
                "test_sharpe_time": round(tm.sharpe_ratio_time_based, 4),
                "test_sharpe_equity": round(tm.sharpe_ratio_equity_based, 4),
            }

            # OOS Monte Carlo per split (test trades only)
            test_trades_dicts = [t._asdict() if hasattr(t, '_asdict') else t for t in te_r.trades]
            if len(test_trades_dicts) >= MONTE_CARLO_MIN_TRADES:
                mc_split = monte_carlo_validation(
                    test_trades_dicts, n_simulations=MONTE_CARLO_SIMULATIONS,
                    seed=seed + i, backtest_days=te_r.backtest_days,
                )
                split_mc_results.append(mc_split)
                split_entry["mc_sharpe_ci_low"] = mc_split.sharpe_ci_low
                split_entry["mc_confidence"] = mc_split.confidence_level

            split_metrics.append(split_entry)
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
        "sharpe_time": round(full_metrics.sharpe_ratio_time_based, 4),
        "sharpe_equity": round(full_metrics.sharpe_ratio_equity_based, 4),
        "sortino": round(full_metrics.sortino_ratio, 4),
        "calmar": round(full_metrics.calmar_ratio, 4),
        "calmar_raw": round(full_metrics.calmar_ratio, 4),
        "log_calmar": round(math.log(1.0 + max(0.0, full_metrics.calmar_ratio)), 4),
        "trade_ramp": round(min(1.0, max(0.0, full_metrics.trades_per_year / TARGET_TRADES_YEAR)), 4),
        "objective_type": "log_calmar",
        "recovery_factor": round(full_metrics.recovery_factor, 4),
        "profit_factor": round(full_metrics.profit_factor, 4),
        "expectancy": round(full_metrics.expectancy, 2),
        "profitable_months_ratio": round(full_metrics.profitable_months_ratio, 4),
        "time_in_market": round(full_metrics.time_in_market, 4),
        "split_results": split_metrics,
        "run_timestamp": run_timestamp,
        "trades_file": f"trades_{symbol.replace('/', '_')}_1h_FULL.csv",
        "start_equity": STARTING_EQUITY,
    })

    if last_train_metrics:
        best_params.update({
            "train_equity": round(last_train_metrics.equity, 2),
            "train_trades": last_train_metrics.trades,
            "train_sharpe_time": round(last_train_metrics.sharpe_ratio_time_based, 4),
            "train_sharpe_equity": round(last_train_metrics.sharpe_ratio_equity_based, 4),
        })
    if last_test_metrics:
        best_params.update({
            "test_equity": round(last_test_metrics.equity, 2),
            "test_trades": last_test_metrics.trades,
            "test_sharpe_time": round(last_test_metrics.sharpe_ratio_time_based, 4),
            "test_sharpe_equity": round(last_test_metrics.sharpe_ratio_equity_based, 4),
        })
    if last_train_metrics and last_test_metrics:
        best_params["test_train_ratio"] = (
            round(last_test_metrics.equity / last_train_metrics.equity, 4)
            if last_train_metrics.equity > 0 else 0
        )

    # 5. Monte Carlo (OOS — aggregated from per-split test trades)
    mc = aggregate_mc_results(split_mc_results)
    logger.info(
        "OOS Monte Carlo: %d/%d splits evaluated, confidence=%s",
        len(split_mc_results), len(splits), mc.confidence_level,
    )
    best_params.update({
        "mc_sharpe_mean": mc.sharpe_mean,
        "mc_sharpe_ci_low": mc.sharpe_ci_low,
        "mc_sharpe_ci_high": mc.sharpe_ci_high,
        "mc_max_dd_mean": mc.max_dd_mean,
        "mc_max_dd_ci_low": mc.max_dd_ci_low,
        "mc_max_dd_ci_high": mc.max_dd_ci_high,
        "mc_confidence": mc.confidence_level,
        "mc_robustness": mc.robustness_score,
        "mc_source": "oos_per_split",
        "mc_splits_evaluated": len(split_mc_results),
    })
    best_params["strategy_param_keys"] = strategy_param_keys

    # 6. Save results
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "best_params.json", best_params)
    save_trades_csv(outdir / f"trades_{symbol.replace('/', '_')}_1h_FULL.csv", full_result.trades)

    # 7. HTML report
    trades_dicts = [t._asdict() if hasattr(t, '_asdict') else t for t in full_result.trades]
    optuna_history = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            optuna_history.append({"number": trial.number, "value": trial.value})
    save_report(outdir / f"report_{base}.html", best_params, trades_dicts, optuna_history=optuna_history)

    logger.info(
        "Done %s: Equity=$%s, Sharpe(time)=%.2f, Sharpe(equity)=%.2f, Trades=%d",
        symbol, f"{full_metrics.equity:,.2f}",
        full_metrics.sharpe_ratio_time_based,
        full_metrics.sharpe_ratio_equity_based,
        full_metrics.trades,
    )

    # 8. Notifications
    notify_complete(best_params)

    best_params["run_dir"] = str(outdir.parent)

    return best_params


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="QRE Optimizer — Quant Whale Strategy AWF")
    parser.add_argument("--symbol", type=str, default="BTC/USDC", choices=["BTC/USDC", "SOL/USDC"])
    parser.add_argument("--hours", type=int, default=8760)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--skip-recent", type=int, default=0,
                        help="Skip most recent N hours of data (e.g., 720 = skip last month)")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--test-size", type=float, default=0.20,
                        help="Test window size as fraction (default: 0.20 = 20%%)")
    parser.add_argument("--allow-flip", type=int, default=0, choices=[0, 1],
                        help="Allow position flipping: 1=always-in, 0=selective (default: 0)")
    parser.add_argument("--warm-start", type=str, default=None, metavar="PATH",
                        help="Path to best_params.json for warm-starting (enqueues as first trial)")

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
        run_tag=args.tag,
        skip_recent_hours=args.skip_recent,
        test_size=args.test_size,
        allow_flip=args.allow_flip,
        warm_start_path=args.warm_start,
    )

    print(f"\nResult: {result['symbol']}")
    print(f"  Equity: ${result['equity']:,.2f}")
    print(f"  Sharpe (time):   {result['sharpe_time']:.4f}")
    print(f"  Sharpe (equity): {result['sharpe_equity']:.4f}")
    print(f"  Trades: {result['trades']}")
    print(f"  Max DD: {result['max_drawdown']:.2f}%")


if __name__ == "__main__":
    main()
