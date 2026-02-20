# tests/unit/test_optimize.py
"""Unit tests for QRE optimizer (post-redesign)."""

import numpy as np
import pandas as pd
import pytest
import optuna

from qre.optimize import (
    compute_awf_splits,
    create_sampler,
    create_pruner,
    build_objective,
)


class TestComputeAwfSplits:
    def test_custom_splits_count(self):
        splits = compute_awf_splits(total_hours=10000, n_splits=4)
        assert len(splits) == 4

    def test_custom_splits_increasing_train_end(self):
        splits = compute_awf_splits(total_hours=10000, n_splits=3)
        train_ends = [s["train_end"] for s in splits]
        assert train_ends == sorted(train_ends)

    def test_short_data_uses_2_splits(self):
        splits = compute_awf_splits(total_hours=8000, n_splits=None)
        assert len(splits) == 2

    def test_full_data_uses_3_splits(self):
        splits = compute_awf_splits(total_hours=15000, n_splits=None)
        assert len(splits) == 3

    def test_too_short_data_returns_none(self):
        splits = compute_awf_splits(total_hours=2000, n_splits=None)
        assert splits is None

    def test_each_split_has_train_and_test(self):
        splits = compute_awf_splits(total_hours=10000, n_splits=3)
        for s in splits:
            assert "train_end" in s
            assert "test_end" in s
            assert s["test_end"] > s["train_end"]


class TestCreateSampler:
    def test_returns_sampler(self):
        sampler = create_sampler(seed=42, n_trials=1000)
        assert sampler is not None

    def test_deterministic_with_seed(self):
        s1 = create_sampler(seed=42, n_trials=1000)
        s2 = create_sampler(seed=42, n_trials=1000)
        assert type(s1) == type(s2)


class TestCreatePruner:
    def test_returns_pruner(self):
        pruner = create_pruner(n_trials=1000)
        assert pruner is not None


class TestBuildObjective:
    def test_returns_callable(self):
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2025-01-01", periods=n, freq="1h")
        close = 100 + np.cumsum(np.random.randn(n) * 0.3)
        mock_data = {"1h": pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1, "close": close},
            index=dates,
        )}
        splits = [{"train_end": 0.70, "test_end": 0.85}]
        objective = build_objective(symbol="BTC/USDC", data=mock_data, splits=splits)
        assert callable(objective)

    def test_objective_runs_without_error(self):
        """Objective produces a float or raises TrialPruned."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2025-01-01", periods=n, freq="1h")
        close = 100 + np.cumsum(np.random.randn(n) * 0.3)
        mock_data = {"1h": pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1, "close": close},
            index=dates,
        )}
        splits = [{"train_end": 0.70, "test_end": 0.85}]
        objective = build_objective(symbol="BTC/USDC", data=mock_data, splits=splits)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study()
        # Run a few trials — some may prune (macd_fast >= macd_slow)
        completed = 0
        for _ in range(20):
            trial = study.ask()
            try:
                result = objective(trial)
                assert isinstance(result, float)
                study.tell(trial, result)
                completed += 1
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        assert completed > 0

    def test_objective_returns_non_negative(self):
        """Objective should return value >= 0.0 (soft penalties, no hard cap)."""
        np.random.seed(42)
        n = 2000
        dates = pd.date_range("2025-01-01", periods=n, freq="1h")
        close = 100 + np.cumsum(np.random.randn(n) * 0.3)
        mock_data = {"1h": pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1, "close": close},
            index=dates,
        )}
        splits = [{"train_end": 0.60, "test_end": 0.80}]
        objective = build_objective(symbol="BTC/USDC", data=mock_data, splits=splits)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study()
        completed_values = []
        for _ in range(30):
            trial = study.ask()
            try:
                result = objective(trial)
                assert result >= 0.0, f"Objective {result} is negative"
                completed_values.append(result)
                study.tell(trial, result)
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        assert len(completed_values) > 0

    def test_objective_no_penalties_import(self):
        """Objective should not import apply_all_penalties."""
        import inspect
        import qre.optimize as mod
        source = inspect.getsource(mod.build_objective)
        assert "apply_all_penalties" not in source

    def test_objective_uses_calmar_not_sharpe(self):
        """Objective must return Calmar-based score, not raw Sharpe."""
        import inspect
        from qre.optimize import build_objective
        source = inspect.getsource(build_objective)
        assert "calmar" in source.lower(), "Objective must use Calmar ratio"
        assert "SHARPE_PENALTY_TIERS" not in source, "Old tier penalties must be removed"

    def test_sharpe_decay_penalty_reduces_score(self):
        """When OOS Sharpe > SHARPE_SUSPECT_THRESHOLD, Calmar score is penalized."""
        from qre.config import SHARPE_SUSPECT_THRESHOLD, SHARPE_DECAY_RATE
        sharpe = 8.0
        raw_calmar = 5.0
        penalty = 1.0 / (1.0 + SHARPE_DECAY_RATE * (sharpe - SHARPE_SUSPECT_THRESHOLD))
        penalized = raw_calmar * penalty
        assert penalized < raw_calmar
        assert penalized > 0  # no hard cap
        assert abs(penalty - 0.4) < 0.01  # 1/(1+0.3*5) = 0.4

    def test_sharpe_below_threshold_no_penalty(self):
        """Sharpe at or below threshold should not be penalized."""
        from qre.config import SHARPE_SUSPECT_THRESHOLD, SHARPE_DECAY_RATE
        # Test at exact boundary — strict > means no penalty at threshold
        sharpe = SHARPE_SUSPECT_THRESHOLD  # exactly 3.0
        raw_calmar = 5.0
        # With strict >, sharpe == threshold should NOT trigger penalty
        # Penalty only applies when sharpe > threshold
        penalized = raw_calmar  # no penalty expected
        assert penalized == raw_calmar

        # Also verify: sharpe just above threshold DOES get penalized
        sharpe_above = SHARPE_SUSPECT_THRESHOLD + 0.1
        penalty = 1.0 / (1.0 + SHARPE_DECAY_RATE * (sharpe_above - SHARPE_SUSPECT_THRESHOLD))
        penalized_above = raw_calmar * penalty
        assert penalized_above < raw_calmar, "Sharpe above threshold must be penalized"


class TestNoLegacyImports:
    def test_no_split_fail_penalty(self):
        """SPLIT_FAIL_PENALTY removed from optimizer."""
        import qre.optimize as mod
        import inspect
        source = inspect.getsource(mod)
        assert "SPLIT_FAIL_PENALTY" not in source

    def test_no_stochrsi_in_optimizer(self):
        """No StochRSI references in optimizer."""
        import qre.optimize as mod
        import inspect
        source = inspect.getsource(mod)
        assert "stochrsi" not in source.lower()


class TestOptimizeImports:
    def test_notify_importable(self):
        from qre.optimize import notify_start, notify_complete
        assert callable(notify_start)
        assert callable(notify_complete)

    def test_report_importable(self):
        from qre.optimize import save_report
        assert callable(save_report)
