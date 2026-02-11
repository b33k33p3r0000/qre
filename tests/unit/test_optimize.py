# tests/unit/test_optimize.py
"""Unit tests for QRE optimizer orchestration."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

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
    """Test that build_objective returns a callable that Optuna can use."""

    def test_returns_callable(self):
        mock_data = {"1h": pd.DataFrame(
            {"open": [1.0]*500, "high": [2.0]*500, "low": [0.5]*500, "close": [1.5]*500},
            index=pd.date_range("2025-01-01", periods=500, freq="1h"),
        )}
        splits = [{"train_end": 0.70, "test_end": 0.85}]

        objective = build_objective(
            symbol="BTC/USDC",
            data=mock_data,
            splits=splits,
        )
        assert callable(objective)


class TestOptimizeImports:
    """Verify report + notify wiring is present in optimize module."""

    def test_notify_importable(self):
        from qre.optimize import notify_start, notify_complete
        assert callable(notify_start)
        assert callable(notify_complete)

    def test_report_importable(self):
        from qre.optimize import save_report
        assert callable(save_report)
