# tests/unit/test_hooks.py
"""Unit tests for QRE hook system."""

from unittest.mock import patch

import pytest

from qre.hooks import _hooks, run_pre_hooks, run_post_hooks, register_hook, clear_hooks
from qre.hooks.auto_diagnose import register_auto_diagnose


@pytest.fixture(autouse=True)
def reset_hooks():
    clear_hooks()
    yield
    clear_hooks()


class TestHookSystem:
    def test_register_pre_hook(self):
        """Can register a pre_run hook."""
        called = []
        def my_hook(config):
            called.append(True)

        register_hook("pre_run", my_hook)
        run_pre_hooks({"symbol": "BTC/USDC"})
        assert len(called) == 1

    def test_register_post_hook(self):
        """Can register a post_run hook."""
        results = []
        def my_hook(result):
            results.append(result)

        register_hook("post_run", my_hook)
        run_post_hooks({"equity": 51000})
        assert len(results) == 1

    def test_hooks_dont_crash_pipeline(self):
        """A failing hook logs error but doesn't crash."""
        def bad_hook(config):
            raise ValueError("broken")

        register_hook("pre_run", bad_hook)
        # Should NOT raise
        run_pre_hooks({"symbol": "BTC/USDC"})

    def test_unknown_hook_type_raises(self):
        with pytest.raises(ValueError):
            register_hook("unknown_type", lambda x: x)


class TestAutoDiagnoseHook:
    def test_registers_post_run_hook(self):
        """register_auto_diagnose() adds exactly one post_run hook."""
        register_auto_diagnose()
        assert len(_hooks["post_run"]) == 1

    def test_hook_calls_analyze_run(self):
        """post_run hook calls analyze_run with run_dir from result dict."""
        register_auto_diagnose()

        with patch("qre.hooks.auto_diagnose.analyze_run") as mock_analyze:
            run_post_hooks({"run_dir": "/tmp/results/2026-02-14_test"})

        mock_analyze.assert_called_once_with("/tmp/results/2026-02-14_test")

    def test_hook_skips_when_no_run_dir(self):
        """Hook does nothing when result dict has no run_dir key."""
        register_auto_diagnose()

        with patch("qre.hooks.auto_diagnose.analyze_run") as mock_analyze:
            run_post_hooks({"equity": 51000})

        mock_analyze.assert_not_called()
