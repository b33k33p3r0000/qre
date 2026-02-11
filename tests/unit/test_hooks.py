# tests/unit/test_hooks.py
"""Unit tests for QRE hook system."""

import pytest

from qre.hooks import run_pre_hooks, run_post_hooks, register_hook, clear_hooks


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
