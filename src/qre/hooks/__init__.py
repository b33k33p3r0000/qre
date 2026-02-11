"""
QRE Hook System
===============
Extensibility points for future agent integration.

Usage:
    from qre.hooks import register_hook, run_pre_hooks, run_post_hooks

    def my_pre_hook(config: dict):
        # Validate config before run
        pass

    register_hook("pre_run", my_pre_hook)
"""

import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger("qre.hooks")

VALID_HOOK_TYPES = {"pre_run", "post_run"}

_hooks: Dict[str, List[Callable]] = {
    "pre_run": [],
    "post_run": [],
}


def register_hook(hook_type: str, fn: Callable) -> None:
    """Register a hook function."""
    if hook_type not in VALID_HOOK_TYPES:
        raise ValueError(f"Unknown hook type: {hook_type}. Valid: {VALID_HOOK_TYPES}")
    _hooks[hook_type].append(fn)


def run_pre_hooks(config: Dict[str, Any]) -> None:
    """Run all pre_run hooks. Errors are logged, not raised."""
    for fn in _hooks["pre_run"]:
        try:
            fn(config)
        except Exception as e:
            logger.error(f"Pre-run hook {fn.__name__} failed: {e}")


def run_post_hooks(result: Dict[str, Any]) -> None:
    """Run all post_run hooks. Errors are logged, not raised."""
    for fn in _hooks["post_run"]:
        try:
            fn(result)
        except Exception as e:
            logger.error(f"Post-run hook {fn.__name__} failed: {e}")


def clear_hooks() -> None:
    """Clear all registered hooks (useful for testing)."""
    for key in _hooks:
        _hooks[key].clear()
