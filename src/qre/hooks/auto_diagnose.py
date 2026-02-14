"""Auto-diagnose hook — runs analyze pipeline after each optimizer run."""
from __future__ import annotations

import logging

from qre.analyze import analyze_run
from qre.hooks import register_hook

log = logging.getLogger(__name__)


def _on_post_run(result: dict) -> None:
    run_dir = result.get("run_dir")
    if not run_dir:
        log.warning("auto_diagnose: no run_dir in result, skipping")
        return
    log.info("auto_diagnose: analyzing %s", run_dir)
    analyze_run(run_dir)


def register_auto_diagnose() -> None:
    register_hook("post_run", _on_post_run)
