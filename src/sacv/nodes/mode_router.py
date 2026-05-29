"""
nodes/mode_router.py
====================
Detects or confirms the project mode (GREENFIELD / BROWNFIELD) and
applies the corresponding ModeConfig to the state.

Mode is set once at the start of the session and never mutated.
Detection heuristics (git history depth, legacy package presence, etc.)
are deterministic Python — no LLM call.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from sacv.orchestration.state import ProjectMode, WorkflowPhase
from sacv.modes.greenfield import GreenfieldConfig
from sacv.modes.brownfield import BrownfieldConfig

if TYPE_CHECKING:
    from sacv.orchestration.graph import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

# Indicators that the repo is brownfield
_BROWNFIELD_SIGNALS: list[str] = [
    "pom.xml",          # existing Maven build = pre-existing codebase
    "package-lock.json",
    ".git/refs/heads",  # has existing git history
]

_GREENFIELD_MAX_COMMITS = 5   # fewer than this = likely new project


def _detect_mode(cwd: Path) -> ProjectMode:
    """
    Heuristic detection: pure function, no I/O beyond filesystem stat calls.
    Returns BROWNFIELD if any signal file exists; GREENFIELD otherwise.
    """
    for signal in _BROWNFIELD_SIGNALS:
        candidate = cwd / signal
        if candidate.exists():
            # Additional check: if it's a git repo, count commits
            git_dir = cwd / ".git"
            if git_dir.is_dir():
                try:
                    import subprocess
                    result = subprocess.run(
                        ["git", "rev-list", "--count", "HEAD"],
                        capture_output=True, text=True, cwd=str(cwd), timeout=5,
                    )
                    commit_count = int(result.stdout.strip() or "0")
                    if commit_count <= _GREENFIELD_MAX_COMMITS:
                        return ProjectMode.GREENFIELD
                except Exception:
                    pass
            return ProjectMode.BROWNFIELD
    return ProjectMode.GREENFIELD


def make_mode_router_node(deps: "NodeDeps"):

    async def mode_router_node(state: "WorkflowState") -> dict:
        # Honour explicitly provided mode; auto-detect only if absent
        provided = state.get("project_mode")
        if provided in (ProjectMode.GREENFIELD.value, ProjectMode.BROWNFIELD.value):
            mode_str = provided
        else:
            mode = _detect_mode(Path.cwd())
            mode_str = mode.value

        log.info("mode_router.resolved", mode=mode_str, task_id=state["task_id"])

        return {
            "project_mode":  mode_str,
            "current_phase": WorkflowPhase.SCOUT.value,
        }

    return mode_router_node
