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

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import structlog

from sacv.orchestration.state import ProjectMode, WorkflowPhase
from sacv.nodes._node_context import bind_node_context
from sacv.nodes._node_timer import node_timer
from sacv.modes.greenfield import GreenfieldConfig
from sacv.modes.brownfield import BrownfieldConfig

if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

# Indicators that the repo is brownfield
_BROWNFIELD_SIGNALS: list[str] = [
    "pom.xml",          # existing Maven build = pre-existing codebase
    "package-lock.json",
    ".git/refs/heads",  # has existing git history
]

_GREENFIELD_MAX_COMMITS = 5   # fewer than this = likely new project


async def _detect_mode(cwd: Path) -> ProjectMode:
    """
    Heuristic detection: pure function, no I/O beyond filesystem stat calls.
    Returns BROWNFIELD if any signal file exists and commit count exceeds
    the greenfield threshold; GREENFIELD otherwise.
    """
    git_dir = cwd / ".git"
    has_git = git_dir.is_dir()

    for signal in _BROWNFIELD_SIGNALS:
        candidate = cwd / signal

        # Skip git-specific signals when no git repo
        if signal == ".git/refs/heads" and not has_git:
            continue

        if not candidate.exists():
            continue

        # If we have git, consult commit count before declaring brownfield
        if has_git:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "git", "rev-list", "--count", "HEAD",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(cwd),
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                commit_count = int(stdout.decode().strip() or "0")
                if commit_count <= _GREENFIELD_MAX_COMMITS:
                    return ProjectMode.GREENFIELD
            except Exception:
                pass

        return ProjectMode.BROWNFIELD

    return ProjectMode.GREENFIELD


def make_mode_router_node(deps: "NodeDeps") -> "Callable[[WorkflowState], Coroutine[Any, Any, dict[str, object]]]":

    async def mode_router_node(state: "WorkflowState") -> dict[str, object]:
        bind_node_context(state, "mode_router")
        async with node_timer("mode_router", state=state) as timing:
            # Honour explicitly provided mode; auto-detect only if absent
            provided = state.get("project_mode")
            if provided in (ProjectMode.GREENFIELD.value, ProjectMode.BROWNFIELD.value):
                mode_str = provided
            else:
                # Use git root from BranchManager, not process CWD
                project_root = deps.git.repo_root
                mode     = await _detect_mode(project_root)
                mode_str = mode.value

            log.info("mode_router.resolved", mode=mode_str, task_id=state["task_id"])

            timing["mode"] = mode_str
            return {
                "project_mode":  mode_str,
                "current_phase": WorkflowPhase.SCOUT.value,
            }

    return mode_router_node
