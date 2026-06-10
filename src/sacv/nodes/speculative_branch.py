"""
nodes/speculative_branch.py
===========================
When the Actor fails a second time on the main branch, the orchestrator
forks the state into multiple isolated git branches and evaluates alternative
strategies concurrently (throttled to ≤max_parallel_branches via asyncio.gather).

Each branch runs a complete Actor → Critics → Verifier sub-cycle.
The first branch to produce a PASS verdict wins; its diff is committed
and all other branches are stashed and logged.

If all branches fail, the node signals HITL escalation.
"""
from __future__ import annotations

import asyncio
import copy
import tempfile
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import structlog

from sacv.git.branch_manager import sanitize_branch_name
from sacv.nodes._node_context import bind_node_context
from sacv.nodes._node_timer import node_timer
from sacv.orchestration.state import WorkflowPhase, VerifierVerdict, DiagnosticVerdict, CRITIC_RESET

if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState, StrategyCandidate

log = structlog.get_logger(__name__)


def _merge_branch_state(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """
    Merge node output into branch_state, correctly handling CRITIC_RESET.

    Uses deep copy to prevent nested mutable objects (dicts, lists) from
    being shared between the returned merged dict and the original base dict.

    Replaces CRITIC_RESET with [] so branch_state always holds a list.
    This mirrors what the LangGraph _merge_lists reducer does.
    """
    merged = copy.deepcopy(base)
    merged.update(update)
    if merged.get("critic_findings") is CRITIC_RESET:
        merged["critic_findings"] = []
    return merged


def make_speculative_branch_node(deps: "NodeDeps") -> "Callable[[WorkflowState], Coroutine[Any, Any, dict[str, object]]]":

    async def speculative_branch_node(state: "WorkflowState") -> dict[str, object]:
        bind_node_context(state, "speculative_branch")
        async with node_timer("speculative_branch", state=state) as timing:
            task_id    = state["task_id"]
            candidates = state.get("strategy_candidates", [])
            exhausted  = list(state.get("exhausted_branches", []))
            correction = state["correction_state"]

            log.info(
                "speculative_branch.start",
                task_id=task_id,
                strategies=len(candidates),
            )

            # Stash the current branch before forking
            current_branch = correction.get("branch_name") or "main"
            stash_ref      = await asyncio.to_thread(
                deps.git.stash, f"sacv-speculative-stash-{task_id}"
            )
            exhausted.append(current_branch)
            # Persist stash_ref so downstream nodes (HITL) can restore pre-spec state

            # Sort remaining candidates by composite_score (highest priority first)
            strategies_to_try = [
                c for c in candidates
                if f"agent-task-{sanitize_branch_name(task_id[:8])}-{c['strategy_id']}" not in exhausted
            ]
            strategies_to_try.sort(key=lambda x: x["composite_score"], reverse=True)

            if not strategies_to_try:
                log.warning("speculative_branch.no_strategies_left")
                return {
                    "current_phase":          WorkflowPhase.SPECULATIVE_BRANCH.value,
                    "active_branches":        [],
                    "exhausted_branches":     exhausted,
                    "speculative_stash_ref":  stash_ref,  # for HITL restoration
                    "verifier_verdict":       VerifierVerdict(
                        test_result="FAIL",
                        diagnostic=DiagnosticVerdict.AMBIGUOUS.value,
                        phase1_passed=False,
                        phase2_passed=False,
                        test_failures=[{"message": "All strategies exhausted"}],
                        performance_delta=None,
                        visual_diff_result=None,
                        docker_exit_code=-1,
                        playwright_trace_path=None,
                        otel_trace=None,
                        actuator_snapshot=None,
                        blocked_by_critic=False,
                    ),
                }

            # Evaluate at most max_parallel_branches strategies
            to_evaluate = strategies_to_try[: deps.config.max_parallel_branches]
            remaining   = strategies_to_try[deps.config.max_parallel_branches:]

            results: list[tuple[str, VerifierVerdict | None]] = await asyncio.gather(
                *[
                    _evaluate_branch(state, strategy, deps)
                    for strategy in to_evaluate
                ]
            )

            # Find first passing branch
            winning_branch:   str | None = None
            winning_verdict:  VerifierVerdict | None = None
            new_exhausted     = list(exhausted)

            for branch_name, verdict, reason in results:
                if verdict and verdict["test_result"] == "PASS":
                    winning_branch  = branch_name
                    winning_verdict = verdict
                    break
                else:
                    new_exhausted.append(branch_name)
                    log.warning(
                        "speculative_branch.branch_failed",
                        branch=branch_name,
                        reason=reason,
                    )
                    # _evaluate_branch's finally block already cleaned up the
                    # worktree; no checkout/stash needed here.

            # Prune stale worktree registry so delete_branch doesn't fail on
            # "branch already checked out at <removed_worktree>" errors.
            if new_exhausted:
                await asyncio.to_thread(deps.git.prune_worktrees)

            # Delete failed speculative branches (keep only the winner)
            for exhausted_branch in new_exhausted:
                try:
                    await asyncio.to_thread(
                        deps.git.delete_branch, exhausted_branch, force=True
                    )
                    log.info("speculative_branch.branch_deleted", branch=exhausted_branch)
                except Exception:
                    log.warning("speculative_branch.branch_delete_failed",
                                branch=exhausted_branch, exc_info=True)

            if winning_branch:
                log.info("speculative_branch.winner", branch=winning_branch)
                await asyncio.to_thread(deps.git.checkout, winning_branch)
                return {
                    "current_phase":          WorkflowPhase.SPECULATIVE_BRANCH.value,
                    "active_branches":        [winning_branch],
                    "exhausted_branches":     new_exhausted,
                    "speculative_stash_ref":  stash_ref,  # preserve for HITL
                    "verifier_verdict":       winning_verdict,
                }

            # Queue remaining strategies for next speculative cycle (if any)
            queued_names = [
                f"agent-task-{sanitize_branch_name(task_id[:8])}-{s['strategy_id']}"
                for s in remaining
            ]

            log.warning(
                "speculative_branch.all_failed",
                exhausted=len(new_exhausted),
                queued=len(queued_names),
            )

            return {
                "current_phase":          WorkflowPhase.SPECULATIVE_BRANCH.value,
                "active_branches":        queued_names,
                "exhausted_branches":     new_exhausted,
                "speculative_stash_ref":  stash_ref,  # for HITL restoration
                "verifier_verdict":       VerifierVerdict(
                    test_result="FAIL",
                    diagnostic=DiagnosticVerdict.AMBIGUOUS.value,
                    phase1_passed=False,
                    phase2_passed=False,
                    test_failures=[{"message": "All evaluated branches failed"}],
                    performance_delta=None,
                    visual_diff_result=None,
                    docker_exit_code=-1,
                    playwright_trace_path=None,
                    otel_trace=None,
                    actuator_snapshot=None,
                    blocked_by_critic=False,
                ),
                "critic_findings": CRITIC_RESET,
            }

    return speculative_branch_node


async def _evaluate_branch(
    state:    "WorkflowState",
    strategy: "StrategyCandidate",
    deps:     "NodeDeps",
) -> tuple[str, "VerifierVerdict | None", str]:
    """
    Run a complete Actor → Critics → Verifier cycle for one strategy.

    Returns a 3-tuple: (branch_name, verdict_or_none, failure_reason).
    The failure_reason describes why the branch failed (or 'pass' on success).
    """
    task_id     = state["task_id"]
    branch_name = f"agent-task-{sanitize_branch_name(task_id[:8])}-{strategy['strategy_id']}"
    worktree_path = Path(tempfile.mkdtemp(prefix=f"sacv-spec-{branch_name}-"))

    try:
        # ── Isolated git worktree ────────────────────────────────────────
        await asyncio.to_thread(deps.git.create_worktree, branch_name, worktree_path)

        # ── Isolated deps per worktree ───────────────────────────────────
        from sacv.git.branch_manager import BranchManager
        from sacv.git.diff_engine import DiffEngine

        branch_git = BranchManager(repo_root=worktree_path)
        # Use factory method to preserve port configuration without
        # accessing private attributes (HIGH-005, MED-006 fix).
        branch_sandbox = deps.sandbox.create_isolated_instance(str(worktree_path))
        branch_diff = DiffEngine(repo_root=worktree_path)

        branch_deps = deps.with_git_and_sandbox(
            git=branch_git,
            sandbox=branch_sandbox,
            diff=branch_diff,
        )

        # Shared branch subgraph: actor → preflight → critics → verifier
        from langgraph.checkpoint.memory import MemorySaver
        from sacv.orchestration.graph import build_branch_subgraph

        branch_state = {
            **state,
            "selected_strategy": strategy,
            "correction_state": {
                **state["correction_state"],
                "branch_name":        branch_name,
                "attempt_count":      0,          # each branch gets a clean start
                "stagnation_pattern": "none",     # clear inherited stagnation signal
                "error_history":      [],         # prevent semantic stagnation bleed
                "last_error_hash":    None,
            },
            "critic_findings": CRITIC_RESET,
        }

        # Build, compile, and run the shared mini-workflow for this branch.
        # The subgraph is compiled with an isolated MemorySaver so each
        # branch runs independently without state bleed (WORKER-001).
        subgraph = build_branch_subgraph(branch_deps)
        compiled = subgraph.compile(checkpointer=MemorySaver())

        result = await compiled.ainvoke(branch_state)

        verdict = result.get("verifier_verdict")
        preflight_result = (result.get("preflight_result") or {}).copy()
        if not preflight_result.get("passed", True):
            lsp_count = len(preflight_result.get("lsp_errors", []))
            arch_count = len(preflight_result.get("arch_violations", []))
            reason = f"preflight_failed lsp={lsp_count} arch={arch_count}"
            log.info("speculative_branch.preflight_failed", branch=branch_name, reason=reason)
            return branch_name, None, reason

        if verdict and verdict["test_result"] == "FAIL":
            diag = verdict.get("diagnostic", "?")
            p1 = verdict.get("phase1_passed", False)
            p2 = verdict.get("phase2_passed", False)
            reason = f"verifier_fail diagnostic={diag} phase1={p1} phase2={p2}"
            return branch_name, verdict, reason

        return branch_name, verdict, "pass"

    except Exception as exc:
        reason = f"exception: {type(exc).__name__}: {exc}"
        log.error(
            "speculative_branch.evaluation_error",
            branch=branch_name,
            error=reason,
            exc_info=True,
        )
        return branch_name, None, reason
    finally:
        # Always clean up the worktree and its temp directory
        try:
            await asyncio.to_thread(deps.git.remove_worktree, worktree_path)
        except Exception:
            log.warning(
                "speculative_branch.worktree_cleanup_failed",
                path=str(worktree_path),
                exc_info=True,
            )
        try:
            shutil.rmtree(str(worktree_path), ignore_errors=True)
        except Exception:
            log.warning(
                "speculative_branch.tempdir_cleanup_failed",
                path=str(worktree_path),
                exc_info=True,
            )
