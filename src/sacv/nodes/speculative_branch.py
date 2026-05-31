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
from typing import TYPE_CHECKING

import structlog

from sacv.orchestration.state import WorkflowPhase, VerifierVerdict, DiagnosticVerdict

if TYPE_CHECKING:
    from sacv.orchestration.graph import NodeDeps
    from sacv.orchestration.state import WorkflowState, StrategyCandidate

log = structlog.get_logger(__name__)


def make_speculative_branch_node(deps: "NodeDeps"):

    async def speculative_branch_node(state: "WorkflowState") -> dict:
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
            if f"agent-task-{task_id[:8]}-{c['strategy_id']}" not in exhausted
        ]
        strategies_to_try.sort(key=lambda x: x["composite_score"], reverse=True)

        if not strategies_to_try:
            log.warning("speculative_branch.no_strategies_left")
            return {
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
                    critic_findings=[],
                    docker_exit_code=-1,
                    playwright_trace_path=None,
                    otel_trace=None,
                    actuator_snapshot=None,
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

        for branch_name, verdict in results:
            if verdict and verdict["test_result"] == "PASS":
                winning_branch  = branch_name
                winning_verdict = verdict
                break
            else:
                new_exhausted.append(branch_name)
                # Stash failed branch
                await asyncio.to_thread(deps.git.checkout, branch_name)
                await asyncio.to_thread(deps.git.stash, f"sacv-failed-{branch_name}")

        if winning_branch:
            log.info("speculative_branch.winner", branch=winning_branch)
            await asyncio.to_thread(deps.git.checkout, winning_branch)
            return {
                "current_phase":      WorkflowPhase.MEMORY_CONSOLIDATION.value,
                "active_branches":    [winning_branch],
                "exhausted_branches": new_exhausted,
                "speculative_stash_ref": stash_ref,  # preserve for HITL
                "verifier_verdict":   winning_verdict,
            }

        # Queue remaining strategies for next speculative cycle (if any)
        queued_names = [
            f"agent-task-{task_id[:8]}-{s['strategy_id']}"
            for s in remaining
        ]

        log.warning(
            "speculative_branch.all_failed",
            exhausted=len(new_exhausted),
            queued=len(queued_names),
        )

        return {
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
                critic_findings=[],
                docker_exit_code=-1,
                playwright_trace_path=None,
                otel_trace=None,
                actuator_snapshot=None,
            ),
        }

    return speculative_branch_node


async def _evaluate_branch(
    state:    "WorkflowState",
    strategy: "StrategyCandidate",
    deps:     "NodeDeps",
) -> tuple[str, "VerifierVerdict | None"]:
    """
    Run a complete Actor → Critics → Verifier cycle for one strategy.
    Concurrency is controlled by asyncio.gather in the caller;
    per-critic throttling is handled by the semaphore inside _run_critic.
    """
    task_id     = state["task_id"]
    branch_name = f"agent-task-{task_id[:8]}-{strategy['strategy_id']}"

    try:
        await asyncio.to_thread(deps.git.create_branch, branch_name)
        await asyncio.to_thread(deps.git.checkout, branch_name)

        # Inline mini-workflow: actor → preflight → critics → verifier
        # Note: per-critic semaphore inside _run_critic handles throttling;
        # no outer semaphore here — acquiring it would deadlock with
        # max_parallel_branches >= 2 (see BUG-001 fix).
        from sacv.nodes.actor       import make_actor_node
        from sacv.nodes.preflight_node import make_preflight_node
        from sacv.orchestration.verifier_utils import (
            run_verifier_with_confidence as _run_verifier_with_confidence,
        )
        from sacv.nodes.critics.security    import make_security_critic_node
        from sacv.nodes.critics.style       import make_style_critic_node
        from sacv.nodes.critics.consistency import make_consistency_critic_node

        branch_state = {
            **state,
            "selected_strategy": strategy,
            "correction_state": {
                **state["correction_state"],
                "branch_name": branch_name,
            },
            "critic_findings": [],
        }

        actor_out   = await make_actor_node(deps)(branch_state)
        if not actor_out.get("diff_proposal"):
            return branch_name, None

        branch_state = {**branch_state, **actor_out}

        # Run preflight — if it fails, skip this branch (LSP/compile/arch checks)
        preflight_out = await make_preflight_node(deps)(branch_state)
        preflight_result = preflight_out.get("preflight_result") or {}
        if not preflight_result.get("passed", True):
            log.info(
                "speculative_branch.preflight_failed",
                branch=branch_name,
                lsp_errors=len(preflight_result.get("lsp_errors", [])),
            )
            return branch_name, None  # treat as failed branch

        branch_state = {**branch_state, **preflight_out}

        # Run critics concurrently (respecting per-critic semaphore throttling)
        async def _run_with_semaphore(critic_fn, state):
            async with deps.critic_semaphore:
                return await critic_fn(state)

        sec_out, sty_out, con_out = await asyncio.gather(
            _run_with_semaphore(make_security_critic_node(deps), branch_state),
            _run_with_semaphore(make_style_critic_node(deps), branch_state),
            _run_with_semaphore(make_consistency_critic_node(deps), branch_state),
        )

        branch_state = {
            **branch_state,
            "critic_findings": (
                sec_out.get("critic_findings", [])
                + sty_out.get("critic_findings", [])
                + con_out.get("critic_findings", [])
            ),
        }

        ver_out = await _run_verifier_with_confidence(branch_state, deps)
        return branch_name, ver_out.get("verifier_verdict")

    except Exception as exc:
        log.error(
            "speculative_branch.evaluation_error",
            branch=branch_name,
            error=str(exc),
        )
        return branch_name, None
