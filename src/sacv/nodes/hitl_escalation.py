"""
nodes/hitl_escalation.py
========================
HITL (Human-In-The-Loop) escalation gate.

When MAX_SELF_CORRECTION_CYCLES is reached or all speculative branches fail:
1. Build the full EscalationPayload JSON.
2. Stash the active branch and reset to the last known green commit.
3. Serialise the payload to ``.workflow/escalations/<id>.json``.
4. Call LangGraph ``interrupt()`` — the graph pauses here.

The developer reviews the payload and resumes via:
    ``workflow resume --escalation-id <id> [--branch <branch>]``

All correction choices made by the developer during HITL are later written
back to AgentMemory as procedural constraints (see memory_consolidation.py).
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from langgraph.types import interrupt

from sacv.orchestration.state import WorkflowPhase, EscalationPayload, ResolutionHint
from sacv.interfaces.memory_provider import EpisodicEvent

if TYPE_CHECKING:
    from sacv.orchestration.graph import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

_ESCALATION_DIR = Path(".workflow/escalations")
_WORKFLOW_VERSION = "sacv-1.0"


def make_hitl_escalation_node(deps: "NodeDeps"):

    async def hitl_escalation_node(state: "WorkflowState") -> dict:
        esc_id     = str(uuid.uuid4())
        task_id    = state["task_id"]
        correction = state["correction_state"]
        verdict    = state.get("verifier_verdict")
        exhausted  = state.get("exhausted_branches", [])

        log.warning(
            "hitl.escalating",
            task_id=task_id,
            esc_id=esc_id,
            attempts=correction["attempt_count"],
            branches_exhausted=len(exhausted),
        )

        # ── 1. Build resolution hints ─────────────────────────────────────
        hints: list[ResolutionHint] = _build_hints(verdict, state)

       # ── 2. Capture git state ───────────────────────────────────────────
        stash_ref: str | None = None
        current_branch = (
            correction.get("branch_name")
            or await asyncio.to_thread(deps.git.current_branch)
        )
        if current_branch and current_branch != "main":
            stash_ref = await asyncio.to_thread(
                deps.git.stash, f"sacv-hitl-{esc_id[:8]}"
            )

        green_sha = await asyncio.to_thread(deps.git.get_last_green_commit)
        uncommitted = await asyncio.to_thread(deps.git.uncommitted_files)

        # Reset to last green state — errors are captured but never block escalation
        git_reset_error: str | None = None
        try:
            await asyncio.to_thread(deps.git.reset_hard, green_sha)
            await asyncio.to_thread(deps.git.checkout, "main")
        except Exception as exc:
            git_reset_error = str(exc)
            log.error("hitl.git_reset_failed", error=git_reset_error, green_sha=green_sha)

        git_state = {
            "active_branch":       current_branch,
            "stash_ref":           stash_ref,
            "last_green_commit":   green_sha,
            "stashed_branches":    exhausted,
            "uncommitted_files":   uncommitted,
            "git_reset_failed":    git_reset_error,  # None if git succeeded
            "stash_pop_command":   f"git stash pop {stash_ref}" if stash_ref else None,
            "stash_note":          (
                "Run stash_pop_command to restore pre-speculation work. "
                "If stash_pop fails due to conflicts, use 'git stash list' "
                "and 'git stash drop' to clean up."
            ),
        }

        # ── 3. Build full escalation payload ──────────────────────────────
        payload = EscalationPayload(
            escalation_id=esc_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            workflow_version=_WORKFLOW_VERSION,
            task_id=task_id,
            task_description=state.get("task_description", ""),
            failure_summary={
                "total_attempts":       correction["attempt_count"],
                "branches_exhausted":   exhausted,
                "stagnation_pattern":   correction.get("stagnation_pattern", "none"),
                "last_verifier_output": verdict,
                "critic_findings":      state.get("critic_findings", []),
            },
            git_state=git_state,
            resolution_hints=hints,
            resume_instructions={
                "command": f"workflow resume --escalation-id {esc_id}",
                "state_file": f".workflow/escalations/{esc_id}.json",
                "note": (
                    "Review the failure summary, apply manual corrections if needed, "
                    "then run the resume command."
                ),
            },
        )

        # ── 4. Persist payload ────────────────────────────────────────────
        _ESCALATION_DIR.mkdir(parents=True, exist_ok=True)
        payload_path = _ESCALATION_DIR / f"{esc_id}.json"
        payload_path.write_text(json.dumps(payload, indent=2))

        log.warning("hitl.payload_written", path=str(payload_path))

        # ── 5. Record escalation in AgentMemory ───────────────────────────
        await deps.memory.store_episodic(EpisodicEvent(
            session_id=state["session_id"],
            event_type="hitl_escalation",
            payload={"escalation_id": esc_id, "task_id": task_id},
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

        # ── 6. Pause graph — developer must manually resume ───────────────
        # ``interrupt()`` raises an internal LangGraph signal; the graph
        # checkpoint is saved at this exact state and execution stops.
        interrupt(payload)

        # Code below is unreachable during normal execution but satisfies
        # the type checker and will run when the graph is resumed.
        return {
            "current_phase":    WorkflowPhase.HITL_ESCALATION.value,
            "escalation_payload": payload,
        }

    return hitl_escalation_node


def _build_hints(
    verdict: "VerifierVerdict | None",
    state:   "WorkflowState",
) -> list[ResolutionHint]:
    hints: list[ResolutionHint] = []

    if verdict:
        diag = verdict.get("diagnostic", "")
        if diag == "FIX_IMPL":
            hints.append(ResolutionHint(
                priority=1,
                category="architectural",
                hint=(
                    "Implementation does not satisfy the acceptance criteria. "
                    "Review the selected strategy and consider an alternative approach."
                ),
                automated=False,
            ))
        elif diag == "FIX_TEST":
            hints.append(ResolutionHint(
                priority=1,
                category="test_oracle",
                hint=(
                    "Test expectations may not align with the spec. "
                    "Review the TDD gate output and correct the test assertions."
                ),
                automated=False,
            ))

    blast = state.get("blast_radius_map")
    if blast and blast.get("risk_score", 0) > 0.7:
        hints.append(ResolutionHint(
            priority=2,
            category="blast_radius",
            hint=(
                f"High blast radius risk ({blast['risk_score']:.0%}). "
                "Consider decomposing the task into smaller, lower-risk sub-tasks."
            ),
            automated=False,
        ))

    critics = state.get("critic_findings", [])
    critical_critics = [f for f in critics if f["severity"] == "critical"]
    if critical_critics:
        hints.append(ResolutionHint(
            priority=3,
            category="security",
            hint=(
                f"{len(critical_critics)} critical finding(s) block automated resolution. "
                "Manual security review required."
            ),
            automated=False,
        ))

    # TDD gate failure: oracle couldn't produce red-phase evidence
    if state.get("tdd_gate_attempts", 0) >= 3 and not state.get("red_phase_evidence_path"):
        hints.insert(0, ResolutionHint(
            priority=1,
            category="test_oracle",
            hint=(
                "TDD gate failed to produce red-phase evidence after 3 attempts. "
                "The Test Oracle may be generating tests that pass before implementation, "
                "or the test framework is misconfigured. Review the oracle system prompt."
            ),
            automated=False,
        ))

    return hints
