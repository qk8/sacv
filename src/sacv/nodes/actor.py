"""
nodes/actor.py
==============
Implements the selected strategy via diff-only proposals.

Refactoring additions (debugging + architecture sessions):
  - debug_observations injected into system prompt when available (debugging session).
    Actor receives exact variable values, call stack, root-cause hypothesis.
  - @ai-agent comment instruction added to Build Agent system prompt (approach 3D).
  - File length awareness: system prompt explicitly states 250/200 line limits
    and reminds the agent to split files that exceed them (approach 3C).
"""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import structlog

from sacv.orchestration.state import (
    WorkflowPhase, DiffProposal, UnifiedDiffPayload,
    VerifierVerdict, DiagnosticVerdict, CRITIC_RESET,
)
from sacv.interfaces.agent_provider import AgentConfig
from sacv.interfaces.diff_provider import UnifiedDiff
from sacv.git.branch_manager import sanitize_branch_name
from sacv.nodes._stagnation import check_stagnation
from sacv.orchestration.verifier_utils import add_agent_cost

if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

_ACTOR_SYSTEM = """\
You are a senior {language} Build Agent performing a diff-only code change.

HARD RULES:
- Output ONLY a JSON array of unified diff objects.
- Never rewrite an entire file. Produce targeted, minimal diffs.
- Follow DDD and Clean Architecture layer rules strictly.
- Each object must have: "file_path", "diff_content", "operation", "language".
- Java files must not exceed 250 lines. TypeScript files must not exceed 200 lines.
  If a change would push a file over the limit, split it into smaller modules.

@AI-AGENT COMMENT RULE:
For any non-trivial implementation decision (idempotency, transaction boundary,
async ordering, security constraint, performance trade-off), add a comment:
  Java:       // @ai-agent[<ISO-date>]: <reason>. See: tests/<path>#<test>
  TypeScript: // @ai-agent[<ISO-date>]: <reason>. See: tests/<path>#<test>
This helps future agents understand why the code was written this way.

PROCEDURAL CONSTRAINTS (learned from previous corrections):
{constraints}

AGENTS.MD PROJECT CONVENTIONS:
{agents_md}

PREFLIGHT VIOLATIONS TO FIX:
{preflight_feedback}

DEBUG OBSERVATIONS (live variable state from debugger session):
{debug_feedback}

CRITIC FEEDBACK FROM PREVIOUS ATTEMPT:
{critic_feedback}
"""


def make_actor_node(deps: "NodeDeps"):

    async def actor_node(state: "WorkflowState") -> dict:
        task_id      = state["task_id"]
        strategy     = state.get("selected_strategy")
        correction   = state["correction_state"]
        constraints  = state.get("procedural_constraints", [])
        findings     = state.get("critic_findings", [])
        description  = state.get("task_description", "")
        skeleton     = state.get("context_skeleton") or {}
        agents_md    = state.get("agents_md_context") or "No AGENTS.md found."
        preflight    = state.get("preflight_result") or {}
        debug_obs    = state.get("debug_observations")

        log.info("actor.start", task_id=task_id, attempt=correction["attempt_count"],
                 has_debug_obs=debug_obs is not None)

        # ── 0. Stagnation guard ───────────────────────────────────────────
        stagnation = check_stagnation(correction, deps.config)
        if stagnation:
            log.warning("actor.stagnation", pattern=stagnation)
            # Return a synthetic failing verdict so route_after_verifier
            # sends the graph directly to hitl_escalation (attempt_count
            # == max_self_correction_cycles triggers the max-cycles path).
            return {
                "correction_state": {
                    **correction,
                    "attempt_count":      deps.config.max_self_correction_cycles,
                    "stagnation_pattern": stagnation,
                },
                "verifier_verdict": VerifierVerdict(
                    test_result="FAIL",
                    diagnostic=DiagnosticVerdict.STAGNATION.value,
                    phase1_passed=False,
                    phase2_passed=False,
                    test_failures=[{"message": f"stagnation_detected: {stagnation}"}],
                    performance_delta=None,
                    visual_diff_result=None,
                    docker_exit_code=-1,
                    playwright_trace_path=None,
                    otel_trace=None,
                    actuator_snapshot=None,
                ),
                "diff_proposal": None,
                "cumulative_cost_dollars": state.get("cumulative_cost_dollars", 0.0),
            }

        # ── 1. Git branch ─────────────────────────────────────────────────
        attempt     = correction["attempt_count"]
        branch_name = correction.get("branch_name") or (
            f"agent-task-{sanitize_branch_name(task_id[:8])}-a{attempt}"
        )

        # Guard: branch may have been deleted by speculative_branch cleanup.
        # Check existence and create fresh if missing.
        existing_branches = await asyncio.to_thread(deps.git.list_branches, branch_name)
        branch_exists = branch_name in existing_branches

        if not correction.get("branch_name") or not branch_exists:
            branch_name = f"agent-task-{sanitize_branch_name(task_id[:8])}-a{attempt}"
            await asyncio.to_thread(deps.git.create_branch, branch_name)

        await asyncio.to_thread(deps.git.checkout, branch_name)

        # ── 2. Build prompt ───────────────────────────────────────────────
        language        = "TypeScript" if "frontend" in state["module_type"] else "Java"
        preflight_fb    = _format_preflight(preflight)
        critic_fb       = _format_findings(findings)
        debug_fb        = _format_debug_observations(debug_obs)

        prompt = (
            f"Task: {description}\n\n"
            f"Strategy:\n{json.dumps(strategy, indent=2)}\n\n"
            f"Context skeleton:\n{json.dumps(skeleton, indent=2)}\n\n"
            "Produce diff-only changes to implement this strategy."
        )

        result = await deps.agent.run_task(
            prompt=prompt,
            context={"strategy": strategy, "skeleton": skeleton},
            config=AgentConfig(
                role="build_agent",
                system_prompt=_ACTOR_SYSTEM.format(
                    language=language,
                    constraints="\n".join(f"- {c}" for c in constraints) or "None.",
                    agents_md=agents_md or "No AGENTS.md found.",
                    preflight_feedback=preflight_fb or "None.",
                    debug_feedback=debug_fb or "None (no debug session yet).",
                    critic_feedback=critic_fb or "None.",
                ),
                max_turns=deps.config.iteration_limits.implement_loop,
                # Valid Claude Code SDK tool names (PascalCase).
                # Read/Glob/Grep allow the agent to inspect the codebase;
                # Bash is needed for running git diff and checking context.
                # Write/Edit are intentionally excluded: the agent produces
                # unified diff JSON, it does not write files directly.
                allowed_tools=["Read", "Bash", "Glob", "Grep", "LS"],
            ),
        )

        # ── Token budget tracking (CRIT-002) ──────────────────────────────
        new_cost = add_agent_cost(
            result, state.get("cumulative_cost_dollars", 0.0), deps.config,
        )

        # ── 3. Parse + validate diffs ─────────────────────────────────────
        try:
            raw_diffs: list[dict] = json.loads(result.content)
        except (json.JSONDecodeError, ValueError) as exc:
            log.error("actor.parse_error", error=str(exc))
            raw_diffs = []

        diffs = [
            UnifiedDiffPayload(
                file_path=d["file_path"],
                diff_content=d["diff_content"],
                operation=d.get("operation", "modify"),
                language=d.get("language", language.lower()),
            )
            for d in raw_diffs
        ]

        # ── 3b. Reject empty diffs — prevents phantom DiffProposal bypass ─
        if not diffs:
            log.warning("actor.empty_diff", task_id=task_id, attempt=attempt)
            return {
                "correction_state": {
                    **correction,
                    "attempt_count": correction["attempt_count"] + 1,
                },
                "diff_proposal": None,
                "critic_findings": CRITIC_RESET,   # clear stale findings to avoid misleading next prompt
                "cumulative_cost_dollars": new_cost,
            }

        errors = await deps.diff.validate_no_full_overwrite(
            [UnifiedDiff(**p) for p in diffs]
        )
        if errors:
            log.error("actor.overwrite_rejected", errors=[e.reason for e in errors])
            return {
                "correction_state": {
                    **correction,
                    "attempt_count": correction["attempt_count"] + 1,
                },
               "diff_proposal":   None,
                "critic_findings": CRITIC_RESET,  # clear stale critic feedback from prior diff
                "cumulative_cost_dollars": new_cost,
            }

        apply_result = await deps.diff.apply_diffs([UnifiedDiff(**p) for p in diffs])
        if not apply_result.success:
            log.error("actor.apply_failed", conflicts=apply_result.conflicts)
            return {
                "correction_state": {
                    **correction,
                    "attempt_count": correction["attempt_count"] + 1,
                    "branch_name":   branch_name,
                },
                "diff_proposal":   None,
                "critic_findings": CRITIC_RESET,  # clear stale critic feedback from prior diff
                "cumulative_cost_dollars": new_cost,
            }

        proposal = DiffProposal(
            strategy_id=strategy["strategy_id"] if strategy else "unknown",
            diffs=diffs,
            branch_name=branch_name,
            commit_message=f"sacv: implement {task_id} (attempt {attempt + 1})",
        )
        log.info("actor.complete", branch=branch_name, files=len(diffs))

        return {
            "current_phase":    WorkflowPhase.ACTOR.value,
            "diff_proposal":    proposal,
            "critic_findings":  CRITIC_RESET,
            "preflight_result": None,
            "debug_observations": None,  # reset after actor uses them
            "correction_state": {
                **correction,
                "attempt_count": correction["attempt_count"] + 1,
                "branch_name":   branch_name,
            },
            "cumulative_cost_dollars": new_cost,
        }

    return actor_node


def _format_debug_observations(obs: dict | None) -> str:
    """Format structured debug observations for the Actor system prompt."""
    if not obs:
        return ""
    parts = [f"Error type: {obs.get('error_type', 'UNKNOWN')}"]
    if obs.get("root_cause"):
        parts.append(f"Root cause: {obs['root_cause']}")
    for hit in obs.get("breakpoint_hits", [])[:2]:
        parts.append(f"\nBreakpoint hit at {hit.get('file','')}:{hit.get('line','')}")
        for name, info in list((hit.get("variables") or {}).items())[:8]:
            val = info.get("value", "?") if isinstance(info, dict) else str(info)
            parts.append(f"  {name} = {val}")
        if hit.get("call_stack"):
            parts.append(f"  Stack: {' → '.join(hit['call_stack'][:3])}")
    if obs.get("minimal_payload"):
        parts.append(f"\nMinimal failing payload: {json.dumps(obs['minimal_payload'])}")
    if obs.get("actuator_beans"):
        parts.append("\nSpring Actuator beans snapshot available (DI graph).")
    return "\n".join(parts)


def _format_preflight(preflight: dict) -> str:
    if preflight.get("passed", True):
        return ""
    parts = []
    for e in preflight.get("lsp_errors", [])[:5]:
        parts.append(f"[LSP] {e.get('file','?')}:{e.get('line','?')} {e.get('code','')} — {e.get('message','')}")
    for v in preflight.get("arch_violations", [])[:5]:
        parts.append(f"[ARCH] {v.get('rule','?')}: {v.get('message','')}")
    return "\n".join(parts) if parts else ""


def _format_findings(findings: list[dict]) -> str:
    if not findings:
        return ""
    return "\n".join(
        f"[{f['severity'].upper()}] {f['critic']}: {f['file']}:{f.get('line','?')} "
        f"— {f['message']} → {f['resolution_hint']}"
        for f in findings
    )
