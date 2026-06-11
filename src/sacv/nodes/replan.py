"""
nodes/replan.py
===============
ReplanNode — triggered when all speculative branches fail and we
still have replan budget remaining (approach 2, 4).

Difference from ValueNode:
  - Receives the full failure history of this session (all exhausted branches,
    all critic findings, verifier verdicts, preflight violations).
  - Explicitly told which strategies FAILED and why.
  - Generates a NEW strategy tree that avoids the previously failed paths.
  - Increments replan_count.
  - Routes directly to tdd_gate with pre-selected best candidate.

Plan Agent role: read-only, no tool use. Pure strategy generation.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import structlog

from sacv.orchestration.state import WorkflowPhase, CRITIC_RESET
from sacv.interfaces.agent_provider import AgentConfig
from sacv.nodes._structured_output import extract_structured, StrategyCandidateRaw, StructuredOutputError
from sacv.nodes._node_context import bind_node_context
from sacv.nodes._node_timer import node_timer
from sacv.nodes._audit import make_audit_entry

if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

_REPLAN_SYSTEM_VERSION = "2026-06-11-v1"

_REPLAN_SYSTEM = "# prompt_version: " + _REPLAN_SYSTEM_VERSION + "\n" + """\
You are a Principal Architect performing a root-cause analysis and replanning session.

Previous implementation attempts have all FAILED. Your task:
1. Analyse the failure history provided.
2. Identify WHY each strategy failed (architectural constraint? wrong abstraction? test oracle mismatch?)
3. Generate {n} ALTERNATIVE strategies that avoid the failure patterns.

Output ONLY a JSON array of strategy objects:
[{{"strategy_id": "r1", "description": "...", "affected_files": ["..."], "avoids": "..."}}]

Field "avoids" is mandatory: explain in one sentence why this strategy will not repeat the failure.
No explanation. No markdown. Only the JSON array.
"""


def make_replan_node(deps: "NodeDeps") -> "Callable[[WorkflowState], Coroutine[Any, Any, dict[str, object]]]":

    async def replan_node(state: "WorkflowState") -> dict[str, object]:
        bind_node_context(state, "replan")
        async with node_timer("replan", state=state) as timing:
            task_id    = state["task_id"]
            replan_cnt = state.get("replan_count", 0)
            cfg        = deps.config

            log.warning(
                "replan.start",
                task_id=task_id,
                replan_count=replan_cnt,
                exhausted_branches=len(state.get("exhausted_branches", [])),
            )

            # Build failure summary for the Plan Agent
            failure_summary = _build_failure_summary(state)
            context_skeleton = state.get("context_skeleton") or {}
            constraints      = state.get("procedural_constraints", [])

            prompt = (
                f"Task: {state.get('task_description', '')}\n\n"
                f"FAILURE HISTORY:\n{json.dumps(failure_summary, indent=2)}\n\n"
                f"Context skeleton:\n{json.dumps(context_skeleton, indent=2)}\n\n"
                f"Procedural constraints:\n" +
                "\n".join(f"- {c}" for c in constraints) +
                "\n\nGenerate alternative strategies that avoid the above failures."
            )

            # ── 1b. LLM call with structured output + retry ───────────────────
            try:
                structured = await extract_structured(
                    agent=deps.agent,
                    prompt=prompt,
                    response_model=list[StrategyCandidateRaw],
                    system_prompt=_REPLAN_SYSTEM.format(n=cfg.max_strategies),
                    context={"failure_summary": failure_summary},
                    max_retries=3,
                    allowed_tools=[],
                    current_cost=state.get("cumulative_cost_dollars", 0.0),
                    workflow_config=cfg,
                )
                raw: list[dict[str, Any]] = [s.model_dump() for s in structured.data]
                updated_cost = structured.updated_cost
            except StructuredOutputError as exc:
                log.error(
                    "replan.parse_error",
                    error=str(exc),
                    raw_content_preview=exc.last_raw_content[:500],
                    raw_content_len=len(exc.last_raw_content),
                )
                raw = []
                updated_cost = exc.updated_cost

            # Remap to StrategyCandidate format with real scoring
            from sacv.orchestration.state import StrategyCandidate
            from sacv.nodes._scoring import score_strategy

            blast = state.get("blast_radius_map") or {}
            blast_risk = float(blast.get("risk_score", 0.0))
            all_file_sets = [set(r.get("affected_files", [])) for r in raw]

            new_candidates = []
            for i, r in enumerate(raw):
                files = r.get("affected_files", [])
                # Compute collision_ratio: fraction of files shared with other candidates
                my_files = set(files)
                other_files = set().union(*[fs for j, fs in enumerate(all_file_sets) if j != i])
                collision = len(my_files & other_files) / max(len(my_files), 1)
                composite = score_strategy(
                    affected_files=files,
                    collision_ratio=collision,
                    blast_radius_impact=blast_risk,
                    config=cfg,
                )
                new_candidates.append(StrategyCandidate(
                    strategy_id=r.get("strategy_id", f"r{i+1}"),
                    description=r.get("description", "") + f" [avoids: {r.get('avoids','')}]",
                    affected_files=files,
                    token_depth_score=max(0.0, 1.0 - len(files) / max(cfg.max_blast_files, 1)),
                    collision_score=max(0.0, 1.0 - collision),
                    blast_radius_score=max(0.0, 1.0 - blast_risk),
                    composite_score=composite,
                ))

            log.info("replan.complete", new_candidates=len(new_candidates))

            # Prune and pre-select the best candidate so tdd_gate can proceed
            # without going through value_node (BUG-005 fix).
            from sacv.nodes._scoring import prune_strategies
            passing = prune_strategies(new_candidates, config=cfg)
            selected = passing[0] if passing else None

            timing["new_candidates"] = len(passing)

            return {
                "current_phase":             WorkflowPhase.TDD_GATE.value,
                "strategy_candidates":       passing,
                "selected_strategy":         selected,   # pre-selected by replan
                "replan_count":              replan_cnt + 1,
                "exhausted_branches":        [],      # reset for new replan cycle
                "active_branches":           [],
                "correction_state": {
                    **state["correction_state"],
                    "attempt_count":      0,       # reset attempt counter
                    "branch_name":        None,
                    "error_history":      [],      # clear stagnation history for fresh start
                    "last_error_hash":    None,    # clear hash to avoid false stagnation signal
                    "stagnation_pattern": "none",  # reset pattern
                },
                "critic_findings":           CRITIC_RESET,
                "verifier_verdict":          None,
                "preflight_result":          None,
                "diff_proposal":             None,
                "red_phase_evidence_path":   None,   # force tdd_gate to generate new tests
                "test_inventory_paths":      [],     # clear old test inventory for new strategies
                "tdd_gate_attempts":         0,      # reset — prevents immediate HITL escalation after replan
                "empty_diff_retries":        0,      # BUG-003: reset for fresh replan cycle
                "cumulative_cost_dollars":   updated_cost,
                "workflow_audit_trail": [make_audit_entry(
                    "replan",
                    f"new_candidates={len(passing)}",
                    {
                        "new_candidates": len(passing),
                        "selected_id": (passing[0]["strategy_id"] if passing else None),
                        "replan_count": replan_cnt + 1,
                    },
                )],
            }

    return replan_node


def _truncate(text: str, max_chars: int = 5000) -> str:
    """Truncate text to max_chars for inclusion in prompts."""
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + f"... ({len(text)} chars)"


def _build_failure_summary(state: "WorkflowState") -> dict[str, Any]:
    verdict  = state.get("verifier_verdict") or {}
    findings = state.get("critic_findings", [])
    preflight = state.get("preflight_result") or {}
    correction = state.get("correction_state", {})
    diff_proposal = state.get("diff_proposal")

    # Build a list of past diffs to show what the actor tried
    past_diffs: list[dict[str, Any]] = []
    if diff_proposal:
        past_diffs.append({
            "strategy_id": diff_proposal.get("strategy_id"),
            "branch_name": diff_proposal.get("branch_name"),
            "diffs": diff_proposal.get("diffs", [])[:3],  # last 3 diffs max
        })

    # Include debug observations if present
    debug_obs = state.get("debug_observations")
    debug_summary = None
    if debug_obs:
        debug_summary = {
            "error_type": debug_obs.get("error_type"),
            "root_cause": debug_obs.get("root_cause"),
            "has_breakpoint_hits": len(debug_obs.get("breakpoint_hits", [])),
            "has_actuator_snapshot": debug_obs.get("actuator_beans") is not None,
            "has_otel_trace": debug_obs.get("otel_trace") is not None,
        }

    return {
        "exhausted_strategies": [
            s["strategy_id"]
            for s in state.get("strategy_candidates", [])
        ],
        "last_diagnostic":        verdict.get("diagnostic", "UNKNOWN"),
        "last_test_failures":     verdict.get("test_failures", [])[:5],
        "critical_critic_findings": [
            f for f in findings if f.get("severity") == "critical"
        ][:5],
        "arch_violations_found":  preflight.get("arch_violations", [])[:5],
        "stagnation_pattern":     correction.get("stagnation_pattern", "none"),
        "replan_count":           state.get("replan_count", 0),
        "last_diff_proposal":     past_diffs[-2:],  # last 2 diff proposals
        "last_preflight": {
            "passed": preflight.get("passed"),
            "lsp_errors": preflight.get("lsp_errors", [])[:5],
            "cross_stack_errors": preflight.get("cross_stack_errors", [])[:5],
            "blast_errors": preflight.get("blast_errors", [])[:5],
        } if preflight else None,
        "debug_observations":     debug_summary,
        "error_history":          correction.get("error_history", [])[-5:],  # last 5 error hashes
    }
