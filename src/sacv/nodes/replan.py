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

if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

_REPLAN_SYSTEM = """\
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
            )
            raw: list[dict] = [s.model_dump() for s in structured.data]
        except StructuredOutputError:
            log.error("replan.parse_error")
            raw = []

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
            "cumulative_cost_dollars":   state.get("cumulative_cost_dollars", 0.0),
        }

    return replan_node


def _build_failure_summary(state: "WorkflowState") -> dict:
    verdict  = state.get("verifier_verdict") or {}
    findings = state.get("critic_findings", [])
    preflight = state.get("preflight_result") or {}

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
        "stagnation_pattern":     state["correction_state"].get("stagnation_pattern", "none"),
        "replan_count":           state.get("replan_count", 0),
    }
