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
  - Routes back to value_node (which rescores and prunes the new candidates).

Plan Agent role: read-only, no tool use. Pure strategy generation.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from sacv.orchestration.state import WorkflowPhase
from sacv.interfaces.agent_provider import AgentConfig

if TYPE_CHECKING:
    from sacv.orchestration.graph import NodeDeps
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


def make_replan_node(deps: "NodeDeps"):

    async def replan_node(state: "WorkflowState") -> dict:
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

        result = await deps.agent.run_task(
            prompt=prompt,
            context={"failure_summary": failure_summary},
            config=AgentConfig(
                role="plan_agent_replan",
                system_prompt=_REPLAN_SYSTEM.format(n=cfg.max_strategies),
                max_turns=1,
                allowed_tools=[],   # read-only Plan Agent — no writes
            ),
        )

        try:
            raw: list[dict] = json.loads(result.content)
        except (json.JSONDecodeError, ValueError) as exc:
            log.error("replan.parse_error", error=str(exc))
            raw = []

        # Remap to StrategyCandidate format (scoring happens in value_node)
        from sacv.orchestration.state import StrategyCandidate
        new_candidates = [
            StrategyCandidate(
                strategy_id=r.get("strategy_id", f"r{i+1}"),
                description=r.get("description", "") + f" [avoids: {r.get('avoids','')}]",
                affected_files=r.get("affected_files", []),
                token_depth_score=0.5,   # placeholder — value_node will rescore
                collision_score=0.5,
                blast_radius_score=0.5,
                composite_score=0.5,
            )
            for i, r in enumerate(raw)
        ]

        log.info("replan.complete", new_candidates=len(new_candidates))

        return {
            "current_phase":         WorkflowPhase.VALUE_NODE.value,
            "strategy_candidates":   new_candidates,
            "selected_strategy":     None,    # value_node will select
            "replan_count":          replan_cnt + 1,
            "exhausted_branches":    [],      # reset for new replan cycle
            "active_branches":       [],
            "correction_state": {
                **state["correction_state"],
                "attempt_count":      0,       # reset attempt counter
                "branch_name":        None,
                "error_history":      [],      # clear stagnation history for fresh start
                "last_error_hash":    None,    # clear hash to avoid false stagnation signal
                "stagnation_pattern": "none",  # reset pattern
            },
            "critic_findings":   [],
            "verifier_verdict":  None,
            "preflight_result":  None,
            "diff_proposal":     None,
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
