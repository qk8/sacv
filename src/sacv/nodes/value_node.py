"""
nodes/value_node.py
===================
Generates a Tree of Strategies (LLM call) then scores and prunes them
with pure, deterministic Python functions.

The LLM is asked ONLY to generate candidate strategies in JSON format.
All scoring arithmetic happens in Python — never inside the LLM prompt.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from sacv.orchestration.state import WorkflowPhase, StrategyCandidate
from sacv.interfaces.agent_provider import AgentConfig
from sacv.nodes._scoring import score_strategy, prune_strategies, detect_collision_pairs
from sacv.nodes._structured_output import extract_structured, StrategyCandidateRaw, StructuredOutputError

if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)

_STRATEGY_SYSTEM_PROMPT = """\
You are a senior software architect. Given a task description, a context skeleton \
(AST sub-graph, dependency map, schema alignment), and a list of procedural \
constraints, generate {n} distinct implementation strategies as a JSON array.

Each strategy must be an object with:
  "strategy_id":    string (e.g. "s1", "s2")
  "description":    string  — what this strategy does
  "affected_files": array of file paths that will be modified

Respond with ONLY the JSON array. No explanation, no markdown fences.
"""

_MAX_STRATEGIES_TO_GENERATE = 4


def make_value_node(deps: "NodeDeps"):

    async def value_node_fn(state: "WorkflowState") -> dict:
        cfg          = deps.config
        skeleton     = state.get("context_skeleton") or {}
        blast        = state.get("blast_radius_map")
        constraints  = state.get("procedural_constraints", [])
        description  = state.get("task_description", "")
        mode         = state["project_mode"]

        log.info("value_node.start", task_id=state["task_id"])

        # ── 1. Ask LLM to generate candidate strategies ───────────────────
        # Build replan context block if this is a retry after failures
        replan_section = ""
        if state.get("replan_count", 0) > 0 or state.get("exhausted_branches"):
            verdict        = state.get("verifier_verdict") or {}
            exhausted      = list(state.get("exhausted_branches", []))
            last_failures  = verdict.get("test_failures", [])[:5]
            last_diagnostic = verdict.get("diagnostic", "unknown")
            replan_section = (
                "\n\n## IMPORTANT — Previous Attempts Failed\n"
                f"The following strategies have already been tried and FAILED. "
                "Do NOT repeat or closely resemble them:\n"
                + "\n".join(f"- {sid}" for sid in exhausted)
                + f"\n\nLast failure diagnostic: {last_diagnostic}\n"
                f"Last test failures (sample):\n"
                + "\n".join(f"  - {f.get('message', '')}" for f in last_failures)
                + "\n\nGenerate strategies that take a fundamentally different approach."
            )

        prompt = (
            f"Task: {description}\n\n"
            f"Context skeleton:\n{json.dumps(skeleton, indent=2)}\n\n"
            f"Procedural constraints:\n" +
            "\n".join(f"- {c}" for c in constraints) +
            replan_section +
            f"\n\nGenerate {_MAX_STRATEGIES_TO_GENERATE} distinct strategies."
        )

        # ── 1b. LLM call with structured output + retry ───────────────────
        try:
            structured = await extract_structured(
                agent=deps.agent,
                prompt=prompt,
                response_model=list[StrategyCandidateRaw],
                system_prompt=_STRATEGY_SYSTEM_PROMPT.format(
                    n=_MAX_STRATEGIES_TO_GENERATE
                ),
                context={"skeleton": skeleton, "blast": blast},
                max_retries=3,
                allowed_tools=[],
            )
            raw_strategies: list[dict] = [s.model_dump() for s in structured.data]
        except StructuredOutputError:
            log.error("value_node.parse_error")
            raw_strategies = []

        # ── 3. Score each strategy (deterministic) ────────────────────────
        blast_files = set(blast["affected_files"]) if blast else set()
        all_files: list[list[str]] = [s.get("affected_files", []) for s in raw_strategies]

        candidates: list[StrategyCandidate] = []
        for i, raw in enumerate(raw_strategies):
            affected = raw.get("affected_files", [])

            # Collision ratio: what fraction of this strategy's files appear
            # in other strategies' affected file sets.
            other_files: set[str] = set()
            for j, other in enumerate(all_files):
                if j != i:
                    other_files.update(other)
            collisions = len(set(affected) & other_files)
            collision_ratio = collisions / max(len(affected), 1)

            # Blast radius impact: fraction of affected files in blast zone.
            blast_overlap = len(set(affected) & blast_files)
            blast_impact  = blast_overlap / max(len(blast_files), 1) if blast_files else 0.0

            composite = score_strategy(
                affected_files=affected,
                collision_ratio=collision_ratio,
                blast_radius_impact=blast_impact,
                config=cfg,
            )

            candidates.append(StrategyCandidate(
                strategy_id=raw.get("strategy_id", f"s{i+1}"),
                description=raw.get("description", ""),
                affected_files=affected,
                token_depth_score=max(0.0, 1.0 - len(affected) / cfg.max_blast_files),
                collision_score=max(0.0, 1.0 - collision_ratio),
                blast_radius_score=max(0.0, 1.0 - blast_impact),
                composite_score=composite,
            ))

        # ── 4. Prune low-scoring strategies ──────────────────────────────
        pruned   = [c for c in candidates if c["composite_score"] < cfg.min_strategy_score]
        passing  = prune_strategies(candidates, config=cfg)

        log.info(
            "value_node.complete",
            generated=len(candidates),
            passing=len(passing),
            pruned=len(pruned),
        )

        # ── 5. Select highest-scoring strategy as primary ─────────────────
        selected = passing[0] if passing else None

        return {
            "current_phase":      WorkflowPhase.TDD_GATE.value,
            "strategy_candidates": passing,
            "selected_strategy":   selected,
            "pruned_strategies":   pruned,
            "cumulative_cost_dollars": state.get("cumulative_cost_dollars", 0.0),
        }

    return value_node_fn
