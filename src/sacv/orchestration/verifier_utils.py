"""Shared verifier utilities used by both graph.py and speculative_branch.py."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sacv.orchestration.graph import NodeDeps
    from sacv.orchestration.state import WorkflowState, AgentResult


def accumulate_cost(
    last_tokens: "AgentResult | None",
    state:       dict,
    config:      "WorkflowConfig",
) -> float:
    """
    Calculate the new cumulative cost after adding this result's token usage.

    Returns the updated cumulative_cost_dollars value.
    """
    if last_tokens is None:
        return state.get("cumulative_cost_dollars", 0.0)
    cost = (
        last_tokens.input_tokens  / 1_000_000 * config.token_budget.cost_per_m_input
        + last_tokens.output_tokens / 1_000_000 * config.token_budget.cost_per_m_output
    )
    return state.get("cumulative_cost_dollars", 0.0) + cost


async def run_verifier_with_confidence(
    state: "WorkflowState",
    deps:  "NodeDeps",
) -> dict:
    """Run the verifier node and compute confidence_score in the returned dict."""
    from sacv.nodes.verifier import make_verifier_node
    from sacv.orchestration.edges import compute_confidence_score

    _inner = make_verifier_node(deps)
    out    = await _inner(state)
    merged = {**state, **out}
    score  = compute_confidence_score(merged, deps.config)
    return {**out, "confidence_score": score}
