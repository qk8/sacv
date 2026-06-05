"""Shared verifier utilities used by both graph.py and speculative_branch.py."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState
    from sacv.interfaces.agent_provider import AgentResult
    from sacv.orchestration.config import WorkflowConfig


def accumulate_cost(
    last_tokens: "AgentResult | None",
    state:       dict,
    config:      "WorkflowConfig",
) -> float:
    """
    Calculate the new cumulative cost after adding this result's token usage.

    Prefers SDK-reported cost (total_cost_usd) when available, as it is
    always accurate and model-agnostic. Falls back to token-count
    estimation otherwise.

    Returns the updated cumulative_cost_dollars value.
    """
    if last_tokens is None:
        return state.get("cumulative_cost_dollars", 0.0)

    # Prefer SDK-reported cost when available (always accurate)
    sdk_cost = getattr(last_tokens, "total_cost_usd", None)
    if sdk_cost is not None:
        return state.get("cumulative_cost_dollars", 0.0) + sdk_cost

    # Fallback: estimate from token counts
    cost = (
        last_tokens.input_tokens  / 1_000_000 * config.token_budget.cost_per_m_input
        + last_tokens.output_tokens / 1_000_000 * config.token_budget.cost_per_m_output
    )
    return state.get("cumulative_cost_dollars", 0.0) + cost


def add_agent_cost(
    result: "AgentResult",
    current_cost: float,
    config: "WorkflowConfig",
) -> float:
    """
    Add the cost of a single AgentResult to the running total.

    Call this after every ``deps.agent.run_task()`` call and include
    the result in the node's return dict as ``cumulative_cost_dollars``.
    """
    return accumulate_cost(result, {"cumulative_cost_dollars": current_cost}, config)


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
    # The verifier node makes no LLM calls; cost does not change here.
    # Individual agent-calling nodes (actor, tdd_gate, etc.) are responsible
    # for accumulating cost via add_agent_cost() in their own return dicts.
    existing_cost = state.get("cumulative_cost_dollars", 0.0)
    return {**out, "confidence_score": score, "cumulative_cost_dollars": existing_cost}
