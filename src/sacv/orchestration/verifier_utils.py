"""Shared verifier utilities used by both graph.py and speculative_branch.py."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sacv.orchestration.graph import NodeDeps
    from sacv.orchestration.state import WorkflowState


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
