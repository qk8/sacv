"""
nodes/_node_context.py
======================
Utility for binding per-invocation correlation context to structlog.

Call bind_node_context() at the start of every node function.
Preserves any context already bound (e.g., session_id from bootstrap).
"""
from __future__ import annotations

import structlog


def bind_node_context(state: dict, node_name: str) -> None:
    """
    Merge node-specific context into the structlog contextvars store.
    Preserves any context already bound (e.g., session_id from bootstrap).
    """
    structlog.contextvars.bind_contextvars(
        node=node_name,
        phase=state.get("current_phase", "unknown"),
        attempt=state.get("correction_state", {}).get("attempt_count", 0),
    )
