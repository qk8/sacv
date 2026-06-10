"""
nodes/_node_context.py
======================
Utility for binding per-invocation correlation context to structlog.

Call bind_node_context() at the start of every node function.
Clears stale context from previous nodes, then binds primary correlation
IDs and per-invocation metadata from the current state.
"""
from __future__ import annotations

import structlog


def bind_node_context(state: dict, node_name: str) -> None:
    """
    Clear stale context and bind fresh per-invocation context from state.

    IMPORTANT: Calls clear_contextvars() first to prevent leftover keys from
    previous node invocations from bleeding into log output.  Then binds
    primary correlation IDs (session_id, task_id) and per-invocation metadata
    (module_type, replan_count, node, phase, attempt).
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        task_id=state.get("task_id", "unknown"),
        session_id=state.get("session_id", ""),
        module_type=state.get("module_type", ""),
        replan_count=state.get("replan_count", 0),
        node=node_name,
        phase=state.get("current_phase", "unknown"),
        attempt=state.get("correction_state", {}).get("attempt_count", 0),
    )
