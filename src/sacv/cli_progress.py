"""
sacv/cli_progress.py
====================
Real-time progress reporting for the SACV CLI workflow runner.

Replaces ``graph.ainvoke()`` with ``graph.astream_events()`` and prints
structured progress to stderr as nodes complete.

Usage::

    from sacv.cli_progress import run_with_progress
    await run_with_progress(graph, initial_state, config, task_id)
"""
from __future__ import annotations

import json
import logging
import sys

import structlog

from typing import Any, Dict

log = structlog.get_logger(__name__)

# Threshold for summarising scalar strings in delta logs
_DELTA_SUMMARY_THRESHOLD = 200


def _format_delta_summary(value: Any) -> Any:
    """Summarise a value for state-delta logging.

    Scalars and short strings pass through unchanged.
    Long strings, lists, and dicts are replaced with a summary marker.
    """
    if isinstance(value, str) and len(value) <= _DELTA_SUMMARY_THRESHOLD:
        return value
    if isinstance(value, (dict, list)):
        return f"<{type(value).__name__} len={len(value)}>"
    if isinstance(value, str):
        return f"<str len={len(value)}>"
    return value


async def run_with_progress(
    graph: Any,
    initial_state: Dict[str, Any],
    config: Dict[str, Any],
    task_id: str,
) -> Dict[str, Any]:
    """
    Run the workflow graph with real-time progress reporting.

    Streams events via ``astream_events`` and prints node completion
    markers to stderr. Returns the final workflow state as a dict.
    """
    final_state: Dict[str, Any] = {}
    last_phase = ""

    async for event in graph.astream_events(initial_state, config=config, version="v2"):
        kind = event.get("event", "")

        # Node starting
        if kind == "on_chain_start" and event.get("name") not in (
            "LangGraph",
            "__start__",
        ):
            print(f"[sacv] {event.get('name', '?')} STARTED", file=sys.stderr)

        # Node completed — extract phase transition and cost
        if kind == "on_chain_end" and event.get("name") not in (
            "LangGraph",
            "__start__",
        ):
            node_name = event.get("name", "?")
            output = event.get("data", {}).get("output", {}) or {}
            new_phase = output.get("current_phase", "")
            cost = output.get("cumulative_cost_dollars")

            progress_line = f"[sacv] {node_name}"
            if new_phase and new_phase != last_phase:
                progress_line += f" -> {new_phase}"
                last_phase = new_phase
            if cost is not None:
                progress_line += f"  (${cost:.3f})"

            print(progress_line, file=sys.stderr)
            final_state.update(output)

            # State-delta logging at DEBUG level (HIGH-06)
            if output and logging.getLogger().isEnabledFor(logging.DEBUG):
                summary = {
                    k: _format_delta_summary(v) for k, v in output.items()
                }
                log.debug(
                    "node.state_delta",
                    node=node_name,
                    changed_keys=list(output.keys()),
                    delta_summary=summary,
                )

        # Node errored
        elif kind == "on_chain_error":
            node_name = event.get("name", "?")
            error = event.get("data", {}).get("error", "unknown")
            print(f"[sacv] {node_name} ERROR: {error}", file=sys.stderr)

    # Fetch canonical final state with all reducers applied
    snapshot = await graph.aget_state(config)
    if snapshot and snapshot.values:
        return dict(snapshot.values)

    return final_state


def format_result(
    final_state: Dict[str, Any],
    task_id: str,
) -> str:
    """Format the final workflow result as a JSON string for stdout."""
    return json.dumps({
        "phase": final_state.get("current_phase"),
        "task": task_id,
        "cost": final_state.get("cumulative_cost_dollars"),
        "lesson": (final_state.get("lesson_learned") or {}).get("pattern_discovered"),
    })
