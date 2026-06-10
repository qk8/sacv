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
import sys
from typing import Any, Dict


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
