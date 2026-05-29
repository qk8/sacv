"""nodes/bootstrap.py — resets ALL state fields including new debug fields."""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING
import structlog
from sacv.orchestration.state import WorkflowPhase
from sacv.interfaces.memory_provider import EpisodicEvent
if TYPE_CHECKING:
    from sacv.orchestration.graph import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)


def make_bootstrap_node(deps: "NodeDeps"):
    async def bootstrap_node(state: "WorkflowState") -> dict:
        session_id = state.get("session_id") or str(uuid.uuid4())
        log.info("bootstrap.start", session_id=session_id, task_id=state["task_id"])

        constraints = await deps.memory.retrieve_procedural([
            state["module_type"], state["project_mode"], state["task_id"],
        ])
        await deps.memory.store_episodic(EpisodicEvent(
            session_id=session_id,
            event_type="session_start",
            payload={"task_id": state["task_id"], "module_type": state["module_type"],
                     "mode": state["project_mode"], "constraints_loaded": len(constraints)},
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))
        return {
            "session_id":             session_id,
            "current_phase":          WorkflowPhase.SCOUT.value,
            "procedural_constraints": [c.description for c in constraints],
            "critic_findings":        [],
            "active_branches":        [],
            "exhausted_branches":     [],
            "test_inventory_paths":   [],
            "replan_count":           0,
            "confidence_score":       1.0,
            "arch_rules_updated":     False,
            "preflight_result":       None,
            "agents_md_context":      None,
            "debug_observations":     None,   # NEW
            "correction_state": {
                "attempt_count": 0, "branch_name": None,
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            },
        }
    return bootstrap_node
