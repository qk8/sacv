"""nodes/bootstrap.py — resets ALL state fields including new debug fields."""
from __future__ import annotations
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Coroutine
import structlog
from sacv.orchestration.state import WorkflowPhase, CRITIC_RESET
from sacv.nodes._node_context import bind_node_context
from sacv.nodes._node_timer import node_timer
from sacv.interfaces.memory_provider import EpisodicEvent
if TYPE_CHECKING:
    from sacv.orchestration.deps import NodeDeps
    from sacv.orchestration.state import WorkflowState

log = structlog.get_logger(__name__)


def make_bootstrap_node(deps: "NodeDeps") -> "Callable[[WorkflowState], Coroutine[Any, Any, dict[str, object]]]":
    async def bootstrap_node(state: "WorkflowState") -> dict[str, object]:
        bind_node_context(state, "bootstrap")
        async with node_timer("bootstrap", state=state) as timing:
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
            timing["session_id"] = session_id
            return {
                "session_id":             session_id,
                "current_phase":          WorkflowPhase.MODE_ROUTER.value,
                "procedural_constraints": [c.description for c in constraints],
                "check_profile":          state.get("check_profile", "standard"),
                "critic_findings":        CRITIC_RESET,
                "critic_errors":          [],
                "active_branches":        [],
                "exhausted_branches":     [],
                "test_inventory_paths":   [],
                "replan_count":           0,
                "confidence_score":       1.0,
                "arch_rules_updated":     False,
                "preflight_result":       None,
                "agents_md_context":      None,
                "debug_observations":     None,
                "tdd_gate_attempts":      0,
                "skip_tdd_gate":          state.get("skip_tdd_gate", False),
                "correction_state": {
                    "attempt_count": 0, "branch_name": None,
                    "last_error_hash": None, "error_history": [],
                    "stagnation_pattern": "none",
                },
                # ── Fields that bootstrap must also initialise ────────────────────
                "context_skeleton":        None,   # set by Scout
                "blast_radius_map":        None,   # set by Scout (brownfield only)
                "strategy_candidates":     [],     # set by ValueNode
                "selected_strategy":       None,   # set by ValueNode
                "pruned_strategies":       [],     # set by ValueNode
                "red_phase_evidence_path": None,   # set by TDDGate
                "diff_proposal":           None,   # set by Actor
                "empty_diff_retries":      0,      # MED-004: no-diff Actor loop counter
                "verifier_verdict":        None,   # set by Verifier
                "speculative_stash_ref":   None,   # set by SpeculativeBranch
                "escalation_payload":      None,   # set by HITL
                "lesson_learned":          None,   # set by MemoryConsolidation
                "cumulative_cost_dollars": 0.0,       # BUG-008: token budget tracking
                "session_start_ms":          time.time() * 1000,  # BUG-002: session duration tracking
                "workflow_audit_trail":      [],       # HIGH-04: structured audit trail
            }
    return bootstrap_node
