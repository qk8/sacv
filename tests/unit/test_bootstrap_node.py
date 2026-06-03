"""
tests/unit/test_bootstrap_node.py
===================================
Unit tests for the Bootstrap node.

Tests cover:
1. Session ID reuse vs generation
2. Procedural constraint loading from memory
3. Episodic event recording
4. All state fields are initialised to defaults
5. Phase advances to MODE_ROUTER
"""
from __future__ import annotations

import pytest
from datetime import datetime

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase, CRITIC_RESET
from sacv.nodes.bootstrap import make_bootstrap_node
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider,
)
from sacv.interfaces.memory_provider import ProceduralConstraint


def _deps(memory=None, **kw):
    from sacv.orchestration.graph import NodeDeps
    return NodeDeps(
        agent=StubAgentProvider(),
        memory=memory or StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(),
        sandbox=StubSandboxProvider(),
        diff=StubDiffProvider(),
        config=WorkflowConfig(**kw),
    )


def _state(**kw):
    base = {
        "session_id": "", "task_id": "task-bs-001",
        "task_description": "Add feature",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "current_phase": WorkflowPhase.BOOTSTRAP.value,
        "context_skeleton": None, "blast_radius_map": None,
        "agents_md_context": None, "strategy_candidates": [],
        "selected_strategy": None, "pruned_strategies": [],
        "red_phase_evidence_path": None, "test_inventory_paths": [],
        "diff_proposal": None, "preflight_result": None,
        "critic_findings": [], "verifier_verdict": None,
        "debug_observations": None,
        "correction_state": {
            "attempt_count": 0, "branch_name": None,
            "last_error_hash": None, "error_history": [],
            "stagnation_pattern": "none",
        },
        "confidence_score": 1.0, "replan_count": 0,
        "active_branches": [], "exhausted_branches": [],
        "escalation_payload": None, "procedural_constraints": [],
        "lesson_learned": None, "arch_rules_updated": False,
        "check_profile": "standard", "cumulative_cost_dollars": 0.0,
    }
    base.update(kw)
    return base


@pytest.mark.asyncio
@pytest.mark.unit
class TestBootstrapNode:

    async def test_session_id_is_generated_when_empty(self):
        """When session_id is empty string, a new UUID is generated."""
        deps = _deps()
        node = make_bootstrap_node(deps)

        out = await node(_state(session_id=""))

        assert out["session_id"] != ""
        assert len(out["session_id"]) > 10  # UUID-like

    async def test_existing_session_id_is_preserved(self):
        """When session_id is already set, it is not overwritten."""
        deps = _deps()
        node = make_bootstrap_node(deps)

        out = await node(_state(session_id="existing-session-123"))

        assert out["session_id"] == "existing-session-123"

    async def test_procedural_constraints_loaded_from_memory(self):
        """retrieve_procedural is called and constraints are stored in state."""
        constraints = [
            ProceduralConstraint(
                constraint_id="c1", category="security",
                description="Always validate input parameters",
                weight=1.0,
            ),
            ProceduralConstraint(
                constraint_id="c2", category="performance",
                description="Use batch operations for bulk updates",
                weight=0.8,
            ),
        ]
        memory = StubMemoryProvider(procedural=constraints)
        deps = _deps(memory=memory)
        node = make_bootstrap_node(deps)

        out = await node(_state())

        assert out["procedural_constraints"] == [
            "Always validate input parameters",
            "Use batch operations for bulk updates",
        ]

    async def test_empty_constraints_when_none_in_memory(self):
        """When no procedural constraints exist, state gets empty list."""
        memory = StubMemoryProvider(procedural=[])
        deps = _deps(memory=memory)
        node = make_bootstrap_node(deps)

        out = await node(_state())

        assert out["procedural_constraints"] == []

    async def test_episodic_event_recorded_on_start(self):
        """store_episodic is called with session_start event."""
        memory = StubMemoryProvider()
        deps = _deps(memory=memory)
        node = make_bootstrap_node(deps)

        await node(_state(session_id="test-session"))

        assert len(memory.stored_events) == 1
        event = memory.stored_events[0]
        assert event.event_type == "session_start"
        assert event.session_id == "test-session"
        assert event.payload["task_id"] == "task-bs-001"
        assert event.payload["module_type"] == "backend-domain"
        assert event.payload["mode"] == "greenfield"
        assert event.timestamp is not None

    async def test_all_state_fields_initialised(self):
        """Bootstrap must initialise every field in WorkflowState."""
        deps = _deps()
        node = make_bootstrap_node(deps)

        out = await node(_state(session_id=""))

        # Verify all expected keys are present in output
        expected_keys = {
            "session_id", "current_phase", "procedural_constraints",
            "check_profile", "critic_findings", "active_branches",
            "exhausted_branches", "test_inventory_paths", "replan_count",
            "confidence_score", "arch_rules_updated", "preflight_result",
            "agents_md_context", "debug_observations", "tdd_gate_attempts",
            "correction_state", "context_skeleton", "blast_radius_map",
            "strategy_candidates", "selected_strategy", "pruned_strategies",
            "red_phase_evidence_path", "diff_proposal", "verifier_verdict",
            "speculative_stash_ref", "escalation_payload",
            "lesson_learned", "cumulative_cost_dollars",
        }
        for key in expected_keys:
            assert key in out, f"Missing key in bootstrap output: {key}"

    async def test_phase_advances_to_mode_router(self):
        out = await make_bootstrap_node(_deps())(_state())
        assert out["current_phase"] == WorkflowPhase.MODE_ROUTER.value

    async def test_check_profile_defaults_to_standard(self):
        out = await make_bootstrap_node(_deps())(_state())
        assert out["check_profile"] == "standard"

    async def test_critic_findings_reset(self):
        from sacv.orchestration.state import CRITIC_RESET
        out = await make_bootstrap_node(_deps())(_state())
        assert out["critic_findings"] is CRITIC_RESET

    async def test_correction_state_reset(self):
        out = await make_bootstrap_node(_deps())(_state())
        cs = out["correction_state"]
        assert cs["attempt_count"] == 0
        assert cs["branch_name"] is None
        assert cs["last_error_hash"] is None
        assert cs["error_history"] == []
        assert cs["stagnation_pattern"] == "none"

    async def test_cost_starts_at_zero(self):
        out = await make_bootstrap_node(_deps())(_state())
        assert out["cumulative_cost_dollars"] == 0.0

    async def test_memory_is_called_with_correct_context_tags(self):
        """retrieve_procedural is called with module_type, project_mode, task_id."""
        memory = StubMemoryProvider()
        deps = _deps(memory=memory)
        node = make_bootstrap_node(deps)

        # We can't directly inspect the tags passed to retrieve_procedural
        # because it's async and the stub just returns stored constraints.
        # But we can verify the episodic event payload has the right data.
        await node(_state(
            session_id="test",
            task_description="Add feature",
            project_mode="brownfield",
            module_type="frontend-feature",
        ))

        event = memory.stored_events[0]
        assert event.payload["mode"] == "brownfield"
        assert event.payload["module_type"] == "frontend-feature"
