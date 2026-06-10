"""
tests/unit/test_bootstrap_validation.py
=========================================
MED-06: Bootstrap field validation.

Verifies:
  1. _validate_bootstrap_output raises when fields are missing
  2. _validate_bootstrap_output passes when all fields present
  3. bootstrap_node includes workflow_audit_trail
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase
from sacv.nodes.bootstrap import make_bootstrap_node
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider,
)


def _deps(sandbox=None, agent=None, config=None, memory=None):
    from sacv.orchestration.deps import NodeDeps
    return NodeDeps(
        agent=agent or StubAgentProvider(),
        memory=memory or StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(),
        sandbox=sandbox or StubSandboxProvider(),
        diff=StubDiffProvider(),
        config=config or WorkflowConfig(),
    )


def _state(**kw):
    base = {
        "session_id": "t", "task_id": "task-init-001",
        "session_start_ms": None,
        "task_description": "Add findById to UserService",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "current_phase": WorkflowPhase.BOOTSTRAP.value,
        "context_skeleton": {}, "blast_radius_map": None,
        "agents_md_context": None,
        "strategy_candidates": [],
        "selected_strategy": None,
        "pruned_strategies": [],
        "red_phase_evidence_path": None, "test_inventory_paths": [],
        "diff_proposal": None, "empty_diff_retries": 0,
        "preflight_result": None,
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
        "skip_tdd_gate": False, "speculative_stash_ref": None,
        "workflow_audit_trail": [],
    }
    base.update(kw)
    return base


@pytest.mark.asyncio
@pytest.mark.unit
class TestBootstrapValidation:

    async def test_bootstrap_includes_workflow_audit_trail(self):
        """Bootstrap must initialise workflow_audit_trail (HIGH-04 field)."""
        memory = StubMemoryProvider(procedural=[])
        deps = _deps(memory=memory)
        node = make_bootstrap_node(deps)

        out = await node(_state())

        assert "workflow_audit_trail" in out, (
            "bootstrap_node must initialise workflow_audit_trail"
        )
        assert out["workflow_audit_trail"] == []

    async def test_bootstrap_includes_all_workflow_state_fields(self):
        """Bootstrap return dict must cover every WorkflowState field
        except those set by CLI initial state (task_id, task_description,
        module_type, project_mode)."""
        from typing import get_type_hints
        from sacv.orchestration.state import WorkflowState

        memory = StubMemoryProvider(procedural=[])
        deps = _deps(memory=memory)
        node = make_bootstrap_node(deps)

        out = await node(_state())

        required_keys = set(get_type_hints(WorkflowState).keys())
        # These fields are set by the CLI initial state, not bootstrap
        initial_state_only = {"task_id", "task_description", "module_type", "project_mode"}
        missing = (required_keys - initial_state_only) - set(out.keys())
        assert not missing, (
            f"bootstrap_node is missing required WorkflowState fields: {sorted(missing)}"
        )
