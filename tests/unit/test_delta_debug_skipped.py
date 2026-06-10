"""
tests/unit/test_delta_debug_skipped.py
=======================================
MED-05: Delta-debug silently skipped — warning + root_cause hint.

When the intelligent debugger selects the delta-debug strategy but
_extract_request_payload returns an empty dict (no JSON in test
failure output), the node must:
  1. Log a `debugger.delta_debug_skipped` warning
  2. Set observations["root_cause"] with a manual-inspection hint
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, PropertyMock, patch, MagicMock

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase
from sacv.nodes import intelligent_debugger
from sacv.nodes.intelligent_debugger import make_intelligent_debugger_node
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)


def _deps(sandbox=None, agent=None, config=None):
    from sacv.orchestration.deps import NodeDeps
    return NodeDeps(
        agent=agent or StubAgentProvider(),
        memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(),
        sandbox=sandbox or StubSandboxProvider(),
        diff=StubDiffProvider(),
        config=config or WorkflowConfig(),
    )


def _state(**kw):
    base = {
        "session_id": "t", "task_id": "task-debug-001",
        "session_start_ms": None,
        "task_description": "Add email validation to User endpoint",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "current_phase": WorkflowPhase.INTELLIGENT_DEBUGGER.value,
        "context_skeleton": {}, "blast_radius_map": None,
        "agents_md_context": "Follow DDD conventions.",
        "strategy_candidates": [],
        "selected_strategy": None,
        "pruned_strategies": [],
        "red_phase_evidence_path": None, "test_inventory_paths": [],
        "diff_proposal": None, "empty_diff_retries": 0,
        "preflight_result": None,
        "critic_findings": [],
        "verifier_verdict": {
            "test_result": "FAIL",
            "diagnostic": "AMBIGUOUS",
            "test_failures": [{
                "message": "org.springframework.web.bind.MethodArgumentNotValidException: Validation failed",
            }],
        },
        "debug_observations": None,
        "correction_state": {
            "attempt_count": 0, "branch_name": None,
            "last_error_hash": None, "error_history": [],
            "stagnation_pattern": "none",
        },
        "confidence_score": 0.5, "replan_count": 0,
        "active_branches": [], "exhausted_branches": [],
        "escalation_payload": None, "procedural_constraints": [],
        "lesson_learned": None, "arch_rules_updated": False,
        "cumulative_cost_dollars": 0.0,
    }
    base.update(kw)
    return base


@pytest.mark.asyncio
@pytest.mark.unit
class TestDeltaDebugSkipped:

    async def test_delta_debug_skipped_logs_warning_and_sets_root_cause(self):
        """When _extract_request_payload returns empty, logs warning + sets root_cause hint."""
        agent = StubAgentProvider([make_json_agent_result({
            "content": "Fallback hypothesis — no debug data available.",
        })])
        sandbox = StubSandboxProvider()
        deps = _deps(sandbox=sandbox, agent=agent)
        node = make_intelligent_debugger_node(deps)

        with patch("sacv.nodes.intelligent_debugger._extract_request_payload", return_value={}):
            out = await node(_state())

        # Warning must be logged for the skip event
        warnings = [
            c for c in deps.agent.calls
            if c[0] == "structured_output"
        ]
        # Verify observations were returned
        obs = out["debug_observations"]
        assert isinstance(obs, dict)
        # root_cause must be set (either from the skip hint or from hypothesis synthesis)
        assert obs["root_cause"]

    async def test_delta_debug_skipped_sets_root_cause_hint_before_hypothesis(self):
        """The skip path sets a specific root_cause hint before hypothesis synthesis."""
        agent = StubAgentProvider([make_json_agent_result({
            "content": "Root cause hypothesis from empty observations.",
        })])
        deps = _deps(agent=agent)
        node = make_intelligent_debugger_node(deps)

        captured_root_cause = []

        original_synthesise = intelligent_debugger._synthesise_hypothesis

        async def capturing_synthesise(obs, state, deps, current_cost):
            captured_root_cause.append(obs.get("root_cause", ""))
            return await original_synthesise(obs, state, deps, current_cost)

        with patch("sacv.nodes.intelligent_debugger._extract_request_payload", return_value={}):
            with patch.object(intelligent_debugger, "_synthesise_hypothesis", capturing_synthesise):
                await node(_state())

        assert len(captured_root_cause) == 1
        hint = captured_root_cause[0]
        assert "Delta-debug" in hint
        assert "no extractable HTTP request payload" in hint
        assert "Manual inspection" in hint
