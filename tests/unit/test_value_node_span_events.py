"""
tests/unit/test_value_node_span_events.py
============================================
OTEL-001: Verify value_node records span_event() at key transitions.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase
from sacv.nodes.value_node import make_value_node
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.orchestration.deps import NodeDeps


def _make_deps(agent=None):
    return NodeDeps(
        agent=agent or StubAgentProvider(),
        memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(),
        sandbox=StubSandboxProvider(),
        diff=StubDiffProvider(),
        config=WorkflowConfig(),
    )


def _base_state(**kw):
    base = {
        "session_id": "t", "task_id": "task-001",
        "session_start_ms": None,
        "task_description": "Add findById",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "current_phase": WorkflowPhase.VALUE_NODE.value,
        "context_skeleton": {}, "blast_radius_map": None,
        "agents_md_context": "Follow conventions.",
        "strategy_candidates": [], "selected_strategy": None,
        "correction_state": {"attempt_count": 0, "branch_name": None, "error_history": [], "stagnation_pattern": "none"},
        "critic_findings": [], "debug_observations": None, "empty_diff_retries": 0,
        "cumulative_cost_dollars": 0.0, "workflow_audit_trail": [],
        "procedural_constraints": [], "replan_count": 0, "exhausted_branches": [],
    }
    base.update(kw)
    return base


class TestValueNodeSpanEvents:

    def test_records_event_on_strategies_generated(self):
        """span_event('value_node.strategies_generated') is called after scoring."""
        agent = StubAgentProvider()
        agent.enqueue(make_json_agent_result([
            {"strategy_id": "s1", "description": "test", "affected_files": ["a.py"]},
        ]))
        deps = _make_deps(agent=agent)

        with patch("sacv.nodes.value_node.span_event") as mock_span:
            import asyncio
            asyncio.run(make_value_node(deps)(_base_state()))

            event_names = [c[0][0] for c in mock_span.call_args_list]
            assert "value_node.strategies_generated" in event_names, (
                f"Expected 'value_node.strategies_generated' event, got: {event_names}"
            )
