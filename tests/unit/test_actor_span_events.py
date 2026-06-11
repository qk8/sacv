"""
tests/unit/test_actor_span_events.py
======================================
OTEL-001: Verify actor node records span_event() at key transitions.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase
from sacv.nodes.actor import make_actor_node
from sacv.nodes._structured_output import StructuredOutputError
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.orchestration.deps import NodeDeps


def _make_deps(agent=None, diff=None):
    return NodeDeps(
        agent=agent or StubAgentProvider(),
        memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(),
        sandbox=StubSandboxProvider(),
        diff=diff or StubDiffProvider(),
        config=WorkflowConfig(),
    )


def _base_state(**kw):
    base = {
        "session_id": "t", "task_id": "task-001",
        "session_start_ms": None,
        "task_description": "Add findById",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "current_phase": WorkflowPhase.ACTOR.value,
        "context_skeleton": {}, "blast_radius_map": None,
        "agents_md_context": "Follow conventions.",
        "strategy_candidates": [{"strategy_id": "s1", "description": "test", "affected_files": []}],
        "selected_strategy": {"strategy_id": "s1", "description": "test", "affected_files": []},
        "correction_state": {"attempt_count": 0, "branch_name": None, "error_history": [], "stagnation_pattern": "none"},
        "critic_findings": [], "debug_observations": None, "empty_diff_retries": 0,
        "cumulative_cost_dollars": 0.0, "workflow_audit_trail": [],
        "procedural_constraints": [],
    }
    base.update(kw)
    return base


class TestActorSpanEvents:

    def test_records_event_on_successful_diff(self):
        """span_event('actor.diff_applied') is called when diffs are successfully applied."""
        agent = StubAgentProvider()
        agent.enqueue(make_json_agent_result([
            {"file_path": "a.py", "diff_content": "diff", "operation": "modify"},
        ]))
        deps = _make_deps(agent=agent)

        with patch("sacv.nodes.actor.span_event") as mock_span:
            import asyncio
            asyncio.run(make_actor_node(deps)(_base_state()))

            event_names = [c[0][0] for c in mock_span.call_args_list]
            assert "actor.diff_applied" in event_names, (
                f"Expected 'actor.diff_applied' event, got: {event_names}"
            )

    def test_records_event_on_empty_diff(self):
        """span_event('actor.empty_diff') is called when actor produces no diffs."""
        agent = StubAgentProvider()
        agent.enqueue(make_json_agent_result([]))
        deps = _make_deps(agent=agent)

        with patch("sacv.nodes.actor.span_event") as mock_span:
            import asyncio
            asyncio.run(make_actor_node(deps)(_base_state()))

            event_names = [c[0][0] for c in mock_span.call_args_list]
            assert "actor.empty_diff" in event_names, (
                f"Expected 'actor.empty_diff' event, got: {event_names}"
            )

    def test_records_event_on_parse_error(self):
        """span_event('actor.parse_error') is called when structured output fails."""
        agent = StubAgentProvider()
        agent.enqueue(make_json_agent_result("not json"))
        deps = _make_deps(agent=agent)

        with patch("sacv.nodes.actor.span_event") as mock_span, \
             patch("sacv.nodes.actor.extract_structured") as mock_extract:
            mock_extract.side_effect = StructuredOutputError(
                message="bad json", last_raw_content="garbage", updated_cost=0.0,
            )
            import asyncio
            asyncio.run(make_actor_node(deps)(_base_state()))

            event_names = [c[0][0] for c in mock_span.call_args_list]
            assert "actor.parse_error" in event_names, (
                f"Expected 'actor.parse_error' event, got: {event_names}"
            )
