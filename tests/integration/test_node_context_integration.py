"""
tests/integration/test_node_context_integration.py
====================================================
CRIT-01: Verify every node calls bind_node_context and node_timer.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)


def _make_deps(agent):
    from sacv.orchestration.deps import NodeDeps
    return NodeDeps(
        agent=agent, memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(), cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(), sandbox=StubSandboxProvider(),
        diff=StubDiffProvider(), config=WorkflowConfig(),
    )


def _base_state():
    return {
        "session_id": "t", "task_id": "task-001", "task_description": "Create user endpoint",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "current_phase": WorkflowPhase.SCOUT.value,
        "context_skeleton": None, "blast_radius_map": None, "agents_md_context": None,
        "strategy_candidates": [], "selected_strategy": None, "pruned_strategies": [],
        "red_phase_evidence_path": "/p/e.json", "test_inventory_paths": [],
        "diff_proposal": None, "empty_diff_retries": 0,
        "preflight_result": {"passed": True, "lsp_errors": [], "arch_violations": [], "duration_ms": 50},
        "critic_findings": [], "verifier_verdict": None,
        "correction_state": {"attempt_count": 1, "branch_name": "agent-task-abc12345-a1",
                             "last_error_hash": None, "error_history": [], "stagnation_pattern": "none"},
        "confidence_score": 0.8, "replan_count": 0,
        "active_branches": [], "exhausted_branches": [], "escalation_payload": None,
        "procedural_constraints": [], "lesson_learned": None, "arch_rules_updated": False,
    }


@pytest.mark.asyncio
@pytest.mark.integration
class TestNodeCallsContextUtilities:
    """Every node must call bind_node_context and wrap its body in node_timer."""

    async def test_scout_calls_bind_node_context_and_node_timer(self):
        """scout_node calls bind_node_context and node_timer."""
        from sacv.nodes.scout import make_scout_node

        agent = StubAgentProvider([make_json_agent_result({
            "context_skeleton": {"call_graph": {"entry": "X", "nodes": [], "edges": []},
                                 "dependencies": {}, "schema_map": {}, "arch_align": {}},
            "blast_radius_map": None,
        })])
        node = make_scout_node(_make_deps(agent))
        state = {**_base_state(), "current_phase": WorkflowPhase.SCOUT.value}

        with patch("sacv.nodes.scout.bind_node_context") as mock_bind, \
             patch("sacv.nodes.scout.node_timer") as mock_timer:
            mock_timer.return_value.__aenter__ = AsyncMock(return_value={})
            mock_timer.return_value.__aexit__ = AsyncMock(return_value=False)
            await node(state)

        mock_bind.assert_called_once()
        assert mock_bind.call_args[0][1] == "scout"
        mock_timer.assert_called_once()
        assert mock_timer.call_args[0][0] == "scout"

    async def test_bootstrap_calls_bind_node_context_and_node_timer(self):
        """bootstrap_node calls bind_node_context and node_timer."""
        from sacv.nodes.bootstrap import make_bootstrap_node

        agent = StubAgentProvider([make_json_agent_result({
            "project_mode": "greenfield", "task_description": "test",
        })])
        node = make_bootstrap_node(_make_deps(agent))
        state = _base_state()

        with patch("sacv.nodes.bootstrap.bind_node_context") as mock_bind, \
             patch("sacv.nodes.bootstrap.node_timer") as mock_timer:
            mock_timer.return_value.__aenter__ = AsyncMock(return_value={})
            mock_timer.return_value.__aexit__ = AsyncMock(return_value=False)
            await node(state)

        mock_bind.assert_called_once()
        assert mock_bind.call_args[0][1] == "bootstrap"
        mock_timer.assert_called_once()
        assert mock_timer.call_args[0][0] == "bootstrap"

    async def test_mode_router_calls_bind_node_context_and_node_timer(self):
        """mode_router_node calls bind_node_context and node_timer."""
        from sacv.nodes.mode_router import make_mode_router_node

        agent = StubAgentProvider([make_json_agent_result({
            "project_mode": "greenfield",
        })])
        node = make_mode_router_node(_make_deps(agent))
        state = {**_base_state(), "current_phase": WorkflowPhase.MODE_ROUTER.value}

        with patch("sacv.nodes.mode_router.bind_node_context") as mock_bind, \
             patch("sacv.nodes.mode_router.node_timer") as mock_timer:
            mock_timer.return_value.__aenter__ = AsyncMock(return_value={})
            mock_timer.return_value.__aexit__ = AsyncMock(return_value=False)
            await node(state)

        mock_bind.assert_called_once()
        assert mock_bind.call_args[0][1] == "mode_router"
        mock_timer.assert_called_once()
        assert mock_timer.call_args[0][0] == "mode_router"

    async def test_value_node_calls_bind_node_context_and_node_timer(self):
        """value_node_fn calls bind_node_context and node_timer."""
        from sacv.nodes.value_node import make_value_node

        agent = StubAgentProvider([make_json_agent_result([{
            "strategy_id": "s1", "description": "test strategy", "affected_files": ["X.java"],
        }])])
        node = make_value_node(_make_deps(agent))
        state = {**_base_state(), "current_phase": WorkflowPhase.VALUE_NODE.value}

        with patch("sacv.nodes.value_node.bind_node_context") as mock_bind, \
             patch("sacv.nodes.value_node.node_timer") as mock_timer:
            mock_timer.return_value.__aenter__ = AsyncMock(return_value={})
            mock_timer.return_value.__aexit__ = AsyncMock(return_value=False)
            await node(state)

        mock_bind.assert_called_once()
        assert mock_bind.call_args[0][1] == "value_node"
        mock_timer.assert_called_once()
        assert mock_timer.call_args[0][0] == "value_node"

    async def test_preflight_calls_bind_node_context_and_node_timer(self):
        """preflight_node calls bind_node_context and node_timer."""
        from sacv.nodes.preflight_node import make_preflight_node

        node = make_preflight_node(_make_deps(StubAgentProvider()))
        state = {**_base_state(), "current_phase": WorkflowPhase.PREFLIGHT.value}

        with patch("sacv.nodes.preflight_node.bind_node_context") as mock_bind, \
             patch("sacv.nodes.preflight_node.node_timer") as mock_timer:
            mock_timer.return_value.__aenter__ = AsyncMock(return_value={})
            mock_timer.return_value.__aexit__ = AsyncMock(return_value=False)
            await node(state)

        mock_bind.assert_called_once()
        assert mock_bind.call_args[0][1] == "preflight"
        mock_timer.assert_called_once()
        assert mock_timer.call_args[0][0] == "preflight"

    async def test_verifier_calls_bind_node_context_and_node_timer(self):
        """verifier_node calls bind_node_context and node_timer."""
        from sacv.nodes.verifier import make_verifier_node

        agent = StubAgentProvider([make_json_agent_result({
            "verdict": "PASS", "phase1_passed": True, "phase2_passed": True,
            "confidence": 0.9,
        })])
        node = make_verifier_node(_make_deps(agent))
        state = {**_base_state(), "current_phase": WorkflowPhase.VERIFIER.value}

        with patch("sacv.nodes.verifier.bind_node_context") as mock_bind, \
             patch("sacv.nodes.verifier.node_timer") as mock_timer:
            mock_timer.return_value.__aenter__ = AsyncMock(return_value={})
            mock_timer.return_value.__aexit__ = AsyncMock(return_value=False)
            await node(state)

        mock_bind.assert_called_once()
        assert mock_bind.call_args[0][1] == "verifier"
        mock_timer.assert_called_once()
        assert mock_timer.call_args[0][0] == "verifier"

    async def test_replan_calls_bind_node_context_and_node_timer(self):
        """replan_node calls bind_node_context and node_timer."""
        from sacv.nodes.replan import make_replan_node

        agent = StubAgentProvider([make_json_agent_result([{
            "strategy_id": "s1", "description": "try again", "affected_files": ["X.java"],
        }])])
        node = make_replan_node(_make_deps(agent))
        state = {**_base_state(), "current_phase": WorkflowPhase.REPLAN.value}

        with patch("sacv.nodes.replan.bind_node_context") as mock_bind, \
             patch("sacv.nodes.replan.node_timer") as mock_timer:
            mock_timer.return_value.__aenter__ = AsyncMock(return_value={})
            mock_timer.return_value.__aexit__ = AsyncMock(return_value=False)
            await node(state)

        mock_bind.assert_called_once()
        assert mock_bind.call_args[0][1] == "replan"
        mock_timer.assert_called_once()
        assert mock_timer.call_args[0][0] == "replan"

    async def test_intelligent_debugger_calls_bind_node_context_and_node_timer(self):
        """intelligent_debugger_node calls bind_node_context and node_timer."""
        from sacv.nodes.intelligent_debugger import make_intelligent_debugger_node

        agent = StubAgentProvider([make_json_agent_result({
            "root_cause": "null pointer", "suggested_fix": "add null check",
        })])
        node = make_intelligent_debugger_node(_make_deps(agent))
        state = {**_base_state(), "current_phase": WorkflowPhase.INTELLIGENT_DEBUGGER.value}

        with patch("sacv.nodes.intelligent_debugger.bind_node_context") as mock_bind, \
             patch("sacv.nodes.intelligent_debugger.node_timer") as mock_timer:
            mock_timer.return_value.__aenter__ = AsyncMock(return_value={})
            mock_timer.return_value.__aexit__ = AsyncMock(return_value=False)
            await node(state)

        mock_bind.assert_called_once()
        assert mock_bind.call_args[0][1] == "intelligent_debugger"
        mock_timer.assert_called_once()
        assert mock_timer.call_args[0][0] == "intelligent_debugger"

    async def test_hitl_escalation_calls_bind_node_context_and_node_timer(self):
        """hitl_escalation_node calls bind_node_context and node_timer."""
        from sacv.nodes.hitl_escalation import make_hitl_escalation_node

        node = make_hitl_escalation_node(_make_deps(StubAgentProvider()))
        state = {**_base_state(), "current_phase": WorkflowPhase.HITL_ESCALATION.value}

        with patch("sacv.nodes.hitl_escalation.bind_node_context") as mock_bind, \
             patch("sacv.nodes.hitl_escalation.node_timer") as mock_timer:
            mock_timer.return_value.__aenter__ = AsyncMock(return_value={})
            mock_timer.return_value.__aexit__ = AsyncMock(return_value=False)
            await node(state)

        mock_bind.assert_called_once()
        assert mock_bind.call_args[0][1] == "hitl_escalation"
        mock_timer.assert_called_once()
        assert mock_timer.call_args[0][0] == "hitl_escalation"

    async def test_memory_consolidation_calls_bind_node_context_and_node_timer(self):
        """memory_consolidation_node calls bind_node_context and node_timer."""
        from sacv.nodes.memory_consolidation import make_memory_consolidation_node

        node = make_memory_consolidation_node(_make_deps(StubAgentProvider()))
        state = {**_base_state(), "current_phase": WorkflowPhase.MEMORY_CONSOLIDATION.value}

        with patch("sacv.nodes.memory_consolidation.bind_node_context") as mock_bind, \
             patch("sacv.nodes.memory_consolidation.node_timer") as mock_timer:
            mock_timer.return_value.__aenter__ = AsyncMock(return_value={})
            mock_timer.return_value.__aexit__ = AsyncMock(return_value=False)
            await node(state)

        mock_bind.assert_called_once()
        assert mock_bind.call_args[0][1] == "memory_consolidation"
        mock_timer.assert_called_once()
        assert mock_timer.call_args[0][0] == "memory_consolidation"

    async def test_all_critics_node_calls_bind_node_context_and_node_timer(self):
        """all_critics_node calls bind_node_context and node_timer."""
        from sacv.orchestration.graph import _make_all_critics_node

        agent = StubAgentProvider([
            make_json_agent_result([]),
            make_json_agent_result([]),
            make_json_agent_result([]),
        ])
        node = _make_all_critics_node(_make_deps(agent))
        state = {**_base_state(), "current_phase": WorkflowPhase.CRITICS.value}

        with patch("sacv.orchestration.graph.bind_node_context") as mock_bind, \
             patch("sacv.orchestration.graph.node_timer") as mock_timer:
            mock_timer.return_value.__aenter__ = AsyncMock(return_value={})
            mock_timer.return_value.__aexit__ = AsyncMock(return_value=False)
            await node(state)

        mock_bind.assert_called_once()
        assert mock_bind.call_args[0][1] == "all_critics"
        mock_timer.assert_called_once()
        assert mock_timer.call_args[0][0] == "all_critics"

    async def test_verifier_with_confidence_calls_bind_node_context_and_node_timer(self):
        """verifier_with_confidence calls bind_node_context and node_timer."""
        from sacv.orchestration.graph import _inject_confidence

        agent = StubAgentProvider([make_json_agent_result({
            "verdict": "PASS", "phase1_passed": True, "phase2_passed": True,
            "confidence": 0.9,
        })])
        node = _inject_confidence(_make_deps(agent))
        state = {**_base_state(), "current_phase": WorkflowPhase.VERIFIER.value}

        with patch("sacv.orchestration.graph.bind_node_context") as mock_bind, \
             patch("sacv.orchestration.graph.node_timer") as mock_timer:
            mock_timer.return_value.__aenter__ = AsyncMock(return_value={})
            mock_timer.return_value.__aexit__ = AsyncMock(return_value=False)
            await node(state)

        mock_bind.assert_called_once()
        assert mock_bind.call_args[0][1] == "verifier_with_confidence"
        mock_timer.assert_called_once()
        assert mock_timer.call_args[0][0] == "verifier_with_confidence"
