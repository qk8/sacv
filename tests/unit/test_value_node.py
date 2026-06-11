"""
tests/unit/test_value_node.py
================================
Unit tests for the value node — strategy generation, scoring, and pruning.

Tests cover:
1. LLM parse error returns empty strategies
2. Strategies scored and pruned correctly
3. Highest-scoring strategy selected
4. Cost accumulated from agent result
5. Replan context included on retry
6. Phase advances to TDD_GATE
7. Empty candidates handled gracefully
"""
from __future__ import annotations

import json
import pytest

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.interfaces.agent_provider import AgentConfig, AgentResult
from sacv.nodes.value_node import make_value_node


def _deps(agent=None):
    from sacv.orchestration.deps import NodeDeps
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


def _state(**kw):
    base = {
        "session_id": "t", "task_id": "task-vn-001",
        "task_description": "Add UserService.findById method",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "current_phase": WorkflowPhase.VALUE_NODE.value,
        "context_skeleton": {"call_graph": {"nodes": [], "edges": []},
                            "dependencies": {}, "schema_map": {}, "arch_align": {}},
        "blast_radius_map": None,
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
class TestValueNode:

    async def test_parse_error_returns_empty_strategies(self):
        """Invalid JSON from LLM → retries exhausted → empty strategy list."""
        agent = StubAgentProvider([
            AgentResult(content="not json 1", tool_calls=[], finish_reason="stop",
                        input_tokens=10, output_tokens=5),
            AgentResult(content="not json 2", tool_calls=[], finish_reason="stop",
                        input_tokens=10, output_tokens=5),
            AgentResult(content="not json 3", tool_calls=[], finish_reason="stop",
                        input_tokens=10, output_tokens=5),
            AgentResult(content="not json 4", tool_calls=[], finish_reason="stop",
                        input_tokens=10, output_tokens=5),
        ])
        node = make_value_node(_deps(agent))
        out = await node(_state())

        assert out["current_phase"] == WorkflowPhase.TDD_GATE.value
        assert out["strategy_candidates"] == []
        assert out["selected_strategy"] is None

    async def test_non_list_json_returns_empty_strategies(self):
        """LLM returns a dict instead of array → retries exhausted → empty."""
        agent = StubAgentProvider([
            AgentResult(content='{"strategy_id": "s1"}', tool_calls=[], finish_reason="stop",
                        input_tokens=10, output_tokens=5),
            AgentResult(content='{"strategy_id": "s1"}', tool_calls=[], finish_reason="stop",
                        input_tokens=10, output_tokens=5),
            AgentResult(content='{"strategy_id": "s1"}', tool_calls=[], finish_reason="stop",
                        input_tokens=10, output_tokens=5),
            AgentResult(content='{"strategy_id": "s1"}', tool_calls=[], finish_reason="stop",
                        input_tokens=10, output_tokens=5),
        ])
        node = make_value_node(_deps(agent))
        out = await node(_state())

        assert out["strategy_candidates"] == []

    async def test_strategies_scored_and_selected(self):
        """Valid strategy list → scored, pruned, highest selected."""
        strategies = [
            {"strategy_id": "s1", "description": "Simple lookup",
             "affected_files": ["UserService.java"]},
            {"strategy_id": "s2", "description": "Repository pattern",
             "affected_files": ["UserService.java", "UserRepo.java",
                                "UserEntity.java", "UserDTO.java"]},
        ]
        agent = StubAgentProvider([
            AgentResult(content=json.dumps(strategies),
                        tool_calls=[], finish_reason="stop",
                        input_tokens=20, output_tokens=50),
        ])
        node = make_value_node(_deps(agent))
        out = await node(_state())

        assert len(out["strategy_candidates"]) > 0
        assert out["selected_strategy"] is not None
        assert out["selected_strategy"]["strategy_id"] in ["s1", "s2"]

    async def test_highest_scoring_strategy_selected(self):
        """Strategy with lower collision and blast impact scores higher.

        s1 has high collision (all its files shared with s2) → lower score.
        s2 has low collision (only 1/6 files shared) → higher score.
        """
        strategies = [
            {"strategy_id": "s1", "description": "Narrow but colliding",
             "affected_files": ["UserService.java"]},
            {"strategy_id": "s2", "description": "Wide but unique",
             "affected_files": ["UserService.java", "UserRepo.java",
                                "UserEntity.java", "UserDTO.java",
                                "UserMapper.java", "UserValidator.java"]},
        ]
        agent = StubAgentProvider([
            AgentResult(content=json.dumps(strategies),
                        tool_calls=[], finish_reason="stop",
                        input_tokens=20, output_tokens=50),
        ])
        node = make_value_node(_deps(agent))
        out = await node(_state())

        # s2 wins because s1 has 100% collision ratio
        assert out["selected_strategy"]["strategy_id"] == "s2"

    async def test_cost_accumulated(self):
        """Cost accumulated from token counts via extract_structured."""
        agent = StubAgentProvider([
            AgentResult(content="[]",
                        tool_calls=[], finish_reason="stop",
                        input_tokens=100, output_tokens=200),
        ])
        node = make_value_node(_deps(agent))
        out = await node(_state())

        # 100/1M * 5.0 + 200/1M * 30.0 = 0.0065
        assert out["cumulative_cost_dollars"] == pytest.approx(0.0065, abs=0.001)

    async def test_replan_context_included_on_retry(self):
        """When replan_count > 0, previous failures are included in prompt."""
        strategies = [{"strategy_id": "s1", "description": "new approach",
                       "affected_files": ["NewService.java"]}]
        agent = StubAgentProvider([
            AgentResult(content=json.dumps(strategies),
                        tool_calls=[], finish_reason="stop",
                        input_tokens=20, output_tokens=30),
        ])
        # Track what prompt was sent
        received_prompts = []

        class _TrackingAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                received_prompts.append(prompt)
                return await super().run_task(prompt, context, config)

        node = make_value_node(_deps(_TrackingAgent([
            AgentResult(content=json.dumps(strategies),
                        tool_calls=[], finish_reason="stop",
                        input_tokens=20, output_tokens=30),
        ])))
        out = await node(_state(
            replan_count=1,
            exhausted_branches=["s_old"],
            verifier_verdict={
                "test_failures": [{"message": "NPE at line 42"}],
                "diagnostic": "FIX_IMPL",
            },
        ))

        assert out["current_phase"] == WorkflowPhase.TDD_GATE.value
        assert len(received_prompts) == 1
        prompt = received_prompts[0]
        assert "Previous Attempts Failed" in prompt
        assert "s_old" in prompt
        assert "NPE at line 42" in prompt

    async def test_exhausted_branches_included_in_prompt(self):
        """Exhausted branches listed in replan context."""
        strategies = [{"strategy_id": "s1", "description": "different",
                       "affected_files": ["X.java"]}]
        agent = StubAgentProvider([
            AgentResult(content=json.dumps(strategies),
                        tool_calls=[], finish_reason="stop",
                        input_tokens=10, output_tokens=10),
        ])

        received_prompts = []

        class _TrackingAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                received_prompts.append(prompt)
                return await super().run_task(prompt, context, config)

        node = make_value_node(_deps(_TrackingAgent([
            AgentResult(content=json.dumps(strategies),
                        tool_calls=[], finish_reason="stop",
                        input_tokens=10, output_tokens=10),
        ])))
        await node(_state(exhausted_branches=["old-strategy"]))

        assert "old-strategy" in received_prompts[0]

    async def test_empty_candidates_no_crash(self):
        """When LLM returns empty array, selected_strategy is None."""
        agent = StubAgentProvider([
            AgentResult(content="[]",
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=2),
        ])
        node = make_value_node(_deps(agent))
        out = await node(_state())

        assert out["selected_strategy"] is None
        assert out["strategy_candidates"] == []

    async def test_pruned_strategies_recorded(self):
        """Low-scoring strategies are recorded in pruned_strategies."""
        # Create strategies that will be pruned (below min_strategy_score)
        # With default min_strategy_score, strategies with very high collision
        # or blast radius will be pruned
        strategies = [
            {"strategy_id": "s1", "description": "will pass",
             "affected_files": ["A.java"]},
        ]
        agent = StubAgentProvider([
            AgentResult(content=json.dumps(strategies),
                        tool_calls=[], finish_reason="stop",
                        input_tokens=10, output_tokens=10),
        ])
        node = make_value_node(_deps(agent))
        out = await node(_state())

        # Phase should advance regardless
        assert out["current_phase"] == WorkflowPhase.TDD_GATE.value

    async def test_blast_radius_affects_scoring(self):
        """Strategies touching blast radius files score lower."""
        blast = {
            "entry_files": ["UserService.java"],
            "affected_files": ["UserService.java", "UserRepo.java"],
            "dependency_depth": 2, "cross_service_impact": [],
            "schema_impact": [], "risk_score": 0.7,
        }
        # s1 touches blast files, s2 does not
        strategies = [
            {"strategy_id": "s1", "description": "touches blast",
             "affected_files": ["UserService.java"]},
            {"strategy_id": "s2", "description": "avoids blast",
             "affected_files": ["NewService.java"]},
        ]
        agent = StubAgentProvider([
            AgentResult(content=json.dumps(strategies),
                        tool_calls=[], finish_reason="stop",
                        input_tokens=20, output_tokens=30),
        ])
        node = make_value_node(_deps(agent))
        out = await node(_state(blast_radius_map=blast))

        # s2 should be selected since it avoids blast radius
        assert out["selected_strategy"]["strategy_id"] == "s2"

    async def test_agent_called_with_correct_role(self):
        """Agent is called with structured_output role (extract_structured wrapper)."""
        agent = StubAgentProvider([
            AgentResult(content="[]",
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
        ])
        node = make_value_node(_deps(agent))
        await node(_state())

        assert len(agent.calls) == 1
        role, _ = agent.calls[0]
        assert role == "structured_output"

    async def test_all_strategies_pruned_by_scoring(self):
        """When ALL strategies are pruned by scoring, selected_strategy is None and phase advances.

        This tests the edge case where the LLM returns valid strategy objects
        but the deterministic scoring/pruning eliminates all of them (e.g.,
        all have composite_score below min_strategy_score).
        """
        # Create strategies that will all be pruned: high collision + high blast impact
        blast = {
            "entry_files": ["UserService.java", "UserRepo.java", "UserEntity.java",
                            "UserDTO.java", "UserMapper.java", "UserValidator.java"],
            "affected_files": ["UserService.java", "UserRepo.java", "UserEntity.java",
                               "UserDTO.java", "UserMapper.java", "UserValidator.java"],
            "dependency_depth": 6, "cross_service_impact": [],
            "schema_impact": [], "risk_score": 1.0,
        }
        # All strategies touch the same files → 100% collision → low scores
        strategies = [
            {"strategy_id": "s1", "description": "colliding 1",
             "affected_files": ["UserService.java", "UserRepo.java", "UserEntity.java",
                                "UserDTO.java", "UserMapper.java", "UserValidator.java"]},
            {"strategy_id": "s2", "description": "colliding 2",
             "affected_files": ["UserService.java", "UserRepo.java", "UserEntity.java",
                                "UserDTO.java", "UserMapper.java", "UserValidator.java"]},
        ]
        agent = StubAgentProvider([
            AgentResult(content=json.dumps(strategies),
                        tool_calls=[], finish_reason="stop",
                        input_tokens=20, output_tokens=50),
        ])
        # Use a high min_strategy_score so the strategies (scored ~0.35) are pruned
        cfg = WorkflowConfig(min_strategy_score=0.5)
        from sacv.orchestration.deps import NodeDeps
        node = make_value_node(NodeDeps(
            agent=agent,
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=StubSandboxProvider(),
            diff=StubDiffProvider(),
            config=cfg,
        ))
        out = await node(_state(blast_radius_map=blast))

        # All strategies pruned → selected_strategy is None
        assert out["selected_strategy"] is None
        # No strategies pass pruning
        assert out["strategy_candidates"] == []
        # Phase still advances to TDD_GATE
        assert out["current_phase"] == WorkflowPhase.TDD_GATE.value
        # Pruned strategies recorded
        assert len(out["pruned_strategies"]) == 2
        # Audit trail entry has selected_id=None
        audit = out.get("workflow_audit_trail", [])
        assert any(
            entry.get("node") == "value_node" and entry.get("selected_id") is None
            for entry in audit
        ), "audit trail must record selected_id=None when all strategies pruned"

    async def test_selected_strategy_none_when_empty_candidates(self):
        """When LLM returns valid but empty array, selected_strategy is None (not error)."""
        agent = StubAgentProvider([
            AgentResult(content="[]",
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=2),
        ])
        node = make_value_node(_deps(agent))
        out = await node(_state())

        assert out["selected_strategy"] is None
        assert out["strategy_candidates"] == []
        assert out["current_phase"] == WorkflowPhase.TDD_GATE.value
