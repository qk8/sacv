"""
tests/unit/test_style_critic.py
=================================
Unit tests for the Style Critic node.

Tests cover:
1. Returns critic_findings and cumulative_cost_dollars
2. Agent called with style role
3. Agent called with style-specific rules
4. No proposal → returns empty findings
5. Module type context included in prompt
6. Project mode context included in prompt
"""
from __future__ import annotations

import pytest

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.interfaces.agent_provider import AgentResult
from sacv.nodes.critics.style import make_style_critic_node


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
        "session_id": "t", "task_id": "task-sc-001",
        "task_description": "", "project_mode": "greenfield",
        "module_type": "backend-domain",
        "current_phase": WorkflowPhase.CRITICS.value,
        "context_skeleton": None, "blast_radius_map": None,
        "agents_md_context": None, "strategy_candidates": [],
        "selected_strategy": None, "pruned_strategies": [],
        "red_phase_evidence_path": None, "test_inventory_paths": [],
        "diff_proposal": {
            "diffs": [
                {
                    "file_path": "src/main/java/UserService.java",
                    "operation": "modify",
                    "diff_content": "+    public User findById(Long id) { return null; }",
                },
            ],
        },
        "preflight_result": None, "critic_findings": [],
        "verifier_verdict": None, "debug_observations": None,
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
class TestStyleCriticNode:

    async def test_returns_findings_and_cost(self):
        """Style critic returns critic_findings and cumulative_cost_dollars."""
        agent = StubAgentProvider([make_json_agent_result([])])
        deps = _deps(agent=agent)
        node = make_style_critic_node(deps)

        out = await node(_state())

        assert "critic_findings" in out
        assert "cumulative_cost_dollars" in out

    async def test_agent_called_with_style_role(self):
        """Agent is called with the 'style' role."""
        agent = StubAgentProvider([make_json_agent_result([])])
        deps = _deps(agent=agent)
        node = make_style_critic_node(deps)

        await node(_state())

        assert len(agent.calls) == 1
        role, _ = agent.calls[0]
        assert role == "style"

    async def test_agent_called_with_dd_rules(self):
        """Agent receives DDD/Clean Architecture rules in system prompt."""
        received_system_prompts = []

        class _TrackingAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                received_system_prompts.append(config.system_prompt)
                return await super().run_task(prompt, context, config)

        agent = _TrackingAgent([make_json_agent_result([])])
        deps = _deps(agent=agent)
        node = make_style_critic_node(deps)

        await node(_state())

        system_prompt = received_system_prompts[0]
        assert "DDD" in system_prompt
        assert "Clean Architecture" in system_prompt

    async def test_no_proposal_returns_empty(self):
        """When diff_proposal is absent, style critic returns empty findings."""
        agent = StubAgentProvider([])  # Should NOT be called
        deps = _deps(agent=agent)
        node = make_style_critic_node(deps)

        out = await node(_state(diff_proposal=None))

        assert out["critic_findings"] == []
        assert len(agent.calls) == 0

    async def test_agent_called_with_module_type_context(self):
        """Prompt includes module_type in user prompt."""
        received_prompts = []

        class _TrackingAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                received_prompts.append(prompt)
                return await super().run_task(prompt, context, config)

        agent = _TrackingAgent([make_json_agent_result([])])
        deps = _deps(agent=agent)
        node = make_style_critic_node(deps)

        await node(_state(module_type="frontend-feature"))

        prompt = received_prompts[0]
        assert "frontend-feature" in prompt

    async def test_agent_called_with_project_mode_context(self):
        """Prompt includes project_mode in user prompt."""
        received_prompts = []

        class _TrackingAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                received_prompts.append(prompt)
                return await super().run_task(prompt, context, config)

        agent = _TrackingAgent([make_json_agent_result([])])
        deps = _deps(agent=agent)
        node = make_style_critic_node(deps)

        await node(_state(project_mode="brownfield"))

        prompt = received_prompts[0]
        assert "brownfield" in prompt

    async def test_valid_findings_parsed(self):
        """Valid JSON findings from LLM are returned."""
        agent = StubAgentProvider([make_json_agent_result([
            {
                "critic": "style", "severity": "warning",
                "file": "UserService.java", "line": 10,
                "rule_id": "DDD-001", "message": "Domain entity imports infra",
                "resolution_hint": "Move import to service layer",
            },
        ])])
        deps = _deps(agent=agent)
        node = make_style_critic_node(deps)

        out = await node(_state())

        assert len(out["critic_findings"]) == 1
        assert out["critic_findings"][0]["critic"] == "style"
        assert out["critic_findings"][0]["severity"] == "warning"

    async def test_invalid_json_returns_empty_after_exhausted_retries(self):
        """Malformed JSON → retries 2x then returns empty findings."""
        agent = StubAgentProvider([
            AgentResult(content="not json {{{",
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content="still bad",
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content="giving up",
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
        ])
        deps = _deps(agent=agent)
        node = make_style_critic_node(deps)

        out = await node(_state())

        assert out["critic_findings"] == []

    async def test_cost_accumulated(self):
        """Token cost from agent call is accumulated."""
        agent = StubAgentProvider([AgentResult(
            content="[]",
            tool_calls=[], finish_reason="stop",
            input_tokens=100, output_tokens=200,
        )])
        deps = _deps(agent=agent)
        node = make_style_critic_node(deps)

        out = await node(_state())

        assert out["cumulative_cost_dollars"] > 0.0
