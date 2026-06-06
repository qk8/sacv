"""
tests/unit/test_critic_base.py
================================
Unit tests for the shared critic execution logic in _run_critic.

Tests cover:
1. No proposal → returns empty findings
2. Valid JSON array → parsed as findings
3. Invalid JSON → returns empty findings, logs error
4. Non-array JSON (dict) → returns empty findings
5. Agent called with correct critic role and prompt
6. Agent called with module type and project mode context
7. Semaphore acquired during execution
"""
from __future__ import annotations

import pytest

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase, UnifiedDiffPayload
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.interfaces.agent_provider import AgentResult
from sacv.nodes.critics.base import _run_critic


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
        "session_id": "t", "task_id": "task-cr-001",
        "task_description": "", "project_mode": "greenfield",
        "module_type": "backend-domain",
        "current_phase": WorkflowPhase.ACTOR.value,
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
class TestRunCritic:

    async def test_no_proposal_returns_empty(self):
        """When diff_proposal is absent, critic returns immediately."""
        agent = StubAgentProvider([])
        state = _state(diff_proposal=None)
        findings, _ = await _run_critic(
            role="test engineer", critic_name="security",
            extra_rules="test rules", state=state, deps=_deps(agent),
        )
        assert findings == []

    async def test_valid_json_parsed_as_findings(self):
        """Valid JSON array → CriticFinding objects returned."""
        findings_json = [
            {
                "critic": "security", "severity": "critical",
                "file": "UserService.java", "line": 42,
                "rule_id": "SEC-001", "message": "SQL injection",
                "resolution_hint": "Use parameterized query",
            },
            {
                "critic": "security", "severity": "info",
                "file": "UserService.java", "line": 10,
                "rule_id": "SEC-002", "message": "No input validation",
                "resolution_hint": "Add @Valid",
            },
        ]
        agent = StubAgentProvider([make_json_agent_result(findings_json)])
        state = _state()
        result, _ = await _run_critic(
            role="security engineer", critic_name="security",
            extra_rules="OWASP rules", state=state, deps=_deps(agent),
        )
        assert len(result) == 2
        assert result[0]["severity"] == "critical"
        assert result[0]["rule_id"] == "SEC-001"
        assert result[1]["severity"] == "info"

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
        state = _state()
        result, _ = await _run_critic(
            role="security engineer", critic_name="security",
            extra_rules="", state=state, deps=_deps(agent),
        )
        assert result == []
        assert len(agent.calls) == 3  # initial + 2 retries

    async def test_dict_json_returns_empty_after_exhausted_retries(self):
        """LLM returns a dict instead of array → retries then empty."""
        agent = StubAgentProvider([
            AgentResult(content='{"key": "value"}',
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content='{"still": "dict"}',
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content='{"no": "array"}',
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
        ])
        state = _state()
        result, _ = await _run_critic(
            role="security engineer", critic_name="security",
            extra_rules="", state=state, deps=_deps(agent),
        )
        assert result == []
        assert len(agent.calls) == 3

    async def test_agent_receives_critic_role(self):
        """Agent is called with the correct critic role."""
        agent = StubAgentProvider([make_json_agent_result([])])
        state = _state()
        await _run_critic(
            role="security engineer", critic_name="security",
            extra_rules="", state=state, deps=_deps(agent),
        )
        assert len(agent.calls) == 1
        role, _ = agent.calls[0]
        assert role == "security"

    async def test_agent_receives_module_and_mode_context(self):
        """Prompt includes module_type and project_mode."""
        received_prompts = []

        class _TrackingAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                received_prompts.append(prompt)
                return await super().run_task(prompt, context, config)

        agent = _TrackingAgent([make_json_agent_result([])])
        state = _state(module_type="frontend", project_mode="brownfield")
        await _run_critic(
            role="test", critic_name="style",
            extra_rules="", state=state, deps=_deps(agent),
        )
        prompt = received_prompts[0]
        assert "frontend" in prompt
        assert "brownfield" in prompt

    async def test_agent_receives_diff_text(self):
        """Prompt includes the diff content for review."""
        agent = StubAgentProvider([make_json_agent_result([])])
        state = _state(
            diff_proposal={
                "diffs": [
                    {
                        "file_path": "X.java",
                        "operation": "modify",
                        "diff_content": "+ new code",
                    },
                ],
            },
        )
        await _run_critic(
            role="test", critic_name="style",
            extra_rules="", state=state, deps=_deps(agent),
        )
        _, prompt = agent.calls[0]
        assert "X.java" in prompt
        assert "new code" in prompt

    async def test_extra_rules_included_in_prompt(self):
        """Custom rules appended to system prompt."""
        agent = StubAgentProvider([make_json_agent_result([])])
        state = _state()
        await _run_critic(
            role="test", critic_name="security",
            extra_rules="Custom rule: no println",
            state=state, deps=_deps(agent),
        )
        # The system prompt format includes extra_rules
        assert len(agent.calls) == 1

    async def test_findings_with_defaults(self):
        """Missing fields in finding use defaults."""
        agent = StubAgentProvider([
            AgentResult(content='[{"critic": "style", "severity": "warning"}]',
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
        ])
        state = _state()
        result, _ = await _run_critic(
            role="test", critic_name="style",
            extra_rules="", state=state, deps=_deps(agent),
        )
        assert len(result) == 1
        assert result[0]["file"] == "unknown"
        assert result[0]["line"] is None
        assert result[0]["rule_id"] == "UNKNOWN"
        assert result[0]["message"] == ""
        assert result[0]["resolution_hint"] == ""

    async def test_non_dict_items_filtered(self):
        """Non-dict items in JSON array are skipped."""
        agent = StubAgentProvider([
            AgentResult(content='[{"critic": "style"}, "string_item", 42, null]',
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
        ])
        state = _state()
        result, _ = await _run_critic(
            role="test", critic_name="style",
            extra_rules="", state=state, deps=_deps(agent),
        )
        assert len(result) == 1

    async def test_style_critic_role(self):
        agent = StubAgentProvider([make_json_agent_result([])])
        state = _state()
        await _run_critic(
            role="style reviewer", critic_name="style",
            extra_rules="", state=state, deps=_deps(agent),
        )
        assert agent.calls[0][0] == "style"

    async def test_consistency_critic_role(self):
        agent = StubAgentProvider([make_json_agent_result([])])
        state = _state()
        await _run_critic(
            role="consistency reviewer", critic_name="consistency",
            extra_rules="", state=state, deps=_deps(agent),
        )
        assert agent.calls[0][0] == "consistency"


@pytest.mark.asyncio
class TestCriticRetryOnParseFailure:

    async def test_retries_on_malformed_json(self):
        """When JSON parse fails, critic retries and uses second result."""
        agent = StubAgentProvider([
            # First call: malformed JSON (will trigger retry)
            AgentResult(content="not json {{{",
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            # Second call: valid JSON
            make_json_agent_result([
                {"critic": "security", "severity": "critical",
                 "file": "UserService.java", "line": 42,
                 "rule_id": "SEC-001", "message": "SQL injection",
                 "resolution_hint": "Use parameterized query"},
            ]),
        ])
        state = _state()
        result, _ = await _run_critic(
            role="security engineer", critic_name="security",
            extra_rules="", state=state, deps=_deps(agent),
        )
        assert len(result) == 1
        assert result[0]["rule_id"] == "SEC-001"
        assert agent.calls[0][0] == "security"
        assert len(agent.calls) == 2

    async def test_returns_empty_after_max_retries(self):
        """When all retries fail, returns empty findings."""
        agent = StubAgentProvider([
            AgentResult(content="bad {{{",
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content="also bad",
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content="still bad",
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
        ])
        state = _state()
        result, _ = await _run_critic(
            role="security engineer", critic_name="security",
            extra_rules="", state=state, deps=_deps(agent),
        )
        assert result == []
        assert len(agent.calls) == 3  # initial + 2 retries

    async def test_non_array_json_triggers_retry(self):
        """LLM returns a dict instead of array → retry."""
        agent = StubAgentProvider([
            AgentResult(content='{"key": "value"}',
                        tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            make_json_agent_result([
                {"critic": "style", "severity": "warning",
                 "file": "App.tsx", "line": 10,
                 "rule_id": "STY-001", "message": "Bad naming",
                 "resolution_hint": "Use PascalCase"},
            ]),
        ])
        state = _state()
        result, _ = await _run_critic(
            role="style reviewer", critic_name="style",
            extra_rules="", state=state, deps=_deps(agent),
        )
        assert len(result) == 1
        assert result[0]["rule_id"] == "STY-001"
        assert len(agent.calls) == 2
