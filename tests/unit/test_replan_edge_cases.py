"""
tests/unit/test_replan_edge_cases.py
=====================================
Unit tests for replan_node error handling and edge cases.

Tests cover:
1. StructuredOutputError → raw=[], updated_cost propagated, zero candidates
2. All strategies pruned → selected_strategy=None, passing=[]
3. _truncate helper — empty input, short input, long input
4. _truncate helper — truncation at max_chars with suffix
"""
from __future__ import annotations

import json
import pytest

from sacv.nodes.replan import make_replan_node, _truncate
from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.interfaces.agent_provider import AgentResult
from sacv.nodes._structured_output import StructuredOutputError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_deps() -> dict:
    from sacv.orchestration.deps import NodeDeps
    return NodeDeps(
        agent=StubAgentProvider(),
        memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(),
        sandbox=StubSandboxProvider(),
        diff=StubDiffProvider(),
        config=WorkflowConfig(),
    )


def _make_state(
    task_id: str = "task-replan-001",
    exhausted: list = None,
    candidates: list = None,
    replan_count: int = 0,
    critic_findings: list = None,
    verifier_verdict: dict = None,
    preflight_result: dict = None,
    correction_state: dict = None,
    debug_observations: dict = None,
    diff_proposal: dict = None,
    cost: float = 0.0,
) -> dict:
    return {
        "session_id": "sess-1",
        "task_id": task_id,
        "project_mode": "greenfield",
        "module_type": "backend-domain",
        "current_phase": "replan",
        "task_description": "Add findById method",
        "context_skeleton": {"call_graph": {"nodes": [], "edges": []},
                            "dependencies": {}, "schema_map": {}, "arch_align": {}},
        "blast_radius_map": {"risk_score": 0.3},
        "agents_md_context": None,
        "strategy_candidates": candidates or [],
        "selected_strategy": None,
        "pruned_strategies": [],
        "red_phase_evidence_path": None,
        "test_inventory_paths": [],
        "tdd_gate_attempts": 0,
        "diff_proposal": diff_proposal or {
            "strategy_id": "s1", "branch_name": "agent-task-abc12345-a1",
            "commit_message": "sacv: implement find-by-id",
            "diffs": [{"file_path": "src/main/X.java", "diff_content": "+method", "operation": "modify", "language": "java"}],
        },
        "preflight_result": preflight_result,
        "critic_findings": critic_findings or [],
        "verifier_verdict": verifier_verdict,
        "debug_observations": debug_observations,
        "correction_state": correction_state or {
            "attempt_count": 3, "branch_name": "agent-task-abc12345-a1",
            "last_error_hash": "abc123", "error_history": ["abc123", "def456"],
            "stagnation_pattern": "outcome",
        },
        "confidence_score": 0.3,
        "replan_count": replan_count,
        "active_branches": [],
        "exhausted_branches": exhausted or [
            {"strategy_id": "s1", "error": "compile fail"},
            {"strategy_id": "s2", "error": "test fail"},
        ],
        "speculative_stash_ref": None,
        "escalation_payload": None,
        "procedural_constraints": ["no-deletion"],
        "lesson_learned": None,
        "arch_rules_updated": False,
        "check_profile": "standard",
        "cumulative_cost_dollars": cost,
    }


# ── _truncate tests ───────────────────────────────────────────────────────────

class TestTruncate:

    def test_empty_input_returns_empty(self):
        assert _truncate("") == ""

    def test_none_input_returns_empty(self):
        assert _truncate(None) == ""

    def test_short_input_unchanged(self):
        result = _truncate("hello world")
        assert result == "hello world"

    def test_exact_max_length_unchanged(self):
        text = "x" * 5000
        result = _truncate(text, max_chars=5000)
        assert result == text

    def test_over_max_truncated_with_suffix(self):
        text = "x" * 6000
        result = _truncate(text, max_chars=5000)
        # Result is 5000 chars + suffix "... (6000 chars)" = 5016 chars total
        assert result.startswith("x")
        assert result.endswith("... (6000 chars)")
        assert len(result) == 5000 + len("... (6000 chars)")

    def test_truncation_includes_char_count(self):
        text = "a" * 100
        result = _truncate(text, max_chars=50)
        assert "(100 chars)" in result

    def test_default_max_chars_is_5000(self):
        text = "x" * 5001
        result = _truncate(text)
        # Result is 5000 chars + suffix, so it's actually longer than input
        assert result.startswith("x")
        assert result.endswith("... (5001 chars)")

    def test_unicode_truncated_correctly(self):
        text = "你好世界" * 2000  # long unicode string
        result = _truncate(text, max_chars=100)
        assert len(result) <= 100 + len("... (4000 chars)")

    def test_multiline_truncated(self):
        text = "line\n" * 3000  # 15000 chars
        result = _truncate(text, max_chars=100)
        assert result.startswith("line\n")
        assert "(15000 chars)" in result


# ── StructuredOutputError handling ────────────────────────────────────────────

@pytest.mark.asyncio
class TestReplanStructuredOutputError:

    async def test_parse_error_returns_zero_candidates(self):
        """StructuredOutputError → raw=[], new_candidates=[], selected_strategy=None."""
        agent = StubAgentProvider([
            AgentResult(content="not json", tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content="still not json", tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content="more bad", tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content="final bad", tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
        ])
        deps = _make_deps()
        deps.agent = agent
        state = _make_state()
        node = make_replan_node(deps)

        out = await node(state)

        assert out["current_phase"] == WorkflowPhase.TDD_GATE.value
        assert out["strategy_candidates"] == []
        assert out["selected_strategy"] is None
        # Agent was called 4 times (1 initial + 3 retries)
        assert len(agent.calls) == 4

    async def test_parse_error_propagates_updated_cost(self):
        """StructuredOutputError.updated_cost is propagated to cumulative_cost_dollars."""
        agent = StubAgentProvider([
            AgentResult(content="bad 1", tool_calls=[], finish_reason="stop",
                        input_tokens=100, output_tokens=200),
            AgentResult(content="bad 2", tool_calls=[], finish_reason="stop",
                        input_tokens=100, output_tokens=200),
            AgentResult(content="bad 3", tool_calls=[], finish_reason="stop",
                        input_tokens=100, output_tokens=200),
            AgentResult(content="bad 4", tool_calls=[], finish_reason="stop",
                        input_tokens=100, output_tokens=200),
        ])
        deps = _make_deps()
        deps.agent = agent
        state = _make_state(cost=1.0)
        node = make_replan_node(deps)

        out = await node(state)

        # Cost should have increased from the 4 failed agent calls
        assert out["cumulative_cost_dollars"] >= 1.0

    async def test_parse_error_emits_audit_entry(self):
        """StructuredOutputError → audit trail records the parse_error event."""
        agent = StubAgentProvider([
            AgentResult(content="not json", tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content="not json", tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content="not json", tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            AgentResult(content="not json", tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
        ])
        deps = _make_deps()
        deps.agent = agent
        state = _make_state()
        node = make_replan_node(deps)

        out = await node(state)

        trail = out.get("workflow_audit_trail", [])
        assert len(trail) == 1
        assert trail[0]["node"] == "replan"
        # The decision should reference zero candidates
        assert "new_candidates=0" in trail[0]["decision"]


# ── Empty pruning edge case ───────────────────────────────────────────────────

@pytest.mark.asyncio
class TestReplanEmptyPruning:

    async def test_all_strategies_pruned_selected_is_none(self):
        """When all candidates are pruned -> selected_strategy=None, passing=[]."""
        # Use a high min_strategy_score (0.99) so even good candidates get pruned
        agent = StubAgentProvider([
            make_json_agent_result([{
                "strategy_id": "r1", "description": "new approach",
                "affected_files": ["X.java"], "avoids": "previous error",
            }])
        ])
        deps = _make_deps()
        deps.agent = agent
        deps.config = WorkflowConfig(min_strategy_score=0.99)
        state = _make_state()
        node = make_replan_node(deps)

        out = await node(state)

        assert out["current_phase"] == WorkflowPhase.TDD_GATE.value
        assert out["strategy_candidates"] == []
        assert out["selected_strategy"] is None

    async def test_empty_candidates_audit_record(self):
        """Empty candidates after pruning → audit records selected_id=None."""
        agent = StubAgentProvider([
            make_json_agent_result([{
                "strategy_id": "r1", "description": "new",
                "affected_files": ["X.java"], "avoids": "prev",
            }])
        ])
        deps = _make_deps()
        deps.agent = agent
        deps.config = WorkflowConfig(min_strategy_score=0.99)
        state = _make_state()
        node = make_replan_node(deps)

        out = await node(state)

        trail = out.get("workflow_audit_trail", [])
        assert len(trail) == 1
        assert trail[0]["key_values"]["selected_id"] is None

    async def test_replan_count_increments(self):
        """replan_count increments by 1 on each call."""
        agent = StubAgentProvider([
            make_json_agent_result([{
                "strategy_id": "r1", "description": "new approach",
                "affected_files": ["X.java"], "avoids": "previous error",
            }])
        ])
        deps = _make_deps()
        deps.agent = agent
        state = _make_state(replan_count=2)
        node = make_replan_node(deps)

        out = await node(state)

        assert out["replan_count"] == 3

    async def test_exhausted_branches_reset(self):
        """exhausted_branches reset to empty list after replan."""
        agent = StubAgentProvider([
            make_json_agent_result([{
                "strategy_id": "r1", "description": "new approach",
                "affected_files": ["X.java"], "avoids": "previous error",
            }])
        ])
        deps = _make_deps()
        deps.agent = agent
        state = _make_state(
            exhausted=[
                {"strategy_id": "s1", "error": "fail 1"},
                {"strategy_id": "s2", "error": "fail 2"},
            ]
        )
        node = make_replan_node(deps)

        out = await node(state)

        assert out["exhausted_branches"] == []
        assert out["active_branches"] == []


# ── Correction state reset ────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestReplanCorrectionStateReset:

    async def test_correction_state_reset_on_replan(self):
        """correction_state is reset: attempt_count=0, branch_name=None, etc."""
        agent = StubAgentProvider([
            make_json_agent_result([{
                "strategy_id": "r1", "description": "new approach",
                "affected_files": ["X.java"], "avoids": "previous error",
            }])
        ])
        deps = _make_deps()
        deps.agent = agent
        state = _make_state(
            correction_state={
                "attempt_count": 5,
                "branch_name": "agent-task-old-branch",
                "last_error_hash": "oldhash",
                "error_history": ["h1", "h2", "h3"],
                "stagnation_pattern": "outcome",
            }
        )
        node = make_replan_node(deps)

        out = await node(state)

        reset = out["correction_state"]
        assert reset["attempt_count"] == 0
        assert reset["branch_name"] is None
        assert reset["error_history"] == []
        assert reset["last_error_hash"] is None
        assert reset["stagnation_pattern"] == "none"

    async def test_verdict_and_preflight_reset(self):
        """verifier_verdict, preflight_result, diff_proposal reset to None."""
        agent = StubAgentProvider([
            make_json_agent_result([{
                "strategy_id": "r1", "description": "new",
                "affected_files": ["X.java"], "avoids": "prev",
            }])
        ])
        deps = _make_deps()
        deps.agent = agent
        state = _make_state(
            verifier_verdict={"test_result": "FAIL", "diagnostic": "RUN_CODE"},
            preflight_result={"passed": False},
        )
        node = make_replan_node(deps)

        out = await node(state)

        assert out["verifier_verdict"] is None
        assert out["preflight_result"] is None
        assert out["diff_proposal"] is None

    async def test_tdd_gate_attempts_reset(self):
        """tdd_gate_attempts reset to 0 on replan."""
        agent = StubAgentProvider([
            make_json_agent_result([{
                "strategy_id": "r1", "description": "new",
                "affected_files": ["X.java"], "avoids": "prev",
            }])
        ])
        deps = _make_deps()
        deps.agent = agent
        state = _make_state()
        state["tdd_gate_attempts"] = 5
        node = make_replan_node(deps)

        out = await node(state)

        assert out["tdd_gate_attempts"] == 0

    async def test_empty_diff_retries_reset(self):
        """empty_diff_retries reset to 0 on replan."""
        agent = StubAgentProvider([
            make_json_agent_result([{
                "strategy_id": "r1", "description": "new",
                "affected_files": ["X.java"], "avoids": "prev",
            }])
        ])
        deps = _make_deps()
        deps.agent = agent
        state = _make_state()
        state["empty_diff_retries"] = 3
        node = make_replan_node(deps)

        out = await node(state)

        assert out["empty_diff_retries"] == 0


# ── Failure summary ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestReplanFailureSummary:

    async def test_failure_summary_includes_exhausted_strategies(self):
        """Failure summary includes strategy IDs from strategy_candidates."""
        received_prompts = []

        class TrackingAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                received_prompts.append(prompt)
                return await super().run_task(prompt, context, config)

        agent = TrackingAgent([
            make_json_agent_result([{
                "strategy_id": "r1", "description": "new",
                "affected_files": ["X.java"], "avoids": "prev",
            }])
        ])
        deps = _make_deps()
        deps.agent = agent
        state = _make_state(
            candidates=[
                {"strategy_id": "s1"},
                {"strategy_id": "s2"},
                {"strategy_id": "s3"},
            ]
        )
        node = make_replan_node(deps)

        await node(state)

        assert len(received_prompts) == 1
        assert "s1" in received_prompts[0]
        assert "s2" in received_prompts[0]
        assert "s3" in received_prompts[0]

    async def test_failure_summary_includes_debug_observations(self):
        """Debug observations included in failure summary when present."""
        received_prompts = []

        class TrackingAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                received_prompts.append(prompt)
                return await super().run_task(prompt, context, config)

        agent = TrackingAgent([
            make_json_agent_result([{
                "strategy_id": "r1", "description": "new",
                "affected_files": ["X.java"], "avoids": "prev",
            }])
        ])
        deps = _make_deps()
        deps.agent = agent
        state = _make_state(
            debug_observations={
                "error_type": "COMPILATION",
                "root_cause": "missing import",
                "breakpoint_hits": [],
            }
        )
        node = make_replan_node(deps)

        await node(state)

        assert "debug" in received_prompts[0].lower()
        assert "missing import" in received_prompts[0]

    async def test_failure_summary_includes_critic_findings(self):
        """Critical critic findings included in failure summary."""
        received_prompts = []

        class TrackingAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                received_prompts.append(prompt)
                return await super().run_task(prompt, context, config)

        agent = TrackingAgent([
            make_json_agent_result([{
                "strategy_id": "r1", "description": "new",
                "affected_files": ["X.java"], "avoids": "prev",
            }])
        ])
        deps = _make_deps()
        deps.agent = agent
        state = _make_state(
            critic_findings=[
                {"critic": "security", "severity": "critical", "rule_id": "SEC-1",
                 "file": "X.java", "line": 1, "message": "injection",
                 "resolution_hint": "use params"},
                {"critic": "style", "severity": "info", "rule_id": "STY-1",
                 "file": "Y.java", "line": 5, "message": "naming",
                 "resolution_hint": "rename"},
            ]
        )
        node = make_replan_node(deps)

        await node(state)

        # Critical finding should be in summary; info should not be in critical list
        assert "injection" in received_prompts[0]
        # The summary should reference critical findings
        assert "SEC-1" in received_prompts[0] or "critical" in received_prompts[0]
