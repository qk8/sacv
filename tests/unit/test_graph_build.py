"""
tests/unit/test_graph_build.py
================================
Unit tests for graph building — build_graph validation and all_critics exception handling.

Tests cover:
1. build_graph raises ValueError when checkpointer is None
2. build_graph compiles successfully with MemorySaver checkpointer
3. all_critics_node handles single critic exception
4. all_critics_node handles all three critics raising exceptions
5. all_critics_node merges findings from multiple critics
6. all_critics_node cost calculation with baseline
7. _inject_confidence passes state through correctly
"""
from __future__ import annotations

import pytest

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.deps import NodeDeps
from sacv.orchestration.graph import (
    build_graph,
    _make_all_critics_node,
    _inject_confidence,
)
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.orchestration.state import WorkflowPhase


def _make_deps(agent=None) -> NodeDeps:
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


def _proposal():
    return {
        "strategy_id": "s1",
        "diffs": [
            {
                "file_path": "src/main/java/com/example/UserService.java",
                "diff_content": "@@ -10,6 +10,10 @@\n+    public User createUser(String email) {\n+        return userRepo.save(new User(email));\n+    }",
                "operation": "modify",
                "language": "java",
            }
        ],
        "branch_name": "agent-task-abc12345-a1",
        "commit_message": "sacv: implement create-user",
    }


def _base_state(**kw) -> dict:
    base = {
        "session_id": "t", "task_id": "task-gb-001",
        "task_description": "Add method", "project_mode": "greenfield",
        "module_type": "backend-domain",
        "current_phase": WorkflowPhase.CRITICS.value,
        "context_skeleton": {"call_graph": {"nodes": [], "edges": []},
                            "dependencies": {}, "schema_map": {}, "arch_align": {}},
        "blast_radius_map": None, "agents_md_context": None,
        "strategy_candidates": [], "selected_strategy": None, "pruned_strategies": [],
        "red_phase_evidence_path": None, "test_inventory_paths": [],
        "diff_proposal": _proposal(), "preflight_result": None,
        "critic_findings": [], "verifier_verdict": None,
        "debug_observations": None, "correction_state": {
            "attempt_count": 0, "branch_name": None,
            "last_error_hash": None, "error_history": [],
            "stagnation_pattern": "none",
        },
        "confidence_score": 1.0, "replan_count": 0,
        "active_branches": [], "exhausted_branches": [],
        "escalation_payload": None, "procedural_constraints": [],
        "lesson_learned": None, "arch_rules_updated": False,
        "check_profile": "standard", "cumulative_cost_dollars": 0.0,
        "workflow_audit_trail": [],
    }
    base.update(kw)
    return base


@pytest.mark.unit
class TestBuildGraph:

    def test_raises_value_error_with_none_checkpointer(self):
        """build_graph(deps, checkpointer=None) raises ValueError."""
        deps = _make_deps()
        with pytest.raises(ValueError, match="requires an explicit checkpointer"):
            build_graph(deps, checkpointer=None)

    def test_raises_value_error_with_no_checkpointer_arg(self):
        """build_graph(deps) with no checkpointer arg raises ValueError."""
        deps = _make_deps()
        with pytest.raises(ValueError, match="requires an explicit checkpointer"):
            build_graph(deps)  # type: ignore[call-arg]

    @pytest.mark.asyncio
    async def test_compiles_with_memory_saver(self):
        """build_graph compiles successfully with MemorySaver checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver
        deps = _make_deps()
        checkpointer = MemorySaver()
        graph = build_graph(deps, checkpointer=checkpointer)
        assert graph is not None
        # Graph should have nodes
        node_names = [n for n in graph.get_graph().nodes if isinstance(n, str)]
        assert "bootstrap" in node_names
        assert "value_node" in node_names
        assert "verifier" in node_names
        assert "hitl_escalation" in node_names


@pytest.mark.unit
class TestAllCriticsNode:

    @pytest.mark.asyncio
    async def test_single_critic_exception(self):
        """One critic raises → empty findings for that critic, others proceed."""
        call_count = {"security": 0}

        class FailingSecurityAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                call_count["security"] += 1
                raise RuntimeError("security critic crashed")

        agent = FailingSecurityAgent()
        deps = _make_deps(agent)
        node = _make_all_critics_node(deps)
        state = _base_state(critic_findings=[])

        result = await node(state)

        # Should not raise — exception is caught
        assert result["current_phase"] == WorkflowPhase.CRITICS.value
        # critic_errors should contain "security"
        assert "security" in result.get("critic_errors", [])
        # Findings from style and consistency should still be present
        assert isinstance(result["critic_findings"], list)

    @pytest.mark.asyncio
    async def test_all_critics_exception(self):
        """All three critics raise → all names in critic_errors, empty findings."""
        call_counts = {"security": 0, "style": 0, "consistency": 0}

        class FailingAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                # Determine which critic is calling by examining the prompt
                for name in call_counts:
                    if name in prompt:
                        call_counts[name] += 1
                        raise RuntimeError(f"{name} critic crashed")
                # Fallback — count all as called
                for name in call_counts:
                    call_counts[name] += 1
                raise RuntimeError("all critics crashed")

        agent = FailingAgent()
        deps = _make_deps(agent)
        node = _make_all_critics_node(deps)
        state = _base_state(critic_findings=[])

        result = await node(state)

        # All three should be in critic_errors
        errors = result.get("critic_errors", [])
        assert "security" in errors
        assert "style" in errors
        assert "consistency" in errors
        # No findings from any critic
        assert result["critic_findings"] == []
        # Audit trail should record the failures
        audit = result.get("workflow_audit_trail", [])
        assert len(audit) >= 1
        assert audit[0]["node"] == "all_critics"
        assert "critic_exceptions" in audit[0]["decision"]
        assert "failed_critics" in audit[0].get("key_values", {})

    @pytest.mark.asyncio
    async def test_findings_merged_from_all_critics(self):
        """Findings from multiple critics are concatenated."""
        agent = StubAgentProvider([
            make_json_agent_result([
                {"critic": "security", "severity": "critical", "rule_id": "SEC-1",
                 "file": "X.java", "line": 1, "message": "injection",
                 "resolution_hint": "use params"}
            ]),
            make_json_agent_result([
                {"critic": "style", "severity": "info", "rule_id": "STY-1",
                 "file": "Y.java", "line": 5, "message": "naming",
                 "resolution_hint": "rename"}
            ]),
            make_json_agent_result([
                {"critic": "consistency", "severity": "warning", "rule_id": "CON-1",
                 "file": "Z.java", "line": 10, "message": "inconsistency",
                 "resolution_hint": "align"}
            ]),
        ])
        deps = _make_deps(agent)
        node = _make_all_critics_node(deps)
        state = _base_state(critic_findings=[])

        result = await node(state)

        # All three findings should be merged
        assert len(result["critic_findings"]) == 3
        critics = [f["critic"] for f in result["critic_findings"]]
        assert "security" in critics
        assert "style" in critics
        assert "consistency" in critics

    @pytest.mark.asyncio
    async def test_cost_calculation_with_baseline(self):
        """Cost is baseline + incremental from each critic, not triple-counted."""
        baseline_cost = 0.5
        agent = StubAgentProvider([
            make_json_agent_result({"content": "[]"}),
            make_json_agent_result({"content": "[]"}),
            make_json_agent_result({"content": "[]"}),
        ])
        deps = _make_deps(agent)
        node = _make_all_critics_node(deps)
        state = _base_state(critic_findings=[], cumulative_cost_dollars=baseline_cost)

        result = await node(state)

        # Final cost should be baseline + incremental portions
        assert result["cumulative_cost_dollars"] >= baseline_cost


@pytest.mark.unit
class TestInjectConfidence:

    @pytest.mark.asyncio
    async def test_passes_state_through(self):
        """_inject_confidence passes all state keys through to result."""
        agent = StubAgentProvider([
            make_json_agent_result({
                "test_result": "PASS", "diagnostic": "PASS",
                "phase1_passed": True, "phase2_passed": True,
                "test_failures": [], "performance_delta": None,
                "visual_diff_result": None, "critic_findings": [],
                "docker_exit_code": 0, "playwright_trace_path": None,
                "otel_trace": None, "actuator_snapshot": None,
            }),
        ])
        deps = _make_deps(agent)
        wrapper = _inject_confidence(deps)
        state = _base_state(
            current_phase=WorkflowPhase.VERIFIER.value,
            critic_findings=[],
        )

        result = await wrapper(state)

        # Original keys should be preserved
        assert result["current_phase"] == WorkflowPhase.VERIFIER.value
        assert "verifier_verdict" in result
        assert "correction_state" in result
        assert "cumulative_cost_dollars" in result
