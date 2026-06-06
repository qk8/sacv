"""
tests/unit/test_tdd_gate.py
============================
Unit tests for the tdd_gate_node function itself.

Tests cover:
1. No strategy selected → returns tdd_gate_attempts + 1
2. JSON parse failure → returns tdd_gate_attempts + 1
3. Tests pass unexpectedly (green phase) → returns tdd_gate_attempts + 1
4. Red phase confirmed → advances to ACTOR, records evidence, writes test files
5. skip_tdd_gate → returns early without agent/sandbox calls
6. Token cost accumulation
7. Multiple test files written to canonical paths
8. Frontend module uses e2e test path
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from sacv.nodes.tdd_gate import make_tdd_gate_node
from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.interfaces.sandbox_provider import ExecResult
from sacv.interfaces.agent_provider import AgentResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_deps(
    agent: StubAgentProvider | None = None,
    sandbox: StubSandboxProvider | None = None,
) -> dict:
    """Return NodeDeps with minimal stubs."""
    from sacv.orchestration.deps import NodeDeps
    return NodeDeps(
        agent=agent or StubAgentProvider(),
        memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=StubGitProvider(),
        sandbox=sandbox or StubSandboxProvider(),
        diff=StubDiffProvider(),
        config=WorkflowConfig(),
    )


def _make_state(
    task_id: str = "task-tg-001",
    strategy: dict | None = None,
    module: str = "backend-domain",
    skip: bool = False,
    cost: float = 0.0,
) -> dict:
    return {
        "session_id":             "sess-1",
        "task_id":                task_id,
        "project_mode":           "greenfield",
        "module_type":            module,
        "current_phase":          "tdd_gate",
        "task_description":       "Add findById method",
        "context_skeleton":       None,
        "blast_radius_map":       None,
        "agents_md_context":      None,
        "strategy_candidates":    [],
        "selected_strategy":      strategy,
        "pruned_strategies":      [],
        "red_phase_evidence_path": None,
        "test_inventory_paths":   [],
        "tdd_gate_attempts":      0,
        "diff_proposal":          None,
        "preflight_result":       None,
        "critic_findings":        [],
        "verifier_verdict":       None,
        "debug_observations":     None,
        "correction_state":       {
            "attempt_count": 0, "branch_name": None,
            "last_error_hash": None, "error_history": [],
            "stagnation_pattern": "none",
        },
        "confidence_score":       1.0,
        "replan_count":           0,
        "active_branches":        [],
        "exhausted_branches":     [],
        "speculative_stash_ref":  None,
        "escalation_payload":     None,
        "procedural_constraints": [],
        "lesson_learned":         None,
        "arch_rules_updated":     False,
        "check_profile":          "standard",
        "cumulative_cost_dollars": cost,
        "skip_tdd_gate":          skip,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestTddGateNode:

    async def test_no_strategy_returns_attempts_increment(self):
        """When strategy is None, skip agent call and increment attempts."""
        deps = _make_deps()
        state = _make_state(strategy=None)
        node = make_tdd_gate_node(deps)

        out = await node(state)

        assert out["red_phase_evidence_path"] is None
        assert out["test_inventory_paths"] == []
        assert out["tdd_gate_attempts"] == 1
        # Agent should NOT have been called
        assert len(deps.agent.calls) == 0

    async def test_json_parse_failure_returns_attempts_increment(self):
        """Invalid JSON from agent → increment attempts."""
        agent = StubAgentProvider([
            AgentResult(
                content="not valid json {{{",
                tool_calls=[], finish_reason="stop",
                input_tokens=5, output_tokens=5,
            )
        ])
        deps = _make_deps(agent=agent)
        strategy = {"strategy_id": "s1", "description": "test"}
        state = _make_state(strategy=strategy)
        node = make_tdd_gate_node(deps)

        out = await node(state)

        assert out["red_phase_evidence_path"] is None
        assert out["test_inventory_paths"] == []
        assert out["tdd_gate_attempts"] == 1
        # Agent was called once
        assert len(deps.agent.calls) == 1

    async def test_tests_pass_unexpectedly_returns_attempts_increment(self):
        """Tests pass (exit_code=0) → green phase detected, increment attempts."""
        agent = StubAgentProvider([
            make_json_agent_result([{
                "file_path": "src/test/java/com/example/FindByIdTest.java",
                "content": "public class FindByIdTest {}",
            }])
        ])
        sandbox = StubSandboxProvider(
            default_exit_code=0,  # tests passed — unexpected!
            default_stdout="Tests run: 1, Failures: 0",
        )
        deps = _make_deps(agent=agent, sandbox=sandbox)
        strategy = {"strategy_id": "s1", "description": "test"}
        state = _make_state(strategy=strategy)
        node = make_tdd_gate_node(deps)

        out = await node(state)

        assert out["red_phase_evidence_path"] is None
        assert out["test_inventory_paths"] == []
        assert out["tdd_gate_attempts"] == 1
        # Agent was called to generate tests
        assert len(deps.agent.calls) == 1

    async def test_red_phase_confirmed_advances_to_actor(self, tmp_path, monkeypatch):
        """Tests fail (exit_code!=0) → red phase confirmed, advance to ACTOR."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        agent = StubAgentProvider([
            make_json_agent_result([{
                "file_path": "src/test/java/com/example/FindByIdTest.java",
                "content": "public class FindByIdTest {}",
            }])
        ])
        sandbox = StubSandboxProvider(
            default_exit_code=1,  # tests fail — expected!
            default_stdout="FAILURE",
        )
        deps = _make_deps(agent=agent, sandbox=sandbox)
        strategy = {"strategy_id": "s1", "description": "test"}
        state = _make_state(strategy=strategy)
        node = make_tdd_gate_node(deps)

        out = await node(state)

        assert out["current_phase"] == WorkflowPhase.ACTOR.value
        assert out["red_phase_evidence_path"] is not None
        assert "task-tg-001" in out["red_phase_evidence_path"]
        assert out["test_inventory_paths"] == [
            "src/test/java/com/example/FindByIdTest.java",
        ]
        # Evidence file should exist on disk
        evidence = json.loads(Path(out["red_phase_evidence_path"]).read_text())
        assert evidence["task_id"] == "task-tg-001"
        assert evidence["permanent_paths"] == [
            "src/test/java/com/example/FindByIdTest.java",
        ]
        assert "FAILURE" in evidence["failure_output"]

    async def test_skip_tdd_gate_returns_early(self):
        """skip_tdd_gate=True → return immediately without agent/sandbox calls."""
        deps = _make_deps()
        state = _make_state(skip=True)
        node = make_tdd_gate_node(deps)

        out = await node(state)

        assert out["red_phase_evidence_path"] == ".workflow/tdd-evidence/skipped.json"
        assert out["test_inventory_paths"] == []
        assert len(deps.agent.calls) == 0

    async def test_cost_accumulation(self):
        """Token costs from agent call are added to cumulative_cost_dollars."""
        agent = StubAgentProvider([
            AgentResult(
                content=json.dumps([{
                    "file_path": "src/test/X.java",
                    "content": "class X {}",
                }]),
                tool_calls=[], finish_reason="stop",
                input_tokens=100, output_tokens=200,
            )
        ])
        sandbox = StubSandboxProvider(default_exit_code=1, default_stdout="FAIL")
        deps = _make_deps(agent=agent, sandbox=sandbox)
        strategy = {"strategy_id": "s1", "description": "test"}
        state = _make_state(strategy=strategy, cost=0.5)
        node = make_tdd_gate_node(deps)

        out = await node(state)

        assert out["cumulative_cost_dollars"] > 0.5

    async def test_multiple_test_files_written(self):
        """Multiple test files from agent → all written to canonical paths."""
        agent = StubAgentProvider([
            make_json_agent_result([
                {
                    "file_path": "src/test/java/com/example/FindByIdTest.java",
                    "content": "class FindByIdTest {}",
                },
                {
                    "file_path": "src/test/java/com/example/FindAllTest.java",
                    "content": "class FindAllTest {}",
                },
            ])
        ])
        sandbox = StubSandboxProvider(default_exit_code=1, default_stdout="FAIL")
        deps = _make_deps(agent=agent, sandbox=sandbox)
        strategy = {"strategy_id": "s1", "description": "test"}
        state = _make_state(strategy=strategy)
        node = make_tdd_gate_node(deps)

        out = await node(state)

        assert out["test_inventory_paths"] == [
            "src/test/java/com/example/FindByIdTest.java",
            "src/test/java/com/example/FindAllTest.java",
        ]
        # Verify sandbox mkdir was called for both files
        mkdir_calls = [
            c for c in sandbox.exec_calls
            if "mkdir" in c
        ]
        assert len(mkdir_calls) == 2

    async def test_frontend_uses_e2e_path(self):
        """Frontend module rewrites non-e2e paths to tests/e2e/features/."""
        agent = StubAgentProvider([
            make_json_agent_result([{
                "file_path": "tests/unit/login.spec.ts",  # wrong path
                "content": "describe('login', () => {})",
            }])
        ])
        sandbox = StubSandboxProvider(default_exit_code=1, default_stdout="FAIL")
        deps = _make_deps(agent=agent, sandbox=sandbox)
        strategy = {"strategy_id": "s1", "description": "test"}
        state = _make_state(strategy=strategy, module="frontend-feature")
        node = make_tdd_gate_node(deps)

        out = await node(state)

        # Path should be rewritten to tests/e2e/features/
        assert out["test_inventory_paths"] == [
            "tests/e2e/features/task-tg-001.spec.ts",
        ]

    async def test_empty_test_file_list_skips_sandbox(self):
        """Empty test file list → no sandbox calls, still confirms red phase
        with empty test_inventory_paths."""
        agent = StubAgentProvider([
            make_json_agent_result([]),  # empty list
        ])
        sandbox = StubSandboxProvider(default_exit_code=1, default_stdout="FAIL")
        deps = _make_deps(agent=agent, sandbox=sandbox)
        strategy = {"strategy_id": "s1", "description": "test"}
        state = _make_state(strategy=strategy)
        node = make_tdd_gate_node(deps)

        out = await node(state)

        # Empty test list → no files written, but sandbox still runs test command
        assert out["test_inventory_paths"] == []
        # Sandbox exec_in_container is still called for the test command
        assert len(sandbox.exec_calls) >= 1

    async def test_missing_content_field_skipped(self):
        """Test file entries missing content field are silently skipped."""
        agent = StubAgentProvider([
            make_json_agent_result([
                {"file_path": "X.java", "content": "class X {}"},
                {"file_path": "Y.java"},  # missing content
                {"content": "class Z {}"},  # missing file_path
            ])
        ])
        sandbox = StubSandboxProvider(default_exit_code=1, default_stdout="FAIL")
        deps = _make_deps(agent=agent, sandbox=sandbox)
        strategy = {"strategy_id": "s1", "description": "test"}
        state = _make_state(strategy=strategy)
        node = make_tdd_gate_node(deps)

        out = await node(state)

        # Only the valid entry (file_path + content) is written; canonical path applied
        assert len(out["test_inventory_paths"]) == 1
        assert "task-tg-001Test.java" in out["test_inventory_paths"][0]

    async def test_git_commit_on_red_phase(self, tmp_path, monkeypatch):
        """Red phase confirmed → test files are staged and committed."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        agent = StubAgentProvider([
            make_json_agent_result([{
                "file_path": "src/test/java/com/example/FindByIdTest.java",
                "content": "public class FindByIdTest {}",
            }])
        ])
        sandbox = StubSandboxProvider(
            default_exit_code=1,
            default_stdout="FAILURE",
        )
        deps = _make_deps(agent=agent, sandbox=sandbox)
        strategy = {"strategy_id": "s1", "description": "test"}
        state = _make_state(strategy=strategy)
        node = make_tdd_gate_node(deps)

        out = await node(state)

        assert out["current_phase"] == WorkflowPhase.ACTOR.value
        # Git should have staged and committed the test file
        assert ("stage_file", "src/test/java/com/example/FindByIdTest.java") in deps.git.calls
        assert ("commit", "sacv: add test inventory for task-tg-001 [tests]") in deps.git.calls

    async def test_no_git_commit_on_empty_test_list(self, tmp_path, monkeypatch):
        """Empty test file list → no git staging or committing."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        agent = StubAgentProvider([
            make_json_agent_result([]),  # empty list
        ])
        sandbox = StubSandboxProvider(default_exit_code=1, default_stdout="FAIL")
        deps = _make_deps(agent=agent, sandbox=sandbox)
        strategy = {"strategy_id": "s1", "description": "test"}
        state = _make_state(strategy=strategy)
        node = make_tdd_gate_node(deps)

        out = await node(state)

        # No files written → no git operations for test inventory
        git_calls = [c for c in deps.git.calls if c[0] in ("stage_file", "commit")]
        assert len(git_calls) == 0
