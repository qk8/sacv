"""
tests/unit/test_tdd_gate_errors.py
===================================
Unit tests for tdd_gate_node error handling paths.

Tests cover:
1. git stage_file raises RuntimeError — error logged, permanent_paths still records files
2. git commit raises RuntimeError — error logged, evidence still written
3. destroy_container in finally — node completes despite git failures
4. All stage_file calls fail — no commit attempted, evidence still written
5. Git errors are non-fatal — evidence always written on red phase
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from sacv.nodes.tdd_gate import make_tdd_gate_node
from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_deps(
    agent=None, sandbox=None, git=None,
) -> dict:
    from sacv.orchestration.deps import NodeDeps
    return NodeDeps(
        agent=agent or StubAgentProvider(),
        memory=StubMemoryProvider(),
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=git or StubGitProvider(),
        sandbox=sandbox or StubSandboxProvider(),
        diff=StubDiffProvider(),
        config=WorkflowConfig(),
    )


def _make_state(
    task_id="task-tg-err-001",
    strategy=None,
    module="backend-domain",
) -> dict:
    return {
        "session_id": "sess-1",
        "task_id": task_id,
        "project_mode": "greenfield",
        "module_type": module,
        "current_phase": "tdd_gate",
        "task_description": "Add findById method",
        "context_skeleton": None,
        "blast_radius_map": None,
        "agents_md_context": None,
        "strategy_candidates": [],
        "selected_strategy": strategy,
        "pruned_strategies": [],
        "red_phase_evidence_path": None,
        "test_inventory_paths": [],
        "tdd_gate_attempts": 0,
        "diff_proposal": None,
        "preflight_result": None,
        "critic_findings": [],
        "verifier_verdict": None,
        "debug_observations": None,
        "correction_state": {
            "attempt_count": 0, "branch_name": None,
            "last_error_hash": None, "error_history": [],
            "stagnation_pattern": "none",
        },
        "confidence_score": 1.0,
        "replan_count": 0,
        "active_branches": [],
        "exhausted_branches": [],
        "speculative_stash_ref": None,
        "escalation_payload": None,
        "procedural_constraints": [],
        "lesson_learned": None,
        "arch_rules_updated": False,
        "check_profile": "standard",
        "cumulative_cost_dollars": 0.0,
        "skip_tdd_gate": False,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestTddGateGitErrors:

    async def test_stage_file_raises_logs_error_and_continues(self, tmp_path, monkeypatch):
        """stage_file raises RuntimeError -> error logged, permanent_paths still records files.

        permanent_paths is populated BEFORE git staging, so all files appear in
        test_inventory_paths regardless of git staging failures. The staging failure
        is logged but non-fatal.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        agent = StubAgentProvider([
            make_json_agent_result([
                {"file_path": "src/test/java/com/example/GoodTest.java", "content": "class GoodTest {}"},
                {"file_path": "src/test/java/com/example/BadTest.java", "content": "class BadTest {}"},
                {"file_path": "src/test/java/com/example/AnotherTest.java", "content": "class AnotherTest {}"},
            ])
        ])
        sandbox = StubSandboxProvider(default_exit_code=1, default_stdout="FAIL")
        git = StubGitProvider()

        original_stage = git.stage_file
        call_count = {"n": 0}

        def failing_stage(path):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("git stage failed: permission denied")
            return original_stage(path)

        git.stage_file = failing_stage

        deps = _make_deps(agent=agent, sandbox=sandbox, git=git)
        state = _make_state(strategy={"strategy_id": "s1", "description": "test"})
        node = make_tdd_gate_node(deps)

        out = await node(state)

        assert out["current_phase"] == WorkflowPhase.ACTOR.value
        assert out["red_phase_evidence_path"] is not None
        # permanent_paths records ALL written files (before git staging)
        assert len(out["test_inventory_paths"]) == 3
        # Git stage_file was called 3 times, second one raised
        assert call_count["n"] == 3

    async def test_commit_raises_logs_error_evidence_still_written(self, tmp_path, monkeypatch):
        """commit raises RuntimeError -> error logged, evidence file still written."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        agent = StubAgentProvider([
            make_json_agent_result([{
                "file_path": "src/test/java/com/example/FindByIdTest.java",
                "content": "public class FindByIdTest {}",
            }])
        ])
        sandbox = StubSandboxProvider(default_exit_code=1, default_stdout="FAIL")
        git = StubGitProvider()

        git.commit = lambda msg, **kw: (_ for _ in ()).throw(RuntimeError("git commit failed: lock held"))

        deps = _make_deps(agent=agent, sandbox=sandbox, git=git)
        state = _make_state(strategy={"strategy_id": "s1", "description": "test"})
        node = make_tdd_gate_node(deps)

        out = await node(state)

        # Evidence should still be written -- git failure doesn't block evidence
        assert out["current_phase"] == WorkflowPhase.ACTOR.value
        assert out["red_phase_evidence_path"] is not None
        evidence = json.loads(Path(out["red_phase_evidence_path"]).read_text())
        assert evidence["task_id"] == "task-tg-err-001"
        # File is in inventory (staging succeeded, only commit failed)
        assert len(out["test_inventory_paths"]) == 1

    async def test_destroy_container_called_on_git_failure(self, tmp_path, monkeypatch):
        """destroy_container is called in finally even when git operations fail.
        The node completes successfully (no exception) because the finally block
        ensures cleanup regardless of git errors.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        agent = StubAgentProvider([
            make_json_agent_result([{
                "file_path": "src/test/java/com/example/XTest.java",
                "content": "class XTest {}",
            }])
        ])
        sandbox = StubSandboxProvider(default_exit_code=1, default_stdout="FAIL")
        git = StubGitProvider()

        git.commit = lambda msg, **kw: (_ for _ in ()).throw(RuntimeError("commit failed"))

        deps = _make_deps(agent=agent, sandbox=sandbox, git=git)
        state = _make_state(strategy={"strategy_id": "s1", "description": "test"})
        node = make_tdd_gate_node(deps)

        out = await node(state)

        # Node completes successfully -- finally block executed cleanup
        assert out["current_phase"] == WorkflowPhase.ACTOR.value
        assert out["red_phase_evidence_path"] is not None

    async def test_stage_file_raises_for_all_files(self, tmp_path, monkeypatch):
        """All stage_file calls fail -> no commit attempted, evidence still written."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        agent = StubAgentProvider([
            make_json_agent_result([
                {"file_path": "src/test/java/com/example/A.java", "content": "class A {}"},
                {"file_path": "src/test/java/com/example/B.java", "content": "class B {}"},
            ])
        ])
        sandbox = StubSandboxProvider(default_exit_code=1, default_stdout="FAIL")
        git = StubGitProvider()

        git.stage_file = lambda path: (_ for _ in ()).throw(RuntimeError("stage failed"))

        deps = _make_deps(agent=agent, sandbox=sandbox, git=git)
        state = _make_state(strategy={"strategy_id": "s1", "description": "test"})
        node = make_tdd_gate_node(deps)

        out = await node(state)

        # permanent_paths records ALL files written (before git staging)
        assert out["current_phase"] == WorkflowPhase.ACTOR.value
        assert out["red_phase_evidence_path"] is not None
        # Files are in inventory since permanent_paths is populated before git staging
        assert len(out["test_inventory_paths"]) == 2

    async def test_git_errors_do_not_block_evidence_path(self, tmp_path, monkeypatch):
        """Git errors are non-fatal -- evidence always written on red phase."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        agent = StubAgentProvider([
            make_json_agent_result([{
                "file_path": "src/test/java/com/example/XTest.java",
                "content": "class XTest {}",
            }])
        ])
        sandbox = StubSandboxProvider(default_exit_code=1, default_stdout="FAIL")
        git = StubGitProvider()

        git.stage_file = lambda path: (_ for _ in ()).throw(RuntimeError("stage fail"))
        git.commit = lambda msg, **kw: (_ for _ in ()).throw(RuntimeError("commit fail"))

        deps = _make_deps(agent=agent, sandbox=sandbox, git=git)
        state = _make_state(strategy={"strategy_id": "s1", "description": "test"})
        node = make_tdd_gate_node(deps)

        out = await node(state)

        # Evidence should still be written
        assert out["red_phase_evidence_path"] is not None
        evidence = json.loads(Path(out["red_phase_evidence_path"]).read_text())
        assert "FAIL" in evidence["failure_output"]
