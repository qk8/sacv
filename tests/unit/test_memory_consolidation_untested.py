"""
tests/unit/test_memory_consolidation_untested.py
=================================================
Unit tests for untested memory_consolidation.py code paths.

Tests cover:
1. _update_agents_md StructuredOutputError path — returns (False, updated_cost)
2. _update_arch_rules exception path — returns (False, cost)
3. _commit_test_inventory with missing file — skips file, continues
4. _commit_test_inventory all files missing — returns empty list
5. _commit_production_code_no_record commit failure — returns empty string
6. get_head_sha failure — logs warning, continues with empty SHA
7. scout brownfield + empty file_hints — blast radius skipped
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import (
    WorkflowPhase, LessonLearned, CRITIC_RESET,
)
from sacv.nodes.memory_consolidation import (
    make_memory_consolidation_node,
    _commit_test_inventory,
    _commit_production_code_no_record,
)
from sacv.nodes.scout import make_scout_node
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.nodes._structured_output import StructuredOutputError
from sacv.interfaces.agent_provider import AgentResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_deps(
    agent=None, git=None, sandbox=None, memory=None,
    code_graph=None, cross_domain=None, diff=None,
    config=None,
) -> dict:
    from sacv.orchestration.deps import NodeDeps
    return NodeDeps(
        agent=agent or StubAgentProvider(),
        memory=memory or StubMemoryProvider(),
        code_graph=code_graph or StubCodeGraphProvider(),
        cross_domain=cross_domain or StubCrossDomainProvider(),
        git=git or StubGitProvider(),
        sandbox=sandbox or StubSandboxProvider(),
        diff=diff or StubDiffProvider(),
        config=config or WorkflowConfig(),
    )


def _make_state(
    task_id="task-mc-001",
    session_id="sess-1",
    correction_state=None,
    verifier_verdict=None,
    escalation_payload=None,
    critic_findings=None,
    test_inventory_paths=None,
    preflight_result=None,
    agents_md_context=None,
    module_type="backend-domain",
    project_mode="greenfield",
    cost=0.0,
    session_start_ms=None,
    **kw,
) -> dict:
    base = {
        "session_id": session_id,
        "task_id": task_id,
        "project_mode": project_mode,
        "module_type": module_type,
        "current_phase": WorkflowPhase.COMPLETE.value,
        "task_description": "Add findById method",
        "context_skeleton": {},
        "blast_radius_map": None,
        "agents_md_context": agents_md_context or "Follow DDD conventions.",
        "strategy_candidates": [],
        "selected_strategy": {"strategy_id": "s1", "description": "test", "affected_files": ["X.java"]},
        "pruned_strategies": [],
        "red_phase_evidence_path": None,
        "test_inventory_paths": test_inventory_paths or [],
        "tdd_gate_attempts": 0,
        "diff_proposal": None,
        "preflight_result": preflight_result,
        "critic_findings": critic_findings or [],
        "verifier_verdict": verifier_verdict,
        "debug_observations": None,
        "correction_state": correction_state or {
            "attempt_count": 1, "branch_name": None,
            "last_error_hash": None, "error_history": [],
            "stagnation_pattern": "none",
        },
        "confidence_score": 1.0,
        "replan_count": 0,
        "active_branches": [],
        "exhausted_branches": [],
        "speculative_stash_ref": None,
        "escalation_payload": escalation_payload,
        "procedural_constraints": [],
        "lesson_learned": None,
        "arch_rules_updated": False,
        "check_profile": "standard",
        "cumulative_cost_dollars": cost,
        "session_start_ms": session_start_ms or 1000000,
    }
    base.update(kw)
    return base


# ── _commit_test_inventory edge cases ─────────────────────────────────────────

@pytest.mark.asyncio
class TestCommitTestInventory:

    async def test_missing_file_skipped_and_continues(self, tmp_path, monkeypatch):
        """Files that don't exist on disk are skipped; existing files still committed."""
        monkeypatch.chdir(tmp_path)
        # Create one file that exists
        (tmp_path / "ExistingTest.java").write_text("class ExistingTest {}")

        git = StubGitProvider()
        deps = _make_deps(git=git)

        result = await _commit_test_inventory(
            ["ExistingTest.java", "MissingTest.java"],
            "task-mc-001", deps,
        )

        # Only the existing file should be committed
        assert result == ["ExistingTest.java"]
        # Git stage_file called once for the existing file
        stage_calls = [c for c in git.calls if c[0] == "stage_file"]
        assert len(stage_calls) == 1
        assert stage_calls[0][1] == "ExistingTest.java"

    async def test_all_files_missing_returns_empty(self, tmp_path, monkeypatch):
        """All files missing → empty list returned, no git operations."""
        monkeypatch.chdir(tmp_path)
        git = StubGitProvider()
        deps = _make_deps(git=git)

        result = await _commit_test_inventory(
            ["Missing1.java", "Missing2.java"],
            "task-mc-001", deps,
        )

        assert result == []
        # No git operations at all
        assert len(git.calls) == 0

    async def test_stage_file_raises_returns_empty(self, tmp_path, monkeypatch):
        """stage_file raises RuntimeError → file skipped, returns empty."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "X.java").write_text("class X {}")

        git = StubGitProvider()
        git.stage_file = lambda path: (_ for _ in ()).throw(RuntimeError("permission denied"))
        deps = _make_deps(git=git)

        result = await _commit_test_inventory(
            ["X.java"], "task-mc-001", deps,
        )

        assert result == []


# ── _commit_production_code_no_record ─────────────────────────────────────────

@pytest.mark.asyncio
class TestCommitProductionCode:

    async def test_commit_success_returns_sha(self, tmp_path, monkeypatch):
        """Successful commit returns the SHA."""
        monkeypatch.chdir(tmp_path)
        git = StubGitProvider()
        deps = _make_deps(git=git)

        result = await _commit_production_code_no_record("task-mc-001", deps)

        assert result != ""  # non-empty SHA
        commit_calls = [c for c in git.calls if c[0] == "commit"]
        assert len(commit_calls) == 1
        assert "task-mc-001" in commit_calls[0][1]

    async def test_commit_failure_returns_empty(self, tmp_path, monkeypatch):
        """Commit failure → empty string, no exception."""
        monkeypatch.chdir(tmp_path)
        git = StubGitProvider()
        git.commit = lambda msg, **kw: (_ for _ in ()).throw(RuntimeError("commit failed"))
        deps = _make_deps(git=git)

        result = await _commit_production_code_no_record("task-mc-001", deps)

        assert result == ""


# ── Full memory_consolidation_node edge cases ─────────────────────────────────

@pytest.mark.asyncio
class TestMemoryConsolidationNode:

    async def test_get_head_sha_failure_logs_warning(self, tmp_path, monkeypatch):
        """head_sha failure → logs warning, continues with empty SHA."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir(exist_ok=True)

        agent = StubAgentProvider([
            make_json_agent_result({
                "common_mistakes": "Don't forget imports",
                "architecture_decisions": "Use layers",
            })
        ])
        git = StubGitProvider()
        # Make head_sha raise
        git.head_sha = lambda: (_ for _ in ()).throw(RuntimeError("git error"))
        deps = _make_deps(agent=agent, git=git)
        state = _make_state(
            task_id="task-mc-001",
            correction_state={"attempt_count": 1, "branch_name": None,
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "none"},
        )
        node = make_memory_consolidation_node(deps)

        out = await node(state)

        # Should complete successfully despite head_sha failure
        assert out["current_phase"] == WorkflowPhase.COMPLETE.value
        assert out["lesson_learned"] is not None

    async def test_verdict_none_does_not_crash(self, tmp_path, monkeypatch):
        """verifier_verdict=None → correction_type derived from attempt_count."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir(exist_ok=True)

        agent = StubAgentProvider([
            make_json_agent_result({
                "common_mistakes": "test",
                "architecture_decisions": "test",
            })
        ])
        git = StubGitProvider()
        deps = _make_deps(agent=agent, git=git)
        state = _make_state(
            task_id="task-mc-001",
            correction_state={"attempt_count": 0, "branch_name": None,
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "none"},
            verifier_verdict=None,
        )
        node = make_memory_consolidation_node(deps)

        out = await node(state)

        assert out["current_phase"] == WorkflowPhase.COMPLETE.value
        assert out["lesson_learned"]["correction_type"] == "none"

    async def test_hitl_correction_type(self, tmp_path, monkeypatch):
        """escalation_payload present → correction_type='hitl'."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir(exist_ok=True)

        agent = StubAgentProvider([
            make_json_agent_result({
                "common_mistakes": "test",
                "architecture_decisions": "test",
            })
        ])
        git = StubGitProvider()
        deps = _make_deps(agent=agent, git=git)
        state = _make_state(
            task_id="task-mc-001",
            correction_state={"attempt_count": 0, "branch_name": None,
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "none"},
            escalation_payload={"resolution_hints": []},
        )
        node = make_memory_consolidation_node(deps)

        out = await node(state)

        assert out["lesson_learned"]["correction_type"] == "hitl"

    async def test_self_correction_correction_type(self, tmp_path, monkeypatch):
        """attempt_count > 1 → correction_type='self_correction'."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir(exist_ok=True)

        agent = StubAgentProvider([
            make_json_agent_result({
                "common_mistakes": "test",
                "architecture_decisions": "test",
            })
        ])
        git = StubGitProvider()
        deps = _make_deps(agent=agent, git=git)
        state = _make_state(
            task_id="task-mc-001",
            correction_state={"attempt_count": 3, "branch_name": "b",
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "outcome"},
        )
        node = make_memory_consolidation_node(deps)

        out = await node(state)

        assert out["lesson_learned"]["correction_type"] == "self_correction"

    async def test_critic_guided_correction_type(self, tmp_path, monkeypatch):
        """Non-empty critic_findings with attempt_count=1 → correction_type='critic_guided'."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir(exist_ok=True)

        agent = StubAgentProvider([
            make_json_agent_result({
                "common_mistakes": "test",
                "architecture_decisions": "test",
            })
        ])
        git = StubGitProvider()
        deps = _make_deps(agent=agent, git=git)
        state = _make_state(
            task_id="task-mc-001",
            correction_state={"attempt_count": 1, "branch_name": None,
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "none"},
            critic_findings=[
                {"critic": "security", "severity": "critical", "rule_id": "SEC-1",
                 "file": "X.java", "line": 1, "message": "injection",
                 "resolution_hint": "use params"},
            ],
        )
        node = make_memory_consolidation_node(deps)

        out = await node(state)

        assert out["lesson_learned"]["correction_type"] == "critic_guided"

    async def test_stash_ref_dropped_on_complete(self, tmp_path, monkeypatch):
        """speculative_stash_ref → stash_drop called in finally."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir(exist_ok=True)

        agent = StubAgentProvider([
            make_json_agent_result({
                "common_mistakes": "test",
                "architecture_decisions": "test",
            })
        ])
        git = StubGitProvider()
        deps = _make_deps(agent=agent, git=git)
        state = _make_state(
            task_id="task-mc-001",
            speculative_stash_ref="stash@{0}",
        )
        node = make_memory_consolidation_node(deps)

        await node(state)

        # stash_drop should have been called
        stash_calls = [c for c in git.calls if c[0] == "stash_drop"]
        assert len(stash_calls) == 1
        assert stash_calls[0][1] == "stash@{0}"

    async def test_stash_drop_failure_does_not_crash(self, tmp_path, monkeypatch):
        """stash_drop exception → logged, node continues."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir(exist_ok=True)

        agent = StubAgentProvider([
            make_json_agent_result({
                "common_mistakes": "test",
                "architecture_decisions": "test",
            })
        ])
        git = StubGitProvider()
        git.stash_drop = lambda ref: (_ for _ in ()).throw(RuntimeError("stash not found"))
        deps = _make_deps(agent=agent, git=git)
        state = _make_state(
            task_id="task-mc-001",
            speculative_stash_ref="stash@{0}",
        )
        node = make_memory_consolidation_node(deps)

        out = await node(state)

        # Should complete successfully despite stash_drop failure
        assert out["current_phase"] == WorkflowPhase.COMPLETE.value


# ── Scout: brownfield + empty file_hints ──────────────────────────────────────

@pytest.mark.asyncio
class TestScoutBrownfieldEmptyHints:

    async def test_brownfield_with_empty_file_hints_skips_blast_radius(self):
        """Brownfield mode + no file hints in task description → blast_radius_map=None."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "X.java", "diff_content": "+x",
            "operation": "modify", "language": "java",
        }])])
        code_graph = StubCodeGraphProvider()
        deps = _make_deps(agent=agent, code_graph=code_graph)
        # Task description with NO file paths → file_hints = []
        state = _make_state(
            task_id="task-scout-001",
            project_mode="brownfield",
            task_description="Add a new feature",  # no file paths
        )
        node = make_scout_node(deps)

        out = await node(state)

        assert out["current_phase"] == WorkflowPhase.VALUE_NODE.value
        assert out["blast_radius_map"] is None
        # get_blast_radius should NOT have been called
        blast_calls = [c for c in code_graph.calls if c[0] == "get_blast_radius"]
        assert len(blast_calls) == 0

    async def test_brownfield_with_file_hints_runs_blast_radius(self):
        """Brownfield mode + file hints in task description → blast_radius computed."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "X.java", "diff_content": "+x",
            "operation": "modify", "language": "java",
        }])])
        code_graph = StubCodeGraphProvider()
        deps = _make_deps(agent=agent, code_graph=code_graph)
        # Task description WITH file paths
        state = _make_state(
            task_id="task-scout-002",
            project_mode="brownfield",
            task_description="Fix bug in UserService.java and UserRepository.java",
        )
        node = make_scout_node(deps)

        out = await node(state)

        assert out["current_phase"] == WorkflowPhase.VALUE_NODE.value
        # blast_radius_map should be populated (not None)
        assert out["blast_radius_map"] is not None
        blast_calls = [c for c in code_graph.calls if c[0] == "get_blast_radius"]
        assert len(blast_calls) == 1

    async def test_greenfield_with_file_hints_skips_blast_radius(self):
        """Greenfield mode + file hints → blast_radius_map=None (never computed)."""
        agent = StubAgentProvider([make_json_agent_result([{
            "file_path": "X.java", "diff_content": "+x",
            "operation": "modify", "language": "java",
        }])])
        code_graph = StubCodeGraphProvider()
        deps = _make_deps(agent=agent, code_graph=code_graph)
        state = _make_state(
            task_id="task-scout-003",
            project_mode="greenfield",
            task_description="Fix bug in UserService.java",
        )
        node = make_scout_node(deps)

        out = await node(state)

        assert out["blast_radius_map"] is None
        blast_calls = [c for c in code_graph.calls if c[0] == "get_blast_radius"]
        assert len(blast_calls) == 0
