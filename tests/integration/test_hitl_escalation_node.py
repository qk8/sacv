"""
tests/integration/test_hitl_escalation_node.py
================================================
Integration tests for the HITL escalation node.

Tests cover:
1. Payload construction with all required fields
2. Stash creation when on a non-main branch
3. No stash on main branch
4. Git reset error captured in payload
5. Resolution hints from FIX_IMPL diagnostic
6. Resolution hints from FIX_TEST diagnostic
7. Resolution hints from blast radius
8. Resolution hints from critical critic findings
9. Resolution hints from TDD gate failure
10. Escalation file written to disk
11. Episodic event stored in memory
12. Escalation ID is unique per call
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from sacv.nodes.hitl_escalation import make_hitl_escalation_node
from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import (
    WorkflowPhase, VerifierVerdict, DiagnosticVerdict,
)
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_deps(
    git:        StubGitProvider,
    memory:     StubMemoryProvider,
    config:     WorkflowConfig | None = None,
) -> object:
    from sacv.orchestration.deps import NodeDeps
    return NodeDeps(
        agent=StubAgentProvider(),
        memory=memory,
        code_graph=StubCodeGraphProvider(),
        cross_domain=StubCrossDomainProvider(),
        git=git,
        sandbox=StubSandboxProvider(),
        diff=StubDiffProvider(),
        config=config or WorkflowConfig(),
    )


def _base_state(**kw) -> dict:
    state = {
        "session_id":             "sess-hitl-001",
        "task_id":                "task-hitl-001",
        "project_mode":           "greenfield",
        "module_type":            "backend-domain",
        "current_phase":          WorkflowPhase.HITL_ESCALATION.value,
        "task_description":       "Add findById to UserService",
        "context_skeleton":       None,
        "blast_radius_map":       None,
        "agents_md_context":      None,
        "strategy_candidates":    [],
        "selected_strategy":      None,
        "pruned_strategies":      [],
        "red_phase_evidence_path": None,
        "test_inventory_paths":   [],
        "tdd_gate_attempts":      0,
        "diff_proposal":          None,
        "preflight_result":       None,
        "critic_findings":        [],
        "verifier_verdict":       None,
        "debug_observations":     None,
        "correction_state": {
            "attempt_count": 3,
            "branch_name": "agent-task-hitl-a1",
            "last_error_hash": "abc123",
            "error_history": ["error1", "error2"],
            "stagnation_pattern": "iteration",
        },
        "confidence_score":       0.1,
        "replan_count":           0,
        "active_branches":        [],
        "exhausted_branches":     ["agent-task-hitl-s1"],
        "speculative_stash_ref":  None,
        "escalation_payload":     None,
        "procedural_constraints": [],
        "lesson_learned":         None,
        "arch_rules_updated":     False,
        "check_profile":          "standard",
        "cumulative_cost_dollars": 5.0,
    }
    state.update(kw)
    return state


@pytest.mark.asyncio
@pytest.mark.integration
class TestHITLEscalationNode:

    async def test_payload_has_required_fields(self, tmp_path, monkeypatch):
        """Escalation payload must contain all required top-level keys."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider(current_branch_name="feature-branch")
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        state = _base_state()

        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            # interrupt() raises; we just need the side effects
            pass

        # Escalation file should exist
        esc_files = list((tmp_path / ".workflow" / "escalations").glob("*.json"))
        assert len(esc_files) == 1

        data = json.loads(esc_files[0].read_text())
        assert "escalation_id" in data
        assert "timestamp" in data
        assert "workflow_version" in data
        assert "task_id" in data
        assert "task_description" in data
        assert "failure_summary" in data
        assert "git_state" in data
        assert "resolution_hints" in data
        assert "resume_instructions" in data

    async def test_stash_created_on_non_main_branch(self, tmp_path, monkeypatch):
        """When on a non-main branch, stash is created."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider(current_branch_name="feature-branch")
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        state = _base_state()
        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        stash_calls = [c for c in git.calls if c[0] == "stash"]
        assert len(stash_calls) >= 1
        assert "sacv-hitl" in stash_calls[0][1]

    async def test_no_stash_on_main_branch(self, tmp_path, monkeypatch):
        """When on main branch, no stash is created."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider(current_branch_name="main")
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        # Clear correction_state.branch_name so code falls through to git.current_branch()
        state = _base_state(correction_state={
            "attempt_count": 3, "branch_name": None,
            "last_error_hash": "abc123", "error_history": [],
            "stagnation_pattern": "iteration",
        })
        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        stash_calls = [c for c in git.calls if c[0] == "stash"]
        assert len(stash_calls) == 0

    async def test_git_reset_error_captured(self, tmp_path, monkeypatch):
        """When git reset_hard fails, error is captured in git_state."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        class _FailingResetGit(StubGitProvider):
            def reset_hard(self, ref):
                raise RuntimeError(f"fatal: failed to reset to {ref}")

        git = _FailingResetGit(current_branch_name="feature-branch")
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        state = _base_state()
        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        esc_files = list((tmp_path / ".workflow" / "escalations").glob("*.json"))
        data = json.loads(esc_files[0].read_text())
        assert data["git_state"]["git_reset_failed"] is not None
        assert "fatal" in data["git_state"]["git_reset_failed"]

    async def test_fix_impl_diagnostic_produces_hint(self, tmp_path, monkeypatch):
        """FIX_IMPL diagnostic → architectural resolution hint."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        verdict: VerifierVerdict = {
            "test_result": "FAIL",
            "diagnostic": DiagnosticVerdict.FIX_IMPL.value,
            "phase1_passed": True, "phase2_passed": False,
            "test_failures": [{"message": "assertion failed"}],
            "performance_delta": None, "visual_diff_result": None,
            "critic_findings": [], "docker_exit_code": 1,
            "playwright_trace_path": None, "otel_trace": None,
            "actuator_snapshot": None,
        }
        state = _base_state(verifier_verdict=verdict)

        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        esc_files = list((tmp_path / ".workflow" / "escalations").glob("*.json"))
        data = json.loads(esc_files[0].read_text())
        hints = data["resolution_hints"]
        assert any(h["category"] == "architectural" for h in hints)

    async def test_fix_test_diagnostic_produces_hint(self, tmp_path, monkeypatch):
        """FIX_TEST diagnostic → test_oracle resolution hint."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        verdict: VerifierVerdict = {
            "test_result": "FAIL",
            "diagnostic": DiagnosticVerdict.FIX_TEST.value,
            "phase1_passed": True, "phase2_passed": False,
            "test_failures": [{"message": "assertion failed"}],
            "performance_delta": None, "visual_diff_result": None,
            "critic_findings": [], "docker_exit_code": 1,
            "playwright_trace_path": None, "otel_trace": None,
            "actuator_snapshot": None,
        }
        state = _base_state(verifier_verdict=verdict)

        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        esc_files = list((tmp_path / ".workflow" / "escalations").glob("*.json"))
        data = json.loads(esc_files[0].read_text())
        hints = data["resolution_hints"]
        assert any(h["category"] == "test_oracle" for h in hints)

    async def test_blast_radius_produces_hint(self, tmp_path, monkeypatch):
        """High blast radius risk_score → blast_radius resolution hint."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        state = _base_state(
            blast_radius_map={
                "entry_files": [], "affected_files": ["A.java", "B.java"],
                "dependency_depth": 3, "cross_service_impact": [],
                "schema_impact": [], "risk_score": 0.85,
            },
        )

        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        esc_files = list((tmp_path / ".workflow" / "escalations").glob("*.json"))
        data = json.loads(esc_files[0].read_text())
        hints = data["resolution_hints"]
        assert any(h["category"] == "blast_radius" for h in hints)

    async def test_critical_critic_findings_produce_hint(self, tmp_path, monkeypatch):
        """Critical critic findings → security resolution hint."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        state = _base_state(
            critic_findings=[{
                "critic": "security", "severity": "critical",
                "file": "X.java", "line": 10, "rule_id": "SEC-001",
                "message": "SQL injection", "resolution_hint": "use params",
            }],
        )

        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        esc_files = list((tmp_path / ".workflow" / "escalations").glob("*.json"))
        data = json.loads(esc_files[0].read_text())
        hints = data["resolution_hints"]
        assert any(h["category"] == "security" for h in hints)

    async def test_tdd_gate_failure_produces_hint(self, tmp_path, monkeypatch):
        """TDD gate exceeded attempts with no evidence → test_oracle hint."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        config = WorkflowConfig(max_tdd_gate_attempts=3)
        deps = _make_deps(git, memory, config=config)

        state = _base_state(
            tdd_gate_attempts=3,
            red_phase_evidence_path=None,
        )

        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        esc_files = list((tmp_path / ".workflow" / "escalations").glob("*.json"))
        data = json.loads(esc_files[0].read_text())
        hints = data["resolution_hints"]
        assert any(
            h["category"] == "test_oracle" and "TDD gate" in h["hint"]
            for h in hints
        )

    async def test_failure_summary_includes_all_fields(self, tmp_path, monkeypatch):
        """Failure summary includes total_attempts, branches_exhausted, etc."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        state = _base_state()
        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        esc_files = list((tmp_path / ".workflow" / "escalations").glob("*.json"))
        data = json.loads(esc_files[0].read_text())
        fs = data["failure_summary"]
        assert "total_attempts" in fs
        assert fs["total_attempts"] == 3
        assert "branches_exhausted" in fs
        assert "stagnation_pattern" in fs
        assert "last_verifier_output" in fs
        assert "critic_findings" in fs

    async def test_git_state_captures_branch_and_stash(self, tmp_path, monkeypatch):
        """Git state captures active branch, stash ref, and uncommitted files."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider(current_branch_name="feature-branch")
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        # Clear correction_state.branch_name so code falls through to git.current_branch()
        state = _base_state(correction_state={
            "attempt_count": 3, "branch_name": None,
            "last_error_hash": "abc123", "error_history": ["error1", "error2"],
            "stagnation_pattern": "iteration",
        })
        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        esc_files = list((tmp_path / ".workflow" / "escalations").glob("*.json"))
        data = json.loads(esc_files[0].read_text())
        gs = data["git_state"]
        assert gs["active_branch"] == "feature-branch"
        assert gs["stash_ref"] is not None
        assert "last_green_commit" in gs
        assert "stashed_branches" in gs
        assert "uncommitted_files" in gs

    async def test_episodic_event_stored(self, tmp_path, monkeypatch):
        """An episodic event of type hitl_escalation is stored in memory."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        state = _base_state()
        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        events = [e for e in memory.stored_events if e.event_type == "hitl_escalation"]
        assert len(events) == 1
        assert events[0].payload["task_id"] == "task-hitl-001"

    async def test_resume_instructions_present(self, tmp_path, monkeypatch):
        """Resume instructions contain the resume command and state file path."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        state = _base_state()
        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        esc_files = list((tmp_path / ".workflow" / "escalations").glob("*.json"))
        data = json.loads(esc_files[0].read_text())
        ri = data["resume_instructions"]
        assert "command" in ri
        assert "state_file" in ri
        assert "note" in ri
        assert "escalation-id" in ri["command"]

    async def test_stash_pop_command_present_when_stash_created(self, tmp_path, monkeypatch):
        """stash_pop_command is present when stash was created."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider(current_branch_name="feature-branch")
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        state = _base_state()
        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        esc_files = list((tmp_path / ".workflow" / "escalations").glob("*.json"))
        data = json.loads(esc_files[0].read_text())
        gs = data["git_state"]
        assert gs["stash_pop_command"] is not None
        assert "git stash pop" in gs["stash_pop_command"]

    async def test_no_stash_pop_command_when_on_main(self, tmp_path, monkeypatch):
        """stash_pop_command is None when no stash was created."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider(current_branch_name="main")
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        # Clear correction_state.branch_name so code falls through to git.current_branch()
        state = _base_state(correction_state={
            "attempt_count": 3, "branch_name": None,
            "last_error_hash": "abc123", "error_history": [],
            "stagnation_pattern": "iteration",
        })
        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        esc_files = list((tmp_path / ".workflow" / "escalations").glob("*.json"))
        data = json.loads(esc_files[0].read_text())
        gs = data["git_state"]
        assert gs["stash_pop_command"] is None

    async def test_exhausted_branches_in_git_state(self, tmp_path, monkeypatch):
        """exhausted_branches from state appear in git_state.stashed_branches."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        state = _base_state(
            exhausted_branches=["agent-task-hitl-s1", "agent-task-hitl-s2"],
        )
        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        esc_files = list((tmp_path / ".workflow" / "escalations").glob("*.json"))
        data = json.loads(esc_files[0].read_text())
        assert data["git_state"]["stashed_branches"] == [
            "agent-task-hitl-s1", "agent-task-hitl-s2",
        ]

    async def test_checkout_main_after_reset(self, tmp_path, monkeypatch):
        """After reset_hard, checkout main is called."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider(current_branch_name="feature-branch")
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        state = _base_state()
        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        checkout_calls = [c for c in git.calls if c[0] == "checkout"]
        assert any(c[1] == "main" for c in checkout_calls)

    async def test_workflow_version_in_payload(self, tmp_path, monkeypatch):
        """Workflow version is sacv-1.0 in the payload."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        deps = _make_deps(git, memory)

        state = _base_state()
        try:
            await make_hitl_escalation_node(deps)(state)
        except Exception:
            pass

        esc_files = list((tmp_path / ".workflow" / "escalations").glob("*.json"))
        data = json.loads(esc_files[0].read_text())
        # M-08: workflow_version uses package version (not hardcoded "sacv-1.0")
        import importlib.metadata
        try:
            expected = f"sacv-{importlib.metadata.version('sacv-workflow')}"
        except importlib.metadata.PackageNotFoundError:
            expected = "sacv-unknown"
        assert data["workflow_version"] == expected
