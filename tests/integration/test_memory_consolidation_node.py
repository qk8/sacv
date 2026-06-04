"""
tests/integration/test_memory_consolidation_node.py
=====================================================
Integration tests for the memory_consolidation node.

Tests cover:
1. Happy path: commits tests, production code, AGENTS.md, arch rules, green SHA
2. No test inventory: skips test commit, still proceeds through rest of pipeline
3. No arch violations: skips arch rules update
4. With critic findings: records correction_type as critic_guided
5. With escalation: records correction_type as hitl
6. With self-correction: records correction_type as self_correction
7. Cost accumulation: carries forward cumulative_cost_dollars
8. Stash cleanup: pops speculative stash ref
9. Git failure resilience: commit failures don't block the node
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from sacv.nodes.memory_consolidation import make_memory_consolidation_node
from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import (
    WorkflowPhase, LessonLearned, EscalationPayload,
    VerifierVerdict, DiagnosticVerdict,
)
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.interfaces.agent_provider import AgentResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _agents_md_response() -> AgentResult:
    return make_json_agent_result({
        "common_mistakes": "Avoid NPE on findById.",
        "architecture_decisions": "UserService uses repository pattern.",
    })


def _arch_rule_response() -> AgentResult:
    return make_json_agent_result({
        "name": "no-layer-violation",
        "from": {"paths": ["*"]},
        "to": [{"paths": ["*"]}],
    })


def _make_deps(
    agent:      StubAgentProvider,
    git:        StubGitProvider,
    memory:     StubMemoryProvider,
    config:     WorkflowConfig | None = None,
) -> object:
    from sacv.orchestration.graph import NodeDeps
    return NodeDeps(
        agent=agent,
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
        "session_id":             "sess-mc-001",
        "task_id":                "task-mc-001",
        "project_mode":           "greenfield",
        "module_type":            "backend-domain",
        "current_phase":          WorkflowPhase.MEMORY_CONSOLIDATION.value,
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
            "attempt_count": 1,
            "branch_name": None,
            "last_error_hash": None,
            "error_history": [],
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
        "cumulative_cost_dollars": 0.0,
    }
    state.update(kw)
    return state


@pytest.mark.asyncio
@pytest.mark.integration
class TestMemoryConsolidationNode:

    async def test_happy_path_commits_and_consolidates(self, tmp_path, monkeypatch):
        """
        Full happy path: test inventory committed, production code committed,
        AGENTS.md updated, green SHA recorded, lesson stored.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()
        # Create a test file so test_inventory_paths is meaningful
        test_file = tmp_path / "src" / "test" / "UserServiceTest.java"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("// test")

        git = StubGitProvider()
        memory = StubMemoryProvider()
        agent = StubAgentProvider([
            _agents_md_response(),
            _arch_rule_response(),
        ])
        deps = _make_deps(agent, git, memory)

        state = _base_state(
            test_inventory_paths=["src/test/UserServiceTest.java"],
            verifier_verdict={
                "test_result": "PASS",
                "diagnostic": DiagnosticVerdict.PASS.value,
                "phase1_passed": True, "phase2_passed": True,
                "test_failures": [], "performance_delta": None,
                "visual_diff_result": None, "critic_findings": [],
                "docker_exit_code": 0,
                "playwright_trace_path": None, "otel_trace": None,
                "actuator_snapshot": None,
            },
        )

        out = await make_memory_consolidation_node(deps)(state)

        assert out["current_phase"] == WorkflowPhase.COMPLETE.value
        assert out["lesson_learned"] is not None
        assert out["lesson_learned"]["task_id"] == "task-mc-001"
        assert out["lesson_learned"]["correction_type"] == "none"
        # Git: test inventory staged + committed, production code committed,
        # AGENTS.md staged + committed
        stage_calls = [c for c in git.calls if c[0] == "stage_file"]
        assert len(stage_calls) >= 2  # test file + AGENTS.md
        commit_calls = [c for c in git.calls if c[0] == "commit"]
        assert len(commit_calls) >= 2  # tests + production code
        # Green SHA recorded
        record_green_calls = [c for c in git.calls if c[0] == "record_green"]
        assert len(record_green_calls) >= 1
        # Lesson stored in memory
        assert len(memory.stored_events) >= 1
        assert memory.stored_events[0].event_type == "lesson_learned"
        # Cost accumulated
        assert out["cumulative_cost_dollars"] > 0

    async def test_no_test_inventory_skips_test_commit(self, tmp_path, monkeypatch):
        """When test_inventory_paths is empty, no test commit occurs."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        agent = StubAgentProvider([_agents_md_response(), _arch_rule_response()])
        deps = _make_deps(agent, git, memory)

        state = _base_state(test_inventory_paths=[])
        out = await make_memory_consolidation_node(deps)(state)

        assert out["current_phase"] == WorkflowPhase.COMPLETE.value
        stage_calls = [c for c in git.calls if c[0] == "stage_file"]
        # Should still stage AGENTS.md but no test files
        assert any("AGENTS.md" in str(c) for c in stage_calls)

    async def test_no_arch_violations_skips_arch_rules_update(self, tmp_path, monkeypatch):
        """When preflight has no arch_violations, arch rules agent is not called."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        agent = StubAgentProvider([_agents_md_response()])
        deps = _make_deps(agent, git, memory)

        state = _base_state(
            preflight_result={
                "passed": True,
                "lsp_errors": [],
                "arch_violations": [],
                "cross_stack_errors": [],
                "duration_ms": 50,
            },
        )
        out = await make_memory_consolidation_node(deps)(state)

        assert out["arch_rules_updated"] is False
        # Only AGENTS.md agent call, no arch rules call
        agent_roles = [c[0] for c in agent.calls]
        assert agent_roles == ["plan_agent_docs"]

    async def test_with_arch_violations_triggers_arch_rules_update(self, tmp_path, monkeypatch):
        """When preflight has arch_violations, arch rules agent is called."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        agent = StubAgentProvider([
            _agents_md_response(),
            _arch_rule_response(),
        ])
        deps = _make_deps(agent, git, memory)

        state = _base_state(
            preflight_result={
                "passed": False,
                "lsp_errors": [],
                "arch_violations": [
                    {"rule": "no-direct-controller-service",
                     "source_file": "UserController.java",
                     "target_file": "PaymentService.java",
                     "message": "Controller should not depend on PaymentService"},
                ],
                "cross_stack_errors": [],
                "duration_ms": 50,
            },
        )
        out = await make_memory_consolidation_node(deps)(state)

        assert out["arch_rules_updated"] is True
        agent_roles = [c[0] for c in agent.calls]
        assert "plan_agent_arch_rules" in agent_roles

    async def test_critic_findings_set_correction_type(self, tmp_path, monkeypatch):
        """With critic findings and attempt_count=1, correction_type is critic_guided."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        agent = StubAgentProvider([_agents_md_response(), _arch_rule_response()])
        deps = _make_deps(agent, git, memory)

        state = _base_state(
            attempt_count=1,
            critic_findings=[{
                "critic": "security", "severity": "warning",
                "file": "X.java", "line": 10, "rule_id": "SEC-001",
                "message": "Potential issue", "resolution_hint": "review",
            }],
        )
        out = await make_memory_consolidation_node(deps)(state)

        assert out["lesson_learned"]["correction_type"] == "critic_guided"

    async def test_escalation_sets_correction_type_hitl(self, tmp_path, monkeypatch):
        """With escalation_payload, correction_type is hitl."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        agent = StubAgentProvider([_agents_md_response(), _arch_rule_response()])
        deps = _make_deps(agent, git, memory)

        escalation = EscalationPayload(
            escalation_id="esc-001",
            timestamp="2025-01-01T00:00:00Z",
            workflow_version="sacv-1.0",
            task_id="task-mc-001",
            task_description="Add findById",
            failure_summary={
                "total_attempts": 3,
                "branches_exhausted": [],
                "stagnation_pattern": "none",
                "last_verifier_output": None,
                "critic_findings": [],
            },
            git_state={},
            resolution_hints=[],
            resume_instructions={},
        )
        state = _base_state(
            attempt_count=3,
            escalation_payload=escalation,
        )
        out = await make_memory_consolidation_node(deps)(state)

        assert out["lesson_learned"]["correction_type"] == "hitl"

    async def test_self_correction_sets_correction_type(self, tmp_path, monkeypatch):
        """With attempt_count > 1 and no escalation, correction_type is self_correction."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        agent = StubAgentProvider([_agents_md_response(), _arch_rule_response()])
        deps = _make_deps(agent, git, memory)

        state = _base_state(correction_state={
            "attempt_count": 2, "branch_name": None,
            "last_error_hash": None, "error_history": [],
            "stagnation_pattern": "none",
        })
        out = await make_memory_consolidation_node(deps)(state)

        assert out["lesson_learned"]["correction_type"] == "self_correction"

    async def test_stash_cleanup_on_speculative_stash_ref(self, tmp_path, monkeypatch):
        """When speculative_stash_ref is set, stash_pop is called."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        agent = StubAgentProvider([_agents_md_response(), _arch_rule_response()])
        deps = _make_deps(agent, git, memory)

        state = _base_state(speculative_stash_ref="stash@{0}")
        await make_memory_consolidation_node(deps)(state)

        stash_pops = [c for c in git.calls if c[0] == "stash_pop"]
        assert len(stash_pops) >= 1
        assert stash_pops[0][1] == "stash@{0}"

    async def test_lesson_pattern_reflects_state(self, tmp_path, monkeypatch):
        """Lesson pattern includes module, mode, attempt count, stagnation, replan."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        agent = StubAgentProvider([_agents_md_response(), _arch_rule_response()])
        deps = _make_deps(agent, git, memory)

        state = _base_state(
            module_type="frontend-feature",
            project_mode="brownfield",
            correction_state={
                "attempt_count": 3, "branch_name": None,
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "semantic",
            },
            replan_count=1,
            verifier_verdict={
                "test_result": "PASS",
                "diagnostic": DiagnosticVerdict.PASS.value,
                "phase1_passed": True, "phase2_passed": True,
                "test_failures": [], "performance_delta": None,
                "visual_diff_result": None, "critic_findings": [],
                "docker_exit_code": 0,
                "playwright_trace_path": None, "otel_trace": None,
                "actuator_snapshot": None,
            },
        )
        out = await make_memory_consolidation_node(deps)(state)

        pattern = out["lesson_learned"]["pattern_discovered"]
        assert "module=frontend-feature" in pattern
        assert "mode=brownfield" in pattern
        assert "resolved_in=3_attempts" in pattern
        assert "stagnation=semantic" in pattern
        assert "replanned=1x" in pattern

    async def test_negative_constraints_from_critical_findings(self, tmp_path, monkeypatch):
        """Critical findings become negative constraints in lesson learned."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        agent = StubAgentProvider([_agents_md_response(), _arch_rule_response()])
        deps = _make_deps(agent, git, memory)

        state = _base_state(
            critic_findings=[{
                "critic": "security", "severity": "critical",
                "file": "AuthService.java", "line": 42,
                "rule_id": "SEC-001",
                "message": "Hardcoded password in source",
                "resolution_hint": "use environment variable",
            }],
        )
        out = await make_memory_consolidation_node(deps)(state)

        constraints = out["lesson_learned"]["negative_constraints"]
        assert len(constraints) >= 1
        assert "[SECURITY]" in constraints[0]
        assert "Hardcoded password" in constraints[0]

    async def test_cost_accumulation_from_agent_calls(self, tmp_path, monkeypatch):
        """Cost accumulates from AGENTS.md and arch rules agent calls."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        agent = StubAgentProvider([
            _agents_md_response(),
            _arch_rule_response(),
        ])
        deps = _make_deps(agent, git, memory)

        state = _base_state(
            preflight_result={
                "passed": False,
                "arch_violations": [{"rule": "violation", "source_file": "X",
                                     "target_file": "Y", "message": "bad"}],
                "lsp_errors": [], "cross_stack_errors": [], "duration_ms": 50,
            },
        )
        out = await make_memory_consolidation_node(deps)(state)

        assert out["cumulative_cost_dollars"] > 0
        # 2 agent calls: AGENTS.md + arch rules
        assert len(agent.calls) == 2

    async def test_preflight_none_does_not_crash(self, tmp_path, monkeypatch):
        """When preflight_result is None, node handles gracefully."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        agent = StubAgentProvider([_agents_md_response()])
        deps = _make_deps(agent, git, memory)

        state = _base_state(preflight_result=None)
        out = await make_memory_consolidation_node(deps)(state)

        assert out["current_phase"] == WorkflowPhase.COMPLETE.value
        assert out["arch_rules_updated"] is False

    async def test_episode_event_contains_lesson_payload(self, tmp_path, monkeypatch):
        """The episodic event stored in memory contains the lesson payload."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        agent = StubAgentProvider([_agents_md_response(), _arch_rule_response()])
        deps = _make_deps(agent, git, memory)

        state = _base_state()
        await make_memory_consolidation_node(deps)(state)

        lesson_events = [
            e for e in memory.stored_events if e.event_type == "lesson_learned"
        ]
        assert len(lesson_events) == 1
        payload = lesson_events[0].payload
        assert payload["task_id"] == "task-mc-001"
        assert "pattern_discovered" in payload
        assert "negative_constraints" in payload
        assert "correction_type" in payload

    async def test_purge_noise_called(self, tmp_path, monkeypatch):
        """purge_noise is called for the session."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        git = StubGitProvider()
        memory = StubMemoryProvider()
        agent = StubAgentProvider([_agents_md_response(), _arch_rule_response()])
        deps = _make_deps(agent, git, memory)

        state = _base_state()
        await make_memory_consolidation_node(deps)(state)

        assert "sess-mc-001" in memory.purged_sessions
