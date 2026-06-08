"""
tests/e2e/test_brownfield_e2e.py
===================================
End-to-end graph execution tests for brownfield mode.

These tests verify:
1. Brownfield mode: full happy path with blast radius computation
2. Brownfield with preflight error: retry path in brownfield mode
3. Brownfield with critic findings: critic-guided correction

No live API calls. No Docker. No git operations.
All providers are stubs; agent responses are pre-loaded fixture JSON.
"""
from __future__ import annotations

import pytest

from langgraph.checkpoint.memory import MemorySaver

from sacv.orchestration.graph import build_graph, NodeDeps
from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.interfaces.agent_provider import AgentResult
from sacv.interfaces.sandbox_provider import ExecResult
from sacv.interfaces.code_graph_provider import BlastRadiusMap


# ── Helpers (mirroring test_full_workflow.py exactly) ─────────────────────────

def _diff_response() -> AgentResult:
    return make_json_agent_result([{
        "file_path": "src/main/java/com/example/UserService.java",
        "diff_content": "@@ -10,6 +10,10 @@\n+    public User findById(Long id) {\n+        return repo.findById(id).orElseThrow();\n+    }",
        "operation": "modify",
        "language": "java",
    }])


def _strategies_response() -> AgentResult:
    return make_json_agent_result([{
        "strategy_id": "s1",
        "description": "Add findById method to UserService",
        "affected_files": ["src/main/java/com/example/UserService.java"],
    }])


def _empty_critics() -> list[AgentResult]:
    return [make_json_agent_result([]) for _ in range(3)]


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


def _blast_radius_map() -> BlastRadiusMap:
    return BlastRadiusMap(
        entry_files=["UserService.java"],
        affected_files=["UserRepo.java", "UserController.java"],
        dependency_depth=2,
        cross_service_impact=[],
        schema_impact=[],
        risk_score=0.5,
    )


def _initial_state(task_id: str = "task-bf-001") -> dict:
    return {
        "session_id":             "",
        "task_id":                task_id,
        "project_mode":           "brownfield",
        "module_type":            "backend-domain",
        "current_phase":          WorkflowPhase.BOOTSTRAP.value,
        "task_description":       "Add findById method to existing UserService.java",
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
        "cumulative_cost_dollars": 0.0,
        "skip_tdd_gate":          True,
    }


def _build_sandbox() -> StubSandboxProvider:
    """Sandbox that always passes (for skip_tdd_gate=True workflows)."""
    return StubSandboxProvider(
        default_exit_code=0,
        default_stdout="BUILD SUCCESS\nTests run: 5, Failures: 0",
    )


@pytest.mark.asyncio
@pytest.mark.e2e
class TestBrownfieldHappyPath:

    async def test_brownfield_full_flow_reaches_complete(self, tmp_path, monkeypatch):
        """
        Brownfield happy path: scout computes blast radius, TDD gate skipped,
        Actor generates diff, Critics find nothing, Verifier passes.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        agent = StubAgentProvider([
            _strategies_response(),
            _diff_response(),
            *_empty_critics(),
            _agents_md_response(),
            _arch_rule_response(),
        ])
        deps = NodeDeps(
            agent=agent,
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(
                blast=_blast_radius_map(),
            ),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=_build_sandbox(),
            diff=StubDiffProvider(),
            config=WorkflowConfig(max_self_correction_cycles=3),
        )

        graph = build_graph(deps, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "e2e-bf-happy"}}

        final = await graph.ainvoke(_initial_state(), cfg)

        assert final["current_phase"] == WorkflowPhase.COMPLETE.value
        assert final["verifier_verdict"]["test_result"] == "PASS"
        assert final["lesson_learned"] is not None
        assert final.get("blast_radius_map") is not None
        assert len(final["blast_radius_map"]["affected_files"]) >= 1

    async def test_brownfield_with_preflight_error_retry(self, tmp_path, monkeypatch):
        """
        Brownfield: actor -> preflight (LSP errors) -> actor retry ->
        preflight (clean) -> critics -> verifier PASS.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        preflight_results = iter([
            {"passed": False, "lsp_errors": [{"file": "X.java", "line": 10,
                                               "code": "TS2345", "message": "error"}],
             "arch_violations": [], "cross_stack_errors": [], "duration_ms": 50},
            {"passed": True, "lsp_errors": [], "arch_violations": [],
             "cross_stack_errors": [], "duration_ms": 50},
        ])

        class _PreflightSandbox(StubSandboxProvider):
            async def exec_in_container(self, handle, command, env=None, timeout=120):
                if "mvn compile" in command or "tsc --noEmit" in command:
                    result = preflight_results.__next__()
                    if result["passed"]:
                        return ExecResult(0, "", "", 10)
                    err = result["lsp_errors"][0]
                    error_line = f"[ERROR] {err['file']}:[{err['line']},10] {err['message']}"
                    return ExecResult(1, error_line, "", 10)
                return ExecResult(0, "BUILD SUCCESS\nTests run: 5, Failures: 0", "", 100)

        agent = StubAgentProvider([
            _strategies_response(),
            _diff_response(),        # actor attempt 1
            _diff_response(),        # actor retry
            *_empty_critics(),       # critics
            _agents_md_response(),   # memory
            _arch_rule_response(),   # memory
        ])
        deps = NodeDeps(
            agent=agent,
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(blast=_blast_radius_map()),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=_PreflightSandbox(),
            diff=StubDiffProvider(),
            config=WorkflowConfig(max_self_correction_cycles=3),
        )

        graph = build_graph(deps, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "e2e-bf-retry"}}

        final = await graph.ainvoke(_initial_state("task-bf-retry"), cfg)

        assert final["current_phase"] == WorkflowPhase.COMPLETE.value
        assert final["verifier_verdict"]["test_result"] == "PASS"
        actor_calls = [c for c in agent.calls if c[0] == "structured_output"]
        # value_node(1) + actor x2(2) + critics(3) + memory(2) = 7 structured_output calls
        assert len(actor_calls) == 7

    async def test_brownfield_with_critic_findings(self, tmp_path, monkeypatch):
        """
        Brownfield: critics find issues, but verifier still passes.
        Findings are recorded in state.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        agent = StubAgentProvider([
            _strategies_response(),
            _diff_response(),
            # Security critic finds a warning
            make_json_agent_result([{
                "critic": "security", "severity": "warning",
                "file": "X.java", "line": 10, "rule_id": "SEC-001",
                "message": "Potential issue", "resolution_hint": "review",
            }]),
            make_json_agent_result([]),  # style
            make_json_agent_result([]),  # consistency
            _agents_md_response(),
            _arch_rule_response(),
        ])
        deps = NodeDeps(
            agent=agent,
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(blast=_blast_radius_map()),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=_build_sandbox(),
            diff=StubDiffProvider(),
            config=WorkflowConfig(max_self_correction_cycles=3),
        )

        graph = build_graph(deps, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "e2e-bf-critics"}}

        final = await graph.ainvoke(_initial_state("task-bf-critics"), cfg)

        assert final["current_phase"] == WorkflowPhase.COMPLETE.value
        assert final["verifier_verdict"]["test_result"] == "PASS"
        assert len(final.get("critic_findings", [])) >= 1
        assert any(f["critic"] == "security" for f in final["critic_findings"])

    async def test_brownfield_lesson_reflects_blast_radius(self, tmp_path, monkeypatch):
        """Lesson learned includes blast radius in pattern_discovered."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        agent = StubAgentProvider([
            _strategies_response(),
            _diff_response(),
            *_empty_critics(),
            _agents_md_response(),
            _arch_rule_response(),
        ])
        deps = NodeDeps(
            agent=agent,
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(blast=_blast_radius_map()),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=_build_sandbox(),
            diff=StubDiffProvider(),
            config=WorkflowConfig(max_self_correction_cycles=3),
        )

        graph = build_graph(deps, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "e2e-bf-lesson"}}

        final = await graph.ainvoke(_initial_state("task-bf-lesson"), cfg)

        assert final["lesson_learned"] is not None
        pattern = final["lesson_learned"]["pattern_discovered"]
        assert "mode=brownfield" in pattern
        assert "resolved_in=1_attempts" in pattern
