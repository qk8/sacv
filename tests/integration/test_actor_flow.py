"""
tests/integration/test_actor_flow.py
======================================
Integration tests for the actor -> preflight -> critics -> verifier flow.

Tests cover:
1. Full actor -> preflight (clean) -> critics (no findings) -> verifier (PASS)
2. Actor -> preflight (errors) -> actor retry -> preflight (clean) -> critics -> verifier (PASS)
3. Actor -> preflight (clean) -> critics (findings) -> verifier (PASS with findings recorded)
4. Actor produces no diff -> self-loop -> actor produces diff -> verifier (PASS)
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from langgraph.checkpoint.memory import MemorySaver

from sacv.orchestration.graph import build_graph, NodeDeps
from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase, DiffProposal, UnifiedDiffPayload
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.interfaces.agent_provider import AgentResult
from sacv.interfaces.sandbox_provider import ExecResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _diff_response():
    return make_json_agent_result([{
        "file_path": "src/main/java/com/example/UserService.java",
        "diff_content": "@@ -10,6 +10,10 @@\n+    public User findById(Long id) {\n+        return repo.findById(id).orElseThrow();\n+    }",
        "operation": "modify",
        "language": "java",
    }])


def _empty_critics() -> list[AgentResult]:
    return [make_json_agent_result([]) for _ in range(3)]


def _agents_md_response() -> AgentResult:
    return make_json_agent_result({
        "common_mistakes": "Added findById null-safety pattern.",
        "architecture_decisions": "UserService uses repository pattern.",
    })


def _arch_rule_response() -> AgentResult:
    return make_json_agent_result({
        "name": "no-layer-violation",
        "from": {"paths": ["*"]},
        "to": [{"paths": ["*"]}],
    })


def _initial_state(task_id: str = "task-af-001") -> dict:
    return {
        "session_id":             "",
        "task_id":                task_id,
        "project_mode":           "greenfield",
        "module_type":            "backend-domain",
        "current_phase":          WorkflowPhase.BOOTSTRAP.value,
        "task_description":       "Add findById method to UserService",
        "context_skeleton":       {"call_graph": {"entry": ".", "nodes": [], "edges": []},
                                   "dependencies": {}, "schema_map": {}, "arch_align": {}},
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


def _passing_sandbox() -> StubSandboxProvider:
    s = StubSandboxProvider(
        default_exit_code=0,
        default_stdout="BUILD SUCCESS\nTests run: 5, Failures: 0",
    )
    return s


# ── Full stub chains for tests that start from BOOTSTRAP ─────────────────────
# LangGraph always runs from START → bootstrap → mode_router → scout → value_node
# → tdd_gate → actor → ...  We must provide stubs for every node that calls the agent.


def _actor_chain(diff_response, *extra_responses) -> list[AgentResult]:
    """Chain: value_node + actor + critics + memory (after skipping tdd_gate).

    skip_tdd_gate=True → tdd_gate returns immediately (no agent call)
    value_node → generate strategy (1 agent call)
    actor → produces diff (1 agent call)
    critics → empty (3 agent calls)
    memory_consolidation → 2 agent calls (AGENTS.md + arch rules)
    """
    return [
        # value_node: generate strategy
        make_json_agent_result([{
            "strategy_id": "s1", "description": "Add findById",
            "affected_files": ["UserService.java"],
        }]),
        # actor: produce diff
        diff_response,
        # critics: empty
        *_empty_critics(),
        # memory_consolidation
        _agents_md_response(),
        _arch_rule_response(),
        *extra_responses,
    ]


def _bootstrap_chain_preflight_error(diff_response, *extra_responses) -> list[AgentResult]:
    """Chain with preflight error → actor retry → clean preflight."""
    return [
        # value_node: generate strategy
        make_json_agent_result([{
            "strategy_id": "s1", "description": "Add findById",
            "affected_files": ["UserService.java"],
        }]),
        # actor (first attempt): produce diff
        diff_response,
        # actor (retry after preflight error): produce diff
        diff_response,
        # critics: empty
        *_empty_critics(),
        # memory_consolidation
        _agents_md_response(),
        _arch_rule_response(),
        *extra_responses,
    ]


@pytest.mark.asyncio
@pytest.mark.integration
class TestActorFlow:

    async def test_actor_to_preflight_to_critics_to_verifier_pass(self, tmp_path, monkeypatch):
        """
        Full flow from START: bootstrap → scout → value_node → tdd_gate
        → actor produces diff → preflight clean → critics empty → verifier PASS.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        chain = _actor_chain(_diff_response())
        agent = StubAgentProvider(chain)
        sandbox = _passing_sandbox()
        deps = NodeDeps(
            agent=agent,
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=sandbox,
            diff=StubDiffProvider(),
            config=WorkflowConfig(max_self_correction_cycles=3),
        )

        state = _initial_state("task-af-full")

        graph = build_graph(deps, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "af-full"}}

        final = await graph.ainvoke(state, cfg)

        assert final["current_phase"] == WorkflowPhase.COMPLETE.value
        assert final["verifier_verdict"]["test_result"] == "PASS"
        # Agent should have been called: 1 value_node + 1 actor
        # + 3 critics + 1 memory_consolidation (AGENTS.md only, no arch violations) = 6
        assert len(agent.calls) == 6

    async def test_preflight_errors_trigger_actor_retry(self, tmp_path, monkeypatch):
        """
        Actor -> preflight (LSP errors) -> actor retry -> preflight (clean)
        -> critics -> verifier PASS.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        chain = _bootstrap_chain_preflight_error(_diff_response())
        agent = StubAgentProvider(chain)

        # First preflight has errors, second is clean
        preflight_results = iter([
            {"passed": False, "lsp_errors": [{"file": "X.java", "line": 10, "code": "TS2345", "message": "error"}],
             "arch_violations": [], "cross_stack_errors": [], "duration_ms": 50},
            {"passed": True, "lsp_errors": [], "arch_violations": [], "cross_stack_errors": [], "duration_ms": 50},
        ])

        class _PreflightSandbox(StubSandboxProvider):
            async def exec_in_container(self, handle, command, env=None, timeout=120):
                # Preflight checks: mvn compile -q or tsc --noEmit
                if "mvn compile" in command or "tsc --noEmit" in command:
                    result = preflight_results.__next__()
                    if result["passed"]:
                        return ExecResult(0, "", "", 10)
                    # Return output matching _JAVA_ERROR_RE regex:
                    # [ERROR] <file>.java:<line>,<col> <message>
                    err = result["lsp_errors"][0]
                    error_line = f"[ERROR] {err['file']}:[{err['line']},10] {err['message']}"
                    return ExecResult(1, error_line, "", 10)
                # Verifier: run tests
                if "mvn test" in command:
                    return ExecResult(0, "BUILD SUCCESS\nTests run: 5, Failures: 0", "", 100)
                return ExecResult(0, "BUILD SUCCESS\nTests run: 5, Failures: 0", "", 100)

        sandbox = _PreflightSandbox()
        deps = NodeDeps(
            agent=agent,
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=sandbox,
            diff=StubDiffProvider(),
            config=WorkflowConfig(max_self_correction_cycles=3),
        )

        state = _initial_state("task-af-retry")

        graph = build_graph(deps, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "af-retry"}}

        final = await graph.ainvoke(state, cfg)

        assert final["current_phase"] == WorkflowPhase.COMPLETE.value
        assert final["verifier_verdict"]["test_result"] == "PASS"
        # Should have 2 actor calls (first + retry) + 1 value_node
        # + 3 critics + 2 memory = 8
        actor_calls = [c for c in agent.calls if c[0] == "structured_output"]
        # value_node(1) + actor x2(2) + critics(3) + memory(1) = 7
        assert len(actor_calls) == 7

    async def test_critics_findings_recorded(self, tmp_path, monkeypatch):
        """
        When critics find issues, they are recorded in critic_findings state.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        # Custom chain: critics that find issues
        chain = [
            # value_node: generate strategy
            make_json_agent_result([{
                "strategy_id": "s1", "description": "Add findById",
                "affected_files": ["UserService.java"],
            }]),
            # actor: produce diff
            _diff_response(),
            # security critic finds issue
            make_json_agent_result([{
                "critic": "security", "severity": "warning", "file": "X.java",
                "line": 10, "rule_id": "SEC-001", "message": "Potential issue",
                "resolution_hint": "review",
            }]),
            # style critic: empty
            make_json_agent_result([]),
            # consistency critic: empty
            make_json_agent_result([]),
            # memory_consolidation
            _agents_md_response(),
            _arch_rule_response(),
        ]
        agent = StubAgentProvider(chain)
        deps = NodeDeps(
            agent=agent,
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=_passing_sandbox(),
            diff=StubDiffProvider(),
            config=WorkflowConfig(max_self_correction_cycles=3),
        )

        state = _initial_state("task-af-critics")

        graph = build_graph(deps, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "af-critics"}}

        final = await graph.ainvoke(state, cfg)

        assert final["current_phase"] == WorkflowPhase.COMPLETE.value
        # Critic findings should have been recorded (at least 1 from security critic)
        assert len(final.get("critic_findings", [])) >= 1

    async def test_actor_self_loop_on_empty_diff(self, tmp_path, monkeypatch):
        """
        Actor produces empty diff -> self-loop -> actor produces valid diff -> verifier PASS.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        chain = [
            # value_node: generate strategy
            make_json_agent_result([{
                "strategy_id": "s1", "description": "Add findById",
                "affected_files": ["UserService.java"],
            }]),
            # actor (first attempt): empty diff
            AgentResult(content="[]", tool_calls=[], finish_reason="stop",
                        input_tokens=5, output_tokens=5),
            # actor (retry): produce diff
            _diff_response(),
            # critics: empty
            *_empty_critics(),
            # memory_consolidation
            _agents_md_response(),
            _arch_rule_response(),
        ]
        agent = StubAgentProvider(chain)
        deps = NodeDeps(
            agent=agent,
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=_passing_sandbox(),
            diff=StubDiffProvider(),
            config=WorkflowConfig(max_self_correction_cycles=3),
        )

        state = _initial_state("task-af-empty")

        graph = build_graph(deps, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "af-empty"}}

        final = await graph.ainvoke(state, cfg)

        assert final["current_phase"] == WorkflowPhase.COMPLETE.value
        assert final["verifier_verdict"]["test_result"] == "PASS"
        # Should have 2 actor calls (empty diff + retry)
        actor_calls = [c for c in agent.calls if c[0] == "structured_output"]
        # value_node(1) + actor x2(2, one empty + one retry) + critics(3) + memory(1) = 7
        assert len(actor_calls) == 7

    async def test_cost_accumulation_across_nodes(self, tmp_path, monkeypatch):
        """
        Token costs accumulate across value_node, tdd_gate, actor, critics,
        and memory_consolidation.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        chain = _actor_chain(_diff_response())
        agent = StubAgentProvider(chain)
        deps = NodeDeps(
            agent=agent,
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=_passing_sandbox(),
            diff=StubDiffProvider(),
            config=WorkflowConfig(
                max_self_correction_cycles=3,
                token_budget=WorkflowConfig().token_budget,
            ),
        )

        state = _initial_state("task-af-cost")

        graph = build_graph(deps, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "af-cost"}}

        final = await graph.ainvoke(state, cfg)

        assert final["current_phase"] == WorkflowPhase.COMPLETE.value
        # Cost should be > 0 (token tracking is working)
        assert final["cumulative_cost_dollars"] > 0
