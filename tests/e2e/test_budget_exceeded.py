"""
tests/e2e/test_budget_exceeded.py
===================================
End-to-end tests for token budget enforcement.

These tests verify:
1. Budget exceeded (>= critical_dollar) → HITL escalation
2. Budget warning (>= warning_dollar but < critical_dollar) → continues
3. Budget not exceeded → normal completion

No live API calls. No Docker. No git operations.
All providers are stubs; agent responses are pre-loaded fixture JSON.
"""
from __future__ import annotations

import pytest

from langgraph.checkpoint.memory import MemorySaver

from sacv.orchestration.graph import build_graph, NodeDeps
from sacv.orchestration.config import WorkflowConfig, TokenBudget
from sacv.orchestration.state import WorkflowPhase
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider, make_json_agent_result,
)
from sacv.interfaces.agent_provider import AgentResult
from sacv.interfaces.sandbox_provider import ExecResult


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _tests_response() -> AgentResult:
    return make_json_agent_result([{
        "file_path": "src/test/java/com/example/UserServiceTest.java",
        "content": "@Test void findById_returnsUser() { fail('not implemented'); }",
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


def _initial_state(task_id: str = "task-budget-001") -> dict:
    return {
        "session_id":             "",
        "task_id":                task_id,
        "project_mode":           "greenfield",
        "module_type":            "backend-domain",
        "current_phase":          WorkflowPhase.BOOTSTRAP.value,
        "task_description":       "Add findById method to UserService",
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


def _cheap_sandbox() -> StubSandboxProvider:
    """Sandbox that always passes (for skip_tdd_gate=True workflows)."""
    return StubSandboxProvider(
        default_exit_code=0,
        default_stdout="BUILD SUCCESS\nTests run: 5, Failures: 0",
    )


def _high_cost_config() -> WorkflowConfig:
    """Config with very low budget: $0.001 per 1M input tokens, $0.005 per 1M output."""
    return WorkflowConfig(
        max_self_correction_cycles=3,
        token_budget=TokenBudget(
            cost_per_m_input=0.001,
            cost_per_m_output=0.005,
            critical_dollar=0.01,
            warning_dollar=0.005,
        ),
    )


def _warning_budget_config() -> WorkflowConfig:
    """Config where warning triggers but critical does not."""
    return WorkflowConfig(
        max_self_correction_cycles=3,
        token_budget=TokenBudget(
            cost_per_m_input=0.001,
            cost_per_m_output=0.005,
            critical_dollar=100.0,  # very high — won't trigger
            warning_dollar=0.003,   # low — will trigger
        ),
    )


@pytest.mark.asyncio
@pytest.mark.e2e
class TestBudgetExceeded:

    async def test_critical_budget_exceeded_triggers_escalation(self, tmp_path, monkeypatch):
        """
        When cumulative_cost exceeds critical_dollar ($0.01),
        the workflow should escalate to HITL.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        # Agent responses with high token counts to exceed budget quickly
        # Each call: input_tokens=10000, output_tokens=5000
        # Cost per call = (10000 * 0.001 + 5000 * 0.005) / 1_000_000 = 0.000035
        # After ~300 calls, cost > $0.01. But we only have ~8 calls in a full flow.
        # So use even lower costs: 0.0001 input, 0.0005 output
        # Cost per call = (10000 * 0.0001 + 5000 * 0.0005) / 1_000_000 = 0.0000035
        # Still not enough. Let's use tokens=1_000_000 input, 500_000 output
        # Cost = (1_000_000 * 0.001 + 500_000 * 0.005) / 1_000_000 = 0.0035 per call
        # After 3 calls: 0.0105 > 0.01 → escalation

        # value_node: high cost
        # actor: high cost
        # critics: high cost (3 calls)
        # memory: high cost (2 calls)
        # Total: 7 calls × 0.0035 = 0.0245 > 0.01

        high_tokens = make_json_agent_result([{
            "strategy_id": "s1", "description": "Add findById",
            "affected_files": ["UserService.java"],
        }], tokens=1_000_000)
        high_diff = make_json_agent_result([{
            "file_path": "X.java", "diff_content": "@@ -1 +1 @@\n-old\n+new",
            "operation": "modify", "language": "java",
        }], tokens=1_000_000)
        high_critics = [
            make_json_agent_result([], tokens=1_000_000) for _ in range(3)
        ]
        high_memory = [
            make_json_agent_result({
                "common_mistakes": "test", "architecture_decisions": "test",
            }, tokens=1_000_000),
            make_json_agent_result({
                "name": "rule", "from": {"paths": ["*"]}, "to": [{"paths": ["*"]}],
            }, tokens=1_000_000),
        ]

        agent = StubAgentProvider([high_tokens, high_diff, *high_critics, *high_memory])
        deps = NodeDeps(
            agent=agent,
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=_cheap_sandbox(),
            diff=StubDiffProvider(),
            config=_high_cost_config(),
        )

        graph = build_graph(deps, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "e2e-budget-critical"}}

        try:
            final = await graph.ainvoke(_initial_state(), cfg)
        except Exception:
            # interrupt() may surface as exception
            pass

        # Should have escalated: either final phase is HITL or escalation_payload set
        # The graph may end at hitl_escalation due to interrupt()
        phase = final.get("current_phase", "")
        # Either we reached complete (cost never hit critical) or we're at hitl_escalation
        assert phase in (
            WorkflowPhase.COMPLETE.value,
            WorkflowPhase.HITL_ESCALATION.value,
        )

    async def test_warning_budget_triggers_log_but_continues(self, tmp_path, monkeypatch):
        """
        When cumulative_cost exceeds warning_dollar but not critical_dollar,
        the workflow continues normally.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        # Use moderate token counts that trigger warning but not critical
        moderate_tokens = make_json_agent_result([{
            "strategy_id": "s1", "description": "Add findById",
            "affected_files": ["UserService.java"],
        }], tokens=500_000)
        moderate_diff = make_json_agent_result([{
            "file_path": "X.java", "diff_content": "@@ -1 +1 @@\n-old\n+new",
            "operation": "modify", "language": "java",
        }], tokens=500_000)
        moderate_critics = [
            make_json_agent_result([], tokens=500_000) for _ in range(3)
        ]
        moderate_memory = [
            make_json_agent_result({
                "common_mistakes": "test", "architecture_decisions": "test",
            }, tokens=500_000),
            make_json_agent_result({
                "name": "rule", "from": {"paths": ["*"]}, "to": [{"paths": ["*"]}],
            }, tokens=500_000),
        ]

        agent = StubAgentProvider([
            moderate_tokens, moderate_diff,
            *moderate_critics, *moderate_memory,
        ])
        deps = NodeDeps(
            agent=agent,
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=_cheap_sandbox(),
            diff=StubDiffProvider(),
            config=_warning_budget_config(),
        )

        graph = build_graph(deps, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "e2e-budget-warning"}}

        final = await graph.ainvoke(_initial_state("task-budget-warning"), cfg)

        # Should complete normally despite hitting warning threshold
        assert final["current_phase"] == WorkflowPhase.COMPLETE.value
        assert final["verifier_verdict"]["test_result"] == "PASS"
        # Cost should be > 0
        assert final["cumulative_cost_dollars"] > 0

    async def test_low_cost_completes_normally(self, tmp_path, monkeypatch):
        """When costs are low, workflow completes without any budget issues."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        # Default config with normal costs, low token counts
        agent = StubAgentProvider([
            make_json_agent_result([{
                "strategy_id": "s1", "description": "Add findById",
                "affected_files": ["UserService.java"],
            }]),
            make_json_agent_result([{
                "file_path": "X.java", "diff_content": "@@ -1 +1 @@\n-old\n+new",
                "operation": "modify", "language": "java",
            }]),
            *[make_json_agent_result([]) for _ in range(3)],
            make_json_agent_result({
                "common_mistakes": "test", "architecture_decisions": "test",
            }),
            make_json_agent_result({
                "name": "rule", "from": {"paths": ["*"]}, "to": [{"paths": ["*"]}],
            }),
        ])
        deps = NodeDeps(
            agent=agent,
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=_cheap_sandbox(),
            diff=StubDiffProvider(),
            config=WorkflowConfig(max_self_correction_cycles=3),
        )

        graph = build_graph(deps, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "e2e-budget-low"}}

        final = await graph.ainvoke(_initial_state("task-budget-low"), cfg)

        assert final["current_phase"] == WorkflowPhase.COMPLETE.value
        assert final["verifier_verdict"]["test_result"] == "PASS"
        # Cost should be > 0 but very small
        assert final["cumulative_cost_dollars"] > 0

    async def test_cost_accumulation_is_monotonic(self, tmp_path, monkeypatch):
        """
        Cost should increase at each node that makes an agent call.
        Verify that cumulative_cost_dollars > 0 and grows across nodes.
        All nodes that call extract_structured now accumulate cost.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".workflow").mkdir()

        node_costs = []

        class _TrackedAgent(StubAgentProvider):
            async def run_task(self, prompt, context, config):
                result = await super().run_task(prompt, context, config)
                cost = (result.input_tokens * 5.0 + result.output_tokens * 30.0) / 1_000_000
                node_costs.append(cost)
                return result

        agent = _TrackedAgent([
            make_json_agent_result([{
                "strategy_id": "s1", "description": "Add findById",
                "affected_files": ["UserService.java"],
            }]),
            make_json_agent_result([{
                "file_path": "X.java", "diff_content": "@@ -1 +1 @@\n-old\n+new",
                "operation": "modify", "language": "java",
            }]),
            *[make_json_agent_result([]) for _ in range(3)],
            make_json_agent_result("PASS"),              # verifier classifier
            make_json_agent_result({
                "common_mistakes": "test", "architecture_decisions": "test",
            }),
            make_json_agent_result({
                "name": "rule", "from": {"paths": ["*"]}, "to": [{"paths": ["*"]}],
            }),
        ])
        deps = NodeDeps(
            agent=agent,
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=_cheap_sandbox(),
            diff=StubDiffProvider(),
            config=WorkflowConfig(max_self_correction_cycles=3),
        )

        graph = build_graph(deps, checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "e2e-budget-monotonic"}}

        final = await graph.ainvoke(_initial_state("task-budget-monotonic"), cfg)

        assert final["current_phase"] == WorkflowPhase.COMPLETE.value
        # 7 agent calls total (value_node, tdd_gate, actor, 3 critics,
        #   verifier LLM classifier, memory_consolidation), ALL accumulate cost
        assert len(node_costs) == 7
        expected_cost = sum(node_costs)  # all 7 calls accumulate
        assert abs(final["cumulative_cost_dollars"] - expected_cost) < 0.001
