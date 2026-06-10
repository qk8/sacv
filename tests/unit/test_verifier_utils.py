"""
tests/unit/test_verifier_utils.py
===================================
Unit tests for verifier utility functions.

Tests cover:
1. accumulate_cost — with AgentResult
2. accumulate_cost — with None result
3. accumulate_cost — with existing cost
4. add_agent_cost — wraps accumulate_cost
5. Token cost calculation accuracy
6. run_verifier_with_confidence — delegates correctly
"""
from __future__ import annotations

import pytest

from sacv.orchestration.config import WorkflowConfig, TokenBudget
from sacv.orchestration.verifier_utils import accumulate_cost, add_agent_cost
from sacv.interfaces.agent_provider import AgentResult
from sacv.testing.stub_providers import (
    StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
    StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
    StubSandboxProvider,
)


@pytest.mark.unit
class TestAccumulateCost:

    def test_none_result_returns_existing_cost(self):
        """When last_tokens is None, returns existing cumulative_cost_dollars."""
        result = accumulate_cost(None, {"cumulative_cost_dollars": 5.0}, WorkflowConfig())
        assert result == 5.0

    def test_none_result_with_zero_existing(self):
        """When last_tokens is None and no existing cost, returns 0."""
        result = accumulate_cost(None, {}, WorkflowConfig())
        assert result == 0.0

    def test_zero_tokens_adds_no_cost(self):
        """AgentResult with 0 tokens adds no cost."""
        result = accumulate_cost(
            AgentResult(content="x", tool_calls=[], finish_reason="stop",
                        input_tokens=0, output_tokens=0),
            {"cumulative_cost_dollars": 0.0},
            WorkflowConfig(),
        )
        assert result == 0.0

    def test_input_tokens_cost(self):
        """Input tokens cost: tokens / 1M * cost_per_m_input."""
        # Default: cost_per_m_input = 5.0
        # 1000000 input tokens = $5.0
        result = accumulate_cost(
            AgentResult(content="x", tool_calls=[], finish_reason="stop",
                        input_tokens=1_000_000, output_tokens=0),
            {"cumulative_cost_dollars": 0.0},
            WorkflowConfig(),
        )
        assert result == pytest.approx(5.0, abs=1e-6)

    def test_output_tokens_cost(self):
        """Output tokens cost: tokens / 1M * cost_per_m_output."""
        # Default: cost_per_m_output = 30.0
        # 1000000 output tokens = $30.0
        result = accumulate_cost(
            AgentResult(content="x", tool_calls=[], finish_reason="stop",
                        input_tokens=0, output_tokens=1_000_000),
            {"cumulative_cost_dollars": 0.0},
            WorkflowConfig(),
        )
        assert result == pytest.approx(30.0, abs=1e-6)

    def test_both_input_and_output(self):
        """Both input and output tokens are summed."""
        result = accumulate_cost(
            AgentResult(content="x", tool_calls=[], finish_reason="stop",
                        input_tokens=1_000_000, output_tokens=1_000_000),
            {"cumulative_cost_dollars": 0.0},
            WorkflowConfig(),
        )
        # 5.0 + 30.0 = 35.0
        assert result == pytest.approx(35.0, abs=1e-6)

    def test_partial_tokens(self):
        """Partial token counts are proportional."""
        result = accumulate_cost(
            AgentResult(content="x", tool_calls=[], finish_reason="stop",
                        input_tokens=100_000, output_tokens=200_000),
            {"cumulative_cost_dollars": 0.0},
            WorkflowConfig(),
        )
        # 0.1M * 5.0 + 0.2M * 30.0 = 0.5 + 6.0 = 6.5
        assert result == pytest.approx(6.5, abs=1e-6)

    def test_adds_to_existing_cost(self):
        """New cost is added to existing cumulative_cost_dollars."""
        result = accumulate_cost(
            AgentResult(content="x", tool_calls=[], finish_reason="stop",
                        input_tokens=1_000_000, output_tokens=0),
            {"cumulative_cost_dollars": 10.0},
            WorkflowConfig(),
        )
        assert result == pytest.approx(15.0, abs=1e-6)

    def test_custom_token_budget(self):
        """Custom token budget pricing is respected."""
        config = WorkflowConfig(token_budget=TokenBudget(
            cost_per_m_input=10.0, cost_per_m_output=50.0,
            critical_dollar=80.0, warning_dollar=50.0,
        ))
        result = accumulate_cost(
            AgentResult(content="x", tool_calls=[], finish_reason="stop",
                        input_tokens=1_000_000, output_tokens=1_000_000),
            {"cumulative_cost_dollars": 0.0},
            config,
        )
        # 10.0 + 50.0 = 60.0
        assert result == pytest.approx(60.0, abs=1e-6)

    def test_small_token_count(self):
        """Small token counts produce proportionally small costs."""
        result = accumulate_cost(
            AgentResult(content="x", tool_calls=[], finish_reason="stop",
                        input_tokens=100, output_tokens=200),
            {"cumulative_cost_dollars": 0.0},
            WorkflowConfig(),
        )
        # 0.0001M * 5.0 + 0.0002M * 30.0 = 0.0005 + 0.006 = 0.0065
        assert result == pytest.approx(0.0065, abs=1e-8)


@pytest.mark.unit
class TestAddAgentCost:

    def test_wraps_accumulate_cost(self):
        """add_agent_cost correctly wraps accumulate_cost."""
        result = add_agent_cost(
            AgentResult(content="x", tool_calls=[], finish_reason="stop",
                        input_tokens=1_000_000, output_tokens=0),
            0.0,
            WorkflowConfig(),
        )
        assert result == pytest.approx(5.0, abs=1e-6)

    def test_adds_to_running_total(self):
        result = add_agent_cost(
            AgentResult(content="x", tool_calls=[], finish_reason="stop",
                        input_tokens=1_000_000, output_tokens=0),
            10.0,
            WorkflowConfig(),
        )
        assert result == pytest.approx(15.0, abs=1e-6)

    def test_multiple_calls_accumulate(self):
        """Multiple agent calls accumulate cost correctly."""
        config = WorkflowConfig()
        cost = 0.0
        for _ in range(3):
            cost = add_agent_cost(
                AgentResult(content="x", tool_calls=[], finish_reason="stop",
                            input_tokens=500_000, output_tokens=500_000),
                cost,
                config,
            )
        # Each call: 0.5M * 5.0 + 0.5M * 30.0 = 2.5 + 15.0 = 17.5
        # 3 calls: 17.5 * 3 = 52.5
        assert cost == pytest.approx(52.5, abs=1e-6)


@pytest.mark.asyncio
@pytest.mark.unit
class TestRunVerifierWithConfidence:

    async def test_delegates_to_verifier(self):
        """run_verifier_with_confidence calls the inner verifier node."""
        from sacv.orchestration.deps import NodeDeps
        from sacv.testing.stub_providers import (
            StubAgentProvider, StubMemoryProvider, StubCodeGraphProvider,
            StubCrossDomainProvider, StubDiffProvider, StubGitProvider,
            StubSandboxProvider,
        )
        from sacv.orchestration.state import WorkflowPhase

        sandbox = StubSandboxProvider()
        deps = NodeDeps(
            agent=StubAgentProvider(),
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=sandbox,
            diff=StubDiffProvider(),
            config=WorkflowConfig(),
        )

        state = {
            "session_id": "t", "task_id": "t", "task_description": "",
            "project_mode": "greenfield", "module_type": "backend-domain",
            "current_phase": WorkflowPhase.VERIFIER.value,
            "context_skeleton": None, "blast_radius_map": None,
            "agents_md_context": None, "strategy_candidates": [],
            "selected_strategy": None, "pruned_strategies": [],
            "red_phase_evidence_path": None, "test_inventory_paths": [],
            "diff_proposal": None, "preflight_result": None,
            "critic_findings": [], "verifier_verdict": None,
            "debug_observations": None,
            "correction_state": {
                "attempt_count": 0, "branch_name": None,
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            },
            "confidence_score": 1.0, "replan_count": 0,
            "active_branches": [], "exhausted_branches": [],
            "escalation_payload": None, "procedural_constraints": [],
            "lesson_learned": None, "arch_rules_updated": False,
            "cumulative_cost_dollars": 0.0,
        }

        out = await deps.sandbox.warm_container()
        try:
            result = await __import__('sacv.orchestration.verifier_utils', fromlist=['run_verifier_with_confidence']).run_verifier_with_confidence(state, deps)
        finally:
            await deps.sandbox.destroy_container(out)

        # Should have verifier_verdict and confidence_score
        assert "verifier_verdict" in result
        assert "confidence_score" in result

    async def test_verifier_exception_returns_fail_verdict(self):
        """ERR-004: When verifier raises, must return FAIL verdict and increment attempt_count.

        Unhandled exceptions (Docker unavailable, network timeout) should not
        crash the graph. Instead, a synthetic FAIL verdict is returned so
        route_after_verifier can escalate to HITL cleanly.
        """
        from sacv.orchestration.deps import NodeDeps
        from sacv.orchestration.state import WorkflowPhase, DiagnosticVerdict
        from unittest.mock import AsyncMock, patch

        deps = NodeDeps(
            agent=StubAgentProvider(),
            memory=StubMemoryProvider(),
            code_graph=StubCodeGraphProvider(),
            cross_domain=StubCrossDomainProvider(),
            git=StubGitProvider(),
            sandbox=StubSandboxProvider(),
            diff=StubDiffProvider(),
            config=WorkflowConfig(),
        )

        state = {
            "session_id": "t", "task_id": "t", "task_description": "",
            "project_mode": "greenfield", "module_type": "backend-domain",
            "current_phase": WorkflowPhase.VERIFIER.value,
            "context_skeleton": None, "blast_radius_map": None,
            "agents_md_context": None, "strategy_candidates": [],
            "selected_strategy": None, "pruned_strategies": [],
            "red_phase_evidence_path": None, "test_inventory_paths": [],
            "diff_proposal": None, "preflight_result": None,
            "critic_findings": [], "verifier_verdict": None,
            "debug_observations": None,
            "correction_state": {
                "attempt_count": 1, "branch_name": None,
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            },
            "confidence_score": 1.0, "replan_count": 0,
            "active_branches": [], "exhausted_branches": [],
            "escalation_payload": None, "procedural_constraints": [],
            "lesson_learned": None, "arch_rules_updated": False,
            "cumulative_cost_dollars": 0.0,
        }

        async def _failing_verifier(s):
            raise RuntimeError("Docker daemon not running")

        with patch(
            "sacv.nodes.verifier.make_verifier_node",
            return_value=_failing_verifier,
        ):
            result = await __import__(
                'sacv.orchestration.verifier_utils',
                fromlist=['run_verifier_with_confidence'],
            ).run_verifier_with_confidence(state, deps)

        # Must return a FAIL verdict, not propagate the exception
        verdict = result["verifier_verdict"]
        assert isinstance(verdict, dict)
        assert verdict["test_result"] == "FAIL"
        assert verdict["diagnostic"] == DiagnosticVerdict.AMBIGUOUS.value
        # attempt_count must be incremented
        assert result["correction_state"]["attempt_count"] == 2
        # confidence_score must be computed
        assert "confidence_score" in result
        # docker_exit_code should indicate internal error
        assert verdict["docker_exit_code"] == -2
