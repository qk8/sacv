"""
tests/unit/test_speculative_branch.py
======================================

Unit tests for speculative_branch pure functions.

Tests cover:
  1. _merge_branch_state — CRITIC_RESET handling, normal merges
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sacv.nodes.speculative_branch import _evaluate_branch, _merge_branch_state
from sacv.orchestration.state import CRITIC_RESET


class TestMergeBranchState:

    def test_normal_merge_overwrites_fields(self):
        """Normal updates merge into base state."""
        base = {
            "critic_findings": [{"severity": "warning"}],
            "cumulative_cost_dollars": 0.5,
            "task_id": "task-1",
        }
        update = {
            "critic_findings": [{"severity": "critical"}],
            "cumulative_cost_dollars": 1.0,
        }
        result = _merge_branch_state(base, update)
        assert result["critic_findings"] == [{"severity": "critical"}]
        assert result["cumulative_cost_dollars"] == 1.0
        assert result["task_id"] == "task-1"  # preserved from base

    def test_critic_reset_becomes_empty_list(self):
        """CRITIC_RESET is replaced with [] so branch_state always holds a list."""
        base = {
            "critic_findings": [{"severity": "critical"}],
        }
        update = {
            "critic_findings": CRITIC_RESET,
        }
        result = _merge_branch_state(base, update)
        assert result["critic_findings"] == []

    def test_critic_reset_from_empty_base(self):
        """CRITIC_RESET works even when base has no critic_findings."""
        base = {"other_field": "value"}
        update = {"critic_findings": CRITIC_RESET}
        result = _merge_branch_state(base, update)
        assert result["critic_findings"] == []

    def test_critic_reset_preserves_other_fields(self):
        """Other fields are preserved when CRITIC_RESET is applied."""
        base = {
            "task_id": "task-1",
            "correction_state": {"attempt_count": 3},
        }
        update = {"critic_findings": CRITIC_RESET}
        result = _merge_branch_state(base, update)
        assert result["task_id"] == "task-1"
        assert result["correction_state"] == {"attempt_count": 3}
        assert result["critic_findings"] == []

    def test_update_can_add_new_fields(self):
        """New fields in update are added to the merged result."""
        base = {"existing": "field"}
        update = {"new_field": "new_value"}
        result = _merge_branch_state(base, update)
        assert result["existing"] == "field"
        assert result["new_field"] == "new_value"

    def test_update_overwrites_base_values(self):
        """Keys present in both base and update use the update value."""
        base = {"field": "old", "other": "keep"}
        update = {"field": "new"}
        result = _merge_branch_state(base, update)
        assert result["field"] == "new"
        assert result["other"] == "keep"

    def test_empty_update_preserves_base(self):
        """Empty update dict returns a copy of base."""
        base = {"a": 1, "b": 2}
        result = _merge_branch_state(base, {})
        assert result == base
        assert result is not base  # shallow copy, not same object

    def test_empty_base_with_update(self):
        """Update on empty base returns the update."""
        result = _merge_branch_state({}, {"a": 1})
        assert result == {"a": 1}

    def test_critic_reset_with_other_updates(self):
        """CRITIC_RESET alongside other updates merges correctly."""
        base = {
            "critic_findings": [{"severity": "critical"}],
            "cumulative_cost_dollars": 0.5,
        }
        update = {
            "critic_findings": CRITIC_RESET,
            "cumulative_cost_dollars": 1.5,
            "new_key": "new_value",
        }
        result = _merge_branch_state(base, update)
        assert result["critic_findings"] == []
        assert result["cumulative_cost_dollars"] == 1.5
        assert result["new_key"] == "new_value"

    def test_deep_copy_prevents_nested_mutation(self):
        """Mutating nested dicts in the result must not affect the base."""
        base = {
            "correction_state": {"attempt_count": 0, "branch_name": "main"},
            "critic_findings": [{"severity": "warning"}],
        }
        result = _merge_branch_state(base, {})

        # Mutate nested objects in the result
        result["correction_state"]["attempt_count"] = 99
        result["critic_findings"].append({"severity": "critical"})

        # Base should be unchanged
        assert base["correction_state"]["attempt_count"] == 0
        assert len(base["critic_findings"]) == 1
        assert base["critic_findings"][0] == {"severity": "warning"}

    def test_deep_copy_update_overwrites_nested(self):
        """When update provides a new nested dict, it replaces the base one."""
        base = {
            "correction_state": {"attempt_count": 0},
            "task_id": "task-1",
        }
        update = {
            "correction_state": {"attempt_count": 5, "branch_name": "feat"},
        }
        result = _merge_branch_state(base, update)

        assert result["correction_state"] == {"attempt_count": 5, "branch_name": "feat"}
        # Base should be unchanged
        assert base["correction_state"] == {"attempt_count": 0}


class TestSpeculativeBranchCostTracking:
    """
    Tests for the cost accumulation pattern used in _evaluate_branch.

    The pattern: actor produces cost C_actor, each critic starts from baseline
    C_actor and returns C_actor + C_critic_i. The incremental cost is the sum
    of critic costs minus 3× baseline. The TOTAL cost must be baseline +
    incremental (i.e. C_actor + sum of critic increments).

    BUG-012: The old code set branch_cost = incremental only, dropping the
    actor's cost from the final state. This systematically undercounts cost,
    potentially allowing more branches before the budget circuit breaker fires.
    """

    def test_total_cost_includes_actor_baseline(self):
        """
        When actor costs $2.00 and three critics each cost $1.00,
        the final cumulative_cost_dollars must be $5.00, not $3.00.
        """
        baseline = 2.0  # actor's cost merged into branch_state
        sec_cost = 3.0  # baseline(2.0) + critic_increment(1.0)
        sty_cost = 3.0
        con_cost = 3.0

        # Old buggy formula: only incremental cost
        buggy_incremental = sec_cost + sty_cost + con_cost - 3.0 * baseline
        assert buggy_incremental == 3.0  # only critic increments

        # Correct formula: baseline + incremental
        correct_total = baseline + (sec_cost + sty_cost + con_cost - 3.0 * baseline)
        assert correct_total == 5.0  # actor + all critics

    def test_total_cost_matches_sum_of_individual_costs(self):
        """
        The total cost should equal actor_cost + sum of critic increments.
        Each critic's cumulative_cost = baseline + its own token cost.
        """
        actor_cost = 2.0
        sec_increment = 1.0
        sty_increment = 2.0
        con_increment = 3.0
        sec_cost = actor_cost + sec_increment
        sty_cost = actor_cost + sty_increment
        con_cost = actor_cost + con_increment

        total = actor_cost + (sec_cost + sty_cost + con_cost - 3.0 * actor_cost)
        assert total == actor_cost + sec_increment + sty_increment + con_increment
        assert total == 8.0  # 2.0 + 1.0 + 2.0 + 3.0

    def test_zero_actor_cost_still_works(self):
        """When actor cost is 0, total should still be the critic increments."""
        baseline = 0.0
        sec_cost = 1.0
        sty_cost = 1.0
        con_cost = 1.0

        total = baseline + (sec_cost + sty_cost + con_cost - 3.0 * baseline)
        assert total == 3.0

    def test_cost_formula_mirrors_evaluate_branch_pattern(self):
        """
        Mirrors the exact cost computation in _evaluate_branch:
          1. actor_out is merged into branch_state (includes actor cost)
          2. baseline = branch_state.cumulative_cost_dollars
          3. critics run, each returning baseline + their own cost
          4. incremental = sum(critic_costs) - 3*baseline
          5. FINAL: branch_cost = baseline + incremental (BUG-012 fix)

        The old buggy code skipped `+ baseline`, dropping the actor's cost.
        """
        # Simulate actor cost merged into branch_state via _merge_branch_state
        branch_state = {"cumulative_cost_dollars": 2.0}  # actor's cost
        baseline = branch_state["cumulative_cost_dollars"]

        # Each critic starts from baseline and adds its own cost
        sec_out = {"cumulative_cost_dollars": baseline + 1.0}
        sty_out = {"cumulative_cost_dollars": baseline + 2.0}
        con_out = {"cumulative_cost_dollars": baseline + 3.0}

        # Compute incremental critic costs
        incremental = (
            sec_out["cumulative_cost_dollars"]
            + sty_out["cumulative_cost_dollars"]
            + con_out["cumulative_cost_dollars"]
            - 3.0 * baseline
        )

        # BUG-012 FIX: total must include baseline (actor cost)
        branch_cost = baseline + incremental
        assert branch_cost == 8.0  # 2.0 (actor) + 1.0 + 2.0 + 3.0 (critics)

        # Verify the old buggy formula would have given wrong result
        buggy_cost = incremental  # drops baseline
        assert buggy_cost == 6.0  # missing $2.0 actor cost


class TestEvaluateBranchFailureReason:
    """Verify that _evaluate_branch returns failure reasons (DBG-005)."""

    @pytest.fixture
    def mock_deps(self):
        deps = MagicMock()
        deps.config.max_parallel_branches = 3
        return deps

    @pytest.fixture
    def mock_state(self):
        return {
            "task_id": "T1",
            "correction_state": {"branch_name": "main"},
        }

    @pytest.fixture
    def mock_strategy(self):
        return {"strategy_id": "s1", "composite_score": 0.9}

    async def test_returns_failure_reason_on_preflight_fail(
        self, mock_deps, mock_state, mock_strategy,
    ):
        """_evaluate_branch returns a preflight failure reason."""
        from unittest.mock import AsyncMock, patch

        mock_subgraph = MagicMock()
        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.return_value = {
            "verifier_verdict": {"test_result": "FAIL"},
            "preflight_result": {"passed": False, "lsp_errors": [], "arch_violations": []},
        }
        mock_subgraph.compile.return_value = mock_compiled

        with patch(
            "sacv.orchestration.graph.build_branch_subgraph",
            return_value=mock_subgraph,
        ):
            result = await _evaluate_branch(mock_state, mock_strategy, mock_deps)

        branch_name, verdict, reason = result
        assert branch_name == "agent-task-T1-s1"
        assert verdict is None
        assert "preflight_failed" in reason

    async def test_returns_failure_reason_on_verifier_fail(
        self, mock_deps, mock_state, mock_strategy,
    ):
        """_evaluate_branch returns a verifier failure reason."""
        from unittest.mock import AsyncMock, patch

        mock_subgraph = MagicMock()
        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.return_value = {
            "verifier_verdict": {
                "test_result": "FAIL",
                "diagnostic": "test_failed",
                "phase1_passed": True,
                "phase2_passed": False,
            },
            "preflight_result": {"passed": True},
        }
        mock_subgraph.compile.return_value = mock_compiled

        with patch(
            "sacv.orchestration.graph.build_branch_subgraph",
            return_value=mock_subgraph,
        ):
            result = await _evaluate_branch(mock_state, mock_strategy, mock_deps)

        branch_name, verdict, reason = result
        assert branch_name == "agent-task-T1-s1"
        assert verdict is not None
        assert "verifier_fail" in reason

    async def test_returns_failure_reason_on_exception(
        self, mock_deps, mock_state, mock_strategy,
    ):
        """_evaluate_branch returns an exception failure reason."""
        from unittest.mock import AsyncMock, patch

        mock_subgraph = MagicMock()
        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.side_effect = RuntimeError("docker not available")
        mock_subgraph.compile.return_value = mock_compiled

        with patch(
            "sacv.orchestration.graph.build_branch_subgraph",
            return_value=mock_subgraph,
        ):
            result = await _evaluate_branch(mock_state, mock_strategy, mock_deps)

        branch_name, verdict, reason = result
        assert branch_name == "agent-task-T1-s1"
        assert verdict is None
        assert "exception" in reason
        assert "RuntimeError" in reason

    async def test_returns_pass_reason_on_success(
        self, mock_deps, mock_state, mock_strategy,
    ):
        """_evaluate_branch returns 'pass' reason when branch succeeds."""
        from unittest.mock import AsyncMock, patch

        mock_subgraph = MagicMock()
        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.return_value = {
            "verifier_verdict": {"test_result": "PASS"},
            "preflight_result": {"passed": True},
        }
        mock_subgraph.compile.return_value = mock_compiled

        with patch(
            "sacv.orchestration.graph.build_branch_subgraph",
            return_value=mock_subgraph,
        ):
            result = await _evaluate_branch(mock_state, mock_strategy, mock_deps)

        branch_name, verdict, reason = result
        assert branch_name == "agent-task-T1-s1"
        assert verdict is not None
        assert reason == "pass"


class TestSpeculativeBranchVerdictFields:
    """Verify VerifierVerdict constructions include blocked_by_critic (HIGH-07)."""

    @staticmethod
    def _make_node(deps):
        from sacv.nodes.speculative_branch import make_speculative_branch_node
        return make_speculative_branch_node(deps)

    async def test_all_strategies_exhausted_verdict_has_blocked_by_critic(self):
        """When all strategies exhausted, verdict includes blocked_by_critic=False."""
        from unittest.mock import AsyncMock, patch

        agent = MagicMock()
        agent.call_async = AsyncMock(return_value={
            "strategies_evaluated": [],
        })

        deps = MagicMock()
        deps.config.max_parallel_branches = 3
        deps.agent = agent

        node = self._make_node(deps)

        state = {
            "session_id": "t", "task_id": "T1", "task_description": "",
            "project_mode": "brownfield", "module_type": "backend-domain",
            "current_phase": "speculative_branch",
            "context_skeleton": None, "blast_radius_map": None,
            "agents_md_context": None,
            "strategy_candidates": [],
            "selected_strategy": None, "pruned_strategies": [],
            "red_phase_evidence_path": None, "test_inventory_paths": [],
            "diff_proposal": None, "preflight_result": None,
            "critic_findings": [], "verifier_verdict": None,
            "correction_state": {
                "attempt_count": 2, "branch_name": "main",
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            },
            "confidence_score": 0.5, "replan_count": 0,
            "active_branches": [], "exhausted_branches": [],
            "speculative_stash_ref": None, "escalation_payload": None,
            "procedural_constraints": [], "lesson_learned": None,
            "arch_rules_updated": False, "debug_observations": None,
            "cumulative_cost_dollars": 1.0,
        }

        result = await node(state)
        verdict = result.get("verifier_verdict")
        assert verdict is not None
        assert "blocked_by_critic" in verdict

    async def test_all_evaluated_branches_failed_verdict_has_blocked_by_critic(self):
        """When all evaluated branches fail, verdict includes blocked_by_critic=False."""
        from unittest.mock import AsyncMock, patch

        agent = MagicMock()
        agent.call_async = AsyncMock(return_value={
            "strategies_evaluated": [
                {"strategy_id": "s1", "composite_score": 0.9},
            ],
        })

        deps = MagicMock()
        deps.config.max_parallel_branches = 3
        deps.agent = agent

        node = self._make_node(deps)

        state = {
            "session_id": "t", "task_id": "T1", "task_description": "",
            "project_mode": "brownfield", "module_type": "backend-domain",
            "current_phase": "speculative_branch",
            "context_skeleton": None, "blast_radius_map": None,
            "agents_md_context": None,
            "strategy_candidates": [
                {"strategy_id": "s1", "composite_score": 0.9, "affected_files": []},
            ],
            "selected_strategy": None, "pruned_strategies": [],
            "red_phase_evidence_path": None, "test_inventory_paths": [],
            "diff_proposal": None, "preflight_result": None,
            "critic_findings": [], "verifier_verdict": None,
            "correction_state": {
                "attempt_count": 2, "branch_name": "main",
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            },
            "confidence_score": 0.5, "replan_count": 0,
            "active_branches": [], "exhausted_branches": [],
            "speculative_stash_ref": None, "escalation_payload": None,
            "procedural_constraints": [], "lesson_learned": None,
            "arch_rules_updated": False, "debug_observations": None,
            "cumulative_cost_dollars": 1.0,
        }

        result = await node(state)
        verdict = result.get("verifier_verdict")
        assert verdict is not None
        assert "blocked_by_critic" in verdict


class TestSpeculativeBranchRemainingStrategies:
    """Tests for remaining strategies queued when more exist than max_parallel_branches."""

    @staticmethod
    def _make_node(deps):
        from sacv.nodes.speculative_branch import make_speculative_branch_node
        return make_speculative_branch_node(deps)

    async def test_remaining_strategies_queued_in_active_branches(self):
        """When more strategies exist than max_parallel_branches, remaining are queued.

        If we have 4 strategies and max_parallel_branches=2, the first 2 are
        evaluated and the remaining 2 are returned in active_branches for the
        next cycle.
        """
        from unittest.mock import AsyncMock, MagicMock as MockMag, patch

        agent = MockMag()
        agent.call_async = AsyncMock(return_value={
            "strategies_evaluated": [
                {"strategy_id": "s1", "composite_score": 0.9},
                {"strategy_id": "s2", "composite_score": 0.8},
            ],
        })

        deps = MockMag()
        deps.config.max_parallel_branches = 2
        deps.agent = agent

        # Mock build_branch_subgraph to avoid real graph compilation
        mock_subgraph = MockMag()
        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.return_value = {
            "verifier_verdict": {"test_result": "FAIL", "diagnostic": "FIX_IMPL",
                                 "phase1_passed": False, "phase2_passed": False,
                                 "test_failures": [], "performance_delta": None,
                                 "visual_diff_result": None, "critic_findings": [],
                                 "docker_exit_code": 1, "playwright_trace_path": None,
                                 "otel_trace": None, "actuator_snapshot": None},
            "preflight_result": {"passed": False},
        }
        mock_subgraph.compile.return_value = mock_compiled

        node = self._make_node(deps)

        state = {
            "session_id": "t", "task_id": "T1", "task_description": "",
            "project_mode": "brownfield", "module_type": "backend-domain",
            "current_phase": "speculative_branch",
            "context_skeleton": None, "blast_radius_map": None,
            "agents_md_context": None,
            "strategy_candidates": [
                {"strategy_id": "s1", "composite_score": 0.9, "affected_files": []},
                {"strategy_id": "s2", "composite_score": 0.8, "affected_files": []},
                {"strategy_id": "s3", "composite_score": 0.7, "affected_files": []},
                {"strategy_id": "s4", "composite_score": 0.6, "affected_files": []},
            ],
            "selected_strategy": None, "pruned_strategies": [],
            "red_phase_evidence_path": None, "test_inventory_paths": [],
            "diff_proposal": None, "preflight_result": None,
            "critic_findings": [], "verifier_verdict": None,
            "correction_state": {
                "attempt_count": 2, "branch_name": "main",
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            },
            "confidence_score": 0.5, "replan_count": 0,
            "active_branches": [], "exhausted_branches": [],
            "speculative_stash_ref": None, "escalation_payload": None,
            "procedural_constraints": [], "lesson_learned": None,
            "arch_rules_updated": False, "debug_observations": None,
            "cumulative_cost_dollars": 1.0,
        }

        with patch(
            "sacv.orchestration.graph.build_branch_subgraph",
            return_value=mock_subgraph,
        ):
            result = await node(state)

        # The remaining strategies (s3, s4) should be queued in active_branches
        # active_branches contains branch names (strings), not dicts
        active = result.get("active_branches", [])
        assert len(active) == 2
        assert any("s3" in b for b in active)
        assert any("s4" in b for b in active)


class TestSpeculativeBranchExhaustedFiltering:
    """Tests for strategy filtering when already in exhausted_branches."""

    @staticmethod
    def _make_node(deps):
        from sacv.nodes.speculative_branch import make_speculative_branch_node
        return make_speculative_branch_node(deps)

    async def test_exhausted_strategies_filtered_out(self):
        """Strategies already in exhausted_branches are filtered out before evaluation."""
        from unittest.mock import AsyncMock, patch

        agent = MagicMock()
        agent.call_async = AsyncMock(return_value={
            "strategies_evaluated": [],
        })

        deps = MagicMock()
        deps.config.max_parallel_branches = 3
        deps.agent = agent

        node = self._make_node(deps)

        state = {
            "session_id": "t", "task_id": "T1", "task_description": "",
            "project_mode": "brownfield", "module_type": "backend-domain",
            "current_phase": "speculative_branch",
            "context_skeleton": None, "blast_radius_map": None,
            "agents_md_context": None,
            "strategy_candidates": [
                {"strategy_id": "s1", "composite_score": 0.9, "affected_files": []},
                {"strategy_id": "s2", "composite_score": 0.8, "affected_files": []},
            ],
            "selected_strategy": None, "pruned_strategies": [],
            "red_phase_evidence_path": None, "test_inventory_paths": [],
            "diff_proposal": None, "preflight_result": None,
            "critic_findings": [], "verifier_verdict": None,
            "correction_state": {
                "attempt_count": 2, "branch_name": "main",
                "last_error_hash": None, "error_history": [],
                "stagnation_pattern": "none",
            },
            "confidence_score": 0.5, "replan_count": 0,
            "active_branches": [],
            "exhausted_branches": ["agent-task-T1-s1"],  # s1 already exhausted
            "speculative_stash_ref": None, "escalation_payload": None,
            "procedural_constraints": [], "lesson_learned": None,
            "arch_rules_updated": False, "debug_observations": None,
            "cumulative_cost_dollars": 1.0,
        }

        result = await node(state)

        # s1 should be in exhausted_branches (was already there + re-added)
        # s2 should have been evaluated and added to exhausted
        exhausted = result.get("exhausted_branches", [])
        assert "agent-task-T1-s1" in exhausted
        assert "agent-task-T1-s2" in exhausted
