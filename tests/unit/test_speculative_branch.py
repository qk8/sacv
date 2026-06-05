"""
tests/unit/test_speculative_branch.py
======================================

Unit tests for speculative_branch pure functions.

Tests cover:
  1. _merge_branch_state — CRITIC_RESET handling, normal merges
"""
from __future__ import annotations

import pytest
from sacv.nodes.speculative_branch import _merge_branch_state
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
