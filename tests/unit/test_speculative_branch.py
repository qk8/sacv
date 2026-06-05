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
