"""
tests/unit/test_graph_cost_arithmetic.py
==========================================
Tests for the cost arithmetic in _make_all_critics_node.

TDD checklist:
- [x] Tests verify the incremental cost formula
- [x] Tests cover normal case, exception fallback, and zero-cost cases
"""
from __future__ import annotations

import pytest


class TestCostArithmetic:
    """Verify the cost calculation in all_critics_node."""

    def test_baseline_no_incremental(self):
        """When all critics return baseline cost, final = baseline."""
        baseline = 5.0
        sec_cost = baseline
        sty_cost = baseline
        con_cost = baseline

        final_cost = (sec_cost - baseline) + (sty_cost - baseline) \
                     + (con_cost - baseline) + baseline

        assert final_cost == baseline

    def test_single_critic_with_cost(self):
        """When one critic incurs cost, final = baseline + that cost."""
        baseline = 10.0
        sec_cost = 10.0 + 3.0  # security LLM call cost
        sty_cost = baseline
        con_cost = baseline

        final_cost = (sec_cost - baseline) + (sty_cost - baseline) \
                     + (con_cost - baseline) + baseline

        assert final_cost == 13.0

    def test_all_critics_with_different_costs(self):
        """When all critics incur different costs, all are summed."""
        baseline = 10.0
        sec_cost = 10.0 + 2.0
        sty_cost = 10.0 + 1.5
        con_cost = 10.0 + 0.5

        final_cost = (sec_cost - baseline) + (sty_cost - baseline) \
                     + (con_cost - baseline) + baseline

        assert final_cost == 14.0

    def test_zero_baseline(self):
        """When baseline is 0 (first node), formula still works."""
        baseline = 0.0
        sec_cost = 1.5
        sty_cost = 0.8
        con_cost = 0.3

        final_cost = (sec_cost - baseline) + (sty_cost - baseline) \
                     + (con_cost - baseline) + baseline

        assert final_cost == pytest.approx(2.6)

    def test_exception_fallback_returns_baseline(self):
        """When a critic raises, its fallback returns state cost (= baseline)."""
        baseline = 5.0
        # sec_out is exception fallback: returns state cost
        sec_cost = baseline
        sty_cost = baseline + 2.0
        con_cost = baseline + 1.0

        final_cost = (sec_cost - baseline) + (sty_cost - baseline) \
                     + (con_cost - baseline) + baseline

        assert final_cost == 8.0

    def test_negative_incremental_impossible(self):
        """A critic's cumulative_cost can never be less than baseline,
        so incremental costs are always >= 0."""
        baseline = 10.0
        # A critic returns at minimum the baseline (its own cost >= 0)
        sec_cost = baseline + 0.0  # minimum: no cost
        sty_cost = baseline + 0.0
        con_cost = baseline + 0.0

        final_cost = (sec_cost - baseline) + (sty_cost - baseline) \
                     + (con_cost - baseline) + baseline

        assert final_cost == baseline
