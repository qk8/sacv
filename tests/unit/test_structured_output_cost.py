"""
tests/unit/test_structured_output_cost.py
==========================================
MED-02: StructuredOutputError.updated_cost propagated in all callers.

Verifies:
  1. actor.py uses exc.updated_cost in StructuredOutputError handler
  2. value_node.py uses exc.updated_cost
  3. replan.py uses exc.updated_cost
  4. tdd_gate.py uses exc.updated_cost
  5. memory_consolidation.py uses exc.updated_cost
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sacv.orchestration.state import WorkflowPhase
from sacv.nodes._structured_output import StructuredOutputError


# ── actor.py ──────────────────────────────────────────────────────────────────


class TestActorCostPropagation:
    """actor.py must use exc.updated_cost in StructuredOutputError handler."""

    def test_uses_exc_updated_cost(self):
        """actor.py uses exc.updated_cost, not state cumulative_cost_dollars."""
        # We verify by checking that the except block references exc.updated_cost
        import inspect
        from sacv.nodes import actor as actor_module

        source = inspect.getsource(actor_module)
        # The except StructuredOutputError block should contain exc.updated_cost
        assert "exc.updated_cost" in source, (
            "actor.py StructuredOutputError handler must use exc.updated_cost"
        )


# ── value_node.py ─────────────────────────────────────────────────────────────


class TestValueNodeCostPropagation:
    """value_node.py must use exc.updated_cost."""

    def test_uses_exc_updated_cost(self):
        import inspect
        from sacv.nodes import value_node as vn_module

        source = inspect.getsource(vn_module)
        assert "exc.updated_cost" in source, (
            "value_node.py StructuredOutputError handler must use exc.updated_cost"
        )


# ── replan.py ─────────────────────────────────────────────────────────────────


class TestReplanCostPropagation:
    """replan.py must use exc.updated_cost."""

    def test_uses_exc_updated_cost(self):
        import inspect
        from sacv.nodes import replan as replan_module

        source = inspect.getsource(replan_module)
        assert "exc.updated_cost" in source, (
            "replan.py StructuredOutputError handler must use exc.updated_cost"
        )


# ── tdd_gate.py ───────────────────────────────────────────────────────────────


class TestTddGateCostPropagation:
    """tdd_gate.py must use exc.updated_cost."""

    def test_uses_exc_updated_cost(self):
        import inspect
        from sacv.nodes import tdd_gate as tdd_module

        source = inspect.getsource(tdd_module)
        assert "exc.updated_cost" in source, (
            "tdd_gate.py StructuredOutputError handler must use exc.updated_cost"
        )


# ── memory_consolidation.py ──────────────────────────────────────────────────


class TestMemoryConsolidationCostPropagation:
    """memory_consolidation.py must use exc.updated_cost."""

    def test_uses_exc_updated_cost(self):
        import inspect
        from sacv.nodes import memory_consolidation as mc_module

        source = inspect.getsource(mc_module)
        assert "exc.updated_cost" in source, (
            "memory_consolidation.py StructuredOutputError handler must use exc.updated_cost"
        )
