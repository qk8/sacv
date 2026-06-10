"""
tests/unit/test_routing_decision_logging.py
==============================================
LOW-05: Routing decisions are logged.

Verifies that each routing function emits a structured log event
with the destination and key context values.
"""
from __future__ import annotations

import pytest

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import WorkflowPhase
from sacv.orchestration.edges import (
    route_after_preflight,
    route_after_actor,
    route_after_value_node,
    route_after_tdd_gate,
    route_after_verifier,
    route_after_speculative_branch,
    route_after_replan,
)


def _state(**kw):
    base = {
        "session_id": "t", "task_id": "task-route-001",
        "session_start_ms": None,
        "task_description": "Add findById",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "current_phase": WorkflowPhase.ACTOR.value,
        "context_skeleton": {}, "blast_radius_map": None,
        "agents_md_context": None,
        "strategy_candidates": [],
        "selected_strategy": None,
        "pruned_strategies": [],
        "red_phase_evidence_path": None, "test_inventory_paths": [],
        "diff_proposal": None, "empty_diff_retries": 0,
        "preflight_result": None,
        "critic_findings": [],
        "verifier_verdict": None,
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
        "check_profile": "standard", "cumulative_cost_dollars": 0.0,
        "skip_tdd_gate": False, "speculative_stash_ref": None,
        "workflow_audit_trail": [],
    }
    base.update(kw)
    return base


class TestRoutingDecisionLogging:
    """Verify routing functions emit structured log events."""

    def test_route_after_preflight_logs_decision(self, caplog):
        """route_after_preflight logs destination and violation status."""
        result = route_after_preflight(_state(preflight_result={"lsp_errors": [{"message": "error"}]}))
        assert result == "actor"
        assert any("route.preflight_decision" in rec.message for rec in caplog.records)

    def test_route_after_actor_logs_stagnation(self, caplog):
        """route_after_actor logs stagnation path."""
        state = _state(correction_state={
            "attempt_count": 0, "branch_name": None,
            "last_error_hash": None, "error_history": [],
            "stagnation_pattern": "iteration",
        })
        result = route_after_actor(state, WorkflowConfig())
        assert result == "hitl_escalation"
        assert any("route.actor_decision" in rec.message for rec in caplog.records)

    def test_route_after_value_node_logs_no_strategies(self, caplog):
        """route_after_value_node logs when no strategies."""
        result = route_after_value_node(_state(strategy_candidates=[]))
        assert result == "hitl_escalation"
        assert any("route.value_node_decision" in rec.message for rec in caplog.records)

    def test_route_after_tdd_gate_logs_attempts_exceeded(self, caplog):
        """route_after_tdd_gate logs when TDD gate attempts exceeded."""
        cfg = WorkflowConfig(max_tdd_gate_attempts=3)
        state = _state(tdd_gate_attempts=3)
        result = route_after_tdd_gate(state, cfg)
        assert result == "hitl_escalation"
        assert any("route.tdd_gate_decision" in rec.message for rec in caplog.records)

    def test_route_after_verifier_logs_decision(self, caplog):
        """route_after_verifier logs the routing decision with context."""
        cfg = WorkflowConfig()
        state = _state(verifier_verdict={
            "test_result": "FAIL", "diagnostic": "FIX_IMPL",
            "blocked_by_critic": False,
        })
        result = route_after_verifier(state, cfg)
        assert result == "actor"
        assert any("route.verifier_decision" in rec.message for rec in caplog.records)

    def test_route_after_speculative_branch_logs_decision(self, caplog):
        """route_after_speculative_branch logs the routing decision."""
        cfg = WorkflowConfig(max_replan_attempts=1)
        state = _state(
            verifier_verdict={"test_result": "FAIL"},
            replan_count=1,
        )
        result = route_after_speculative_branch(state, cfg)
        assert result == "hitl_escalation"
        assert any("route.speculative_branch_decision" in rec.message
                   for rec in caplog.records)

    def test_route_after_replan_logs_decision(self, caplog):
        """route_after_replan logs the routing decision."""
        state = _state(strategy_candidates=[{"strategy_id": "r1"}])
        result = route_after_replan(state)
        assert result == "tdd_gate"
        assert any("route.replan_decision" in rec.message for rec in caplog.records)
