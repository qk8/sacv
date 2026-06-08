"""
tests/unit/test_route_after_actor_config.py
=============================================
Tests that route_after_actor respects configurable empty-diff retry limits.
"""
from __future__ import annotations

import pytest
from sacv.orchestration.edges import route_after_actor
from sacv.orchestration.config import WorkflowConfig


def _s(**kw):
    base = {
        "session_id": "t", "task_id": "t", "task_description": "",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "context_skeleton": None, "blast_radius_map": None, "agents_md_context": None,
        "strategy_candidates": [], "selected_strategy": None, "pruned_strategies": [],
        "red_phase_evidence_path": None, "test_inventory_paths": [],
        "diff_proposal": None, "preflight_result": None,
        "critic_findings": [], "verifier_verdict": None, "debug_observations": None,
        "correction_state": {
            "attempt_count": 0, "branch_name": None,
            "last_error_hash": None, "error_history": [], "stagnation_pattern": "none",
        },
        "confidence_score": 1.0, "replan_count": 0,
        "active_branches": [], "exhausted_branches": [],
        "escalation_payload": None, "procedural_constraints": [],
        "lesson_learned": None, "arch_rules_updated": False,
    }
    base.update(kw)
    return base


class TestRouteAfterActorEmptyDiffConfig:

    def test_empty_diff_retries_below_config_self_loops(self):
        """When retries < config limit → self-loop to actor."""
        cfg = WorkflowConfig(max_self_correction_cycles=3, max_empty_diff_retries=5)
        s = _s(diff_proposal=None, empty_diff_retries=3)
        assert route_after_actor(s, cfg) == "actor"

    def test_empty_diff_retries_at_config_limit_routes_to_hitl(self):
        """When retries == config limit → hitl_escalation."""
        cfg = WorkflowConfig(max_self_correction_cycles=3, max_empty_diff_retries=5)
        s = _s(diff_proposal=None, empty_diff_retries=5)
        assert route_after_actor(s, cfg) == "hitl_escalation"

    def test_empty_diff_retries_exceeding_config_limit_routes_to_hitl(self):
        """When retries > config limit → hitl_escalation."""
        cfg = WorkflowConfig(max_self_correction_cycles=3, max_empty_diff_retries=5)
        s = _s(diff_proposal=None, empty_diff_retries=10)
        assert route_after_actor(s, cfg) == "hitl_escalation"

    def test_config_limit_lower_than_default_hardcoded(self):
        """With max_empty_diff_retries=1, escalation happens after 1 retry."""
        cfg = WorkflowConfig(max_self_correction_cycles=3, max_empty_diff_retries=1)
        s = _s(diff_proposal=None, empty_diff_retries=1)
        assert route_after_actor(s, cfg) == "hitl_escalation"

    def test_config_limit_higher_allows_more_retries(self):
        """With max_empty_diff_retries=10, retry 9 times before escalation."""
        cfg = WorkflowConfig(max_self_correction_cycles=3, max_empty_diff_retries=10)
        s = _s(diff_proposal=None, empty_diff_retries=9)
        assert route_after_actor(s, cfg) == "actor"

    def test_default_config_uses_three_retries(self):
        """Default config should match the previous hardcoded limit of 3."""
        cfg = WorkflowConfig()
        s = _s(diff_proposal=None, empty_diff_retries=3)
        assert route_after_actor(s, cfg) == "hitl_escalation"
        s2 = _s(diff_proposal=None, empty_diff_retries=2)
        assert route_after_actor(s2, cfg) == "actor"

    def test_stagnation_takes_priority_over_configured_empty_diff(self):
        """Stagnation detection overrides empty-diff retry even with high config limit."""
        cfg = WorkflowConfig(max_self_correction_cycles=3, max_empty_diff_retries=10)
        s = _s(
            diff_proposal=None, empty_diff_retries=5,
            correction_state={"attempt_count": 1, "branch_name": "b",
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "semantic"},
        )
        assert route_after_actor(s, cfg) == "hitl_escalation"
