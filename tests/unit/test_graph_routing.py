"""Unit tests for all conditional edge functions — pure, no I/O."""
from __future__ import annotations
import pytest
from sacv.orchestration.edges import (
    route_after_verifier, route_after_value_node, route_after_tdd_gate,
    route_after_actor, route_after_replan,
)
from sacv.orchestration.config import WorkflowConfig


def _verdict(test_result="FAIL", diagnostic="FIX_IMPL"):
    return {
        "test_result": test_result, "diagnostic": diagnostic,
        "phase1_passed": test_result == "PASS", "phase2_passed": test_result == "PASS",
        "test_failures": [] if test_result == "PASS" else [{"message": "err"}],
        "performance_delta": None, "visual_diff_result": None,
        "critic_findings": [], "docker_exit_code": 0 if test_result == "PASS" else 1,
    }


def _cfg():
    return WorkflowConfig()


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


def _corr(attempt, stagnation="none"):
    return {
        "attempt_count": attempt, "branch_name": "b",
        "last_error_hash": None, "error_history": [], "stagnation_pattern": stagnation,
    }


cfg3 = WorkflowConfig(max_self_correction_cycles=3)


class TestRouteAfterVerifier:

    def test_pass_routes_to_memory(self):
        s = _s(verifier_verdict=_verdict("PASS", "PASS"))
        assert route_after_verifier(s, cfg3) == "memory_consolidation"

    def test_fix_impl_attempt_1_retries_actor(self):
        s = _s(verifier_verdict=_verdict("FAIL", "FIX_IMPL"), correction_state=_corr(1))
        assert route_after_verifier(s, cfg3) == "actor"

    def test_fix_impl_attempt_2_speculative(self):
        s = _s(verifier_verdict=_verdict("FAIL", "FIX_IMPL"), correction_state=_corr(2))
        assert route_after_verifier(s, cfg3) == "speculative_branch"

    def test_fix_impl_attempt_3_hitl(self):
        s = _s(verifier_verdict=_verdict("FAIL", "FIX_IMPL"), correction_state=_corr(3))
        assert route_after_verifier(s, cfg3) == "hitl_escalation"

    def test_ambiguous_routes_to_debugger(self):
        """AMBIGUOUS diagnostic → intelligent_debugger (key new routing)."""
        s = _s(verifier_verdict=_verdict("FAIL", "AMBIGUOUS"), correction_state=_corr(1))
        assert route_after_verifier(s, cfg3) == "intelligent_debugger"

    def test_ambiguous_attempt_2_routes_to_speculative(self):
        """After first debug, AMBIGUOUS falls through to speculative_branch (ARCH-004)."""
        s = _s(verifier_verdict=_verdict("FAIL", "AMBIGUOUS"), correction_state=_corr(2))
        assert route_after_verifier(s, cfg3) == "speculative_branch"

    def test_ambiguous_at_max_hitl(self):
        s = _s(verifier_verdict=_verdict("FAIL", "AMBIGUOUS"), correction_state=_corr(3))
        assert route_after_verifier(s, cfg3) == "hitl_escalation"

    def test_low_confidence_escalates_early(self):
        cfg_tight = WorkflowConfig(
            max_self_correction_cycles=5, confidence_escalation_threshold=0.50,
        )
        s = _s(
            verifier_verdict=_verdict("FAIL", "FIX_IMPL"),
            correction_state=_corr(1, stagnation="semantic"),
        )
        s["confidence_score"] = 0.10
        assert route_after_verifier(s, cfg_tight) == "hitl_escalation"

    def test_custom_max_cycles(self):
        cfg1 = WorkflowConfig(max_self_correction_cycles=1)
        s    = _s(verifier_verdict=_verdict("FAIL", "FIX_IMPL"), correction_state=_corr(1))
        assert route_after_verifier(s, cfg1) == "hitl_escalation"

    def test_none_verdict_routes_to_hitl(self):
        """Missing verifier_verdict now routes to HITL instead of crashing."""
        assert route_after_verifier(_s(verifier_verdict=None), cfg3) == "hitl_escalation"


class TestRouteAfterValueNode:

    def test_empty_candidates_to_hitl(self):
        assert route_after_value_node(_s(strategy_candidates=[])) == "hitl_escalation"

    def test_with_candidates_to_tdd(self):
        c = {"strategy_id": "s1", "description": "x", "affected_files": [],
             "token_depth_score": 0.8, "collision_score": 0.8,
             "blast_radius_score": 0.8, "composite_score": 0.8}
        assert route_after_value_node(_s(strategy_candidates=[c])) == "tdd_gate"


class TestRouteAfterTddGate:

    def test_no_evidence_loops_back(self):
        assert route_after_tdd_gate(_s(red_phase_evidence_path=None), _cfg()) == "tdd_gate"

    def test_with_evidence_proceeds(self):
        assert route_after_tdd_gate(_s(red_phase_evidence_path="/p/e.json"), _cfg()) == "actor"

    def test_max_attempts_routes_to_hitl(self):
        """tdd_gate_attempts >= 3 → hitl_escalation (BUG-004 context)."""
        s = _s(red_phase_evidence_path=None, tdd_gate_attempts=3)
        assert route_after_tdd_gate(s, _cfg()) == "hitl_escalation"

    def test_one_below_max_still_retries(self):
        s = _s(red_phase_evidence_path=None, tdd_gate_attempts=2)
        assert route_after_tdd_gate(s, _cfg()) == "tdd_gate"


class TestRouteAfterActor:

    def test_with_diff_routes_to_preflight(self):
        s = _s(diff_proposal={"strategy_id": "s1", "diffs": []})
        assert route_after_actor(s, cfg3) == "preflight_node"

    def test_no_diff_self_loops(self):
        s = _s(diff_proposal=None, empty_diff_retries=0)
        assert route_after_actor(s, cfg3) == "actor"

    def test_empty_diff_retries_exhausted_routes_to_hitl(self):
        s = _s(diff_proposal=None, empty_diff_retries=3)
        assert route_after_actor(s, cfg3) == "hitl_escalation"

    def test_empty_diff_below_max_self_loops(self):
        s = _s(diff_proposal=None, empty_diff_retries=2)
        assert route_after_actor(s, cfg3) == "actor"

    def test_stagnation_routes_to_hitl(self):
        s = _s(
            diff_proposal={"strategy_id": "s1", "diffs": []},
            correction_state=_corr(1, stagnation="semantic"),
        )
        assert route_after_actor(s, cfg3) == "hitl_escalation"

    def test_stagnation_takes_priority_over_no_diff(self):
        """Stagnation detection overrides empty-diff retry."""
        s = _s(
            diff_proposal=None, empty_diff_retries=0,
            correction_state=_corr(1, stagnation="iteration"),
        )
        assert route_after_actor(s, cfg3) == "hitl_escalation"


class TestRouteAfterReplan:

    def test_with_candidates_routes_to_tdd(self):
        c = {"strategy_id": "r1", "description": "x", "affected_files": [],
             "token_depth_score": 0.8, "collision_score": 0.8,
             "blast_radius_score": 0.8, "composite_score": 0.8}
        s = _s(strategy_candidates=[c])
        assert route_after_replan(s) == "tdd_gate"

    def test_empty_candidates_routes_to_hitl(self):
        s = _s(strategy_candidates=[])
        assert route_after_replan(s) == "hitl_escalation"

    def test_none_candidates_routes_to_hitl(self):
        s = _s(strategy_candidates=None)
        assert route_after_replan(s) == "hitl_escalation"
