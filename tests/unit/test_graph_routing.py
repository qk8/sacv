"""Unit tests for all conditional edge functions — pure, no I/O."""
from __future__ import annotations
import pytest
from sacv.orchestration.edges import (
    route_after_verifier, route_after_value_node, route_after_tdd_gate,
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

    def test_ambiguous_attempt_2_still_debugger(self):
        """AMBIGUOUS preferred over speculative_branch."""
        s = _s(verifier_verdict=_verdict("FAIL", "AMBIGUOUS"), correction_state=_corr(2))
        assert route_after_verifier(s, cfg3) == "intelligent_debugger"

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

    def test_none_verdict_raises(self):
        with pytest.raises(ValueError):
            route_after_verifier(_s(verifier_verdict=None), cfg3)


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
        assert route_after_tdd_gate(_s(red_phase_evidence_path=None)) == "tdd_gate"

    def test_with_evidence_proceeds(self):
        assert route_after_tdd_gate(_s(red_phase_evidence_path="/p/e.json")) == "actor"
