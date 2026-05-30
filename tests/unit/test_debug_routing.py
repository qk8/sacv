"""
tests/unit/test_debug_routing.py
=================================
Unit tests for the new AMBIGUOUS → intelligent_debugger routing.
Pure function tests — no I/O.
"""
from __future__ import annotations
import pytest
from sacv.orchestration.edges import route_after_verifier
from sacv.orchestration.config import WorkflowConfig


def _s(diagnostic="FIX_IMPL", attempt=1, stagnation="none", **kw):
    verdict = {
        "test_result": "FAIL" if diagnostic != "PASS" else "PASS",
        "diagnostic": diagnostic,
        "phase1_passed": diagnostic == "PASS",
        "phase2_passed": diagnostic == "PASS",
        "test_failures": [{"message": "err"}] if diagnostic != "PASS" else [],
        "performance_delta": None, "visual_diff_result": None,
        "critic_findings": [], "docker_exit_code": 1 if diagnostic != "PASS" else 0,
    }
    base = {
        "session_id": "t", "task_id": "t", "task_description": "",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "context_skeleton": None, "blast_radius_map": None, "agents_md_context": None,
        "strategy_candidates": [], "selected_strategy": None, "pruned_strategies": [],
        "red_phase_evidence_path": None, "test_inventory_paths": [],
        "diff_proposal": None, "preflight_result": None,
        "critic_findings": [], "verifier_verdict": verdict,
        "debug_observations": None,
        "correction_state": {
            "attempt_count": attempt, "branch_name": "b",
            "last_error_hash": None, "error_history": [],
            "stagnation_pattern": stagnation,
        },
        "confidence_score": 0.8, "replan_count": 0,
        "active_branches": [], "exhausted_branches": [],
        "escalation_payload": None, "procedural_constraints": [],
        "lesson_learned": None, "arch_rules_updated": False,
    }
    base.update(kw)
    return base


cfg = WorkflowConfig(max_self_correction_cycles=3)


class TestAmbiguousRouting:

    def test_ambiguous_routes_to_debugger(self):
        """AMBIGUOUS diagnostic → intelligent_debugger (not blind actor retry)."""
        s = _s(diagnostic="AMBIGUOUS", attempt=1)
        assert route_after_verifier(s, cfg) == "intelligent_debugger"

    def test_fix_impl_routes_to_actor(self):
        """FIX_IMPL still routes to actor on first attempts."""
        s = _s(diagnostic="FIX_IMPL", attempt=1)
        assert route_after_verifier(s, cfg) == "actor"

    def test_fix_test_routes_to_actor(self):
        """FIX_TEST routes to actor."""
        s = _s(diagnostic="FIX_TEST", attempt=1)
        assert route_after_verifier(s, cfg) == "actor"

    def test_ambiguous_at_max_attempts_goes_to_hitl(self):
        """Even AMBIGUOUS escalates to HITL at max cycles."""
        s = _s(diagnostic="AMBIGUOUS", attempt=3)
        assert route_after_verifier(s, cfg) == "hitl_escalation"

    def test_ambiguous_with_low_confidence_goes_to_hitl(self):
        """AMBIGUOUS + stagnation → low confidence → HITL regardless of attempt count."""
        cfg_tight = WorkflowConfig(
            max_self_correction_cycles=5,
            confidence_escalation_threshold=0.5,
        )
        s = _s(diagnostic="AMBIGUOUS", attempt=1, stagnation="semantic")
        s["confidence_score"] = 0.10
        assert route_after_verifier(s, cfg_tight) == "hitl_escalation"

    def test_pass_routes_to_memory(self):
        verdict = {
            "test_result": "PASS", "diagnostic": "PASS",
            "phase1_passed": True, "phase2_passed": True,
            "test_failures": [], "performance_delta": None,
            "visual_diff_result": None, "critic_findings": [],
            "docker_exit_code": 0,
        }
        s = _s(diagnostic="PASS", attempt=1)
        s["verifier_verdict"] = verdict
        assert route_after_verifier(s, cfg) == "memory_consolidation"

    def test_fix_impl_attempt_2_goes_to_speculative(self):
        s = _s(diagnostic="FIX_IMPL", attempt=2)
        assert route_after_verifier(s, cfg) == "speculative_branch"

    def test_ambiguous_attempt_2_still_goes_to_debugger(self):
        """AMBIGUOUS at attempt 2: prefer debugger over speculative branching."""
        s = _s(diagnostic="AMBIGUOUS", attempt=2)
        assert route_after_verifier(s, cfg) == "intelligent_debugger"

    def test_none_verdict_raises(self):
        s = _s()
        s["verifier_verdict"] = None
        with pytest.raises(ValueError):
            route_after_verifier(s, cfg)
