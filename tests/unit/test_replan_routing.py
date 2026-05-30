"""
tests/unit/test_replan_routing.py
==================================
Unit tests for the new replan routing introduced in edges.py.
Validates that route_after_speculative_branch correctly offers
a replan before falling through to HITL.
"""
from __future__ import annotations
import pytest
from sacv.orchestration.edges import route_after_speculative_branch
from sacv.orchestration.config import WorkflowConfig


def _s(verdict_result, replan_count=0, active=None, exhausted=None):
    verdict = None
    if verdict_result:
        verdict = {
            "test_result": verdict_result,
            "diagnostic":"FIX_IMPL" if verdict_result=="FAIL" else "PASS",
            "phase1_passed": verdict_result=="PASS",
            "phase2_passed": verdict_result=="PASS",
            "test_failures":[], "performance_delta":None,
            "visual_diff_result":None, "critic_findings":[], "docker_exit_code":0,
        }
    return {
        "session_id":"t","task_id":"t","task_description":"",
        "project_mode":"greenfield","module_type":"backend-domain",
        "context_skeleton":None,"blast_radius_map":None,"agents_md_context":None,
        "strategy_candidates":[],"selected_strategy":None,"pruned_strategies":[],
        "red_phase_evidence_path":None,"test_inventory_paths":[],
        "diff_proposal":None,"preflight_result":None,
        "critic_findings":[],"verifier_verdict":verdict,
        "correction_state":{"attempt_count":3,"branch_name":"b",
                            "last_error_hash":None,"error_history":[],"stagnation_pattern":"none"},
        "confidence_score":0.4,"replan_count":replan_count,
        "active_branches":active or [],"exhausted_branches":exhausted or ["b1","b2"],
        "escalation_payload":None,"procedural_constraints":[],
        "lesson_learned":None,"arch_rules_updated":False,
    }


cfg = WorkflowConfig(max_replan_attempts=1)


class TestRouteAfterSpeculativeBranch:

    def test_pass_routes_to_memory(self):
        assert route_after_speculative_branch(_s("PASS"), cfg) == "memory_consolidation"

    def test_first_all_fail_routes_to_replan(self):
        """First all-fail: replan_count=0 < max_replan_attempts=1 → replan."""
        assert route_after_speculative_branch(_s("FAIL", replan_count=0), cfg) == "replan"

    def test_second_all_fail_routes_to_hitl(self):
        """Second all-fail: replan_count=1 >= max_replan_attempts=1 → HITL."""
        assert route_after_speculative_branch(_s("FAIL", replan_count=1), cfg) == "hitl_escalation"

    def test_zero_replan_budget_goes_directly_to_hitl(self):
        cfg0 = WorkflowConfig(max_replan_attempts=0)
        assert route_after_speculative_branch(_s("FAIL", replan_count=0), cfg0) == "hitl_escalation"

    def test_two_replan_budget(self):
        cfg2 = WorkflowConfig(max_replan_attempts=2)
        assert route_after_speculative_branch(_s("FAIL", replan_count=0), cfg2) == "replan"
        assert route_after_speculative_branch(_s("FAIL", replan_count=1), cfg2) == "replan"
        assert route_after_speculative_branch(_s("FAIL", replan_count=2), cfg2) == "hitl_escalation"

    def test_none_verdict_with_no_active_branches_goes_to_replan(self):
        state = _s(None, replan_count=0, active=[], exhausted=["b1"])
        assert route_after_speculative_branch(state, cfg) == "replan"
