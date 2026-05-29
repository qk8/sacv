"""
tests/unit/test_confidence_score.py
=====================================
Unit tests for compute_confidence_score — pure function, no I/O.
"""
from __future__ import annotations
import pytest
from sacv.orchestration.edges import compute_confidence_score
from sacv.orchestration.config import WorkflowConfig, StagnationConfig


def _s(attempt=0, stagnation="none", risk=0.0, critical_count=0):
    findings = [
        {"severity":"critical","critic":"security","file":"X.java","line":1,
         "rule_id":"r","message":"m","resolution_hint":"h"}
        for _ in range(critical_count)
    ]
    return {
        "session_id":"t","task_id":"t","task_description":"",
        "project_mode":"greenfield","module_type":"backend-domain","check_profile":"standard",
        "current_phase":"verifier",
        "context_skeleton":None,"blast_radius_map":{"risk_score":risk} if risk else None,
        "agents_md_context":None,
        "strategy_candidates":[],"selected_strategy":None,"pruned_strategies":[],
        "red_phase_evidence_path":None,"test_inventory_paths":[],
        "diff_proposal":None,"preflight_result":None,
        "critic_findings":findings,"verifier_verdict":None,
        "correction_state":{"attempt_count":attempt,"branch_name":None,
                            "last_error_hash":None,"error_history":[],"stagnation_pattern":stagnation},
        "confidence_score":1.0,"replan_count":0,
        "active_branches":[],"exhausted_branches":[],"escalation_payload":None,
        "procedural_constraints":[],"lesson_learned":None,"arch_rules_updated":False,
    }

cfg3 = WorkflowConfig(max_self_correction_cycles=3)


class TestComputeConfidenceScore:

    def test_fresh_state_scores_one(self):
        assert compute_confidence_score(_s(0), cfg3) == pytest.approx(1.0)

    def test_one_attempt_reduces_score(self):
        s = compute_confidence_score(_s(1), cfg3)
        assert 0.6 < s < 0.9

    def test_max_attempts_scores_zero(self):
        s = compute_confidence_score(_s(3), cfg3)
        assert s == pytest.approx(0.0)

    def test_stagnation_penalises_heavily(self):
        no_stag = compute_confidence_score(_s(0, stagnation="none"), cfg3)
        stag    = compute_confidence_score(_s(0, stagnation="semantic"), cfg3)
        assert stag < no_stag
        assert stag == pytest.approx(no_stag - 0.40)

    def test_blast_radius_risk_penalises(self):
        low  = compute_confidence_score(_s(0, risk=0.0), cfg3)
        high = compute_confidence_score(_s(0, risk=1.0), cfg3)
        assert high < low
        assert high == pytest.approx(low - 0.30)

    def test_one_critical_finding_penalises(self):
        no_crit = compute_confidence_score(_s(0, critical_count=0), cfg3)
        one     = compute_confidence_score(_s(0, critical_count=1), cfg3)
        assert one == pytest.approx(no_crit - 0.10)

    def test_critical_findings_capped_at_three(self):
        three = compute_confidence_score(_s(0, critical_count=3), cfg3)
        ten   = compute_confidence_score(_s(0, critical_count=10), cfg3)
        assert three == ten   # both capped at 0.30 penalty

    def test_score_never_goes_below_zero(self):
        s = compute_confidence_score(_s(3, stagnation="semantic", risk=1.0, critical_count=10), cfg3)
        assert s == 0.0

    def test_below_threshold_triggers_early_escalation(self):
        """Verify the threshold constant is respected by edges.route_after_verifier."""
        from sacv.orchestration.edges import route_after_verifier
        state = _s(attempt=2, stagnation="semantic", risk=0.8, critical_count=2)
        state["verifier_verdict"] = {
            "test_result":"FAIL","diagnostic":"FIX_IMPL",
            "phase1_passed":False,"phase2_passed":False,
            "test_failures":[{"message":"err"}],
            "performance_delta":None,"visual_diff_result":None,
            "critic_findings":[],"docker_exit_code":1,
        }
        # With attempt=2 and low confidence, should escalate immediately
        cfg_low = WorkflowConfig(
            max_self_correction_cycles=3,
            confidence_escalation_threshold=0.50,
        )
        result = route_after_verifier(state, cfg_low)
        assert result == "hitl_escalation"
