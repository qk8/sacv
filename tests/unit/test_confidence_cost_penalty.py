"""
tests/unit/test_confidence_cost_penalty.py
===========================================
Unit tests for cost-based penalty in compute_confidence_score.

Issue HIGH-001: compute_confidence_score has no awareness of cumulative cost.
The routing uses a hard $80 gate in route_after_verifier, but there is no
graduated penalty — an agent could hit $79.50 with full confidence and get
one more expensive LLM call before the hard stop.

Tests verify that cumulative_cost_dollars reduces confidence as it approaches
warning_dollar ($50), and the reduction is factored into route_after_verifier
decisions.
"""
from __future__ import annotations

import pytest
from sacv.orchestration.edges import (
    compute_confidence_score, route_after_verifier,
)
from sacv.orchestration.config import WorkflowConfig


def _s(
    attempt=0,
    stagnation="none",
    risk=0.0,
    critical_count=0,
    cumulative_cost_dollars=0.0,
    confidence_threshold=0.25,
):
    findings = [
        {
            "severity": "critical",
            "critic": "security",
            "file": "X.java",
            "line": 1,
            "rule_id": "r",
            "message": "m",
            "resolution_hint": "h",
        }
        for _ in range(critical_count)
    ]
    return {
        "session_id": "t",
        "task_id": "t",
        "task_description": "",
        "project_mode": "greenfield",
        "module_type": "backend-domain",
        "current_phase": "verifier",
        "context_skeleton": None,
        "blast_radius_map": {"risk_score": risk} if risk else None,
        "agents_md_context": None,
        "strategy_candidates": [],
        "selected_strategy": None,
        "pruned_strategies": [],
        "red_phase_evidence_path": None,
        "test_inventory_paths": [],
        "diff_proposal": None,
        "preflight_result": None,
        "critic_findings": findings,
        "verifier_verdict": None,
        "correction_state": {
            "attempt_count": attempt,
            "branch_name": None,
            "last_error_hash": None,
            "error_history": [],
            "stagnation_pattern": stagnation,
        },
        "confidence_score": 1.0,
        "replan_count": 0,
        "active_branches": [],
        "exhausted_branches": [],
        "escalation_payload": None,
        "procedural_constraints": [],
        "lesson_learned": None,
        "arch_rules_updated": False,
        "cumulative_cost_dollars": cumulative_cost_dollars,
    }


class TestComputeConfidenceScoreWithCost:

    def test_zero_cost_does_not_penalise(self):
        """When cumulative_cost is 0, confidence should be unaffected by cost."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=3,
            confidence_escalation_threshold=0.25,
        )
        s = _s(attempt=0, cumulative_cost_dollars=0.0)
        score = compute_confidence_score(s, cfg)
        # Should be 1.0 (no penalties) — cost penalty should be 0
        assert score == pytest.approx(1.0)

    def test_cost_below_warning_threshold_no_penalty(self):
        """Cost below $50 (warning_dollar) should not penalise confidence."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=3,
            confidence_escalation_threshold=0.25,
        )
        s = _s(attempt=0, cumulative_cost_dollars=40.0)
        score = compute_confidence_score(s, cfg)
        assert score == pytest.approx(1.0)

    def test_cost_above_warning_threshold_reduces_confidence(self):
        """Cost just above $50 (warning_dollar) should reduce confidence below 1.0."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=3,
            confidence_escalation_threshold=0.25,
        )
        s = _s(attempt=0, cumulative_cost_dollars=51.0)
        score = compute_confidence_score(s, cfg)
        assert score < 1.0
        assert score > 0.0

    def test_cost_near_critical_reduces_confidence_more(self):
        """Cost at $75 (near critical $80) should reduce confidence more than $50."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=3,
            confidence_escalation_threshold=0.25,
        )
        s_50 = _s(attempt=0, cumulative_cost_dollars=50.0)
        s_75 = _s(attempt=0, cumulative_cost_dollars=75.0)
        score_50 = compute_confidence_score(s_50, cfg)
        score_75 = compute_confidence_score(s_75, cfg)
        assert score_75 < score_50

    def test_cost_above_critical_scores_zero(self):
        """Cost at or above $80 (critical_dollar) should score 0.0."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=3,
            confidence_escalation_threshold=0.25,
        )
        s = _s(attempt=0, cumulative_cost_dollars=80.0)
        score = compute_confidence_score(s, cfg)
        assert score == pytest.approx(0.0)

    def test_cost_above_critical_scores_zero(self):
        """Cost above $80 should also score 0.0."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=3,
            confidence_escalation_threshold=0.25,
        )
        s = _s(attempt=0, cumulative_cost_dollars=90.0)
        score = compute_confidence_score(s, cfg)
        assert score == pytest.approx(0.0)

    def test_cost_penalty_composes_with_other_penalties(self):
        """Cost + stagnation + blast radius should all compose correctly."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=3,
            confidence_escalation_threshold=0.25,
        )
        s = _s(
            attempt=1,
            stagnation="semantic",
            risk=1.0,
            cumulative_cost_dollars=60.0,
        )
        score = compute_confidence_score(s, cfg)
        # attempt_penalty ~= 1/3 = 0.333
        # stagnation_penalty = 0.40
        # blast_penalty = 0.30
        # cost_penalty > 0 (between 50 and 80)
        # Total penalty > 1.0, so score should be 0.0
        assert score == pytest.approx(0.0)

    def test_cost_penalty_is_linear_between_warning_and_critical(self):
        """Cost penalty should scale linearly between warning_dollar and critical_dollar."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=3,
            confidence_escalation_threshold=0.25,
        )
        # At $50: penalty should be some value P
        # At $65 (midpoint): penalty should be ~2*P
        # At $80: penalty should be 1.0 (full)
        s_50 = _s(attempt=0, cumulative_cost_dollars=50.0)
        s_65 = _s(attempt=0, cumulative_cost_dollars=65.0)
        s_80 = _s(attempt=0, cumulative_cost_dollars=80.0)

        score_50 = compute_confidence_score(s_50, cfg)
        score_65 = compute_confidence_score(s_65, cfg)
        score_80 = compute_confidence_score(s_80, cfg)

        assert score_80 == pytest.approx(0.0)
        assert score_65 < score_50
        assert score_65 > score_80

    def test_custom_token_budget_respected(self):
        """Custom warning_dollar and critical_dollar values should be used."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=3,
            confidence_escalation_threshold=0.25,
            token_budget=WorkflowConfig().token_budget.__class__(
                cost_per_m_input=5.0,
                cost_per_m_output=30.0,
                critical_dollar=100.0,
                warning_dollar=75.0,
            ),
        )
        # At $70: below warning, no penalty
        s_70 = _s(attempt=0, cumulative_cost_dollars=70.0)
        score_70 = compute_confidence_score(s_70, cfg)
        assert score_70 == pytest.approx(1.0)

        # At $80: between warning(75) and critical(100), should be penalised
        s_80 = _s(attempt=0, cumulative_cost_dollars=80.0)
        score_80 = compute_confidence_score(s_80, cfg)
        assert score_80 < score_70


class TestRouteAfterVerifierWithCostPenalty:

    def test_low_cost_confidence_allows_retry(self):
        """Low cost + high confidence → actor retry."""
        cfg = WorkflowConfig(max_self_correction_cycles=3)
        s = _s(
            attempt=1,
            cumulative_cost_dollars=10.0,
        )
        s["verifier_verdict"] = {
            "test_result": "FAIL",
            "diagnostic": "FIX_IMPL",
            "phase1_passed": False,
            "phase2_passed": False,
            "test_failures": [{"message": "err"}],
            "performance_delta": None,
            "visual_diff_result": None,
            "critic_findings": [],
            "docker_exit_code": 1,
        }
        s["confidence_score"] = compute_confidence_score(s, cfg)
        assert route_after_verifier(s, cfg) == "actor"

    def test_high_cost_confidence_triggers_early_escalation(self):
        """High cost reduces confidence enough to trigger escalation at attempt 1."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=3,
            confidence_escalation_threshold=0.50,  # generous threshold
        )
        s = _s(
            attempt=1,
            cumulative_cost_dollars=70.0,  # near critical
        )
        s["verifier_verdict"] = {
            "test_result": "FAIL",
            "diagnostic": "FIX_IMPL",
            "phase1_passed": False,
            "phase2_passed": False,
            "test_failures": [{"message": "err"}],
            "performance_delta": None,
            "visual_diff_result": None,
            "critic_findings": [],
            "docker_exit_code": 1,
        }
        s["confidence_score"] = compute_confidence_score(s, cfg)
        # Confidence should be very low due to cost + attempt
        assert s["confidence_score"] < cfg.confidence_escalation_threshold
        assert route_after_verifier(s, cfg) == "hitl_escalation"
