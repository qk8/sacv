"""
tests/unit/test_edges.py
========================
Unit tests for pure edge routing functions not covered by test_graph_routing.py:
- compute_confidence_score
- route_after_preflight
- route_after_speculative_branch
"""
from __future__ import annotations

import pytest

from sacv.orchestration.edges import (
    compute_confidence_score,
    route_after_preflight,
    route_after_speculative_branch,
    route_after_actor,
    route_after_verifier,
    route_after_value_node,
    route_after_tdd_gate,
    route_after_replan,
)
from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import PreflightResult


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


# ── compute_confidence_score ──────────────────────────────────────────────────


class TestComputeConfidenceScore:

    def test_no_penalties_gives_1_0(self):
        """Fresh state with no penalties → confidence = 1.0."""
        s = _s(
            correction_state={"attempt_count": 0, "stagnation_pattern": "none"},
            blast_radius_map=None, critic_findings=[],
            cumulative_cost_dollars=0.0,
        )
        cfg = WorkflowConfig(
            max_self_correction_cycles=5,
            token_budget=WorkflowConfig().token_budget,
        )
        assert compute_confidence_score(s, cfg) == pytest.approx(1.0, abs=1e-9)

    def test_attempt_penalty_scales_with_cycles(self):
        """Each attempt reduces confidence proportionally."""
        cfg = WorkflowConfig(max_self_correction_cycles=5)
        s = _s(
            correction_state={"attempt_count": 2, "stagnation_pattern": "none"},
            blast_radius_map=None, critic_findings=[],
            cumulative_cost_dollars=0.0,
        )
        score = compute_confidence_score(s, cfg)
        # attempt_penalty = min(1.0, 2/5) = 0.4
        assert score == pytest.approx(0.6, abs=1e-9)

    def test_attempt_at_max_gives_0_5(self):
        """attempt == max_cycles → attempt_penalty = 1.0 → score = 0.0."""
        cfg = WorkflowConfig(max_self_correction_cycles=3)
        s = _s(
            correction_state={"attempt_count": 3, "stagnation_pattern": "none"},
            blast_radius_map=None, critic_findings=[],
            cumulative_cost_dollars=0.0,
        )
        assert compute_confidence_score(s, cfg) == pytest.approx(0.0, abs=1e-9)

    def test_stagnation_penalty(self):
        """Non-none stagnation adds 0.40 penalty."""
        cfg = WorkflowConfig(max_self_correction_cycles=10)
        s = _s(
            correction_state={"attempt_count": 0, "stagnation_pattern": "semantic"},
            blast_radius_map=None, critic_findings=[],
            cumulative_cost_dollars=0.0,
        )
        assert compute_confidence_score(s, cfg) == pytest.approx(0.6, abs=1e-9)

    def test_blast_radius_penalty(self):
        """High blast radius score reduces confidence."""
        cfg = WorkflowConfig(max_self_correction_cycles=10)
        s = _s(
            correction_state={"attempt_count": 0, "stagnation_pattern": "none"},
            blast_radius_map={"risk_score": 0.8},
            critic_findings=[], cumulative_cost_dollars=0.0,
        )
        # blast_penalty = 0.8 * 0.30 = 0.24
        assert compute_confidence_score(s, cfg) == pytest.approx(0.76, abs=1e-9)

    def test_critical_critic_penalty(self):
        """Each critical finding adds 0.10, capped at 0.30."""
        cfg = WorkflowConfig(max_self_correction_cycles=10)
        findings = [
            {"severity": "critical", "critic": "security", "message": "x",
             "file": "a.java", "line": 1, "rule_id": "r1", "resolution_hint": "fix"},
            {"severity": "critical", "critic": "security", "message": "y",
             "file": "b.java", "line": 1, "rule_id": "r2", "resolution_hint": "fix"},
            {"severity": "warning", "critic": "style", "message": "z",
             "file": "c.java", "line": 1, "rule_id": "r3", "resolution_hint": "fix"},
        ]
        s = _s(
            correction_state={"attempt_count": 0, "stagnation_pattern": "none"},
            blast_radius_map=None, critic_findings=findings,
            cumulative_cost_dollars=0.0,
        )
        # critic_penalty = min(0.30, 2 * 0.10) = 0.20
        assert compute_confidence_score(s, cfg) == pytest.approx(0.80, abs=1e-9)

    def test_cost_penalty_linear_ramp(self):
        """Cost between warning and critical → linear penalty."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=10,
            token_budget=WorkflowConfig().token_budget,
        )
        warning = cfg.token_budget.warning_dollar
        critical = cfg.token_budget.critical_dollar
        mid_cost = (warning + critical) / 2
        s = _s(
            correction_state={"attempt_count": 0, "stagnation_pattern": "none"},
            blast_radius_map=None, critic_findings=[],
            cumulative_cost_dollars=mid_cost,
        )
        # cost_penalty = (mid - warning) / (critical - warning) = 0.5
        assert compute_confidence_score(s, cfg) == pytest.approx(0.5, abs=1e-9)

    def test_cost_at_critical_gives_full_penalty(self):
        """Cost >= critical → cost_penalty = 1.0 → score = 0.0."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=10,
            token_budget=WorkflowConfig().token_budget,
        )
        s = _s(
            correction_state={"attempt_count": 0, "stagnation_pattern": "none"},
            blast_radius_map=None, critic_findings=[],
            cumulative_cost_dollars=cfg.token_budget.critical_dollar * 2,
        )
        assert compute_confidence_score(s, cfg) == pytest.approx(0.0, abs=1e-9)

    def test_cost_below_warning_no_penalty(self):
        """Cost below warning_dollar → no cost penalty."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=10,
            token_budget=WorkflowConfig().token_budget,
        )
        s = _s(
            correction_state={"attempt_count": 0, "stagnation_pattern": "none"},
            blast_radius_map=None, critic_findings=[],
            cumulative_cost_dollars=cfg.token_budget.warning_dollar * 0.5,
        )
        assert compute_confidence_score(s, cfg) == pytest.approx(1.0, abs=1e-9)

    def test_combined_penalties_clamp_to_zero(self):
        """Multiple penalties together can drive score below 0 → clamped."""
        cfg = WorkflowConfig(
            max_self_correction_cycles=3,
            token_budget=WorkflowConfig().token_budget,
        )
        s = _s(
            correction_state={"attempt_count": 3, "stagnation_pattern": "semantic"},
            blast_radius_map={"risk_score": 0.9},
            critic_findings=[{"severity": "critical", "critic": "s", "message": "m",
                              "file": "x.java", "line": 1, "rule_id": "r", "resolution_hint": "h"}],
            cumulative_cost_dollars=cfg.token_budget.critical_dollar,
        )
        # attempt=1.0 + stagnation=0.4 + blast=0.27 + critic=0.10 + cost=1.0 = 2.77
        assert compute_confidence_score(s, cfg) == pytest.approx(0.0, abs=1e-9)


# ── route_after_preflight ─────────────────────────────────────────────────────


class TestRouteAfterPreflight:

    def test_preflight_passed_routes_to_critics(self):
        """All checks pass → all_critics."""
        result = PreflightResult(
            passed=True, lsp_errors=[], arch_violations=[],
            cross_stack_errors=[], blast_errors=[], duration_ms=100,
        )
        s = _s(preflight_result=result)
        assert route_after_preflight(s) == "all_critics"

    def test_lsp_errors_routes_to_actor(self):
        """LSP compile errors → actor (fix code first)."""
        s = _s(preflight_result=PreflightResult(
            passed=False, lsp_errors=[{"file": "a.java", "line": 1, "code": "E", "message": "err"}],
            arch_violations=[], cross_stack_errors=[], blast_errors=[], duration_ms=100,
        ))
        assert route_after_preflight(s) == "actor"

    def test_arch_violations_routes_to_actor(self):
        """Architecture violations → actor."""
        s = _s(preflight_result=PreflightResult(
            passed=False, lsp_errors=[],
            arch_violations=[{"rule": "r", "source_file": "a", "target_file": "b", "message": "v"}],
            cross_stack_errors=[], blast_errors=[], duration_ms=100,
        ))
        assert route_after_preflight(s) == "actor"

    def test_cross_stack_errors_routes_to_actor(self):
        """Cross-stack type errors → actor."""
        s = _s(preflight_result=PreflightResult(
            passed=False, lsp_errors=[], arch_violations=[],
            cross_stack_errors=[{"file": "a.ts", "line": 1, "code": "E", "message": "m", "source": "cs"}],
            blast_errors=[], duration_ms=100,
        ))
        assert route_after_preflight(s) == "actor"

    def test_blast_errors_routes_to_actor(self):
        """Blast radius errors → actor."""
        s = _s(preflight_result=PreflightResult(
            passed=False, lsp_errors=[], arch_violations=[],
            cross_stack_errors=[], blast_errors=[{"rule": "blast", "message": "m"}],
            duration_ms=100,
        ))
        assert route_after_preflight(s) == "actor"

    def test_none_preflight_routes_to_critics(self):
        """Missing preflight_result → treated as passed."""
        s = _s(preflight_result=None)
        assert route_after_preflight(s) == "all_critics"


# ── route_after_speculative_branch ────────────────────────────────────────────


class TestRouteAfterSpeculativeBranch:

    def test_pass_routes_to_memory(self):
        """A passing branch → memory_consolidation."""
        cfg = WorkflowConfig(max_replan_attempts=3)
        s = _s(
            verifier_verdict={"test_result": "PASS", "diagnostic": "PASS",
                              "phase1_passed": True, "phase2_passed": True,
                              "test_failures": [], "performance_delta": None,
                              "visual_diff_result": None, "docker_exit_code": 0},
        )
        assert route_after_speculative_branch(s, cfg) == "memory_consolidation"

    def test_fail_below_max_replan_routes_to_replan(self):
        """Failed branch with remaining replan attempts → replan."""
        cfg = WorkflowConfig(max_replan_attempts=3)
        s = _s(
            replan_count=1,
            verifier_verdict={"test_result": "FAIL", "diagnostic": "FIX_IMPL",
                              "phase1_passed": False, "phase2_passed": False,
                              "test_failures": [{"message": "err"}],
                              "performance_delta": None, "visual_diff_result": None,
                              "docker_exit_code": 1},
        )
        assert route_after_speculative_branch(s, cfg) == "replan"

    def test_fail_at_max_replan_routes_to_hitl(self):
        """Failed branch with no replan attempts left → hitl_escalation."""
        cfg = WorkflowConfig(max_replan_attempts=2)
        s = _s(
            replan_count=2,
            verifier_verdict={"test_result": "FAIL", "diagnostic": "FIX_IMPL",
                              "phase1_passed": False, "phase2_passed": False,
                              "test_failures": [{"message": "err"}],
                              "performance_delta": None, "visual_diff_result": None,
                              "docker_exit_code": 1},
        )
        assert route_after_speculative_branch(s, cfg) == "hitl_escalation"

    def test_none_verdict_routes_to_replan(self):
        """No verdict (e.g. no strategies) → replan (replan decides next)."""
        cfg = WorkflowConfig(max_replan_attempts=3)
        s = _s(verifier_verdict=None, replan_count=0)
        assert route_after_speculative_branch(s, cfg) == "replan"


# ── route_after_actor ─────────────────────────────────────────────────────────


class TestRouteAfterActor:

    def test_stagnation_routes_to_hitl(self):
        """Stagnation detected → HITL escalation."""
        cfg = WorkflowConfig()
        s = _s(
            correction_state={"attempt_count": 1, "branch_name": None,
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "semantic"},
        )
        assert route_after_actor(s, cfg) == "hitl_escalation"

    def test_no_diff_retries_within_limit(self):
        """No diff, retries < max → actor (self-loop)."""
        cfg = WorkflowConfig(max_empty_diff_retries=3)
        s = _s(diff_proposal=None, empty_diff_retries=1)
        assert route_after_actor(s, cfg) == "actor"

    def test_no_diff_exhausted_retries_routes_to_hitl(self):
        """No diff, retries >= max → HITL escalation."""
        cfg = WorkflowConfig(max_empty_diff_retries=3)
        s = _s(diff_proposal=None, empty_diff_retries=3)
        assert route_after_actor(s, cfg) == "hitl_escalation"

    def test_has_diff_routes_to_preflight(self):
        """Actor produced a diff → preflight_node."""
        cfg = WorkflowConfig()
        s = _s(diff_proposal={"strategy_id": "s1", "diffs": [],
                              "branch_name": "b", "commit_message": "m"})
        assert route_after_actor(s, cfg) == "preflight_node"

    def test_stagnation_takes_priority_over_no_diff(self):
        """Both stagnation and no diff → stagnation wins (HITL)."""
        cfg = WorkflowConfig(max_empty_diff_retries=0)
        s = _s(
            diff_proposal=None, empty_diff_retries=0,
            correction_state={"attempt_count": 1, "branch_name": None,
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "iteration"},
        )
        assert route_after_actor(s, cfg) == "hitl_escalation"


# ── route_after_verifier ──────────────────────────────────────────────────────


class TestRouteAfterVerifier:

    def _verdict(self, test_result="FAIL", diagnostic="FIX_IMPL", **kw):
        base = {
            "test_result": test_result, "diagnostic": diagnostic,
            "phase1_passed": False, "phase2_passed": True,
            "test_failures": [], "performance_delta": None,
            "visual_diff_result": None, "docker_exit_code": 1,
            "playwright_trace_path": None, "otel_trace": None,
            "actuator_snapshot": None, "blocked_by_critic": False,
        }
        base.update(kw)
        return base

    def test_pass_routes_to_memory_consolidation(self):
        """PASS verdict → memory_consolidation."""
        cfg = WorkflowConfig()
        s = _s(verifier_verdict=self._verdict(test_result="PASS", phase1_passed=True,
                                               phase2_passed=True))
        assert route_after_verifier(s, cfg) == "memory_consolidation"

    def test_missing_verdict_routes_to_hitl(self):
        """No verdict → HITL escalation (safety)."""
        cfg = WorkflowConfig()
        s = _s(verifier_verdict=None)
        assert route_after_verifier(s, cfg) == "hitl_escalation"

    def test_fail_low_confidence_routes_to_hitl(self):
        """FAIL + confidence below threshold → HITL escalation."""
        cfg = WorkflowConfig(confidence_escalation_threshold=0.5)
        s = _s(
            verifier_verdict=self._verdict(),
            confidence_score=0.3,
        )
        assert route_after_verifier(s, cfg) == "hitl_escalation"

    def test_fail_first_attempt_routes_to_actor(self):
        """FAIL + attempt 0 → actor (retry with critic feedback)."""
        cfg = WorkflowConfig()
        s = _s(
            correction_state={"attempt_count": 0, "branch_name": None,
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "none"},
            verifier_verdict=self._verdict(),
        )
        assert route_after_verifier(s, cfg) == "actor"

    def test_fail_second_attempt_routes_to_actor(self):
        """FAIL + attempt 1 → actor."""
        cfg = WorkflowConfig()
        s = _s(
            correction_state={"attempt_count": 1, "branch_name": None,
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "none"},
            verifier_verdict=self._verdict(),
        )
        assert route_after_verifier(s, cfg) == "actor"

    def test_fail_max_attempts_routes_to_hitl(self):
        """FAIL + attempt >= max → HITL escalation."""
        cfg = WorkflowConfig(max_self_correction_cycles=3)
        s = _s(
            correction_state={"attempt_count": 3, "branch_name": None,
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "none"},
            verifier_verdict=self._verdict(),
        )
        assert route_after_verifier(s, cfg) == "hitl_escalation"

    def test_ambiguous_first_attempt_routes_to_debugger(self):
        """AMBIGUOUS + attempt <= 1 → intelligent_debugger."""
        cfg = WorkflowConfig()
        s = _s(
            correction_state={"attempt_count": 0, "branch_name": None,
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "none"},
            verifier_verdict=self._verdict(diagnostic="AMBIGUOUS",
                                            phase1_passed=True, phase2_passed=False),
        )
        assert route_after_verifier(s, cfg) == "intelligent_debugger"

    def test_ambiguous_second_attempt_routes_to_debugger(self):
        """AMBIGUOUS + attempt == 1 → intelligent_debugger."""
        cfg = WorkflowConfig()
        s = _s(
            correction_state={"attempt_count": 1, "branch_name": None,
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "none"},
            verifier_verdict=self._verdict(diagnostic="AMBIGUOUS",
                                            phase1_passed=True, phase2_passed=False),
        )
        assert route_after_verifier(s, cfg) == "intelligent_debugger"

    def test_ambiguous_attempt_2_routes_to_speculative(self):
        """AMBIGUOUS + attempt >= 2 → speculative_branch (prevents starvation)."""
        cfg = WorkflowConfig()
        s = _s(
            correction_state={"attempt_count": 2, "branch_name": None,
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "none"},
            verifier_verdict=self._verdict(diagnostic="AMBIGUOUS",
                                            phase1_passed=True, phase2_passed=False),
        )
        assert route_after_verifier(s, cfg) == "speculative_branch"

    def test_budget_exceeded_routes_to_hitl(self):
        """Cost >= critical_dollar → HITL escalation."""
        cfg = WorkflowConfig(token_budget=WorkflowConfig().token_budget)
        s = _s(
            cumulative_cost_dollars=cfg.token_budget.critical_dollar * 2,
            verifier_verdict=self._verdict(),
        )
        assert route_after_verifier(s, cfg) == "hitl_escalation"

    def test_attempt_2_routes_to_speculative(self):
        """FAIL + attempt >= 2 → speculative_branch."""
        cfg = WorkflowConfig()
        s = _s(
            correction_state={"attempt_count": 2, "branch_name": None,
                              "last_error_hash": None, "error_history": [],
                              "stagnation_pattern": "none"},
            verifier_verdict=self._verdict(),
        )
        assert route_after_verifier(s, cfg) == "speculative_branch"


# ── route_after_value_node ────────────────────────────────────────────────────


class TestRouteAfterValueNode:

    def test_no_strategies_routes_to_hitl(self):
        """Empty strategy_candidates → HITL escalation."""
        s = _s(strategy_candidates=[])
        assert route_after_value_node(s) == "hitl_escalation"

    def test_strategies_exist_routes_to_tdd_gate(self):
        """Non-empty strategy_candidates → TDD gate."""
        s = _s(strategy_candidates=[{"strategy_id": "s1"}])
        assert route_after_value_node(s) == "tdd_gate"


# ── route_after_tdd_gate ──────────────────────────────────────────────────────


class TestRouteAfterTddGate:

    def test_evidence_path_routes_to_actor(self):
        """Red phase confirmed → actor."""
        cfg = WorkflowConfig()
        s = _s(red_phase_evidence_path=".workflow/tdd-evidence/test.json")
        assert route_after_tdd_gate(s, cfg) == "actor"

    def test_max_attempts_routes_to_hitl(self):
        """TDD gate attempts >= max → HITL escalation."""
        cfg = WorkflowConfig(max_tdd_gate_attempts=3)
        s = _s(tdd_gate_attempts=3, red_phase_evidence_path=None)
        assert route_after_tdd_gate(s, cfg) == "hitl_escalation"

    def test_below_max_attempts_no_evidence_retries_gate(self):
        """No evidence, attempts < max → retry TDD gate."""
        cfg = WorkflowConfig(max_tdd_gate_attempts=3)
        s = _s(tdd_gate_attempts=1, red_phase_evidence_path=None)
        assert route_after_tdd_gate(s, cfg) == "tdd_gate"


# ── route_after_replan ────────────────────────────────────────────────────────


class TestRouteAfterReplan:

    def test_strategies_exist_routes_to_tdd_gate(self):
        """Replan produced strategies → TDD gate."""
        s = _s(strategy_candidates=[{"strategy_id": "s1"}])
        assert route_after_replan(s) == "tdd_gate"

    def test_no_strategies_routes_to_hitl(self):
        """Replan produced no strategies → HITL escalation."""
        s = _s(strategy_candidates=[])
        assert route_after_replan(s) == "hitl_escalation"
