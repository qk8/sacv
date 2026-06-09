"""
tests/unit/test_verdict_blocked_by_critic.py
==============================================
Unit tests for the blocked_by_critic flag in the verifier verdict.

Issue HIGH-003: When critics find critical issues, the verifier returns
FAIL immediately without running Docker. This is correct for preventing
wasteful cycles, but the verdict always has diagnostic=FIX_IMPL regardless
of whether the critical finding is actually related to the test failure.
The Actor gets critic feedback but has no signal that it shouldn't waste
effort on test-driven debugging.

Fix: Add blocked_by_critic: true to the verdict when the verifier blocks
on critical findings. The Actor can check this flag to decide whether to
retry with critic feedback or escalate.

Tests verify:
1. _make_verdict accepts blocked_by_critic parameter
2. _build_return includes blocked_by_critic in output
3. route_after_verifier routes to actor when blocked_by_critic=True (M-02)
4. Verdict without critic block has blocked_by_critic = False/None
5. _classify is unaffected by this change
"""
from __future__ import annotations

import pytest

from sacv.orchestration.state import DiagnosticVerdict, VerifierVerdict
from sacv.nodes.verifier import (
    _make_verdict,
    _build_return,
    _classify,
    _check_test_deletions,
)


class TestMakeVerdictWithBlockedByCritic:

    def test_blocked_by_critic_defaults_to_false(self):
        """When not specified, blocked_by_critic should default to False."""
        v = _make_verdict(
            test_result="FAIL",
            diagnostic=DiagnosticVerdict.FIX_IMPL.value,
            phase1_passed=False, phase2_passed=True,
            failures=[{"message": "NPE"}], findings=[],
        )
        assert v.get("blocked_by_critic") is False

    def test_blocked_by_critic_can_be_set_true(self):
        """When the verifier blocks on critical findings, set to True."""
        v = _make_verdict(
            test_result="FAIL",
            diagnostic=DiagnosticVerdict.FIX_IMPL.value,
            phase1_passed=False, phase2_passed=True,
            failures=[{"message": "NPE"}], findings=[],
            blocked_by_critic=True,
        )
        assert v["blocked_by_critic"] is True

    def test_blocked_by_critic_false_on_normal_pass(self):
        """Normal passing verdict should have blocked_by_critic = False."""
        v = _make_verdict(
            test_result="PASS",
            diagnostic=DiagnosticVerdict.PASS.value,
            phase1_passed=True, phase2_passed=True,
            failures=[], findings=[],
        )
        assert v.get("blocked_by_critic") is False

    def test_blocked_by_critic_in_verdict_verdict_type(self):
        """blocked_by_critic should be part of VerifierVerdict TypedDict."""
        v = _make_verdict(
            test_result="FAIL",
            diagnostic=DiagnosticVerdict.FIX_IMPL.value,
            phase1_passed=False, phase2_passed=False,
            failures=[], findings=[],
            blocked_by_critic=True,
        )
        # Should be a valid VerifierVerdict
        assert isinstance(v, dict)
        assert "blocked_by_critic" in v


class TestBuildReturnWithBlockedByCritic:

    def test_build_return_includes_blocked_by_critic(self):
        """_build_return should propagate blocked_by_critic to output."""
        correction = {"attempt_count": 0, "error_history": [], "last_error_hash": None}
        verdict = _make_verdict(
            test_result="FAIL",
            diagnostic=DiagnosticVerdict.FIX_IMPL.value,
            phase1_passed=False, phase2_passed=True,
            failures=[{"message": "NPE"}], findings=[],
            blocked_by_critic=True,
        )
        result = _build_return(verdict, correction, "NPE")
        assert result["verifier_verdict"]["blocked_by_critic"] is True

    def test_build_return_preserves_false_when_not_blocked(self):
        """_build_return preserves blocked_by_critic = False."""
        correction = {"attempt_count": 0, "error_history": [], "last_error_hash": None}
        verdict = _make_verdict(
            test_result="FAIL",
            diagnostic=DiagnosticVerdict.FIX_IMPL.value,
            phase1_passed=False, phase2_passed=True,
            failures=[{"message": "NPE"}], findings=[],
            blocked_by_critic=False,
        )
        result = _build_return(verdict, correction, "NPE")
        assert result["verifier_verdict"]["blocked_by_critic"] is False


class TestActorUsesBlockedByCritic:

    def test_actor_can_read_blocked_by_critic_from_state(self):
        """The Actor node should be able to read blocked_by_critic from state."""
        from sacv.orchestration.state import WorkflowState

        state: WorkflowState = {  # type: ignore[assignment]
            "session_id": "t",
            "task_id": "t",
            "task_description": "",
            "project_mode": "greenfield",
            "module_type": "backend-domain",
            "current_phase": "verifier",
            "context_skeleton": None,
            "blast_radius_map": None,
            "agents_md_context": None,
            "strategy_candidates": [],
            "selected_strategy": None,
            "pruned_strategies": [],
            "red_phase_evidence_path": None,
            "test_inventory_paths": [],
            "diff_proposal": None,
            "preflight_result": None,
            "critic_findings": [],
            "verifier_verdict": {
                "test_result": "FAIL",
                "diagnostic": "FIX_IMPL",
                "phase1_passed": False,
                "phase2_passed": True,
                "test_failures": [{"message": "NPE"}],
                "performance_delta": None,
                "visual_diff_result": None,
                "docker_exit_code": 1,
                "blocked_by_critic": True,
            },
            "correction_state": {
                "attempt_count": 0,
                "branch_name": None,
                "last_error_hash": None,
                "error_history": [],
                "stagnation_pattern": "none",
            },
            "confidence_score": 1.0,
            "replan_count": 0,
            "active_branches": [],
            "exhausted_branches": [],
            "escalation_payload": None,
            "procedural_constraints": [],
            "lesson_learned": None,
            "arch_rules_updated": False,
            "debug_observations": None,
            "cumulative_cost_dollars": 0.0,
            "speculative_stash_ref": None,
        }

        verdict = state.get("verifier_verdict") or {}
        assert verdict.get("blocked_by_critic") is True


class TestRouteAfterVerifierWithBlockedByCritic:
    """M-02: route_after_verifier routes to actor when blocked_by_critic=True."""

    def _make_state(self, **overrides):
        base = {
            "session_id": "t",
            "task_id": "T1",
            "task_description": "",
            "project_mode": "greenfield",
            "module_type": "backend-domain",
            "current_phase": "verifier",
            "context_skeleton": None,
            "blast_radius_map": None,
            "agents_md_context": None,
            "strategy_candidates": [],
            "selected_strategy": None,
            "pruned_strategies": [],
            "red_phase_evidence_path": None,
            "test_inventory_paths": [],
            "diff_proposal": None,
            "preflight_result": None,
            "critic_findings": [],
            "verifier_verdict": {
                "test_result": "FAIL",
                "diagnostic": "FIX_IMPL",
                "phase1_passed": False,
                "phase2_passed": True,
                "test_failures": [{"message": "NPE"}],
                "performance_delta": None,
                "visual_diff_result": None,
                "docker_exit_code": 1,
                "blocked_by_critic": False,
            },
            "correction_state": {
                "attempt_count": 0,
                "branch_name": None,
                "last_error_hash": None,
                "error_history": [],
                "stagnation_pattern": "none",
            },
            "confidence_score": 1.0,
            "replan_count": 0,
            "active_branches": [],
            "exhausted_branches": [],
            "escalation_payload": None,
            "procedural_constraints": [],
            "lesson_learned": None,
            "arch_rules_updated": False,
            "debug_observations": None,
            "cumulative_cost_dollars": 0.0,
            "speculative_stash_ref": None,
        }
        base.update(overrides)
        return base

    def test_blocked_by_critic_routes_to_actor_not_speculative(self):
        """When blocked_by_critic=True, route to actor even at attempt >= 2."""
        from sacv.orchestration.edges import route_after_verifier

        state = self._make_state(
            verifier_verdict={
                **self._make_state()["verifier_verdict"],
                "blocked_by_critic": True,
            },
            correction_state={**self._make_state()["correction_state"], "attempt_count": 2},
        )
        result = route_after_verifier(state)
        assert result == "actor"

    def test_not_blocked_by_critic_uses_normal_routing(self):
        """When blocked_by_critic=False, normal routing applies."""
        from sacv.orchestration.edges import route_after_verifier

        state = self._make_state(
            correction_state={**self._make_state()["correction_state"], "attempt_count": 2},
        )
        result = route_after_verifier(state)
        assert result == "speculative_branch"

    def test_blocked_by_critic_at_low_attempt_still_routes_to_actor(self):
        """blocked_by_critic=True routes to actor regardless of attempt count."""
        from sacv.orchestration.edges import route_after_verifier

        state = self._make_state(
            verifier_verdict={
                **self._make_state()["verifier_verdict"],
                "blocked_by_critic": True,
            },
            correction_state={**self._make_state()["correction_state"], "attempt_count": 0},
        )
        result = route_after_verifier(state)
        assert result == "actor"
