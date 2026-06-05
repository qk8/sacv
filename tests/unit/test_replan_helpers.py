"""
tests/unit/test_replan_helpers.py
===================================
Unit tests for replan helper functions.

Tests cover:
1. _build_failure_summary — builds correct failure summary dict
2. _build_failure_summary — handles missing verdict gracefully
3. _build_failure_summary — truncates test failures to 5
4. _build_failure_summary — truncates critic findings to 5
5. _build_failure_summary — truncates arch violations to 5
6. _build_failure_summary — includes all correction state fields
"""
from __future__ import annotations

import pytest

from sacv.nodes.replan import _build_failure_summary


def _s(**kw):
    base = {
        "session_id": "t", "task_id": "t", "task_description": "",
        "project_mode": "greenfield", "module_type": "backend-domain",
        "current_phase": "tdd_gate",
        "context_skeleton": None, "blast_radius_map": None,
        "agents_md_context": None,
        "strategy_candidates": [
            {"strategy_id": "s1", "description": "d1", "affected_files": []},
            {"strategy_id": "s2", "description": "d2", "affected_files": []},
        ],
        "selected_strategy": None, "pruned_strategies": [],
        "red_phase_evidence_path": None, "test_inventory_paths": [],
        "diff_proposal": None, "preflight_result": None,
        "critic_findings": [], "verifier_verdict": None,
        "debug_observations": None,
        "correction_state": {
            "attempt_count": 2, "branch_name": "b",
            "last_error_hash": "abc", "error_history": [],
            "stagnation_pattern": "semantic",
        },
        "confidence_score": 0.5, "replan_count": 1,
        "active_branches": [], "exhausted_branches": [],
        "escalation_payload": None, "procedural_constraints": [],
        "lesson_learned": None, "arch_rules_updated": False,
    }
    base.update(kw)
    return base


class TestBuildFailureSummary:

    def test_basic_structure(self):
        """Failure summary has all expected keys."""
        state = _s(
            verifier_verdict={
                "diagnostic": "FIX_IMPL",
                "test_failures": [{"message": "err1"}],
            },
            critic_findings=[{"severity": "critical", "critic": "security"}],
            preflight_result={"arch_violations": [{"rule": "no-circle"}]},
        )
        result = _build_failure_summary(state)
        assert "exhausted_strategies" in result
        assert "last_diagnostic" in result
        assert "last_test_failures" in result
        assert "critical_critic_findings" in result
        assert "arch_violations_found" in result
        assert "stagnation_pattern" in result
        assert "replan_count" in result

    def test_includes_exhausted_strategies(self):
        """All strategy IDs from strategy_candidates are listed."""
        state = _s(
            strategy_candidates=[
                {"strategy_id": "s1"},
                {"strategy_id": "s2"},
                {"strategy_id": "s3"},
            ],
            verifier_verdict={"diagnostic": "FIX_IMPL", "test_failures": []},
        )
        result = _build_failure_summary(state)
        assert result["exhausted_strategies"] == ["s1", "s2", "s3"]

    def test_no_strategies_returns_empty_list(self):
        """When no strategy_candidates, exhausted_strategies is empty."""
        state = _s(strategy_candidates=[])
        result = _build_failure_summary(state)
        assert result["exhausted_strategies"] == []

    def test_missing_strategy_candidates_returns_empty(self):
        """When strategy_candidates key is absent, returns empty list."""
        state = _s()
        del state["strategy_candidates"]
        result = _build_failure_summary(state)
        assert result["exhausted_strategies"] == []

    def test_includes_last_diagnostic(self):
        state = _s(verifier_verdict={"diagnostic": "AMBIGUOUS", "test_failures": []})
        result = _build_failure_summary(state)
        assert result["last_diagnostic"] == "AMBIGUOUS"

    def test_unknown_diagnostic_when_no_verdict(self):
        state = _s(verifier_verdict=None)
        result = _build_failure_summary(state)
        assert result["last_diagnostic"] == "UNKNOWN"

    def test_unknown_diagnostic_when_no_diagnostic_key(self):
        state = _s(verifier_verdict={})
        result = _build_failure_summary(state)
        assert result["last_diagnostic"] == "UNKNOWN"

    def test_truncates_test_failures_to_5(self):
        failures = [{"message": f"err{i}"} for i in range(10)]
        state = _s(verifier_verdict={"diagnostic": "FIX_IMPL", "test_failures": failures})
        result = _build_failure_summary(state)
        assert len(result["last_test_failures"]) == 5

    def test_fewer_than_5_failures_kept_as_is(self):
        failures = [{"message": f"err{i}"} for i in range(3)]
        state = _s(verifier_verdict={"diagnostic": "FIX_IMPL", "test_failures": failures})
        result = _build_failure_summary(state)
        assert len(result["last_test_failures"]) == 3

    def test_includes_critical_critic_findings(self):
        findings = [
            {"severity": "critical", "critic": "security", "message": "S1"},
            {"severity": "warning", "critic": "style", "message": "W1"},
            {"severity": "critical", "critic": "consistency", "message": "S2"},
        ]
        state = _s(critic_findings=findings)
        result = _build_failure_summary(state)
        assert len(result["critical_critic_findings"]) == 2
        assert result["critical_critic_findings"][0]["message"] == "S1"
        assert result["critical_critic_findings"][1]["message"] == "S2"

    def test_filters_non_critical_findings(self):
        findings = [
            {"severity": "warning", "critic": "style", "message": "W1"},
            {"severity": "info", "critic": "consistency", "message": "I1"},
        ]
        state = _s(critic_findings=findings)
        result = _build_failure_summary(state)
        assert result["critical_critic_findings"] == []

    def test_truncates_critical_findings_to_5(self):
        findings = [
            {"severity": "critical", "critic": "security", "message": f"S{i}"}
            for i in range(10)
        ]
        state = _s(critic_findings=findings)
        result = _build_failure_summary(state)
        assert len(result["critical_critic_findings"]) == 5

    def test_includes_arch_violations(self):
        violations = [{"rule": "no-circle", "message": "C"}]
        state = _s(preflight_result={"arch_violations": violations})
        result = _build_failure_summary(state)
        assert len(result["arch_violations_found"]) == 1

    def test_truncates_arch_violations_to_5(self):
        violations = [{"rule": f"rule{i}"} for i in range(10)]
        state = _s(preflight_result={"arch_violations": violations})
        result = _build_failure_summary(state)
        assert len(result["arch_violations_found"]) == 5

    def test_missing_preflight_returns_empty_violations(self):
        state = _s(preflight_result=None)
        result = _build_failure_summary(state)
        assert result["arch_violations_found"] == []

    def test_includes_stagnation_pattern(self):
        state = _s(correction_state={"stagnation_pattern": "iteration"})
        result = _build_failure_summary(state)
        assert result["stagnation_pattern"] == "iteration"

    def test_includes_replan_count(self):
        state = _s(replan_count=3)
        result = _build_failure_summary(state)
        assert result["replan_count"] == 3

    def test_minimal_state_produces_valid_summary(self):
        """Minimal state with correction_state produces valid summary."""
        result = _build_failure_summary({
            "correction_state": {"stagnation_pattern": "none"},
            "replan_count": 0,
        })
        assert result["exhausted_strategies"] == []
        assert result["last_diagnostic"] == "UNKNOWN"
        assert result["last_test_failures"] == []
        assert result["critical_critic_findings"] == []
        assert result["arch_violations_found"] == []
        assert result["stagnation_pattern"] == "none"
        assert result["replan_count"] == 0
