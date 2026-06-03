"""
tests/unit/test_hitl_escalation_helpers.py
=============================================
Unit tests for HITL escalation helper functions.

Tests cover:
1. _build_hints — resolution hint generation for various scenarios
"""
from __future__ import annotations

import pytest

from sacv.nodes.hitl_escalation import _build_hints
from sacv.orchestration.config import WorkflowConfig


@pytest.mark.unit
class TestBuildHints:

    def _verdict(self, diagnostic):
        return {
            "test_result": "FAIL",
            "diagnostic": diagnostic,
            "phase1_passed": False, "phase2_passed": False,
            "test_failures": [{"message": "err"}],
            "performance_delta": None, "visual_diff_result": None,
            "critic_findings": [], "docker_exit_code": 1,
        }

    def _state(self, **kw):
        base = {
            "blast_radius_map": None,
            "critic_findings": [],
            "tdd_gate_attempts": 0,
            "red_phase_evidence_path": None,
        }
        base.update(kw)
        return base

    def test_no_verdict_returns_empty_hints(self):
        config = WorkflowConfig()
        hints = _build_hints(None, self._state(), config)
        assert hints == []

    def test_fix_impl_diagnostic_adds_architectural_hint(self):
        config = WorkflowConfig()
        verdict = self._verdict("FIX_IMPL")
        hints = _build_hints(verdict, self._state(), config)
        assert len(hints) == 1
        assert hints[0]["category"] == "architectural"
        assert hints[0]["priority"] == 1
        assert "Implementation does not satisfy" in hints[0]["hint"]

    def test_fix_test_diagnostic_adds_test_oracle_hint(self):
        config = WorkflowConfig()
        verdict = self._verdict("FIX_TEST")
        hints = _build_hints(verdict, self._state(), config)
        assert len(hints) == 1
        assert hints[0]["category"] == "test_oracle"
        assert hints[0]["priority"] == 1
        assert "Test expectations may not align" in hints[0]["hint"]

    def test_high_blast_radius_adds_decomposition_hint(self):
        config = WorkflowConfig()
        state = self._state(blast_radius_map={"risk_score": 0.85})
        hints = _build_hints(None, state, config)
        assert len(hints) == 1
        assert hints[0]["category"] == "blast_radius"
        assert hints[0]["priority"] == 2
        assert "0.85" in hints[0]["hint"] or "85%" in hints[0]["hint"]

    def test_low_blast_radius_does_not_add_hint(self):
        config = WorkflowConfig()
        state = self._state(blast_radius_map={"risk_score": 0.3})
        hints = _build_hints(None, state, config)
        assert hints == []

    def test_critical_critic_findings_add_security_hint(self):
        config = WorkflowConfig()
        state = self._state(critic_findings=[
            {"severity": "critical", "critic": "security", "file": "X.java",
             "line": 1, "rule_id": "SEC-001", "message": "SQL injection",
             "resolution_hint": "use params"},
            {"severity": "warning", "critic": "style", "file": "Y.java",
             "line": 1, "rule_id": "STY-001", "message": "Long method",
             "resolution_hint": "shorten"},
        ])
        hints = _build_hints(None, state, config)
        assert len(hints) == 1
        assert hints[0]["category"] == "security"
        assert "1 critical finding" in hints[0]["hint"]

    def test_multiple_critical_findings_counted(self):
        config = WorkflowConfig()
        state = self._state(critic_findings=[
            {"severity": "critical", "critic": "security", "file": "X.java",
             "line": 1, "rule_id": "SEC-001", "message": "X", "resolution_hint": "Y"},
            {"severity": "critical", "critic": "security", "file": "Y.java",
             "line": 1, "rule_id": "SEC-002", "message": "Z", "resolution_hint": "W"},
        ])
        hints = _build_hints(None, state, config)
        assert len(hints) == 1
        assert "2 critical finding" in hints[0]["hint"]

    def test_tdd_gate_exhaustion_adds_hint(self):
        config = WorkflowConfig(max_tdd_gate_attempts=3)
        state = self._state(
            tdd_gate_attempts=3,
            red_phase_evidence_path=None,
        )
        hints = _build_hints(None, state, config)
        # TDD gate hint should be first (priority 1, inserted at index 0)
        tdd_hints = [h for h in hints if h["category"] == "test_oracle" and "TDD gate" in h["hint"]]
        assert len(tdd_hints) == 1
        assert tdd_hints[0]["priority"] == 1
        assert "3 attempts" in tdd_hints[0]["hint"]

    def test_tdd_gate_with_evidence_does_not_add_hint(self):
        config = WorkflowConfig(max_tdd_gate_attempts=3)
        state = self._state(
            tdd_gate_attempts=3,
            red_phase_evidence_path="/path/evidence.json",
        )
        hints = _build_hints(None, state, config)
        tdd_hints = [h for h in hints if "TDD gate" in h["hint"]]
        assert len(tdd_hints) == 0

    def test_combined_hints_ordered_by_priority(self):
        config = WorkflowConfig(max_tdd_gate_attempts=3)
        verdict = self._verdict("FIX_IMPL")
        state = self._state(
            blast_radius_map={"risk_score": 0.9},
            critic_findings=[
                {"severity": "critical", "critic": "security", "file": "X.java",
                 "line": 1, "rule_id": "SEC-001", "message": "X", "resolution_hint": "Y"},
            ],
            tdd_gate_attempts=3,
            red_phase_evidence_path=None,
        )
        hints = _build_hints(verdict, state, config)
        assert len(hints) == 4
        # TDD gate hint is inserted at index 0
        assert hints[0]["category"] == "test_oracle"
        assert hints[0]["priority"] == 1
        # Architectural hint is priority 1
        assert any(h["priority"] == 1 for h in hints)

    def test_none_verdict_with_blast_radius(self):
        config = WorkflowConfig()
        state = self._state(blast_radius_map={"risk_score": 0.75})
        hints = _build_hints(None, state, config)
        assert len(hints) == 1
        assert hints[0]["category"] == "blast_radius"

    def test_empty_state_returns_empty(self):
        config = WorkflowConfig()
        hints = _build_hints(None, self._state(), config)
        assert hints == []

    def test_all_hints_are_automated_false(self):
        config = WorkflowConfig(max_tdd_gate_attempts=3)
        verdict = self._verdict("FIX_IMPL")
        state = self._state(
            blast_radius_map={"risk_score": 0.8},
            critic_findings=[{"severity": "critical", "critic": "security",
                              "file": "X.java", "line": 1, "rule_id": "S1",
                              "message": "X", "resolution_hint": "Y"}],
            tdd_gate_attempts=3,
            red_phase_evidence_path=None,
        )
        hints = _build_hints(verdict, state, config)
        for h in hints:
            assert h["automated"] is False
