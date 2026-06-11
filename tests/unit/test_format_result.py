"""
tests/unit/test_format_result.py
=================================
TST-001: Verify format_result outputs rich diagnostic JSON.
"""
from __future__ import annotations

import json

from sacv.cli_progress import format_result


class TestFormatResult:

    def _state(self, **kw):
        base = {
            "current_phase": "complete",
            "cumulative_cost_dollars": 12.34,
            "verifier_verdict": None,
            "correction_state": {"attempt_count": 2},
            "replan_count": 1,
            "escalation_payload": None,
            "critic_findings": [],
            "lesson_learned": None,
            "stagnation": "none",
        }
        base.update(kw)
        return base

    def test_pass_result_includes_result_field(self):
        """PASS workflow should include result='PASS'."""
        state = self._state(verifier_verdict={"test_result": "PASS"})
        out = json.loads(format_result(state, "T-001"))
        assert out["result"] == "PASS"

    def test_hitl_result_includes_result_field(self):
        """HITL escalation should include result='HITL'."""
        state = self._state(
            escalation_payload={"escalation_id": "esc-123"},
            verifier_verdict={"test_result": "FAIL"},
        )
        out = json.loads(format_result(state, "T-001"))
        assert out["result"] == "HITL"

    def test_fail_result_includes_result_field(self):
        """FAIL workflow should include result='FAIL'."""
        state = self._state(verifier_verdict={"test_result": "FAIL"})
        out = json.loads(format_result(state, "T-001"))
        assert out["result"] == "FAIL"

    def test_includes_attempt_count(self):
        """Output should include attempt count."""
        state = self._state(correction_state={"attempt_count": 3})
        out = json.loads(format_result(state, "T-001"))
        assert out["attempts"] == 3

    def test_includes_replan_count(self):
        """Output should include replan count."""
        state = self._state(replan_count=2)
        out = json.loads(format_result(state, "T-001"))
        assert out["replan_count"] == 2

    def test_includes_escalation_id(self):
        """HITL output should include escalation ID."""
        state = self._state(escalation_payload={"escalation_id": "esc-abc"})
        out = json.loads(format_result(state, "T-001"))
        assert out["escalation_id"] == "esc-abc"

    def test_includes_critic_counts(self):
        """Output should include critic finding counts."""
        state = self._state(critic_findings=[
            {"severity": "critical", "critic": "security", "file": "X.java",
             "line": 1, "rule_id": "S1", "message": "X", "resolution_hint": "Y"},
            {"severity": "warning", "critic": "style", "file": "Y.java",
             "line": 1, "rule_id": "S2", "message": "Y", "resolution_hint": "Z"},
        ])
        out = json.loads(format_result(state, "T-001"))
        assert out["critic_findings"] == 2
        assert out["critical_findings"] == 1

    def test_includes_stagnation_pattern(self):
        """Output should include stagnation pattern."""
        state = self._state(correction_state={"stagnation_pattern": "semantic"})
        out = json.loads(format_result(state, "T-001"))
        assert out["stagnation"] == "semantic"

    def test_basic_fields_present(self):
        """Output should always include basic fields."""
        state = self._state()
        out = json.loads(format_result(state, "T-001"))
        assert out["task"] == "T-001"
        assert out["phase"] == "complete"
        assert out["cost"] == 12.34
