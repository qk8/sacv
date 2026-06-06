"""
tests/unit/test_actor_helpers.py
=================================
Unit tests for actor.py helper functions:
- _format_debug_observations
- _format_preflight
- _format_findings
"""
from __future__ import annotations

import pytest

from sacv.nodes.actor import (
    _format_debug_observations,
    _format_preflight,
    _format_findings,
)


# ── _format_debug_observations ────────────────────────────────────────────────


class TestFormatDebugObservations:

    def test_none_returns_empty(self):
        """None observations → empty string."""
        assert _format_debug_observations(None) == ""

    def test_empty_dict_returns_empty(self):
        """Empty dict → empty string."""
        assert _format_debug_observations({}) == ""

    def test_error_type_only(self):
        """Just error_type is included."""
        obs = {"error_type": "NULL_REFERENCE"}
        result = _format_debug_observations(obs)
        assert "Error type: NULL_REFERENCE" in result

    def test_root_cause_included(self):
        """Root cause hypothesis is included."""
        obs = {
            "error_type": "NPE",
            "root_cause": "UserService.findById returns null",
        }
        result = _format_debug_observations(obs)
        assert "Root cause: UserService.findById returns null" in result

    def test_breakpoint_hit_included(self):
        """Breakpoint hit shows file, line, and variables."""
        obs = {
            "error_type": "NPE",
            "breakpoint_hits": [{
                "file": "UserService.java",
                "line": 42,
                "variables": {
                    "user": {"value": "null", "type": "User"},
                    "id": {"value": "42", "type": "Long"},
                },
                "call_stack": [
                    "UserService.findById(UserService.java:42)",
                    "UserController.get(UserController.java:20)",
                ],
            }],
        }
        result = _format_debug_observations(obs)
        assert "Breakpoint hit at UserService.java:42" in result
        assert "user = null" in result
        assert "id = 42" in result

    def test_breakpoint_variables_with_non_dict_value(self):
        """Handles variables that are not dicts (plain values)."""
        obs = {
            "error_type": "NPE",
            "breakpoint_hits": [{
                "file": "X.java",
                "line": 1,
                "variables": {"val": "plain_string"},
            }],
        }
        result = _format_debug_observations(obs)
        assert "val = plain_string" in result

    def test_call_stack_truncated_to_3(self):
        """Only the first 3 stack frames are shown."""
        stack = [f"Frame{i}" for i in range(10)]
        obs = {
            "error_type": "NPE",
            "breakpoint_hits": [{"file": "X.java", "line": 1, "call_stack": stack}],
        }
        result = _format_debug_observations(obs)
        assert "Frame0" in result
        assert "Frame2" in result
        assert "Frame3" not in result

    def test_breakpoint_variables_truncated_to_8(self):
        """Only the first 8 variables are shown."""
        variables = {f"var{i}": {"value": f"v{i}", "type": "T"} for i in range(20)}
        obs = {
            "error_type": "NPE",
            "breakpoint_hits": [{"file": "X.java", "line": 1, "variables": variables}],
        }
        result = _format_debug_observations(obs)
        assert "var0 = v0" in result
        assert "var7 = v7" in result
        assert "var8 = v8" not in result

    def test_multiple_breakpoints_only_first_two(self):
        """Only the first 2 breakpoint hits are formatted."""
        hits = [
            {"file": "X.java", "line": i, "variables": {}, "call_stack": []}
            for i in range(5)
        ]
        obs = {"error_type": "NPE", "breakpoint_hits": hits}
        result = _format_debug_observations(obs)
        # Should contain first two
        assert "Breakpoint hit at X.java:0" in result
        assert "Breakpoint hit at X.java:1" in result
        # Third should not appear
        assert "Breakpoint hit at X.java:2" not in result

    def test_minimal_payload_included(self):
        """Minimal failing payload is JSON-serialized."""
        obs = {
            "error_type": "VALIDATION",
            "minimal_payload": {"field": "value", "count": 3},
        }
        result = _format_debug_observations(obs)
        assert '"field": "value"' in result or "'field': 'value'" in result

    def test_actuator_beans_snapshot(self):
        """Spring Actuator beans snapshot is noted."""
        obs = {
            "error_type": "BEAN_CREATION",
            "actuator_beans": {"userService": {"status": "BEAN"}},
        }
        result = _format_debug_observations(obs)
        assert "Spring Actuator beans snapshot available" in result

    def test_combined_all_fields(self):
        """All fields are present in the formatted output."""
        obs = {
            "error_type": "NPE",
            "root_cause": "null ref in service",
            "breakpoint_hits": [{
                "file": "S.java", "line": 10,
                "variables": {"x": {"value": "null", "type": "String"}},
                "call_stack": ["S.m(S.java:10)"],
            }],
            "minimal_payload": {"input": "bad"},
            "actuator_beans": {"userService": {"status": "BEAN"}},
        }
        result = _format_debug_observations(obs)
        assert "Error type: NPE" in result
        assert "Root cause: null ref in service" in result
        assert "Breakpoint hit at S.java:10" in result
        assert "x = null" in result
        assert "Minimal failing payload" in result
        assert "Spring Actuator" in result


# ── _format_preflight ─────────────────────────────────────────────────────────


class TestFormatPreflight:

    def test_passed_returns_empty(self):
        """Passed preflight → empty string."""
        assert _format_preflight({"passed": True}) == ""

    def test_none_passed_defaults_true(self):
        """Missing passed field → defaults to True → empty string."""
        assert _format_preflight({}) == ""

    def test_lsp_errors_formatted(self):
        """LSP errors are formatted as [LSP] lines."""
        pf = {
            "passed": False,
            "lsp_errors": [
                {"file": "a.java", "line": 10, "code": "TS2322", "message": "type mismatch"},
            ],
        }
        result = _format_preflight(pf)
        assert "[LSP]" in result
        assert "a.java:10" in result
        assert "TS2322" in result
        assert "type mismatch" in result

    def test_arch_violations_formatted(self):
        """Arch violations are formatted as [ARCH] lines."""
        pf = {
            "passed": False,
            "arch_violations": [
                {"rule": "no-ui-to-db", "message": "forbidden import"},
            ],
        }
        result = _format_preflight(pf)
        assert "[ARCH]" in result
        assert "no-ui-to-db" in result
        assert "forbidden import" in result

    def test_lsp_truncated_to_5(self):
        """Only first 5 LSP errors are shown."""
        errors = [
            {"file": f"F{i}.java", "line": i, "code": "E", "message": f"err{i}"}
            for i in range(10)
        ]
        pf = {"passed": False, "lsp_errors": errors}
        result = _format_preflight(pf)
        assert "F0.java" in result
        assert "F4.java" in result
        assert "F5.java" not in result

    def test_arch_truncated_to_5(self):
        """Only first 5 arch violations are shown."""
        violations = [
            {"rule": f"R{i}", "message": f"msg{i}"}
            for i in range(10)
        ]
        pf = {"passed": False, "arch_violations": violations}
        result = _format_preflight(pf)
        assert "R0" in result
        assert "R4" in result
        assert "R5" not in result

    def test_empty_lsp_arch_returns_empty(self):
        """No errors → empty string even if passed=False."""
        pf = {"passed": False, "lsp_errors": [], "arch_violations": []}
        result = _format_preflight(pf)
        assert result == ""

    def test_mixed_lsp_and_arch(self):
        """Both LSP and arch violations are shown."""
        pf = {
            "passed": False,
            "lsp_errors": [{"file": "a.java", "line": 1, "code": "E", "message": "err"}],
            "arch_violations": [{"rule": "R", "message": "violation"}],
        }
        result = _format_preflight(pf)
        assert "[LSP]" in result
        assert "[ARCH]" in result


# ── _format_findings ──────────────────────────────────────────────────────────


class TestFormatFindings:

    def test_empty_returns_empty(self):
        """No findings → empty string."""
        assert _format_findings([]) == ""

    def test_single_finding_formatted(self):
        """A single finding is formatted correctly."""
        findings = [{
            "severity": "critical", "critic": "security",
            "file": "a.java", "line": 42,
            "message": "SQL injection", "resolution_hint": "use params",
        }]
        result = _format_findings(findings)
        assert "[CRITICAL]" in result
        assert "security" in result
        assert "a.java:42" in result
        assert "SQL injection" in result
        assert "use params" in result

    def test_multiple_findings_separated_by_newlines(self):
        """Multiple findings are separated by newlines."""
        findings = [
            {"severity": "critical", "critic": "security",
             "file": "a.java", "line": 1, "message": "X", "resolution_hint": "Y"},
            {"severity": "warning", "critic": "style",
             "file": "b.java", "line": 2, "message": "Z", "resolution_hint": "W"},
        ]
        result = _format_findings(findings)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 2
        assert "security" in lines[0]
        assert "style" in lines[1]

    def test_line_defaults_to_question_mark(self):
        """Missing line number → ?."""
        findings = [{
            "severity": "critical", "critic": "security",
            "file": "a.java", "message": "X", "resolution_hint": "Y",
        }]
        result = _format_findings(findings)
        assert "a.java:?" in result

    def test_severity_uppercased(self):
        """Severity is shown in uppercase."""
        findings = [{
            "severity": "critical", "critic": "s",
            "file": "a.java", "line": 1, "message": "X", "resolution_hint": "Y",
        }]
        result = _format_findings(findings)
        assert "[CRITICAL]" in result
