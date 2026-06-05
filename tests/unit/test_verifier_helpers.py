"""
tests/unit/test_verifier_helpers.py
====================================
Unit tests for verifier utility functions.

Tests cover:
1. _check_test_deletions — detects full file deletions
2. _check_test_deletions — detects assertion gutting
3. _check_test_deletions — allows safe test modifications
4. _check_test_deletions — no proposal returns None
5. _classify — PASS when both phases pass
6. _classify — FIX_IMPL when phase 1 fails
7. _classify — AMBIGUOUS when phase 2 fails with no clear signal
8. _fallback_parse — Java BUILD FAILURE
9. _fallback_parse — frontend errors
10. _fallback_parse — no failures found
11. _full_suite_cmd — frontend vs backend
12. _inventory_test_cmd — frontend spec files
13. _inventory_test_cmd — Java test classes
14. _is_bean_error — Spring DI error keywords
15. _has_visual_breakage — passed vs failed
16. _make_verdict — constructs VerifierVerdict
17. _build_return — accumulates error history
"""
from __future__ import annotations

import pytest

from sacv.orchestration.config import WorkflowConfig
from sacv.orchestration.state import DiagnosticVerdict
from sacv.nodes.verifier import (
    _check_test_deletions, _classify, _fallback_parse,
    _full_suite_cmd, _inventory_test_cmd, _has_visual_breakage,
    _is_bean_error, _make_verdict, _build_return,
)
from sacv.interfaces.agent_provider import AgentResult


# ── _check_test_deletions ─────────────────────────────────────────────────────

class TestCheckTestDeletions:

    def test_no_proposal_returns_none(self):
        assert _check_test_deletions({}) is None

    def test_no_diffs_returns_none(self):
        assert _check_test_deletions({"diff_proposal": {"diffs": []}}) is None

    def test_deletes_test_file(self):
        proposal = {
            "diffs": [{
                "file_path": "src/test/java/UserServiceTest.java",
                "operation": "delete",
                "diff_content": "",
            }],
        }
        result = _check_test_deletions({"diff_proposal": proposal})
        assert "UserServiceTest.java" in result

    def test_gutting_assertions(self):
        proposal = {
            "diffs": [{
                "file_path": "src/test/java/UserServiceTest.java",
                "operation": "modify",
                "diff_content": "-    assertThat(service.findById(1L)).isNotNull();\n"
                                "-    assertEquals(42, service.count());\n"
                                "+    // TODO: fix later\n",
            }],
        }
        result = _check_test_deletions({"diff_proposal": proposal})
        assert "assertion gutting" in result

    def test_safe_assertion_replacement_allowed(self):
        """Replacing 2 assertions with 3 is safe (added >= removed // 2)."""
        proposal = {
            "diffs": [{
                "file_path": "src/test/java/UserServiceTest.java",
                "operation": "modify",
                "diff_content": "-    assertThat(x).isTrue();\n"
                                "-    assertThat(y).isFalse();\n"
                                "+    assertThat(x).isEqualTo(true);\n"
                                "+    assertThat(y).isEqualTo(false);\n"
                                "+    assertThat(z).isNotNull();\n",
            }],
        }
        result = _check_test_deletions({"diff_proposal": proposal})
        assert result is None

    def test_non_test_file_ignored(self):
        proposal = {
            "diffs": [{
                "file_path": "src/main/java/UserService.java",
                "operation": "delete",
                "diff_content": "",
            }],
        }
        assert _check_test_deletions({"diff_proposal": proposal}) is None

    def test_multiple_violations_reported(self):
        proposal = {
            "diffs": [
                {
                    "file_path": "src/test/java/ATest.java",
                    "operation": "delete",
                    "diff_content": "",
                },
                {
                    "file_path": "src/test/java/BTest.java",
                    "operation": "delete",
                    "diff_content": "",
                },
            ],
        }
        result = _check_test_deletions({"diff_proposal": proposal})
        assert "ATest.java" in result
        assert "BTest.java" in result


# ── _classify ─────────────────────────────────────────────────────────────────

class TestClassify:

    def test_pass_when_both_phases_pass(self):
        result = _classify(True, True, [], [], {})
        assert result == DiagnosticVerdict.PASS.value

    def test_fix_impl_when_phase1_fails(self):
        result = _classify(False, True, [{"message": "NPE"}], [], {})
        assert result == DiagnosticVerdict.FIX_IMPL.value

    def test_fix_test_when_phase2_assertion_fails(self):
        """p1 passes, p2 fails with assertion errors but no compile errors → FIX_TEST."""
        result = _classify(True, False, [
            {"message": "AssertionError: expected:<2> but was:<3>"},
        ], [], {})
        assert result == DiagnosticVerdict.FIX_TEST.value

    def test_fix_test_javascript_expect_to_be(self):
        """JS assertion style: expect(received).toBe(expected) → FIX_TEST."""
        result = _classify(True, False, [
            {"message": "expect(received).toBe(expected) // Object.is equality"},
        ], [], {})
        assert result == DiagnosticVerdict.FIX_TEST.value

    def test_fix_test_javascript_to_equal(self):
        """JS assertion style: toEqual → FIX_TEST."""
        result = _classify(True, False, [
            {"message": "Expected value to equal: 42"},
        ], [], {})
        # "toequal" is in the assertion keywords
        assert result == DiagnosticVerdict.FIX_TEST.value

    def test_fix_impl_when_phase2_compile_fails(self):
        """p1 passes, p2 fails with compile errors → FIX_IMPL (not FIX_TEST)."""
        result = _classify(True, False, [
            {"message": "error: cannot find symbol"},
        ], [], {})
        assert result == DiagnosticVerdict.FIX_IMPL.value

    def test_fix_impl_when_phase2_has_both_assertion_and_compile(self):
        """p1 passes, p2 fails with both assertion and compile errors → FIX_IMPL takes priority."""
        result = _classify(True, False, [
            {"message": "AssertionError: expected:<2>"},
            {"message": "error: cannot find symbol"},
        ], [], {})
        # has_compile_fail is True, so FIX_TEST is NOT returned
        assert result == DiagnosticVerdict.FIX_IMPL.value

    def test_fix_impl_for_compile_errors(self):
        result = _classify(True, False, [
            {"message": "error: cannot find symbol"},
        ], [], {})
        assert result == DiagnosticVerdict.FIX_IMPL.value

    def test_fix_impl_for_syntax_errors(self):
        result = _classify(True, False, [
            {"message": "error: syntax error on line 5"},
        ], [], {})
        assert result == DiagnosticVerdict.FIX_IMPL.value

    def test_fix_impl_for_module_not_found(self):
        result = _classify(True, False, [
            {"message": "error: module not found"},
        ], [], {})
        assert result == DiagnosticVerdict.FIX_IMPL.value

    def test_fix_impl_when_red_phase_evidence_exists_with_failure(self):
        """red_phase_evidence_path + non-empty failure text → FIX_IMPL."""
        result = _classify(True, False, [{"message": "timeout"}], [], {
            "red_phase_evidence_path": ".workflow/tdd-evidence/e1.json",
        })
        assert result == DiagnosticVerdict.FIX_IMPL.value

    def test_ambiguous_with_red_phase_evidence_but_no_failure_text(self):
        """red_phase_evidence_path alone should NOT force FIX_IMPL.

        When both phases pass but the test runner exits non-zero with no
        parseable failure messages, AMBIGUOUS is the correct diagnostic —
        the Agent has nothing concrete to fix and should escalate to the
        debugger for root-cause analysis.
        """
        result = _classify(True, False, [{"message": ""}], [], {
            "red_phase_evidence_path": ".workflow/tdd-evidence/e1.json",
        })
        assert result == DiagnosticVerdict.AMBIGUOUS.value

    def test_ambiguous_when_both_phases_pass_no_failure_messages(self):
        """Both phases pass, no failure messages, but overall_pass is False
        (e.g. visual breakage) → AMBIGUOUS (not FIX_IMPL).

        This is the key regression test: the old code at verifier.py:241
        unconditionally returned FIX_IMPL when red_phase_evidence_path
        existed, making AMBIGUOUS unreachable after TDD gate.
        """
        result = _classify(True, True, [{"message": ""}], [], {
            "red_phase_evidence_path": ".workflow/tdd-evidence/e1.json",
        }, overall_pass=False)
        assert result == DiagnosticVerdict.AMBIGUOUS.value

    def test_fix_impl_with_critical_findings(self):
        result = _classify(True, False, [], [
            {"severity": "critical", "message": "SQL injection"},
        ], {})
        assert result == DiagnosticVerdict.FIX_IMPL.value

    def test_ambiguous_when_no_failure_messages(self):
        # AMBIGUOUS only when failure_text is empty (no failure messages at all)
        result = _classify(True, False, [
            {"message": ""},
        ], [], {})
        assert result == DiagnosticVerdict.AMBIGUOUS.value

    def test_fix_impl_when_failure_but_no_keyword_match(self):
        # Non-empty failure messages that don't match known keywords → FIX_IMPL
        result = _classify(True, False, [
            {"message": "flaky test timed out"},
        ], [], {})
        assert result == DiagnosticVerdict.FIX_IMPL.value


# ── _fallback_parse ───────────────────────────────────────────────────────────

class TestFallbackParse:

    def test_java_build_failure(self):
        output = "[INFO] BUILD FAILURE\n[ERROR] Tests run: 5, Failures: 2"
        results = _fallback_parse(output, "backend-domain")
        assert len(results) >= 1
        assert "BUILD FAILURE" in results[0]["message"]

    def test_frontend_fail(self):
        output = "FAIL src/Login.test.tsx\n  Error: Timeout - Async..."
        results = _fallback_parse(output, "frontend")
        assert len(results) >= 1
        assert "FAIL" in results[0]["message"]

    def test_no_failures_returns_empty(self):
        results = _fallback_parse("BUILD SUCCESS", "backend-domain")
        assert results == []

    def test_limits_to_ten(self):
        lines = ["Error: fault " + str(i) + "\n" for i in range(20)]
        results = _fallback_parse("".join(lines), "backend-domain")
        assert len(results) <= 10


# ── _full_suite_cmd ───────────────────────────────────────────────────────────

class TestFullSuiteCmd:

    def test_backend_uses_maven(self):
        assert "mvn test" in _full_suite_cmd("backend-domain")

    def test_frontend_uses_playwright(self):
        cmd = _full_suite_cmd("frontend")
        assert "playwright" in cmd


# ── _inventory_test_cmd ───────────────────────────────────────────────────────

class TestInventoryTestCmd:

    def test_frontend_with_spec_files(self):
        cmd = _inventory_test_cmd("frontend", ["src/Login.spec.ts", "src/App.spec.ts"])
        assert "playwright" in cmd
        assert "Login.spec.ts" in cmd

    def test_frontend_without_spec_files(self):
        cmd = _inventory_test_cmd("frontend", ["src/Login.tsx"])
        assert "npm test" in cmd

    def test_java_test_classes(self):
        cmd = _inventory_test_cmd("backend-domain", [
            "src/test/java/UserServiceTest.java",
            "src/test/java/UserRepoTest.java",
        ])
        assert "mvn test" in cmd
        assert "UserServiceTest" in cmd
        assert "UserRepoTest" in cmd

    def test_empty_paths_falls_back_to_full_suite(self):
        cmd = _inventory_test_cmd("backend-domain", [])
        assert "mvn test" in cmd


# ── _is_bean_error ────────────────────────────────────────────────────────────

class TestIsBeanError:

    def test_detects_bean_creation_exception(self):
        assert _is_bean_error("org.springframework.beans.factory.BeanCreationException")

    def test_detects_no_such_bean(self):
        assert _is_bean_error("NoSuchBeanDefinitionException for 'userService'")

    def test_detects_unsatisfied_dependency(self):
        assert _is_bean_error("UnsatisfiedDependencyException")

    def test_detects_no_unique_bean(self):
        assert _is_bean_error("NoUniqueBeanDefinitionException")

    def test_returns_false_for_normal_output(self):
        assert not _is_bean_error("Tests run: 5, Failures: 0")


# ── _has_visual_breakage ──────────────────────────────────────────────────────

class TestHasVisualBreakage:

    def test_none_returns_false(self):
        assert not _has_visual_breakage(None)

    def test_passed_returns_false(self):
        assert not _has_visual_breakage({"passed": True})

    def test_failed_returns_true(self):
        assert _has_visual_breakage({"passed": False})


# ── _make_verdict ─────────────────────────────────────────────────────────────

class TestMakeVerdict:

    def test_minimal_verdict(self):
        v = _make_verdict(
            test_result="PASS",
            diagnostic=DiagnosticVerdict.PASS.value,
            phase1_passed=True, phase2_passed=True,
            failures=[], findings=[],
        )
        assert v["test_result"] == "PASS"
        assert v["phase1_passed"] is True

    def test_verdict_with_all_fields(self):
        v = _make_verdict(
            test_result="FAIL",
            diagnostic=DiagnosticVerdict.AMBIGUOUS.value,
            phase1_passed=True, phase2_passed=False,
            failures=[{"message": "timeout"}],
            findings=[{"severity": "warning"}],
            performance_delta={"delta_ms": 50},
            visual_diff_result={"passed": False},
            playwright_trace_path="/tmp/trace.zip",
            otel_trace={"trace_id": "abc"},
            actuator_snapshot={"beans": []},
            docker_exit_code=1,
        )
        assert v["performance_delta"] == {"delta_ms": 50}
        assert v["visual_diff_result"] == {"passed": False}
        assert v["playwright_trace_path"] == "/tmp/trace.zip"
        assert v["otel_trace"] == {"trace_id": "abc"}
        assert v["actuator_snapshot"] == {"beans": []}
        assert v["docker_exit_code"] == 1


# ── _build_return ─────────────────────────────────────────────────────────────

class TestBuildReturn:

    def test_pass_does_not_accumulate_error_history(self):
        correction = {"attempt_count": 0, "error_history": [], "last_error_hash": None}
        verdict = _make_verdict(
            test_result="PASS", diagnostic=DiagnosticVerdict.PASS.value,
            phase1_passed=True, phase2_passed=True, failures=[], findings=[],
        )
        result = _build_return(verdict, correction, "")
        assert result["current_phase"] == "verifier"
        assert result["correction_state"]["error_history"] == []

    def test_fail_accumulates_error_history(self):
        correction = {"attempt_count": 0, "error_history": [], "last_error_hash": None}
        verdict = _make_verdict(
            test_result="FAIL", diagnostic=DiagnosticVerdict.FIX_IMPL.value,
            phase1_passed=False, phase2_passed=True,
            failures=[{"message": "NPE"}], findings=[],
        )
        result = _build_return(verdict, correction, "NPE")
        assert len(result["correction_state"]["error_history"]) == 1
        assert result["correction_state"]["last_error_hash"] is not None

    def test_fail_with_empty_failure_text_no_history(self):
        correction = {"attempt_count": 0, "error_history": [], "last_error_hash": None}
        verdict = _make_verdict(
            test_result="FAIL", diagnostic=DiagnosticVerdict.FIX_IMPL.value,
            phase1_passed=False, phase2_passed=True,
            failures=[], findings=[],
        )
        result = _build_return(verdict, correction, "")
        assert result["correction_state"]["error_history"] == []

    def test_truncates_error_history_to_five(self):
        correction = {"attempt_count": 0, "error_history": [], "last_error_hash": None}
        verdict = _make_verdict(
            test_result="FAIL", diagnostic=DiagnosticVerdict.FIX_IMPL.value,
            phase1_passed=False, phase2_passed=True,
            failures=[{"message": "err"}], findings=[],
        )
        # Simulate 7 consecutive failures
        for _ in range(7):
            result = _build_return(verdict, correction, "error message")
            correction = result["correction_state"]
        assert len(correction["error_history"]) <= 5
