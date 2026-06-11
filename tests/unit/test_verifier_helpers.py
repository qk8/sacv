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
    classify_confidence,
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


# ── classify_confidence ───────────────────────────────────────────────────────

class TestClassifyConfidence:

    def test_empty_text_returns_zero(self):
        """Empty or whitespace-only text → 0.0."""
        assert classify_confidence("") == pytest.approx(0.0)
        assert classify_confidence("   ") == pytest.approx(0.0)

    def test_assertion_error_high_confidence(self):
        """Classic JUnit assertion error → high confidence."""
        text = "AssertionError: expected:<2> but was:<3>"
        conf = classify_confidence(text)
        assert conf >= 0.8

    def test_junit_assertionerror_high_confidence(self):
        """Fully qualified JUnit assertion error → high confidence."""
        text = "junit.framework.AssertionError: unexpected"
        conf = classify_confidence(text)
        assert conf >= 0.8

    def test_javascript_expect_to_be(self):
        """JS test framework: expect(received).toBe → high confidence."""
        text = "expect(received).toBe(expected) // Object.is equality"
        conf = classify_confidence(text)
        assert conf >= 0.8

    def test_compile_error_medium_confidence(self):
        """Compile error → medium-high confidence."""
        text = "error: cannot find symbol"
        conf = classify_confidence(text)
        assert conf >= 0.6

    def test_syntax_error(self):
        """Syntax error → medium confidence."""
        text = "SyntaxError: unexpected token 'import'"
        conf = classify_confidence(text)
        assert conf >= 0.5

    def test_bean_creation_exception(self):
        """Spring DI error → high confidence."""
        text = "org.springframework.beans.factory.BeanCreationException"
        conf = classify_confidence(text)
        assert conf >= 0.8

    def test_no_matching_keywords(self):
        """Unrecognizable error text → low confidence."""
        text = "flaky test timed out after 30 seconds"
        conf = classify_confidence(text)
        assert conf < 0.4

    def test_vague_error_low_confidence(self):
        """Vague error → low confidence."""
        text = "something went wrong"
        conf = classify_confidence(text)
        assert conf < 0.4

    def test_multiple_keyword_matches_boost(self):
        """Multiple matching keywords boost confidence."""
        text = "AssertionError: expected:<2> but was:<3> - assertion failed"
        conf = classify_confidence(text)
        # Should have more matches than single keyword → higher confidence
        assert conf >= 0.8

    def test_matched_categories_boost(self):
        """Keywords from multiple categories boost confidence."""
        text = "AssertionError: expected compilation error but got different"
        # Matches both assertion and compile keywords
        conf = classify_confidence(text)
        assert conf > 0.0

    def test_non_ascii_penalized(self):
        """Non-ASCII text is penalized."""
        # High-confidence English text
        eng = "AssertionError: expected:<2> but was:<3>"
        eng_conf = classify_confidence(eng)

        # Same text with non-ASCII
        non_ascii = "AssertionError: erwartete:<2> aber war:<3>"
        non_ascii_conf = classify_confidence(non_ascii)

        assert non_ascii_conf < eng_conf

    def test_mixed_signals_reduced(self):
        """Both assertion and compile keywords → reduced confidence."""
        text = "AssertionError: expected compilation to fail"
        conf = classify_confidence(text)
        # Mixed signals reduce confidence by half
        assert conf <= 0.5

    def test_single_keyword_baseline(self):
        """Single keyword match gives baseline confidence."""
        text = "expected"
        conf = classify_confidence(text)
        # avg_weight = 0.6, base = min(0.9, 0.6 + 0.15) = 0.75
        assert conf >= 0.7

    def test_toequal_keyword(self):
        """JS toEqual keyword detected."""
        text = "Expected value to equal 42"
        conf = classify_confidence(text)
        assert conf >= 0.5

    def test_tomatch_keyword(self):
        """JS toMatch keyword detected."""
        text = "Expected string to match pattern"
        conf = classify_confidence(text)
        assert conf >= 0.5

    def test_non_ascii_penalized(self):
        """Non-ASCII text is penalized."""
        text = "AssertionError: expected «valeur» but was «autre»"
        conf = classify_confidence(text)
        # Has non-ASCII → 0.3x penalty; base ~1.0, so conf ~0.3
        assert conf <= 0.3

    def test_whitespace_only(self):
        """Whitespace-only text → 0.0."""
        assert classify_confidence("   \n  ") == 0.0

    def test_multiple_assertion_keywords_boost(self):
        """Multiple assertion keywords boost confidence."""
        text = "AssertionError: expected 5 but was 3, expected 10 but was 0"
        conf = classify_confidence(text)
        assert conf >= 0.8

    def test_compile_with_cannot_find_symbol(self):
        """Compile error with 'cannot find symbol' gets high confidence (two keyword matches)."""
        text = "Compilation failed: cannot find symbol"
        conf = classify_confidence(text)
        # "compilat" + "cannot find symbol" → 2 matches, boosted
        assert conf >= 0.8

    def test_bean_creation_error_high_confidence(self):
        """Spring bean creation error → high confidence."""
        text = "BeanCreationException: Error creating bean 'userService'"
        conf = classify_confidence(text)
        assert conf >= 0.7

    def test_unknown_error_no_keywords(self):
        """Unrecognizable error text → low confidence (no keyword matches)."""
        text = "Something random happened in the service layer"
        conf = classify_confidence(text)
        # No keyword matches → returns 0.15
        assert conf == pytest.approx(0.15)

    def test_exact_0_8_threshold(self):
        """Single assertion keyword gives ≥ 0.8."""
        text = "AssertionError"
        conf = classify_confidence(text)
        # avg_weight = 0.8 (AssertionError weight), base = min(0.9, 0.8 + 0.15) = 0.9
        assert conf >= 0.8

    def test_single_compile_keyword(self):
        """Single compile keyword."""
        text = "compilation error"
        conf = classify_confidence(text)
        assert conf >= 0.5

    def test_high_confidence_junit_error(self):
        """JUnit assertion error → high confidence."""
        text = "junit.framework.AssertionError: expected"
        conf = classify_confidence(text)
        assert conf >= 0.8

    def test_many_keywords_capped_at_1_0(self):
        """Very many keywords should be capped at 1.0."""
        text = "AssertionError expected received but was expected expected received"
        conf = classify_confidence(text)
        assert conf <= 1.0
        assert conf >= 0.9


# ── _classify — diagnostic classification ──────────────────────────────────────

class TestClassify:

    def _state(self, **kw):
        base = {
            "session_id": "t", "task_id": "t", "task_description": "",
            "project_mode": "greenfield", "module_type": "backend-domain",
            "verifier_verdict": None,
        }
        base.update(kw)
        return base

    def test_both_phases_pass(self):
        """Both phases pass → PASS."""
        result = _classify(True, True, [], [], self._state())
        assert result == str(DiagnosticVerdict.PASS.value)

    def test_phase1_fails(self):
        """Phase 1 fails → FIX_IMPL."""
        result = _classify(False, True, [], [], self._state())
        assert result == str(DiagnosticVerdict.FIX_IMPL.value)

    def test_both_phases_fail(self):
        """Both phases fail → FIX_IMPL (p1 dominates)."""
        result = _classify(False, False, [], [], self._state())
        assert result == str(DiagnosticVerdict.FIX_IMPL.value)

    def test_p1_pass_p2_fail_assertion(self):
        """Phase 1 pass, phase 2 fail with assertion → FIX_TEST."""
        failures = [{"message": "AssertionError: expected 200 but got 500"}]
        result = _classify(True, False, failures, [], self._state())
        assert result == str(DiagnosticVerdict.FIX_TEST.value)

    def test_p1_pass_p2_fail_compile(self):
        """Phase 1 pass, phase 2 fail with compile error → FIX_IMPL."""
        failures = [{"message": "Compilation error in test"}]
        result = _classify(True, False, failures, [], self._state())
        assert result == str(DiagnosticVerdict.FIX_IMPL.value)

    def test_p1_pass_p2_fail_with_critical_finding(self):
        """Critical critic finding → FIX_IMPL when p1 passes, p2 fails, no assertion/compile keywords."""
        failures = [{"message": "Some generic failure"}]
        findings = [{"severity": "critical", "critic": "security", "message": "X", "resolution_hint": "Y"}]
        result = _classify(True, False, failures, findings, self._state())
        assert result == str(DiagnosticVerdict.FIX_IMPL.value)

    def test_p1_pass_p2_fail_no_clear_signal(self):
        """p1 pass, p2 fail, no assertion/compile keywords → FIX_IMPL (has failure text)."""
        failures = [{"message": "Something went wrong in the test"}]
        result = _classify(True, False, failures, [], self._state())
        assert result == str(DiagnosticVerdict.FIX_IMPL.value)

    def test_p1_pass_p2_fail_empty_failure_text(self):
        """p1 pass, p2 fail, empty failure text → AMBIGUOUS."""
        failures = [{"message": ""}]
        result = _classify(True, False, failures, [], self._state())
        assert result == str(DiagnosticVerdict.AMBIGUOUS.value)

    def test_both_pass_not_overall_pass(self):
        """Both phases pass but overall_pass=False with failure text → FIX_IMPL (perf/visual broke)."""
        failures = [{"message": "Performance threshold exceeded"}]
        result = _classify(True, True, failures, [], self._state(), overall_pass=False)
        assert result == str(DiagnosticVerdict.FIX_IMPL.value)

    def test_both_pass_not_overall_pass_no_failure_text(self):
        """Both phases pass, overall_pass=False, empty failure → AMBIGUOUS."""
        result = _classify(True, True, [{"message": ""}], [], self._state(), overall_pass=False)
        assert result == str(DiagnosticVerdict.AMBIGUOUS.value)

    def test_compile_keywords_trigger_fix_impl(self):
        """Compile keywords → FIX_IMPL."""
        failures = [{"message": "cannot find symbol UserService"}]
        result = _classify(False, True, failures, [], self._state())
        assert result == str(DiagnosticVerdict.FIX_IMPL.value)

    def test_module_not_found_trigger_fix_impl(self):
        """Module not found → FIX_IMPL."""
        failures = [{"message": "Module './utils' not found"}]
        result = _classify(False, True, failures, [], self._state())
        assert result == str(DiagnosticVerdict.FIX_IMPL.value)

    def test_syntax_error_trigger_fix_impl(self):
        """Syntax error → FIX_IMPL."""
        failures = [{"message": "SyntaxError: unexpected token"}]
        result = _classify(False, True, failures, [], self._state())
        assert result == str(DiagnosticVerdict.FIX_IMPL.value)

    def test_junit_assertion_error(self):
        """JUnit assertion error → FIX_TEST."""
        failures = [{"message": "junit.framework.AssertionError: expected"}]
        result = _classify(True, False, failures, [], self._state())
        assert result == str(DiagnosticVerdict.FIX_TEST.value)

    def test_expect_received_to_be(self):
        """JS expect/received/toBe → FIX_TEST."""
        failures = [{"message": "expect(received).toBe(expected)"}]
        result = _classify(True, False, failures, [], self._state())
        assert result == str(DiagnosticVerdict.FIX_TEST.value)

    def test_expected_received_but_was(self):
        """'expected' and 'but was' → FIX_TEST."""
        failures = [{"message": "expected: 42 but was: 0"}]
        result = _classify(True, False, failures, [], self._state())
        assert result == str(DiagnosticVerdict.FIX_TEST.value)

    def test_expected_angle_bracket(self):
        """'expected:<>' → FIX_TEST."""
        failures = [{"message": "expected:<true> but was:<false>"}]
        result = _classify(True, False, failures, [], self._state())
        assert result == str(DiagnosticVerdict.FIX_TEST.value)

    def test_toequal_trigger_fix_test(self):
        """JS toEqual → FIX_TEST."""
        failures = [{"message": "toEqual failed"}]
        result = _classify(True, False, failures, [], self._state())
        assert result == str(DiagnosticVerdict.FIX_TEST.value)

    def test_tomatch_trigger_fix_test(self):
        """JS toMatch → FIX_TEST."""
        failures = [{"message": "toMatch failed"}]
        result = _classify(True, False, failures, [], self._state())
        assert result == str(DiagnosticVerdict.FIX_TEST.value)

    def test_critical_finding_overrides(self):
        """Critical finding → FIX_IMPL when p1 passes, p2 fails, no assertion/compile keywords."""
        failures = [{"message": "Generic test failure"}]
        findings = [{"severity": "critical", "critic": "security", "message": "X", "resolution_hint": "Y"}]
        result = _classify(True, False, failures, findings, self._state())
        assert result == str(DiagnosticVerdict.FIX_IMPL.value)

    def test_both_phases_pass_returns_pass(self):
        """When both phases pass, critical findings are NOT checked — returns PASS."""
        findings = [{"severity": "critical", "critic": "security", "message": "X", "resolution_hint": "Y"}]
        result = _classify(True, True, [], findings, self._state())
        assert result == str(DiagnosticVerdict.PASS.value)

    def test_warning_finding_does_not_trigger_fix_impl(self):
        """Warning finding does NOT trigger FIX_IMPL."""
        findings = [{"severity": "warning", "critic": "style", "message": "X", "resolution_hint": "Y"}]
        result = _classify(True, True, [], findings, self._state())
        assert result == str(DiagnosticVerdict.PASS.value)

    def test_info_finding_does_not_trigger_fix_impl(self):
        """Info finding does NOT trigger FIX_IMPL."""
        findings = [{"severity": "info", "critic": "style", "message": "X", "resolution_hint": "Y"}]
        result = _classify(True, True, [], findings, self._state())
        assert result == str(DiagnosticVerdict.PASS.value)

    def test_multiple_warning_findings(self):
        """Multiple warning findings do NOT trigger FIX_IMPL."""
        findings = [
            {"severity": "warning", "critic": "style", "message": "X", "resolution_hint": "Y"},
            {"severity": "warning", "critic": "style", "message": "Z", "resolution_hint": "W"},
        ]
        result = _classify(True, True, [], findings, self._state())
        assert result == str(DiagnosticVerdict.PASS.value)

    def test_empty_failure_text_no_clear_signal(self):
        """Empty failure text with p1 pass, p2 fail → AMBIGUOUS."""
        result = _classify(True, False, [], [], self._state())
        assert result == str(DiagnosticVerdict.AMBIGUOUS.value)


# ── _classify_with_llm ─────────────────────────────────────────────────────────


class TestClassifyWithLlm:

    @pytest.fixture
    def mock_deps(self):
        from unittest.mock import AsyncMock
        deps = AsyncMock()
        deps.agent.run_task = AsyncMock(return_value=AgentResult(
            content="FIX_IMPL", tool_calls=[], finish_reason="stop",
            input_tokens=100, output_tokens=10,
        ))
        return deps

    def _deps_with_response(self, response):
        from unittest.mock import AsyncMock
        deps = AsyncMock()
        deps.agent.run_task = AsyncMock(return_value=AgentResult(
            content=response, tool_calls=[], finish_reason="stop",
            input_tokens=100, output_tokens=10,
        ))
        return deps

    async def test_llm_returns_valid_classification(self, mock_deps):
        """LLM returns valid classification → used directly."""
        from sacv.nodes.verifier import _classify_with_llm
        result = await _classify_with_llm(
            p1_passed=True, p2_passed=False,
            failures=[{"message": "compilation error in UserService.java"}],
            findings=[], state={}, overall_pass=False, deps=mock_deps,
        )
        assert result == "FIX_IMPL"

    async def test_llm_pass_when_both_phases_pass(self):
        """Both phases pass → LLM returns PASS."""
        from sacv.nodes.verifier import _classify_with_llm
        deps = self._deps_with_response("PASS")
        result = await _classify_with_llm(
            p1_passed=True, p2_passed=True,
            failures=[], findings=[], state={}, overall_pass=True, deps=deps,
        )
        assert result == "PASS"

    async def test_llm_fix_test_for_phase2_assertion_failure(self):
        """Phase 2 assertion failure → LLM returns FIX_TEST."""
        from sacv.nodes.verifier import _classify_with_llm
        deps = self._deps_with_response("FIX_TEST")
        result = await _classify_with_llm(
            p1_passed=True, p2_passed=False,
            failures=[{"message": "AssertionError: expected:<2> but was:<3>"}],
            findings=[], state={}, overall_pass=False, deps=deps,
        )
        assert result == "FIX_TEST"

    async def test_llm_fallback_on_exception(self, mock_deps):
        """LLM call raises → falls back to keyword _classify."""
        from sacv.nodes.verifier import _classify_with_llm
        from unittest.mock import AsyncMock
        mock_deps.agent.run_task = AsyncMock(side_effect=RuntimeError("network error"))
        # Phase 1 fails → _classify returns FIX_IMPL regardless of LLM
        result = await _classify_with_llm(
            p1_passed=False, p2_passed=True,
            failures=[], findings=[], state={}, overall_pass=False, deps=mock_deps,
        )
        assert result == str(DiagnosticVerdict.FIX_IMPL.value)

    async def test_llm_invalid_output_fallback(self, mock_deps):
        """LLM returns garbage → falls back to keyword _classify."""
        from sacv.nodes.verifier import _classify_with_llm
        from unittest.mock import AsyncMock
        mock_deps.agent.run_task = AsyncMock(return_value=AgentResult(
            content="maybe FIX_IMPL?", tool_calls=[], finish_reason="stop",
            input_tokens=100, output_tokens=10,
        ))
        # Phase 1 fails → fallback _classify returns FIX_IMPL
        result = await _classify_with_llm(
            p1_passed=False, p2_passed=True,
            failures=[], findings=[], state={}, overall_pass=False, deps=mock_deps,
        )
        assert result == str(DiagnosticVerdict.FIX_IMPL.value)

    async def test_llm_amiguous_for_no_failure_text(self):
        """Both phases pass, overall_pass=False, empty text → AMBIGUOUS."""
        from sacv.nodes.verifier import _classify_with_llm
        deps = self._deps_with_response("AMBIGUOUS")
        result = await _classify_with_llm(
            p1_passed=True, p2_passed=True,
            failures=[{"message": ""}], findings=[], state={},
            overall_pass=False, deps=deps,
        )
        assert result == "AMBIGUOUS"

    async def test_llm_returns_ambiguous_for_phase1_failure(self):
        """LLM can return AMBIGUOUS even when p1 fails — the LLM output is trusted.

        This is a design characteristic: the LLM classifier is called regardless
        of phase results. If the LLM decides the failure is ambiguous (e.g.,
        the error message is unparseable), that decision is accepted without
        validation against the known phase results. The fallback _classify()
        is only used when the LLM raises an exception or returns invalid text.
        """
        from sacv.nodes.verifier import _classify_with_llm
        deps = self._deps_with_response("AMBIGUOUS")
        result = await _classify_with_llm(
            p1_passed=False, p2_passed=True,
            failures=[{"message": "Some unparseable error"}],
            findings=[], state={}, overall_pass=False, deps=deps,
        )
        # LLM output is trusted — AMBIGUOUS is returned even though p1 failed
        assert result == "AMBIGUOUS"

    async def test_llm_returns_mixed_case_output(self):
        """LLM returns mixed-case text like 'fix_impl' → uppercased to valid classification."""
        from sacv.nodes.verifier import _classify_with_llm
        deps = self._deps_with_response("fix_impl")
        result = await _classify_with_llm(
            p1_passed=False, p2_passed=True,
            failures=[], findings=[], state={}, overall_pass=False, deps=deps,
        )
        assert result == "FIX_IMPL"

    async def test_llm_returns_mixed_case_with_underscore(self):
        """LLM returns 'Fix_Test' → uppercased to FIX_TEST."""
        from sacv.nodes.verifier import _classify_with_llm
        deps = self._deps_with_response("Fix_Test")
        result = await _classify_with_llm(
            p1_passed=True, p2_passed=False,
            failures=[{"message": "AssertionError"}],
            findings=[], state={}, overall_pass=False, deps=deps,
        )
        assert result == "FIX_TEST"
