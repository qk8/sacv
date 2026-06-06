"""
tests/unit/test_diagnostic_confidence.py
=========================================
Unit tests for classify_confidence — measures how confidently the
keyword-based _classify output matches a known diagnostic category.

Issue MEDIUM-001: _classify uses string matching (assertionerror,
"expected", "compilat", etc.) to determine FIX_IMPL vs FIX_TEST vs
AMBIGUOUS. This works for English-language test output but will fail
with localized test frameworks, custom error messages, or non-ASCII
test output.

Fix: Add classify_confidence() that returns a confidence score [0, 1]
for the keyword-based diagnostic. When confidence is low (< 0.5),
the system can choose to escalate rather than blindly retry.

Tests verify:
1. Strong keyword matches produce high confidence
2. Weak/no keyword matches produce low confidence
3. Mixed signals (both assertion and compile keywords) produce
   lower confidence than clean matches
4. Empty failure messages produce low confidence
5. Non-ASCII / localized messages produce low confidence
"""
from __future__ import annotations

import pytest

from sacv.nodes.verifier import classify_confidence


class TestClassifyConfidence:

    def test_clear_assertion_error_high_confidence(self):
        """'AssertionError: expected:<2> but was:<3>' → clear FIX_TEST signal."""
        result = classify_confidence("AssertionError: expected:<2> but was:<3>")
        assert result >= 0.8

    def test_clear_compile_error_high_confidence(self):
        """'error: cannot find symbol' → clear FIX_IMPL signal."""
        result = classify_confidence("error: cannot find symbol")
        assert result >= 0.8

    def test_clear_syntax_error_high_confidence(self):
        """'error: syntax error on line 5' → clear FIX_IMPL signal."""
        result = classify_confidence("error: syntax error on line 5")
        assert result >= 0.8

    def test_clear_module_not_found_high_confidence(self):
        """'error: module not found' → clear FIX_IMPL signal."""
        result = classify_confidence("error: module not found")
        assert result >= 0.8

    def test_empty_message_low_confidence(self):
        """Empty failure messages produce low confidence."""
        result = classify_confidence("")
        assert result < 0.3

    def test_generic_message_low_confidence(self):
        """Vague messages without clear keywords produce low confidence."""
        result = classify_confidence("flaky test timed out")
        assert result < 0.5

    def test_non_ascii_message_low_confidence(self):
        """Non-ASCII messages produce low confidence (keywords are English)."""
        result = classify_confidence("Fehler: Symbol nicht gefunden")
        assert result < 0.5

    def test_japanese_message_low_confidence(self):
        """Japanese error messages produce low confidence."""
        result = classify_confidence("エラー: シンボルが見つかりません")
        assert result < 0.5

    def test_mixed_assertion_and_compile_lower_confidence(self):
        """When both assertion and compile keywords are present,
        confidence is lower (the classification is ambiguous)."""
        result = classify_confidence(
            "AssertionError: expected:<2> | error: cannot find symbol"
        )
        # Mixed signals reduce confidence
        assert result < 0.7

    def test_js_expect_to_be_high_confidence(self):
        """'expect(received).toBe(expected)' → clear FIX_TEST signal."""
        result = classify_confidence("expect(received).toBe(expected) // Object.is equality")
        assert result >= 0.8

    def test_junit_assertion_error_high_confidence(self):
        """'junit.framework.AssertionError' → clear FIX_TEST signal."""
        result = classify_confidence("junit.framework.AssertionError: expected:<42>")
        assert result >= 0.8

    def test_toequal_high_confidence(self):
        """'toequal' keyword match → clear FIX_TEST signal."""
        result = classify_confidence("Expected value toequal: 42")
        assert result >= 0.8

    def test_single_keyword_match_medium_confidence(self):
        """A single keyword match produces medium confidence."""
        result = classify_confidence("assertionerror")
        assert 0.4 <= result <= 0.9

    def test_multiple_matching_keywords_increases_confidence(self):
        """More matching keywords → higher confidence."""
        single = classify_confidence("assertionerror")
        double = classify_confidence(
            "AssertionError: expected:<2> but was:<3>"
        )
        # double has more keyword hits (assertionerror, expected, but was)
        assert double >= single

    def test_bean_error_keywords_high_confidence(self):
        """Spring DI error keywords produce high confidence."""
        result = classify_confidence("BeanCreationException for 'userService'")
        assert result >= 0.8

    def test_playwright_expect_to_match_high_confidence(self):
        """'toMatch' assertion → clear FIX_TEST signal."""
        result = classify_confidence("expect(received).toMatch(expected)")
        assert result >= 0.8
