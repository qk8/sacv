"""
tests/unit/test_preflight_helpers.py
=====================================
Unit tests for preflight helper functions — pure, no I/O.
"""
from __future__ import annotations

import pytest
from sacv.nodes.preflight_node import (
    _arch_cmd,
    _parse_java_archunit,
)


class TestArchCmd:

    def test_frontend_returns_depcruise(self):
        result = _arch_cmd("frontend-feature")
        assert "dep" in result.lower() or "depcruise" in result.lower()
        assert "mvn" not in result

    def test_frontend_admin_returns_depcruise(self):
        result = _arch_cmd("frontend-admin")
        assert "dep" in result.lower() or "depcruise" in result.lower()

    def test_backend_domain_returns_mvn(self):
        result = _arch_cmd("backend-domain")
        assert "mvn" in result
        assert "ArchTest" in result or "*ArchTest" in result

    def test_backend_api_returns_mvn(self):
        result = _arch_cmd("backend-api")
        assert "mvn" in result

    def test_monorepo_returns_mvn(self):
        result = _arch_cmd("monorepo")
        assert "mvn" in result


class TestParseJavaArchunit:

    def test_empty_output(self):
        assert _parse_java_archunit("") == []

    def test_single_violation(self):
        output = (
            "Architecture Violation - required rule 'STATIC_TYPES': "
            "com.example.UserService should only read from com.example.repo\n"
            "  - com.example.UserService.findById(UserService.java:42)\n"
        )
        violations = _parse_java_archunit(output)
        assert len(violations) == 1
        assert violations[0]["rule"] == "STATIC_TYPES"
        assert "UserService" in violations[0]["message"]

    def test_multiple_violations_same_rule(self):
        output = (
            "Architecture Violation - required rule 'LAYERED_ARCHITECTURE': "
            "domain reads infra\n"
            "  - com.example.UserService.findById(UserService.java:42)\n"
            "Architecture Violation - required rule 'LAYERED_ARCHITECTURE': "
            "domain reads web\n"
            "  - com.example.controller.UserController.getUser(UserController.java:28)\n"
        )
        violations = _parse_java_archunit(output)
        assert len(violations) == 2
        assert violations[0]["rule"] == "LAYERED_ARCHITECTURE"
        assert violations[1]["rule"] == "LAYERED_ARCHITECTURE"

    def test_multiple_different_rules(self):
        output = (
            "Architecture Violation - required rule 'NO_STATIC_IMPORTS': "
            "static import found\n"
            "  - import static com.example.Util.*\n"
            "Architecture Violation - required rule 'PACKAGE_NAMES': "
            "package name invalid\n"
            "  - com.example.invalidPackage\n"
        )
        violations = _parse_java_archunit(output)
        assert len(violations) == 2
        rules = {v["rule"] for v in violations}
        assert "NO_STATIC_IMPORTS" in rules
        assert "PACKAGE_NAMES" in rules

    def test_truncates_to_20(self):
        # Build output with 25 violations
        parts = []
        for i in range(25):
            parts.append(
                f"Architecture Violation - required rule 'RULE-{i}': msg{i}\n"
                f"  - frame{i}\n"
            )
        violations = _parse_java_archunit("".join(parts))
        assert len(violations) == 20

    def test_skips_meta_lines(self):
        """Lines starting with 'There is' are skipped, not treated as violations."""
        output = (
            "Architecture Violation - required rule 'NO_RUNTIME': runtime call\n"
            "  - com.example.Service.doWork(Service.java:10)\n"
            "There are 0 other violations\n"
        )
        violations = _parse_java_archunit(output)
        assert len(violations) == 1
        assert violations[0]["rule"] == "NO_RUNTIME"

    def test_no_violation_detail_after_rule(self):
        """A rule header without a detail line produces no violation."""
        output = (
            "Architecture Violation - required rule 'EMPTY_RULE':\n"
            "Some other line\n"
        )
        violations = _parse_java_archunit(output)
        assert violations == []

    def test_source_and_target_are_unknown(self):
        """ArchUnit parser does not extract source/target — always 'unknown'."""
        output = (
            "Architecture Violation - required rule 'R': msg\n"
            "  - com.example.Foo.bar(Foo.java:1)\n"
        )
        v = _parse_java_archunit(output)[0]
        assert v["source_file"] == "unknown"
        assert v["target_file"] == "unknown"

    def test_rule_with_special_chars(self):
        output = (
            "Architecture Violation - required rule 'no-circle[A,B]': cycle\n"
            "  - A calls B\n"
        )
        violations = _parse_java_archunit(output)
        assert len(violations) == 1
        assert "no-circle" in violations[0]["rule"]

    def test_no_violations_no_output(self):
        output = "No violations found.\n"
        assert _parse_java_archunit(output) == []

    def test_consecutive_blocks_no_cross_contamination(self):
        """Rule from block N does not leak into block N+1."""
        output = (
            "Architecture Violation - required rule 'RULE-A': msg-a\n"
            "  - frame-a\n"
            "Architecture Violation - required rule 'RULE-B': msg-b\n"
            "  - frame-b\n"
        )
        violations = _parse_java_archunit(output)
        assert len(violations) == 2
        assert violations[0]["rule"] == "RULE-A"
        assert violations[1]["rule"] == "RULE-B"
