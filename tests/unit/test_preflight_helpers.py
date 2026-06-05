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
    _parse_lsp,
    _parse_arch,
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


# ── _parse_lsp ────────────────────────────────────────────────────────────────

class TestParseLsp:

    def test_frontend_parses_ts_error(self):
        output = "src/Login.tsx(5,10): error TS2322: Type 'string' is not assignable to type 'number'"
        errors = _parse_lsp(output, "frontend-feature")
        assert len(errors) == 1
        assert errors[0]["file"] == "src/Login.tsx"
        assert errors[0]["line"] == 5
        assert errors[0]["code"] == "TS2322"
        assert "Type 'string'" in errors[0]["message"]

    def test_frontend_parses_tsx_error(self):
        output = "src/App.tsx(10,3): error TS2345: Argument of type 'null'"
        errors = _parse_lsp(output, "frontend-data")
        assert len(errors) == 1
        assert errors[0]["file"] == "src/App.tsx"
        assert errors[0]["line"] == 10

    def test_frontend_parses_multiple_errors(self):
        output = (
            "src/A.tsx(1,2): error TS2322: err1\n"
            "src/B.tsx(3,4): error TS2345: err2\n"
        )
        errors = _parse_lsp(output, "frontend-feature")
        assert len(errors) == 2
        assert errors[0]["file"] == "src/A.tsx"
        assert errors[1]["file"] == "src/B.tsx"

    def test_frontend_limits_to_30(self):
        lines = [f"src/F{i}.tsx(1,1): error TS2322: err\n" for i in range(40)]
        errors = _parse_lsp("".join(lines), "frontend-feature")
        assert len(errors) == 30

    def test_frontend_empty_output(self):
        assert _parse_lsp("", "frontend-feature") == []

    def test_backend_parses_java_error(self):
        output = "[ERROR] UserService.java:[15,10] incompatible types"
        errors = _parse_lsp(output, "backend-domain")
        assert len(errors) == 1
        assert errors[0]["file"] == "UserService.java"
        assert errors[0]["line"] == 15
        assert errors[0]["code"] == "CE"
        assert "incompatible types" in errors[0]["message"]

    def test_backend_parses_multiple_java_errors(self):
        output = (
            "[ERROR] A.java:[1,2] error a\n"
            "[ERROR] B.java:[3,4] error b\n"
        )
        errors = _parse_lsp(output, "backend-api")
        assert len(errors) == 2
        assert errors[0]["file"] == "A.java"
        assert errors[1]["file"] == "B.java"

    def test_backend_limits_to_30(self):
        lines = [f"[ERROR] F{i}.java:[1,1] error\n" for i in range(40)]
        errors = _parse_lsp("".join(lines), "backend-domain")
        assert len(errors) == 30

    def test_backend_empty_output(self):
        assert _parse_lsp("", "backend-domain") == []

    def test_frontend_ignores_backend_format(self):
        output = "[ERROR] A.java:[1,2] error"
        errors = _parse_lsp(output, "frontend-feature")
        assert errors == []

    def test_backend_ignores_frontend_format(self):
        output = "src/A.tsx(1,2): error TS2322: err"
        errors = _parse_lsp(output, "backend-domain")
        assert errors == []


# ── _parse_arch ────────────────────────────────────────────────────────────────

class TestParseArch:

    def test_frontend_no_arch_test_returns_empty(self):
        output = "NO_ARCH_TEST"
        assert _parse_arch(output, "frontend-feature") == []

    def test_frontend_parses_dep_cruise_json(self):
        import json
        data = json.dumps([{
            "source": "src/ui/Login.tsx",
            "violations": [{
                "rule": {"name": "no-ui-to-db"},
                "to": {"resolved": "src/db/UserRepo.ts"},
            }],
        }])
        violations = _parse_arch(data, "frontend-feature")
        assert len(violations) == 1
        assert violations[0]["rule"] == "no-ui-to-db"
        assert violations[0]["source_file"] == "src/ui/Login.tsx"
        assert violations[0]["target_file"] == "src/db/UserRepo.ts"

    def test_frontend_parses_multiple_violations(self):
        import json
        data = json.dumps([{
            "source": "src/ui/A.tsx",
            "violations": [
                {"rule": {"name": "R1"}, "to": {"resolved": "X"}},
                {"rule": {"name": "R2"}, "to": {"resolved": "Y"}},
            ],
        }])
        violations = _parse_arch(data, "frontend-feature")
        assert len(violations) == 2
        assert violations[0]["rule"] == "R1"
        assert violations[1]["rule"] == "R2"

    def test_frontend_limits_to_20(self):
        import json
        violations = [{"rule": {"name": f"R{i}"}, "to": {"resolved": f"F{i}"}} for i in range(30)]
        data = json.dumps([{"source": "src/A.tsx", "violations": violations}])
        result = _parse_arch(data, "frontend-feature")
        assert len(result) == 20

    def test_frontend_invalid_json_returns_empty(self):
        assert _parse_arch("not json", "frontend-feature") == []

    def test_frontend_non_list_data_returns_empty(self):
        import json
        data = json.dumps({"not": "a list"})
        assert _parse_arch(data, "frontend-feature") == []

    def test_backend_delegates_to_java_archunit(self):
        output = "Architecture Violation - required rule 'no-layer': msg\n  - frame"
        violations = _parse_arch(output, "backend-domain")
        assert len(violations) == 1
        assert violations[0]["rule"] == "no-layer"

    def test_backend_no_arch_test_returns_empty(self):
        assert _parse_arch("NO_ARCH_TEST", "backend-domain") == []
