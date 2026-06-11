"""
tests/unit/test_preflight_repair_suggestions.py
=================================================
Unit tests for _compute_repair_suggestions in preflight_node.py.

Tests cover:
1. Symbol extraction from "cannot find symbol" Java errors
2. Symbol extraction from "cannot find name" TypeScript errors
3. Symbol extraction from "cannot find module" TypeScript errors
4. Symbol extraction from "'X' does not exist" errors
5. Fallback symbol extraction when no regex patterns match
6. Unique symbols only (deduplication preserves order)
7. Symbol limit of 5
8. Architecture violation suggestions
9. Cross-stack error suggestions
10. Blast radius suggestions
11. All four categories combined
12. Empty inputs produce empty suggestions
"""
from __future__ import annotations

import pytest

from sacv.nodes.preflight_node import _compute_repair_suggestions


class TestComputeRepairSuggestionsCompile:

    def test_cannot_find_symbol_java(self):
        """Java 'cannot find symbol' error extracts the missing symbol name."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[{
                "file": "UserService.java", "line": 10,
                "code": "CE", "message": "cannot find symbol method findById",
            }],
            arch_violations=[], cross_stack_errors=[], blast_errors=[],
            module="backend-domain", blast_map={},
        )
        assert len(suggestions) == 1
        assert suggestions[0]["category"] == "compile"
        assert "findById" in suggestions[0]["text"]
        assert "UserService.java" in suggestions[0]["text"]

    def test_cannot_find_name_typescript(self):
        """TypeScript 'cannot find name' error extracts the missing name."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[{
                "file": "App.tsx", "line": 20,
                "code": "TS2304", "message": "cannot find name 'UserService'",
            }],
            arch_violations=[], cross_stack_errors=[], blast_errors=[],
            module="frontend-feature", blast_map={},
        )
        assert len(suggestions) == 1
        assert "UserService" in suggestions[0]["text"]

    def test_cannot_find_module_typescript(self):
        """TypeScript 'cannot find module' error extracts the module path."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[{
                "file": "App.tsx", "line": 5,
                "code": "TS2307",
                "message": "cannot find module '@/utils/helpers'",
            }],
            arch_violations=[], cross_stack_errors=[], blast_errors=[],
            module="frontend-feature", blast_map={},
        )
        assert len(suggestions) == 1
        assert "@/utils/helpers" in suggestions[0]["text"]

    def test_does_not_exist_pattern(self):
        """'X' does not exist pattern extracts the symbol name."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[{
                "file": "X.ts", "line": 1,
                "code": "TS2339",
                "message": "'nonexistent' does not exist on type 'Props'",
            }],
            arch_violations=[], cross_stack_errors=[], blast_errors=[],
            module="frontend-feature", blast_map={},
        )
        assert len(suggestions) == 1
        assert "nonexistent" in suggestions[0]["text"]

    def test_fallback_symbol_extraction(self):
        """When no regex pattern matches, extracts alphanumeric words >2 chars."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[{
                "file": "X.java", "line": 1,
                "code": "CE", "message": "some weird error about repository",
            }],
            arch_violations=[], cross_stack_errors=[], blast_errors=[],
            module="backend-domain", blast_map={},
        )
        assert len(suggestions) == 1
        # "repository" is the longest alphanumeric word >2 chars
        assert "repository" in suggestions[0]["text"].lower()

    def test_symbols_deduplicated_order_preserved(self):
        """Duplicate symbols are deduplicated; first occurrence order preserved."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[
                {"file": "X.java", "line": 1, "code": "CE",
                 "message": "cannot find symbol method findById"},
                {"file": "X.java", "line": 2, "code": "CE",
                 "message": "cannot find symbol method findById"},
                {"file": "X.java", "line": 3, "code": "CE",
                 "message": "cannot find symbol method save"},
            ],
            arch_violations=[], cross_stack_errors=[], blast_errors=[],
            module="backend-domain", blast_map={},
        )
        # Only two unique symbols: findById and save
        text = suggestions[0]["text"]
        assert text.count("findById") == 1
        assert "save" in text
        # Should not have "findById, findById"
        assert "findById, findById" not in text

    def test_symbol_limit_of_five(self):
        """At most 5 unique symbols per file."""
        errors = [
            {"file": "X.java", "line": i, "code": "CE",
             "message": f"cannot find symbol method method{i}"}
            for i in range(20)
        ]
        suggestions = _compute_repair_suggestions(
            lsp_errors=errors, arch_violations=[], cross_stack_errors=[],
            blast_errors=[], module="backend-domain", blast_map={},
        )
        text = suggestions[0]["text"]
        # Should contain at most 5 symbols
        for i in range(5):
            assert f"method{i}" in text
        # method5+ should not be in the text (limited to 5)
        assert "method5" not in text

    def test_multiple_files_get_separate_suggestions(self):
        """Each file with errors gets its own suggestion entry."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[
                {"file": "A.java", "line": 1, "code": "CE",
                 "message": "cannot find symbol method foo"},
                {"file": "B.java", "line": 1, "code": "CE",
                 "message": "cannot find symbol method bar"},
            ],
            arch_violations=[], cross_stack_errors=[], blast_errors=[],
            module="backend-domain", blast_map={},
        )
        # Two separate suggestions, one per file
        assert len(suggestions) == 2
        assert "A.java" in suggestions[0]["text"]
        assert "foo" in suggestions[0]["text"]
        assert "B.java" in suggestions[1]["text"]
        assert "bar" in suggestions[1]["text"]

    def test_three_or_more_symbols_changes_hint(self):
        """When 3+ unique symbols, hint says 'Check method signatures'."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[
                {"file": "X.java", "line": 1, "code": "CE",
                 "message": "cannot find symbol method foo"},
                {"file": "X.java", "line": 2, "code": "CE",
                 "message": "cannot find symbol method bar"},
                {"file": "X.java", "line": 3, "code": "CE",
                 "message": "cannot find symbol method baz"},
            ],
            arch_violations=[], cross_stack_errors=[], blast_errors=[],
            module="backend-domain", blast_map={},
        )
        assert "Check method signatures" in suggestions[0]["text"]

    def test_less_than_three_symbols_says_add_import(self):
        """When <3 unique symbols, hint says 'Add import or verify method name'."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[
                {"file": "X.java", "line": 1, "code": "CE",
                 "message": "cannot find symbol method foo"},
            ],
            arch_violations=[], cross_stack_errors=[], blast_errors=[],
            module="backend-domain", blast_map={},
        )
        assert "Add import or verify method name" in suggestions[0]["text"]


class TestComputeRepairSuggestionsArch:

    def test_arch_violation_suggestion(self):
        """Arch violations produce architecture category suggestions."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[],
            arch_violations=[{
                "rule": "no-circle", "source_file": "A.java",
                "target_file": "B.java", "message": "Circular dependency",
            }],
            cross_stack_errors=[], blast_errors=[],
            module="backend-domain", blast_map={},
        )
        assert len(suggestions) == 1
        assert suggestions[0]["category"] == "architecture"
        assert "no-circle" in suggestions[0]["text"]
        assert "A.java" in suggestions[0]["text"]
        assert "B.java" in suggestions[0]["text"]

    def test_arch_violation_limits_to_five(self):
        """Only first 5 arch violations produce suggestions."""
        violations = [
            {"rule": f"R{i}", "source_file": f"A{i}.java",
             "target_file": f"B{i}.java", "message": f"violation {i}"}
            for i in range(10)
        ]
        suggestions = _compute_repair_suggestions(
            lsp_errors=[], arch_violations=violations,
            cross_stack_errors=[], blast_errors=[],
            module="backend-domain", blast_map={},
        )
        assert len(suggestions) == 5


class TestComputeRepairSuggestionsCrossStack:

    def test_cross_stack_error_suggestion(self):
        """Cross-stack errors produce cross_stack category suggestions."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[], arch_violations=[],
            cross_stack_errors=[{
                "file": "frontend/src/api/types.ts",
                "message": "Type mismatch: User.id should be string",
            }],
            blast_errors=[],
            module="backend-domain", blast_map={},
        )
        assert len(suggestions) == 1
        assert suggestions[0]["category"] == "cross_stack"
        assert "Type mismatch" in suggestions[0]["text"]
        assert "frontend/src/api/types.ts" in suggestions[0]["text"]

    def test_cross_stack_error_limits_to_five(self):
        """Only first 5 cross-stack errors produce suggestions."""
        errors = [
            {"file": f"T{i}.ts", "message": f"mismatch {i}"}
            for i in range(10)
        ]
        suggestions = _compute_repair_suggestions(
            lsp_errors=[], arch_violations=[],
            cross_stack_errors=errors, blast_errors=[],
            module="backend-domain", blast_map={},
        )
        assert len(suggestions) == 5


class TestComputeRepairSuggestionsBlast:

    def test_blast_radius_suggestion(self):
        """Blast radius errors produce blast_radius category suggestions."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[], arch_violations=[],
            cross_stack_errors=[],
            blast_errors=[{
                "rule": "blast_radius_limit",
                "message": "Change affects 60 files (limit: 50).",
            }],
            module="backend-domain",
            blast_map={"affected_files": [f"F{i}.java" for i in range(60)]},
        )
        assert len(suggestions) == 1
        assert suggestions[0]["category"] == "blast_radius"
        assert "60 files" in suggestions[0]["text"]
        assert "Consider splitting" in suggestions[0]["text"]

    def test_blast_radius_truncates_at_ten(self):
        """Blast radius suggestion lists at most 10 files."""
        affected = [f"F{i}.java" for i in range(20)]
        suggestions = _compute_repair_suggestions(
            lsp_errors=[], arch_violations=[],
            cross_stack_errors=[],
            blast_errors=[{"rule": "blast_radius_limit", "message": "too wide"}],
            module="backend-domain", blast_map={"affected_files": affected},
        )
        text = suggestions[0]["text"]
        for i in range(10):
            assert f"F{i}.java" in text
        # Should have ellipsis
        assert "..." in text


class TestComputeRepairSuggestionsCombined:

    def test_all_categories_produce_separate_suggestions(self):
        """All four categories produce independent suggestions."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[{
                "file": "X.java", "line": 1, "code": "CE",
                "message": "cannot find symbol method foo",
            }],
            arch_violations=[{
                "rule": "no-dep", "source_file": "A.java",
                "target_file": "B.java", "message": "bad dep",
            }],
            cross_stack_errors=[{
                "file": "types.ts", "message": "type mismatch",
            }],
            blast_errors=[{"rule": "blast_radius_limit", "message": "too wide"}],
            module="backend-domain",
            blast_map={"affected_files": [f"F{i}.java" for i in range(60)]},
        )
        categories = [s["category"] for s in suggestions]
        assert categories == ["compile", "architecture", "cross_stack", "blast_radius"]

    def test_empty_inputs_produce_empty_suggestions(self):
        """All empty lists → no suggestions."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[], arch_violations=[],
            cross_stack_errors=[], blast_errors=[],
            module="backend-domain", blast_map={},
        )
        assert suggestions == []


class TestComputeRepairSuggestionsEdgeCases:

    def test_lsp_error_without_symbol_produces_fallback(self):
        """LSP error message without recognizable patterns uses fallback."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[{
                "file": "X.java", "line": 1, "code": "CE",
                "message": "incompatible types: expected String but got int",
            }],
            arch_violations=[], cross_stack_errors=[], blast_errors=[],
            module="backend-domain", blast_map={},
        )
        assert len(suggestions) == 1
        # Fallback takes last alphanumeric word >2 chars: "int"
        assert "int" in suggestions[0]["text"].lower()

    def test_lsp_error_with_quoted_symbol(self):
        """Symbol in quotes is extracted by the 'does not exist' pattern."""
        suggestions = _compute_repair_suggestions(
            lsp_errors=[{
                "file": "X.ts", "line": 1, "code": "TS",
                "message": "'myVar' does not exist",
            }],
            arch_violations=[], cross_stack_errors=[], blast_errors=[],
            module="frontend-feature", blast_map={},
        )
        assert "myVar" in suggestions[0]["text"]

    def test_limits_lsp_errors_to_ten_for_symbol_extraction(self):
        """Only first 10 LSP errors are processed for symbol extraction."""
        errors = [
            {"file": "X.java", "line": i, "code": "CE",
             "message": f"cannot find symbol method method{i}"}
            for i in range(15)
        ]
        suggestions = _compute_repair_suggestions(
            lsp_errors=errors, arch_violations=[], cross_stack_errors=[],
            blast_errors=[], module="backend-domain", blast_map={},
        )
        # Should only process first 10 errors
        text = suggestions[0]["text"]
        for i in range(5):  # 5 unique symbols max
            assert f"method{i}" in text
        # method10+ should not be present (beyond the 10-error limit)
        assert "method10" not in text
