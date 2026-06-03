"""
tests/unit/test_memory_consolidation_helpers.py
=================================================
Unit tests for memory_consolidation helper functions.

Tests cover:
1. _derive_pattern — pattern string generation
2. _extract_constraints — constraint extraction from findings/escalation
3. _extract_section — AGENTS.md section extraction
4. _splice_sections — AGENTS.md section replacement
5. _inject_depcruiser_rule — validation and injection
6. _inject_archunit_rule — validation and injection
7. _default_agents_md and _default_archunit_class
"""
from __future__ import annotations

import pytest

from sacv.nodes.memory_consolidation import (
    _derive_pattern, _extract_constraints,
    _extract_section, _splice_sections,
    _inject_depcruiser_rule, _inject_archunit_rule,
    _default_agents_md, _default_archunit_class,
)
from pathlib import Path


@pytest.mark.unit
class TestDerivePattern:

    def test_basic_module_and_mode(self):
        state = {
            "module_type": "backend-domain",
            "project_mode": "greenfield",
            "correction_state": {"attempt_count": 1},
            "verifier_verdict": {"test_result": "PASS"},
        }
        result = _derive_pattern(state)
        assert "module=backend-domain" in result
        assert "mode=greenfield" in result
        assert "resolved_in=1_attempts" in result

    def test_self_correction_reflected(self):
        state = {
            "module_type": "backend-domain",
            "project_mode": "greenfield",
            "correction_state": {"attempt_count": 3},
            "verifier_verdict": {"test_result": "PASS"},
        }
        result = _derive_pattern(state)
        assert "resolved_in=3_attempts" in result

    def test_stagnation_pattern_included(self):
        state = {
            "module_type": "backend-domain",
            "project_mode": "greenfield",
            "correction_state": {"attempt_count": 3, "stagnation_pattern": "semantic"},
            "verifier_verdict": {"test_result": "PASS"},
        }
        result = _derive_pattern(state)
        assert "stagnation=semantic" in result

    def test_replan_count_included(self):
        state = {
            "module_type": "backend-domain",
            "project_mode": "greenfield",
            "correction_state": {"attempt_count": 1},
            "verifier_verdict": {"test_result": "PASS"},
            "replan_count": 2,
        }
        result = _derive_pattern(state)
        assert "replanned=2x" in result

    def test_multiple_signals_combined(self):
        state = {
            "module_type": "frontend-feature",
            "project_mode": "brownfield",
            "correction_state": {"attempt_count": 2, "stagnation_pattern": "iteration"},
            "verifier_verdict": {"test_result": "PASS"},
            "replan_count": 1,
        }
        result = _derive_pattern(state)
        assert "module=frontend-feature" in result
        assert "mode=brownfield" in result
        assert "stagnation=iteration" in result
        assert "replanned=1x" in result
        assert "resolved_in=2_attempts" in result

    def test_failed_verdict_excludes_resolution(self):
        state = {
            "module_type": "backend-domain",
            "project_mode": "greenfield",
            "correction_state": {"attempt_count": 1},
            "verifier_verdict": {"test_result": "FAIL"},
        }
        result = _derive_pattern(state)
        assert "resolved_in" not in result
        assert "module=backend-domain" in result


@pytest.mark.unit
class TestExtractConstraints:

    def test_critical_finding_becomes_constraint(self):
        findings = [{
            "severity": "critical", "critic": "security",
            "message": "SQL injection in user query",
            "resolution_hint": "use parameterized queries",
        }]
        result = _extract_constraints(findings, None)
        assert len(result) == 1
        assert "[SECURITY]" in result[0]
        assert "SQL injection" in result[0]
        assert "use parameterized queries" in result[0]

    def test_non_critical_finding_ignored(self):
        findings = [{
            "severity": "warning", "critic": "style",
            "message": "Long method name",
            "resolution_hint": "shorten",
        }]
        result = _extract_constraints(findings, None)
        assert result == []

    def test_escalation_hints_become_constraints(self):
        escalation = {
            "resolution_hints": [
                {"hint": "Review the strategy"},
                {"hint": "Consider decomposition"},
            ]
        }
        result = _extract_constraints([], escalation)
        assert len(result) == 2
        assert "[HITL] Review the strategy" in result[0]
        assert "[HITL] Consider decomposition" in result[1]

    def test_both_findings_and_escalation(self):
        findings = [{
            "severity": "critical", "critic": "security",
            "message": "XSS vulnerability",
            "resolution_hint": "escape output",
        }]
        escalation = {"resolution_hints": [{"hint": "Manual review required"}]}
        result = _extract_constraints(findings, escalation)
        assert len(result) == 2
        assert "[SECURITY]" in result[0]
        assert "[HITL]" in result[1]

    def test_empty_when_no_findings_or_escalation(self):
        result = _extract_constraints([], None)
        assert result == []


@pytest.mark.unit
class TestExtractSection:

    def test_existing_section_extracted(self):
        content = "# AGENTS.md\n\n## Common Mistakes\nSome mistake here.\n\n## Other Section"
        result = _extract_section(content, "Common Mistakes")
        assert "## Common Mistakes:" in result
        assert "Some mistake here." in result

    def test_missing_section_returns_placeholder(self):
        content = "# AGENTS.md\n\n## Other Section"
        result = _extract_section(content, "Common Mistakes")
        assert "## Common Mistakes: _not present_" == result

    def test_truncated_to_500_chars(self):
        long_content = "x" * 600
        content = f"## Common Mistakes\n{long_content}\n\n## Other"
        result = _extract_section(content, "Common Mistakes")
        # Should be truncated
        assert len(result) <= 500 + len("## Common Mistakes: _not present_")

    def test_empty_content(self):
        result = _extract_section("", "Common Mistakes")
        assert "_not present_" in result

    def test_section_with_newlines_preserved(self):
        content = "## Common Mistakes\nLine 1\nLine 2\n\n## Other"
        result = _extract_section(content, "Common Mistakes")
        assert "Line 1" in result
        assert "Line 2" in result


@pytest.mark.unit
class TestSpliceSections:

    def test_replaces_common_mistakes(self):
        content = "## Common Mistakes\nOld content.\n\n## Architecture Decisions\nOld arch."
        updates = {"Common Mistakes": "New mistake content."}
        result = _splice_sections(content, updates)
        assert "New mistake content." in result
        assert "Old content." not in result

    def test_replaces_architecture_decisions(self):
        content = "## Common Mistakes\nOld mistakes.\n\n## Architecture Decisions\nOld arch."
        updates = {"Architecture Decisions": "New arch content."}
        result = _splice_sections(content, updates)
        assert "New arch content." in result
        assert "Old arch." not in result

    def test_replaces_both_sections(self):
        content = "## Common Mistakes\nOld.\n\n## Architecture Decisions\nOld arch."
        updates = {
            "Common Mistakes": "New mistakes.",
            "Architecture Decisions": "New arch.",
        }
        result = _splice_sections(content, updates)
        assert "New mistakes." in result
        assert "New arch." in result
        assert "Old." not in result

    def test_empty_update_skipped(self):
        content = "## Common Mistakes\nOld.\n\n## Architecture Decisions\nOld arch."
        updates = {"Common Mistakes": ""}
        result = _splice_sections(content, updates)
        assert "Old." in result

    def test_unknown_section_key_ignored(self):
        content = "## Common Mistakes\nOld.\n\n## Architecture Decisions\nOld arch."
        updates = {"unknown_key": "new content"}
        result = _splice_sections(content, updates)
        assert "Old." in result
        assert "new content" not in result

    def test_preserves_other_content(self):
        content = "# Header\n\n## Common Mistakes\nOld.\n\n## Other\nOther content.\n\n## Architecture Decisions\nOld arch."
        updates = {"Common Mistakes": "New."}
        result = _splice_sections(content, updates)
        assert "# Header" in result
        assert "Other content." in result
        assert "## Architecture Decisions" in result
        assert "New." in result


@pytest.mark.unit
class TestInjectDepcruiserRule:

    def test_injects_valid_rule(self, tmp_path):
        config_file = tmp_path / ".dependency-cruiser.json"
        config_file.write_text('{"forbidden": []}')
        rule = json.loads('{"name": "no-layer", "from": {"paths": ["*"]}, "to": [{"paths": ["*"]}]}')
        _inject_depcruiser_rule(config_file, json.dumps(rule))
        config = json.loads(config_file.read_text())
        assert len(config["forbidden"]) == 1
        assert config["forbidden"][0]["name"] == "no-layer"

    def test_invalid_json_skipped(self, tmp_path):
        config_file = tmp_path / ".dependency-cruiser.json"
        config_file.write_text('{"forbidden": []}')
        _inject_depcruiser_rule(config_file, "not valid json")
        config = json.loads(config_file.read_text())
        assert config["forbidden"] == []

    def test_missing_required_fields_skipped(self, tmp_path):
        config_file = tmp_path / ".dependency-cruiser.json"
        config_file.write_text('{"forbidden": []}')
        rule = {"name": "no-layer"}  # missing "from" and "to"
        _inject_depcruiser_rule(config_file, json.dumps(rule))
        config = json.loads(config_file.read_text())
        assert config["forbidden"] == []

    def test_creates_forbidden_array_if_missing(self, tmp_path):
        config_file = tmp_path / ".dependency-cruiser.json"
        config_file.write_text('{}')
        rule = {"name": "no-layer", "from": {"paths": ["*"]}, "to": [{"paths": ["*"]}]}
        _inject_depcruiser_rule(config_file, json.dumps(rule))
        config = json.loads(config_file.read_text())
        assert "forbidden" in config
        assert len(config["forbidden"]) == 1

    def test_appends_to_existing_rules(self, tmp_path):
        config_file = tmp_path / ".dependency-cruiser.json"
        config_file.write_text('{"forbidden": [{"name": "existing"}]}')
        rule = {"name": "new-rule", "from": {"paths": ["*"]}, "to": [{"paths": ["*"]}]}
        _inject_depcruiser_rule(config_file, json.dumps(rule))
        config = json.loads(config_file.read_text())
        assert len(config["forbidden"]) == 2
        assert config["forbidden"][0]["name"] == "existing"
        assert config["forbidden"][1]["name"] == "new-rule"


@pytest.mark.unit
class TestInjectArchunitRule:

    def test_injects_valid_rule(self, tmp_path):
        config_file = tmp_path / "ArchitectureTest.java"
        existing = 'class ArchitectureTest {\n\n    // existing\n}\n'
        config_file.write_text(existing)
        rule = '    @ArchTest public void noLayerViolation(ArchRule rule) { rule.check(this); }'
        _inject_archunit_rule(config_file, rule, user_package="com.sacv")
        content = config_file.read_text()
        assert "@ArchTest" in content
        assert "noLayerViolation" in content

    def test_no_annotation_skipped(self, tmp_path):
        config_file = tmp_path / "ArchitectureTest.java"
        config_file.write_text("class X {}")
        rule = "    public void test() {}"  # no @ArchTest
        _inject_archunit_rule(config_file, rule)
        content = config_file.read_text()
        assert content == "class X {}"

    def test_unbalanced_braces_skipped(self, tmp_path):
        config_file = tmp_path / "ArchitectureTest.java"
        config_file.write_text("class X {}")
        rule = "    @ArchTest public void test() { check(this); }"  # missing closing brace
        _inject_archunit_rule(config_file, rule)
        content = config_file.read_text()
        assert content == "class X {}"

    def test_missing_archrule_type_skipped(self, tmp_path):
        config_file = tmp_path / "ArchitectureTest.java"
        config_file.write_text("class X {}")
        rule = "    @ArchTest void test() { assert true; }"  # no ArchRule type
        _inject_archunit_rule(config_file, rule)
        content = config_file.read_text()
        assert content == "class X {}"

    def test_creates_file_when_missing(self, tmp_path):
        config_file = tmp_path / "ArchitectureTest.java"
        rule = "    @ArchTest public void layered(ArchRule rule) { rule.check(this); }"
        _inject_archunit_rule(config_file, rule, user_package="com.example")
        content = config_file.read_text()
        assert "package com.example;" in content
        assert "@ArchTest" in content
        assert "layered" in content
        assert "ArchRule" in content

    def test_inserts_before_closing_brace(self, tmp_path):
        config_file = tmp_path / "ArchitectureTest.java"
        existing = 'class ArchitectureTest {\n\n    @ArchTest void existing() {}\n}\n'
        config_file.write_text(existing)
        rule = "    @ArchTest void newRule(ArchRule r) { r.check(this); }"
        _inject_archunit_rule(config_file, rule)
        content = config_file.read_text()
        # Should have both rules
        assert "existing" in content
        assert "newRule" in content


@pytest.mark.unit
class TestDefaultContent:

    def test_default_agents_md_structure(self):
        content = _default_agents_md()
        assert "# AGENTS.md" in content
        assert "## Architecture Overview" in content
        assert "## Common Mistakes" in content
        assert "## Architecture Decisions" in content
        assert "## Module Conventions" in content

    def test_default_archunit_class_structure(self):
        rule = "    @ArchTest void test(ArchRule r) {}"
        content = _default_archunit_class(rule, user_package="com.example")
        assert "package com.example;" in content
        assert "import com.tngtech.archunit.junit.ArchTest;" in content
        assert "import com.tngtech.archunit.lang.ArchRule;" in content
        assert rule in content
        assert "class ArchitectureTest" in content


# Need to import json for the depcruiser tests
import json
