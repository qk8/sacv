"""
tests/unit/test_memory_consolidation.py
=========================================

Unit tests for memory_consolidation helper functions.

Tests cover:
  1. _derive_pattern — pattern string from state
  2. _extract_constraints — critical findings → constraints
  3. _extract_section — AGENTS.md section extraction
  4. _splice_sections — section replacement in AGENTS.md
  5. _default_agents_md — default content
  6. _inject_depcruiser_rule — depcruiser rule injection
  7. _inject_archunit_rule — ArchUnit rule injection
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sacv.nodes.memory_consolidation import (
    _derive_pattern,
    _extract_constraints,
    _extract_section,
    _splice_sections,
    _default_agents_md,
    _inject_depcruiser_rule,
    _inject_archunit_rule,
)


# ── _derive_pattern ───────────────────────────────────────────────────────────


class TestDerivePattern:

    def test_basic_pattern(self, base_state):
        state = base_state(
            module_type="backend-domain",
            project_mode="greenfield",
            correction_state={"attempt_count": 1},
        )
        result = _derive_pattern(state)
        assert "module=backend-domain" in result
        assert "mode=greenfield" in result

    def test_pass_on_first_attempt(self, base_state):
        state = base_state(
            module_type="backend-domain",
            project_mode="greenfield",
            correction_state={"attempt_count": 1},
            verifier_verdict={"test_result": "PASS"},
        )
        result = _derive_pattern(state)
        assert "resolved_in=1_attempts" in result

    def test_pass_on_multiple_attempts(self, base_state):
        state = base_state(
            module_type="backend-domain",
            project_mode="brownfield",
            correction_state={"attempt_count": 3},
            verifier_verdict={"test_result": "PASS"},
        )
        result = _derive_pattern(state)
        assert "resolved_in=3_attempts" in result

    def test_stagnation_iteration(self, base_state):
        state = base_state(
            module_type="backend-domain",
            project_mode="greenfield",
            correction_state={"attempt_count": 2, "stagnation_pattern": "iteration"},
        )
        result = _derive_pattern(state)
        assert "stagnation=iteration" in result

    def test_stagnation_semantic(self, base_state):
        state = base_state(
            module_type="backend-domain",
            project_mode="greenfield",
            correction_state={"attempt_count": 2, "stagnation_pattern": "semantic"},
        )
        result = _derive_pattern(state)
        assert "stagnation=semantic" in result

    def test_no_stagnation(self, base_state):
        state = base_state(
            module_type="backend-domain",
            project_mode="greenfield",
            correction_state={"attempt_count": 2, "stagnation_pattern": "none"},
        )
        result = _derive_pattern(state)
        assert "stagnation=" not in result

    def test_replanned(self, base_state):
        state = base_state(
            module_type="backend-domain",
            project_mode="greenfield",
            correction_state={"attempt_count": 1},
            replan_count=2,
        )
        result = _derive_pattern(state)
        assert "replanned=2x" in result

    def test_replanned_zero(self, base_state):
        """replan_count=0 should not appear in pattern."""
        state = base_state(
            module_type="backend-domain",
            project_mode="greenfield",
            correction_state={"attempt_count": 1},
            replan_count=0,
        )
        result = _derive_pattern(state)
        assert "replanned=" not in result

    def test_blast_radius_included(self, base_state):
        """blast_radius_map is stored in LessonLearned, not in pattern string."""
        state = base_state(
            module_type="backend-domain",
            project_mode="brownfield",
            correction_state={"attempt_count": 1},
            blast_radius_map={"affected_files": ["a.java", "b.java"]},
        )
        result = _derive_pattern(state)
        # Pattern string contains module/mode/corrections but NOT blast_radius
        assert "module=backend-domain" in result

    def test_blast_radius_none(self, base_state):
        """Greenfield with no blast radius should still work."""
        state = base_state(
            module_type="backend-domain",
            project_mode="greenfield",
            correction_state={"attempt_count": 1},
            blast_radius_map=None,
        )
        result = _derive_pattern(state)
        assert "module=backend-domain" in result


# ── _extract_constraints ──────────────────────────────────────────────────────


class TestExtractConstraints:

    def test_no_findings_returns_empty(self):
        assert _extract_constraints([], None) == []

    def test_critical_finding_becomes_constraint(self):
        findings = [{
            "critic": "security",
            "severity": "critical",
            "message": "SQL injection risk",
            "resolution_hint": "Use parameterized queries",
        }]
        result = _extract_constraints(findings, None)
        assert len(result) == 1
        assert "[SECURITY]" in result[0]
        assert "SQL injection risk" in result[0]
        assert "Use parameterized queries" in result[0]

    def test_warning_finding_ignored(self):
        findings = [{
            "critic": "style",
            "severity": "warning",
            "message": "Long method",
            "resolution_hint": "Break it up",
        }]
        assert _extract_constraints(findings, None) == []

    def test_info_finding_ignored(self):
        findings = [{
            "critic": "consistency",
            "severity": "info",
            "message": "Naming convention",
            "resolution_hint": "Rename",
        }]
        assert _extract_constraints(findings, None) == []

    def test_multiple_critical_findings(self):
        findings = [
            {
                "critic": "security",
                "severity": "critical",
                "message": "XSS risk",
                "resolution_hint": "Escape output",
            },
            {
                "critic": "security",
                "severity": "critical",
                "message": "Hardcoded secret",
                "resolution_hint": "Use env var",
            },
        ]
        result = _extract_constraints(findings, None)
        assert len(result) == 2

    def test_hitl_resolution_hints(self):
        escalation = {
            "resolution_hints": [
                {"priority": 1, "category": "architectural",
                 "hint": "Review the strategy", "automated": False},
            ]
        }
        result = _extract_constraints([], escalation)
        assert len(result) == 1
        assert "[HITL]" in result[0]
        assert "Review the strategy" in result[0]

    def test_hitl_and_critical_combined(self):
        findings = [{
            "critic": "security",
            "severity": "critical",
            "message": "XSS",
            "resolution_hint": "Escape",
        }]
        escalation = {
            "resolution_hints": [
                {"priority": 1, "category": "architectural",
                 "hint": "Check strategy", "automated": False},
            ]
        }
        result = _extract_constraints(findings, escalation)
        assert len(result) == 2

    def test_hitl_empty_hints(self):
        escalation = {"resolution_hints": []}
        assert _extract_constraints([], escalation) == []


# ── _extract_section ──────────────────────────────────────────────────────────


class TestExtractSection:

    def test_existing_section(self):
        content = "## Common Mistakes\nSome mistake here\n## Other Section"
        result = _extract_section(content, "Common Mistakes")
        assert "## Common Mistakes:" in result
        assert "Some mistake here" in result

    def test_missing_section(self):
        content = "## Other Section\nNothing here"
        result = _extract_section(content, "Common Mistakes")
        assert "## Common Mistakes: _not present_" in result

    def test_section_truncated_at_500_chars(self):
        long_content = "## Common Mistakes\n" + "x" * 600 + "\n## Next"
        result = _extract_section(long_content, "Common Mistakes")
        # Should be truncated but include the header
        assert "## Common Mistakes:" in result
        # The content should be limited
        assert len(result) <= 500 + len("## Common Mistakes: _not present_\n")

    def test_section_at_end_of_file(self):
        content = "## Common Mistakes\nLast section"
        result = _extract_section(content, "Common Mistakes")
        assert "Last section" in result

    def test_empty_content(self):
        result = _extract_section("", "Common Mistakes")
        assert "_not present_" in result


# ── _splice_sections ──────────────────────────────────────────────────────────


class TestSpliceSections:

    def test_replace_common_mistakes(self):
        content = "## Common Mistakes\nOld stuff\n## Architecture\nOld arch"
        updates = {"Common Mistakes": "New mistakes here"}
        result = _splice_sections(content, updates)
        assert "New mistakes here" in result
        assert "Old stuff" not in result

    def test_replace_architecture_decisions(self):
        content = "## Common Mistakes\nOld\n## Architecture Decisions\nOld arch"
        updates = {"Architecture Decisions": "New arch"}
        result = _splice_sections(content, updates)
        assert "New arch" in result
        assert "Old arch" not in result

    def test_replace_both_sections(self):
        content = "## Common Mistakes\nOld\n## Architecture Decisions\nOld arch"
        updates = {
            "Common Mistakes": "New mistakes",
            "Architecture Decisions": "New arch",
        }
        result = _splice_sections(content, updates)
        assert "New mistakes" in result
        assert "New arch" in result
        assert "Old" not in result

    def test_empty_update_ignored(self):
        content = "## Common Mistakes\nOld stuff\n## Other"
        updates = {"Common Mistakes": ""}
        result = _splice_sections(content, updates)
        assert "Old stuff" in result

    def test_missing_section_not_added(self):
        content = "## Other Section\nNothing"
        updates = {"Common Mistakes": "New stuff"}
        result = _splice_sections(content, updates)
        # Should not add a new section if it doesn't exist
        assert "New stuff" not in result

    multiline_content = """\
# Title

## Common Mistakes
Line 1
Line 2

## Architecture Decisions
Decision 1
Decision 2

## Footer
"""

    def test_multiline_section_replaced(self):
        updates = {"Common Mistakes": "Single line replacement"}
        result = _splice_sections(self.multiline_content, updates)
        assert "Single line replacement" in result
        assert "Line 1" not in result
        assert "Line 2" not in result


# ── _default_agents_md ────────────────────────────────────────────────────────


class TestDefaultAgentsMd:

    def test_has_title(self):
        assert "# AGENTS.md" in _default_agents_md()

    def test_has_common_mistakes_section(self):
        assert "## Common Mistakes" in _default_agents_md()

    def test_has_architecture_decisions_section(self):
        assert "## Architecture Decisions" in _default_agents_md()

    def test_has_module_conventions_section(self):
        assert "## Module Conventions" in _default_agents_md()

    def test_has_architecture_overview_section(self):
        assert "## Architecture Overview" in _default_agents_md()


# ── _inject_depcruiser_rule ──────────────────────────────────────────────────


class TestInjectDepcruiserRule:

    def test_injects_valid_rule(self, tmp_path):
        config_file = tmp_path / ".dependency-cruiser.json"
        config_file.write_text('{"forbidden": []}')
        rule = json.dumps({
            "name": "no-domain-to-frontend",
            "from": {"path": ["^src/domain"]},
            "to": {"path": ["^src/frontend"]},
        })
        _inject_depcruiser_rule(config_file, rule)
        config = json.loads(config_file.read_text())
        assert len(config["forbidden"]) == 1
        assert config["forbidden"][0]["name"] == "no-domain-to-frontend"

    def test_rejects_invalid_json(self, tmp_path):
        config_file = tmp_path / ".dependency-cruiser.json"
        config_file.write_text('{"forbidden": []}')
        _inject_depcruiser_rule(config_file, "not json")
        config = json.loads(config_file.read_text())
        assert config["forbidden"] == []

    def test_rejects_rule_missing_fields(self, tmp_path):
        config_file = tmp_path / ".dependency-cruiser.json"
        config_file.write_text('{"forbidden": []}')
        rule = json.dumps({"name": "incomplete"})  # missing "from" and "to"
        _inject_depcruiser_rule(config_file, rule)
        config = json.loads(config_file.read_text())
        assert config["forbidden"] == []

    def test_idempotent_same_name(self, tmp_path):
        config_file = tmp_path / ".dependency-cruiser.json"
        config_file.write_text('{"forbidden": [{"name": "rule1", "from": {}, "to": {}}]}')
        rule = json.dumps({
            "name": "rule1",
            "from": {"path": ["a"]},
            "to": {"path": ["b"]},
        })
        _inject_depcruiser_rule(config_file, rule)
        config = json.loads(config_file.read_text())
        assert len(config["forbidden"]) == 1

    def test_appends_different_name(self, tmp_path):
        config_file = tmp_path / ".dependency-cruiser.json"
        config_file.write_text('{"forbidden": []}')
        rule1 = json.dumps({
            "name": "rule1",
            "from": {"path": ["a"]},
            "to": {"path": ["b"]},
        })
        rule2 = json.dumps({
            "name": "rule2",
            "from": {"path": ["c"]},
            "to": {"path": ["d"]},
        })
        _inject_depcruiser_rule(config_file, rule1)
        _inject_depcruiser_rule(config_file, rule2)
        config = json.loads(config_file.read_text())
        assert len(config["forbidden"]) == 2

    def test_creates_forbidden_array_if_missing(self, tmp_path):
        config_file = tmp_path / ".dependency-cruiser.json"
        config_file.write_text("{}")
        rule = json.dumps({
            "name": "rule1",
            "from": {"path": ["a"]},
            "to": {"path": ["b"]},
        })
        _inject_depcruiser_rule(config_file, rule)
        config = json.loads(config_file.read_text())
        assert "forbidden" in config
        assert len(config["forbidden"]) == 1

    def test_missing_config_file_creates_default(self, tmp_path):
        config_file = tmp_path / ".dependency-cruiser.json"
        rule = json.dumps({
            "name": "rule1",
            "from": {"path": ["a"]},
            "to": {"path": ["b"]},
        })
        _inject_depcruiser_rule(config_file, rule)
        config = json.loads(config_file.read_text())
        assert config["forbidden"][0]["name"] == "rule1"

    def test_invalid_json_config_file_handled(self, tmp_path):
        config_file = tmp_path / ".dependency-cruiser.json"
        config_file.write_text("not json")
        rule = json.dumps({
            "name": "rule1",
            "from": {"path": ["a"]},
            "to": {"path": ["b"]},
        })
        _inject_depcruiser_rule(config_file, rule)
        config = json.loads(config_file.read_text())
        assert len(config["forbidden"]) == 1


# ── _inject_archunit_rule ────────────────────────────────────────────────────


class TestInjectArchunitRule:

    def test_injects_valid_rule(self, tmp_path):
        config_file = tmp_path / "ArchitectureTest.java"
        config_file.write_text("package com.example;\n\nclass ArchitectureTest {\n\n}")
        rule = "    @ArchTest\n    public static final ArchRule myRule = layers()"
        _inject_archunit_rule(config_file, rule, "com.example")
        content = config_file.read_text()
        assert "public static final ArchRule myRule" in content

    def test_rejects_rule_without_annotation(self, tmp_path):
        config_file = tmp_path / "ArchitectureTest.java"
        config_file.write_text("package com.example;\n\nclass ArchitectureTest {\n\n}")
        rule = "    public void myRule() {"
        _inject_archunit_rule(config_file, rule, "com.example")
        content = config_file.read_text()
        # Rule rejected (no @ArchTest) — file remains unchanged
        assert "public void myRule" not in content

    def test_rejects_unbalanced_braces(self, tmp_path):
        config_file = tmp_path / "ArchitectureTest.java"
        config_file.write_text("package com.example;\n\nclass ArchitectureTest {\n\n}")
        rule = "    @ArchTest\n    public static final ArchRule myRule = layers() {"
        _inject_archunit_rule(config_file, rule, "com.example")
        content = config_file.read_text()
        assert "myRule" not in content  # unchanged

    def test_accepts_rule_with_ArchRules_type(self, tmp_path):
        """ArchRules (plural) passes the substring check in _inject_archunit_rule."""
        config_file = tmp_path / "ArchitectureTest.java"
        config_file.write_text("package com.example;\n\nclass ArchitectureTest {\n\n}")
        rule = "    @ArchTest\n    public static final ArchRules myRules = layers()"
        _inject_archunit_rule(config_file, rule, "com.example")
        content = config_file.read_text()
        assert "public static final ArchRules myRules" in content

    def test_idempotent_same_rule_name(self, tmp_path):
        config_file = tmp_path / "ArchitectureTest.java"
        config_file.write_text("package com.example;\n\nclass ArchitectureTest {\n\n}")
        rule = "    @ArchTest\n    public static final ArchRule myRule = layers()"
        _inject_archunit_rule(config_file, rule, "com.example")
        _inject_archunit_rule(config_file, rule, "com.example")
        content = config_file.read_text()
        # Should only appear once
        assert content.count("public static final ArchRule myRule") == 1

    def test_creates_file_if_missing(self, tmp_path):
        config_file = tmp_path / "ArchitectureTest.java"
        rule = "    @ArchTest\n    public static final ArchRule myRule = layers()"
        _inject_archunit_rule(config_file, rule, "com.example")
        content = config_file.read_text()
        assert "package com.example;" in content
        assert "public static final ArchRule myRule" in content

    def test_multiple_different_rules(self, tmp_path):
        config_file = tmp_path / "ArchitectureTest.java"
        config_file.write_text("package com.example;\n\nclass ArchitectureTest {\n\n}")
        rule1 = "    @ArchTest\n    public static final ArchRule layer1 = layers()"
        rule2 = "    @ArchTest\n    public static final ArchRule layer2 = noClasses()"
        _inject_archunit_rule(config_file, rule1, "com.example")
        _inject_archunit_rule(config_file, rule2, "com.example")
        content = config_file.read_text()
        assert "layer1" in content
        assert "layer2" in content
        assert content.count("public static final ArchRule") == 2

    def test_rule_with_archrules_type(self, tmp_path):
        """Rules with ArchRules (plural) type should also be accepted."""
        config_file = tmp_path / "ArchitectureTest.java"
        config_file.write_text("package com.example;\n\nclass ArchitectureTest {\n\n}")
        rule = "    @ArchTest\n    public static final ArchRules myRules = layers()"
        _inject_archunit_rule(config_file, rule, "com.example")
        content = config_file.read_text()
        assert "public static final ArchRules myRules" in content