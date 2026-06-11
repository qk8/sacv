"""
tests/unit/test_system_prompts.py
==================================
CFG-003: Smoke tests that verify all system prompt templates can be
formatted without KeyError and produce non-empty strings.
"""
from __future__ import annotations

import pytest


class TestActorSystemPrompt:

    def test_formats_without_error(self):
        from sacv.nodes.actor import _ACTOR_SYSTEM
        result = _ACTOR_SYSTEM.format(
            language="Java",
            constraints="- no raw SQL",
            agents_md="## Conventions",
            preflight_feedback="None.",
            debug_feedback="None.",
            critic_feedback="None.",
        )
        assert len(result) > 100

    def test_no_unfilled_placeholders(self):
        import re
        from sacv.nodes.actor import _ACTOR_SYSTEM
        result = _ACTOR_SYSTEM.format(
            language="Java",
            constraints="- no raw SQL",
            agents_md="## Conventions",
            preflight_feedback="None.",
            debug_feedback="None.",
            critic_feedback="None.",
        )
        # After .format(), no {identifier} placeholders should remain.
        # Escaped braces {{ → { are fine — they render as literal { in output.
        assert not re.search(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", result), (
            "Unfilled format placeholders found in actor prompt"
        )


class TestReplanSystemPrompt:

    def test_formats_without_error(self):
        from sacv.nodes.replan import _REPLAN_SYSTEM
        result = _REPLAN_SYSTEM.format(n=3)
        assert len(result) > 50

    def test_no_unfilled_placeholders(self):
        import re
        from sacv.nodes.replan import _REPLAN_SYSTEM
        result = _REPLAN_SYSTEM.format(n=3)
        # After .format(n=3), no {name} placeholders should remain
        assert not re.search(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", result), (
            "Unfilled format placeholders found in replan prompt"
        )


class TestTddGateOracleBackend:

    def test_formats_without_error(self):
        """TDD gate backend prompt has no format variables — verify it stays that way."""
        from sacv.nodes.tdd_gate import _ORACLE_BACKEND_SYSTEM
        # Should not raise KeyError
        result = _ORACLE_BACKEND_SYSTEM.format()
        assert len(result) > 50
        # Escaped braces {{...}} become {...} in output — verify they render
        assert "{" in result


class TestTddGateOracleFrontend:

    def test_formats_without_error(self):
        """TDD gate frontend prompt has no format variables."""
        from sacv.nodes.tdd_gate import _ORACLE_FRONTEND_SYSTEM
        result = _ORACLE_FRONTEND_SYSTEM.format()
        assert len(result) > 50
        assert "{" in result


class TestMemoryConsolidationPrompts:

    def test_agents_md_updater_formats(self):
        from sacv.nodes.memory_consolidation import _AGENTS_MD_UPDATER_SYSTEM
        result = _AGENTS_MD_UPDATER_SYSTEM.format()
        assert len(result) > 50

    def test_arch_rule_updater_formats(self):
        from sacv.nodes.memory_consolidation import _ARCH_RULE_UPDATER_SYSTEM
        result = _ARCH_RULE_UPDATER_SYSTEM.format()
        assert len(result) > 50


class TestIntelligentDebuggerPrompt:

    def test_root_cause_system_formats(self):
        from sacv.nodes.intelligent_debugger import _ROOT_CAUSE_SYSTEM
        result = _ROOT_CAUSE_SYSTEM.format()
        assert len(result) > 50
