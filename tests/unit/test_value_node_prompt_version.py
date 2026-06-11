"""
tests/unit/test_value_node_prompt_version.py
==============================================
CFG-003: Verify _STRATEGY_SYSTEM_PROMPT has version prefix.
"""
from __future__ import annotations

import re


class TestValueNodePromptVersion:

    def test_strategy_prompt_has_version_constant(self):
        """_STRATEGY_SYSTEM_PROMPT is prefixed with # prompt_version: <version>."""
        from sacv.nodes.value_node import (
            _STRATEGY_SYSTEM_PROMPT,
            _STRATEGY_SYSTEM_VERSION,
        )
        expected_prefix = f"# prompt_version: {_STRATEGY_SYSTEM_VERSION}\n"
        assert _STRATEGY_SYSTEM_PROMPT.startswith(expected_prefix), (
            f"Expected prompt to start with '{expected_prefix!r}', "
            f"got: {_STRATEGY_SYSTEM_PROMPT[:80]!r}"
        )

    def test_version_constant_is_non_empty_string(self):
        """_STRATEGY_SYSTEM_VERSION is a non-empty string."""
        from sacv.nodes.value_node import _STRATEGY_SYSTEM_VERSION
        assert isinstance(_STRATEGY_SYSTEM_VERSION, str)
        assert len(_STRATEGY_SYSTEM_VERSION) > 0

    def test_strategy_prompt_formats_without_error(self):
        """_STRATEGY_SYSTEM_PROMPT.format() works with n=3."""
        from sacv.nodes.value_node import _STRATEGY_SYSTEM_PROMPT
        result = _STRATEGY_SYSTEM_PROMPT.format(n=3)
        assert len(result) > 100
        # The version prefix should appear in output
        assert "prompt_version" in result

    def test_strategy_prompt_no_unfilled_placeholders(self):
        """After .format(n=3), no {identifier} placeholders remain."""
        import re
        from sacv.nodes.value_node import _STRATEGY_SYSTEM_PROMPT
        result = _STRATEGY_SYSTEM_PROMPT.format(n=3)
        assert not re.search(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", result), (
            "Unfilled format placeholders in value_node prompt"
        )
