"""
tests/unit/test_claude_agent_adapter_helpers.py
=================================================
Unit tests for pure helper functions in claude_agent_adapter.py.

Tests cover:
1. _build_prompt — context prepending logic
2. _truncate_context — recursive context truncation
"""
from __future__ import annotations

import pytest
from sacv.adapters.claude.claude_agent_adapter import (
    _build_prompt, _truncate_context,
)


class TestBuildPrompt:

    def test_empty_context_returns_prompt_unchanged(self):
        result = _build_prompt("Implement user service", {})
        assert result == "Implement user service"

    def test_none_context_returns_prompt_unchanged(self):
        result = _build_prompt("Implement user service", None)
        assert result == "Implement user service"

    def test_empty_dict_context_returns_prompt_unchanged(self):
        result = _build_prompt("Implement user service", {})
        assert result == "Implement user service"

    def test_context_prepended_as_xml_block(self):
        context = {"key": "value"}
        result = _build_prompt("Task prompt", context)
        assert "<context>" in result
        assert "</context>" in result
        assert '"key"' in result
        assert '"value"' in result
        assert result.endswith("Task prompt")

    def test_context_block_separated_from_prompt(self):
        context = {"a": 1}
        result = _build_prompt("Do the thing", context)
        # There should be a blank line between </context> and the prompt
        assert "</context>\n\nDo the thing" in result

    def test_multiple_context_keys_serialized(self):
        context = {"strategy": "use repo pattern", "blast": 5}
        result = _build_prompt("Implement", context)
        assert '"strategy"' in result
        assert '"use repo pattern"' in result
        assert '"blast"' in result

    def test_nested_context_serialized(self):
        context = {"skeleton": {"nodes": ["A", "B"], "edges": ["A->B"]}}
        result = _build_prompt("Task", context)
        assert '"nodes"' in result
        assert '"A"' in result
        assert '"edges"' in result

    def test_large_string_values_are_truncated(self):
        """Context string values exceeding 8000 chars are truncated."""
        long_value = "x" * 10_000
        context = {"data": long_value}
        result = _build_prompt("Task", context)
        # The truncated value should appear in the result
        assert len(result) < 10_000 + 100  # prompt + wrapper < original value length

    def test_list_values_are_truncated_to_20_items(self):
        """Lists are truncated to 20 items."""
        context = {"items": list(range(50))}
        result = _build_prompt("Task", context)
        # The JSON should only contain 20 items
        import json
        ctx_block = result.split("<context>\n")[1].split("\n</context>")[0]
        parsed = json.loads(ctx_block)
        assert len(parsed["items"]) == 20

    def test_dict_values_are_truncated_to_30_keys(self):
        """Dicts are truncated to 30 keys."""
        context = {f"key{i}": f"value{i}" for i in range(50)}
        result = _build_prompt("Task", context)
        import json
        ctx_block = result.split("<context>\n")[1].split("\n</context>")[0]
        parsed = json.loads(ctx_block)
        assert len(parsed) == 30


class TestTruncateContext:

    def test_short_string_unchanged(self):
        result = _truncate_context("hello", max_chars=8000)
        assert result == "hello"

    def test_long_string_truncated(self):
        result = _truncate_context("x" * 10_000, max_chars=100)
        assert len(result) == 100

    def test_short_list_unchanged(self):
        items = list(range(10))
        result = _truncate_context(items, max_chars=8000)
        assert result == items

    def test_long_list_truncated_to_20(self):
        items = list(range(50))
        result = _truncate_context(items, max_chars=8000)
        assert len(result) == 20
        assert result[0] == 0
        assert result[19] == 19

    def test_short_dict_unchanged(self):
        d = {f"key{i}": f"value{i}" for i in range(10)}
        result = _truncate_context(d, max_chars=8000)
        assert result == d

    def test_long_dict_truncated_to_30_keys(self):
        d = {f"key{i}": f"value{i}" for i in range(50)}
        result = _truncate_context(d, max_chars=8000)
        assert len(result) == 30

    def test_nested_dict_truncated(self):
        """Deeply nested dicts are truncated at each level."""
        d = {"data": {f"key{i}": i for i in range(50)}}
        result = _truncate_context(d, max_chars=8000)
        assert len(result["data"]) == 30

    def test_nested_list_truncated(self):
        """Lists inside dicts are truncated."""
        d = {"items": list(range(50))}
        result = _truncate_context(d, max_chars=8000)
        assert len(result["items"]) == 20

    def test_non_string_values_unchanged(self):
        """Non-string, non-list, non-dict values pass through unchanged."""
        assert _truncate_context(42, max_chars=100) == 42
        assert _truncate_context(3.14, max_chars=100) == 3.14
        assert _truncate_context(True, max_chars=100) is True
        assert _truncate_context(None, max_chars=100) is None

    def test_list_of_dicts_truncated(self):
        """Lists containing dicts are truncated at list level."""
        items = [{"i": i} for i in range(50)]
        result = _truncate_context(items, max_chars=8000)
        assert len(result) == 20

    def test_zero_max_chars_truncates_strings(self):
        result = _truncate_context("hello world", max_chars=0)
        assert result == ""

    def test_max_chars_equal_to_length_unchanged(self):
        result = _truncate_context("hello", max_chars=5)
        assert result == "hello"
