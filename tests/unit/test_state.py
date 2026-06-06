"""
tests/unit/test_state.py
=========================
Unit tests for pure state helpers — no I/O, no mocks.

Tests cover:
1. _merge_correction_state — shallow merge with defaults
2. _merge_lists — None (unchanged), CRITIC_RESET, empty list (no-op), append
3. _merge_lists — string values (CRITIC_RESET and unexpected)
4. _merge_lists — existing list preserved when new is empty/None
5. _merge_branches — None preserves existing (node didn't touch)
6. _merge_branches — append new branch names
"""
from __future__ import annotations

import pytest
from sacv.orchestration.state import (
    _merge_correction_state,
    _merge_lists,
    _merge_branches,
    CRITIC_RESET,
)


class TestMergeCorrectionState:

    def test_none_existing_none_new_returns_defaults(self):
        result = _merge_correction_state(None, None)
        assert result == {
            "attempt_count": 0, "branch_name": None,
            "last_error_hash": None, "error_history": [],
            "stagnation_pattern": "none",
        }

    def test_existing_none_new_returns_defaults(self):
        result = _merge_correction_state(None, {"attempt_count": 1})
        # Shallow merge: only attempt_count from new, rest from defaults
        assert result == {"attempt_count": 1}

    def test_existing_with_new_merges(self):
        existing = {
            "attempt_count": 2, "branch_name": "b",
            "last_error_hash": "abc", "error_history": ["e1"],
            "stagnation_pattern": "semantic",
        }
        new = {"attempt_count": 3, "branch_name": "b2"}
        result = _merge_correction_state(existing, new)
        assert result["attempt_count"] == 3
        assert result["branch_name"] == "b2"
        assert result["last_error_hash"] == "abc"  # preserved
        assert result["error_history"] == ["e1"]   # preserved
        assert result["stagnation_pattern"] == "semantic"  # preserved

    def test_existing_with_none_new_returns_existing(self):
        existing = {"attempt_count": 5}
        result = _merge_correction_state(existing, None)
        assert result == existing

    def test_new_overwrites_all_existing(self):
        existing = {"attempt_count": 1}
        new = {"attempt_count": 2, "branch_name": "x", "last_error_hash": "h",
               "error_history": ["e"], "stagnation_pattern": "iteration"}
        result = _merge_correction_state(existing, new)
        assert result == new


class TestMergeLists:

    def test_none_new_returns_existing_or_empty(self):
        assert _merge_lists(["a"], None) == ["a"]
        assert _merge_lists([], None) == []
        assert _merge_lists(None, None) == []

    def test_critic_reset_returns_empty_list(self):
        assert _merge_lists(["a"], CRITIC_RESET) == []
        assert _merge_lists([], CRITIC_RESET) == []
        assert _merge_lists(None, CRITIC_RESET) == []

    def test_unexpected_string_returns_existing(self):
        """Non-CRITIC_RESET strings are logged and return existing list."""
        assert _merge_lists(["a"], "unexpected") == ["a"]

    def test_empty_list_returns_existing_or_empty(self):
        """Empty list = no-op: preserves existing findings."""
        assert _merge_lists(["a", "b"], []) == ["a", "b"]
        assert _merge_lists([], []) == []
        assert _merge_lists(None, []) == []

    def test_non_empty_list_appends(self):
        assert _merge_lists(["a"], ["b"]) == ["a", "b"]
        assert _merge_lists([], ["b"]) == ["b"]
        assert _merge_lists(None, ["b"]) == ["b"]

    def test_multiple_appends(self):
        result = _merge_lists(None, ["a"])
        result = _merge_lists(result, ["b"])
        result = _merge_lists(result, ["c"])
        assert result == ["a", "b", "c"]

    def test_reset_then_append(self):
        result = _merge_lists(["a", "b"], CRITIC_RESET)
        assert result == []
        result = _merge_lists(result, ["c"])
        assert result == ["c"]

    def test_empty_then_append(self):
        result = _merge_lists(["a", "b"], [])
        assert result == ["a", "b"]
        result = _merge_lists(result, ["c"])
        assert result == ["a", "b", "c"]


class TestMergeBranches:

    def test_none_existing_none_new_returns_empty(self):
        """First write: both None → empty list."""
        assert _merge_branches(None, None) == []

    def test_none_existing_list_new_returns_list(self):
        """First write: existing None, new list → list."""
        assert _merge_branches(None, ["branch-1"]) == ["branch-1"]

    def test_existing_none_new_preserves_existing(self):
        """Node didn't touch: existing list, new None → existing."""
        assert _merge_branches(["branch-1"], None) == ["branch-1"]

    def test_existing_empty_list_new_preserves_existing(self):
        """Node didn't touch: existing empty, new None → empty."""
        assert _merge_branches([], None) == []

    def test_existing_list_new_appends(self):
        """Node adds branches: existing + new → merged."""
        result = _merge_branches(["branch-1"], ["branch-2"])
        assert result == ["branch-1", "branch-2"]

    def test_empty_list_is_explicit_clear(self):
        """Empty list = explicit clear: existing + [] → [] (unlike _merge_lists)."""
        result = _merge_branches(["branch-1", "branch-2"], [])
        assert result == []

    def test_multiple_appends_preserve_order(self):
        """Sequential appends preserve insertion order."""
        result = _merge_branches(None, ["a"])
        result = _merge_branches(result, ["b"])
        result = _merge_branches(result, ["c"])
        assert result == ["a", "b", "c"]

    def test_reset_then_append(self):
        """After reset to [], appending works correctly."""
        result = _merge_branches(["branch-1"], [])
        assert result == []
        result = _merge_branches(result, ["branch-2"])
        assert result == ["branch-2"]
