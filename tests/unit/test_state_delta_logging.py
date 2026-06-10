"""
tests/unit/test_state_delta_logging.py
=======================================
HIGH-06: State-delta logging between node invocations.

Verifies:
  1. cli_progress logs state deltas at DEBUG level for each completed node
  2. Large blobs (diff_content, critic_findings) are summarised, not logged raw
  3. format_delta_summary correctly handles nested structures
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from sacv import cli_progress


@pytest.fixture(autouse=True)
def _enable_debug_logging():
    """Enable DEBUG level on the root logger for all tests in this module."""
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)
    yield
    root.setLevel(old_level)


class MockGraph:
    """Minimal mock graph with astream_events."""

    def __init__(self, events):
        self._events = events
        self._canonical_state = None

    async def astream_events(self, *args, **kwargs):
        for event in self._events:
            yield event

    async def aget_state(self, config):
        if self._canonical_state is not None:
            snap = MagicMock()
            snap.values = self._canonical_state
            return snap
        return None

    def set_canonical_state(self, state):
        self._canonical_state = state


class TestStateDeltaLogging:
    """Verify state-delta logging at DEBUG level."""

    def _events(self, events):
        async def gen():
            for e in events:
                yield e
        return gen()

    async def test_logs_state_delta_at_debug_level(self):
        """on_chain_end logs a state-delta event at DEBUG level."""
        graph = MockGraph([
            {
                "event": "on_chain_end",
                "name": "bootstrap",
                "data": {
                    "output": {
                        "current_phase": "scout",
                        "cumulative_cost_dollars": 0.5,
                    },
                },
            },
        ])
        graph.set_canonical_state({"current_phase": "scout"})

        with patch("sacv.cli_progress.log") as mock_log:
            await cli_progress.run_with_progress(
                graph, {}, {"configurable": {"thread_id": "T1"}}, "T1",
            )
            # Should call log.debug with node.state_delta
            debug_calls = [c for c in mock_log.debug.call_args_list
                           if c[0] and c[0][0] == "node.state_delta"]
            assert len(debug_calls) >= 1
            call = debug_calls[0]
            assert call[1]["node"] == "bootstrap"
            assert "current_phase" in call[1]["changed_keys"]

    async def test_logs_state_delta_only_when_debug_enabled(self):
        """State-delta logging is skipped when DEBUG level is not enabled."""
        graph = MockGraph([
            {
                "event": "on_chain_end",
                "name": "bootstrap",
                "data": {"output": {"current_phase": "scout"}},
            },
        ])
        graph.set_canonical_state({"current_phase": "scout"})

        # Temporarily raise root logger level to INFO to disable DEBUG
        root = logging.getLogger()
        old_level = root.level
        root.setLevel(logging.INFO)
        try:
            with patch("sacv.cli_progress.log") as mock_log:
                await cli_progress.run_with_progress(
                    graph, {}, {"configurable": {"thread_id": "T1"}}, "T1",
                )
                # Should NOT call log.debug
                debug_calls = [c for c in mock_log.debug.call_args_list
                               if c[0] and c[0][0] == "node.state_delta"]
                assert len(debug_calls) == 0
        finally:
            root.setLevel(old_level)

    async def test_summarizes_large_outputs(self):
        """Large output fields are summarised, not logged raw."""
        large_diff = "diff --git a/x.java b/x.java\n" * 1000
        graph = MockGraph([
            {
                "event": "on_chain_end",
                "name": "actor",
                "data": {
                    "output": {
                        "current_phase": "verifier",
                        "diff_proposal": {
                            "strategy_id": "s1",
                            "diffs": [{"file_path": "x.java", "diff_content": large_diff}],
                            "branch_name": "main",
                            "commit_message": "test",
                        },
                    },
                },
            },
        ])
        graph.set_canonical_state({"current_phase": "verifier"})

        with patch("sacv.cli_progress.log") as mock_log:
            await cli_progress.run_with_progress(
                graph, {}, {"configurable": {"thread_id": "T1"}}, "T1",
            )
            debug_calls = [c for c in mock_log.debug.call_args_list
                           if c[0] and c[0][0] == "node.state_delta"]
            assert len(debug_calls) >= 1
            # The delta_summary should NOT contain the raw large diff
            summary = debug_calls[0][1]["delta_summary"]
            diff_str = str(summary.get("diff_proposal", ""))
            assert len(large_diff) not in [len(x) for x in diff_str.split() if x.isdigit()]
            # Should contain a summary marker instead
            assert "dict" in diff_str.lower() or "len" in diff_str.lower() or "<" in diff_str


class TestFormatDeltaSummary:
    """Test the delta_summary helper function."""

    def test_returns_scalar_values_unchanged(self):
        """Scalars pass through unchanged."""
        assert cli_progress._format_delta_summary("hello") == "hello"
        assert cli_progress._format_delta_summary(42) == 42
        assert cli_progress._format_delta_summary(1.5) == 1.5
        assert cli_progress._format_delta_summary(None) is None

    def test_short_strings_pass_through(self):
        """Short strings (< 200 chars) pass through unchanged."""
        short = "a" * 199
        assert cli_progress._format_delta_summary(short) == short

    def test_long_strings_are_summarised(self):
        """Long strings are replaced with a summary marker."""
        long_str = "x" * 201
        result = cli_progress._format_delta_summary(long_str)
        assert isinstance(result, str)
        assert len(result) < len(long_str)
        assert "str" in result.lower() or "len" in result.lower() or "<" in result

    def test_lists_are_summarised(self):
        """Lists are replaced with a summary marker."""
        result = cli_progress._format_delta_summary([1, 2, 3])
        assert isinstance(result, str)
        assert "list" in result.lower()

    def test_dicts_are_summarised(self):
        """Dicts are replaced with a summary marker."""
        result = cli_progress._format_delta_summary({"key": "value"})
        assert isinstance(result, str)
        assert "dict" in result.lower()
