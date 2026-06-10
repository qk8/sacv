"""
tests/unit/test_cli_progress.py
================================
Tests for the CLI real-time progress reporting module.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from sacv import cli_progress


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


class TestRunWithProgress:

    async def test_returns_canonical_state_via_aget_state(self):
        """CRIT-07: run_with_progress must return canonical LangGraph state via aget_state."""
        mock_snapshot = MagicMock()
        mock_snapshot.values = {
            "current_phase": "complete",
            "session_id": "sess-canonical",
            "critic_findings": [],  # reduced (not CRITIC_RESET sentinel)
            "correction_state": {"attempt_count": 2},  # fully merged
        }

        async def mock_aget_state(cfg):
            return mock_snapshot

        graph = MagicMock()
        graph.astream_events.return_value = self._events_generator([
            {
                "event": "on_chain_end",
                "name": "bootstrap",
                "data": {"output": {"current_phase": "scout", "session_id": "sess-partial"}},
            },
            {
                "event": "on_chain_end",
                "name": "scout",
                "data": {"output": {"current_phase": "actor"}},
            },
        ])
        graph.aget_state = mock_aget_state

        result = await cli_progress.run_with_progress(
            graph, {}, {"configurable": {"thread_id": "T1"}}, "T1",
        )

        # Must use canonical state, not accumulated partial outputs
        assert result["session_id"] == "sess-canonical"
        assert result["current_phase"] == "complete"

    @staticmethod
    def _events_generator(events):
        async def gen():
            for e in events:
                yield e
        return gen()

    async def test_streams_node_completion_events(self):
        """run_with_progress streams node completion events to stderr."""
        graph = MockGraph([
            {
                "event": "on_chain_end",
                "name": "bootstrap",
                "data": {
                    "output": {
                        "current_phase": "scout",
                        "session_id": "sess-1",
                        "cumulative_cost_dollars": 0.5,
                    },
                },
            },
            {
                "event": "on_chain_end",
                "name": "scout",
                "data": {
                    "output": {
                        "current_phase": "actor",
                        "cumulative_cost_dollars": 1.2,
                    },
                },
            },
        ])
        graph.set_canonical_state({
            "current_phase": "actor",
            "session_id": "sess-1",
            "cumulative_cost_dollars": 1.2,
        })

        result = await cli_progress.run_with_progress(
            graph, {}, {"configurable": {"thread_id": "T1"}}, "T1",
        )

        assert result["current_phase"] == "actor"
        assert result["session_id"] == "sess-1"
        assert result["cumulative_cost_dollars"] == 1.2

    async def test_skips_langgraph_internal_events(self):
        """Internal LangGraph events are skipped."""
        graph = MockGraph([
            {"event": "on_chain_end", "name": "LangGraph", "data": {"output": {}}},
            {"event": "on_chain_end", "name": "__start__", "data": {"output": {}}},
            {
                "event": "on_chain_end",
                "name": "scout",
                "data": {"output": {"current_phase": "actor"}},
            },
        ])
        graph.set_canonical_state({"current_phase": "actor"})

        result = await cli_progress.run_with_progress(
            graph, {}, {"configurable": {"thread_id": "T1"}}, "T1",
        )

        assert result["current_phase"] == "actor"

    async def test_reports_node_errors(self, capsys):
        """on_chain_error events are printed to stderr."""
        graph = MockGraph([
            {
                "event": "on_chain_error",
                "name": "actor",
                "data": {"error": "LLM timeout"},
            },
        ])
        graph.set_canonical_state({})

        await cli_progress.run_with_progress(
            graph, {}, {"configurable": {"thread_id": "T1"}}, "T1",
        )

        captured = capsys.readouterr()
        assert "actor ERROR: LLM timeout" in captured.err

    async def test_tracks_phase_transitions(self, capsys):
        """Phase transitions are shown in progress output."""
        graph = MockGraph([
            {
                "event": "on_chain_end",
                "name": "bootstrap",
                "data": {"output": {"current_phase": "scout", "cumulative_cost_dollars": 0.5}},
            },
            {
                "event": "on_chain_end",
                "name": "scout",
                "data": {"output": {"current_phase": "actor", "cumulative_cost_dollars": 1.0}},
            },
        ])
        graph.set_canonical_state({
            "current_phase": "actor",
            "cumulative_cost_dollars": 1.0,
        })

        await cli_progress.run_with_progress(
            graph, {}, {"configurable": {"thread_id": "T1"}}, "T1",
        )

        captured = capsys.readouterr()
        assert "-> scout" in captured.err
        assert "-> actor" in captured.err

    async def test_accumulates_final_state(self):
        """Final state accumulates output from all completed nodes."""
        graph = MockGraph([
            {
                "event": "on_chain_end",
                "name": "bootstrap",
                "data": {"output": {"session_id": "s1", "cumulative_cost_dollars": 0.1}},
            },
            {
                "event": "on_chain_end",
                "name": "scout",
                "data": {"output": {"blast_radius_map": {}, "cumulative_cost_dollars": 0.3}},
            },
        ])
        graph.set_canonical_state({
            "session_id": "s1",
            "blast_radius_map": {},
            "cumulative_cost_dollars": 0.3,
        })

        result = await cli_progress.run_with_progress(
            graph, {}, {"configurable": {"thread_id": "T1"}}, "T1",
        )

        assert result["session_id"] == "s1"
        assert result["blast_radius_map"] == {}


class TestFormatResult:

    def test_formats_complete_result(self):
        """format_result produces valid JSON with all fields."""
        state = {
            "current_phase": "complete",
            "lesson_learned": {"pattern_discovered": "arch_violation"},
            "cumulative_cost_dollars": 2.5,
        }
        output = cli_progress.format_result(state, "T42")
        parsed = json.loads(output)

        assert parsed["phase"] == "complete"
        assert parsed["task"] == "T42"
        assert parsed["cost"] == 2.5
        assert parsed["lesson"] == "arch_violation"

    def test_handles_missing_optional_fields(self):
        """format_result handles missing optional fields gracefully."""
        state = {"current_phase": "actor"}
        output = cli_progress.format_result(state, "T1")
        parsed = json.loads(output)

        assert parsed["phase"] == "actor"
        assert parsed["task"] == "T1"
        assert parsed["cost"] is None
        assert parsed["lesson"] is None
