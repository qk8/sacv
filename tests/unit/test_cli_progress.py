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

    async def astream_events(self, *args, **kwargs):
        for event in self._events:
            yield event


class TestRunWithProgress:

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
