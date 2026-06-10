"""
tests/unit/test_node_context.py
================================
Tests for the per-node correlation context binding helper.
"""
from __future__ import annotations

import json
import os

import pytest

import structlog

from sacv.nodes._node_context import bind_node_context


class TestBindNodeContext:

    def setup_method(self):
        structlog.contextvars.clear_contextvars()

    def teardown_method(self):
        structlog.contextvars.clear_contextvars()

    def test_clears_stale_context_from_previous_node(self):
        """Stale contextvars from a previous node must not bleed into the next node."""
        # Simulate Node A binding extra context
        structlog.contextvars.bind_contextvars(stale_key="should_not_persist")
        structlog.contextvars.bind_contextvars(files=5)

        state_b = {
            "current_phase": "preflight",
            "correction_state": {"attempt_count": 1},
        }
        bind_node_context(state_b, "preflight")

        ctx = structlog.contextvars.get_contextvars()
        assert "stale_key" not in ctx, (
            "Stale context from previous node leaked into new node"
        )
        assert "files" not in ctx

    def test_binds_session_id_from_state(self):
        """session_id from state must be present on every log line."""
        state = {
            "session_id": "sess-abc123",
            "current_phase": "actor",
            "correction_state": {"attempt_count": 1},
        }
        bind_node_context(state, "actor")
        assert structlog.contextvars.get_contextvars()["session_id"] == "sess-abc123"

    def test_binds_task_id_from_state(self):
        """task_id from state must be present on every log line."""
        state = {
            "task_id": "T-001",
            "current_phase": "scout",
            "correction_state": {"attempt_count": 0},
        }
        bind_node_context(state, "scout")
        assert structlog.contextvars.get_contextvars()["task_id"] == "T-001"

    def test_binds_module_type_from_state(self):
        """module_type from state must be present for module-based log filtering."""
        state = {
            "module_type": "backend-domain",
            "current_phase": "actor",
            "correction_state": {"attempt_count": 1},
        }
        bind_node_context(state, "actor")
        assert structlog.contextvars.get_contextvars()["module_type"] == "backend-domain"

    def test_binds_replan_count_from_state(self):
        """replan_count from state must be present for replan-cycle tracking."""
        state = {
            "replan_count": 2,
            "current_phase": "replan",
            "correction_state": {"attempt_count": 0},
        }
        bind_node_context(state, "replan")
        assert structlog.contextvars.get_contextvars()["replan_count"] == 2

    def test_binds_node_and_phase(self):
        """bind_node_context sets node, phase, and attempt in contextvars."""
        state = {
            "current_phase": "actor",
            "correction_state": {"attempt_count": 2},
        }
        bind_node_context(state, "actor")
        assert structlog.contextvars.get_contextvars()["node"] == "actor"
        assert structlog.contextvars.get_contextvars()["phase"] == "actor"
        assert structlog.contextvars.get_contextvars()["attempt"] == 2

    def test_binds_default_attempt_when_missing(self):
        """When correction_state is missing, attempt defaults to 0."""
        state = {"current_phase": "scout"}
        bind_node_context(state, "scout")
        assert structlog.contextvars.get_contextvars()["attempt"] == 0

    def test_clears_arbitrary_pre_bound_context(self):
        """Arbitrary pre-bound contextvars are cleared; only state-derived fields survive."""
        structlog.contextvars.bind_contextvars(session_id="sess-old")
        structlog.contextvars.bind_contextvars(stale="data")
        bind_node_context(
            {
                "session_id": "sess-new",
                "task_id": "T-999",
                "current_phase": "actor",
                "correction_state": {"attempt_count": 0},
            },
            "actor",
        )
        ctx = structlog.contextvars.get_contextvars()
        # New values from state replace old ones
        assert ctx["session_id"] == "sess-new"
        assert ctx["task_id"] == "T-999"
        # Arbitrary pre-bound keys are gone
        assert "stale" not in ctx

    def test_context_propagates_to_logger(self, capsys):
        """Context variables from state propagate to structlog output."""
        # Use console format for easier testing
        os.environ["LOG_FORMAT"] = "console"
        from sacv.logging_config import configure_logging
        configure_logging()

        bind_node_context(
            {
                "session_id": "sess-xyz",
                "task_id": "T-prop",
                "current_phase": "complete",
                "correction_state": {"attempt_count": 1},
            },
            "memory",
        )

        log = structlog.get_logger("test_ctx_prop")
        log.info("node_done")

        captured = capsys.readouterr()
        assert "node_done" in captured.err
        assert "sess-xyz" in captured.err
        assert "T-prop" in captured.err
        assert "memory" in captured.err
        assert "complete" in captured.err
        os.environ.pop("LOG_FORMAT", None)
