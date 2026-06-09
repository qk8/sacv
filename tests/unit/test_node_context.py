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

    def test_preserves_existing_context(self):
        """Pre-bound contextvars (e.g., session_id) are preserved."""
        structlog.contextvars.bind_contextvars(session_id="sess-abc")
        bind_node_context({"current_phase": "actor", "correction_state": {"attempt_count": 0}}, "actor")
        ctx = structlog.contextvars.get_contextvars()
        assert ctx["session_id"] == "sess-abc"
        assert ctx["node"] == "actor"

    def test_context_propagates_to_logger(self, capsys):
        """Context variables propagate to structlog output."""
        # Use console format for easier testing
        os.environ["LOG_FORMAT"] = "console"
        from sacv.logging_config import configure_logging
        configure_logging()

        structlog.contextvars.bind_contextvars(session_id="sess-xyz")
        bind_node_context({"current_phase": "complete", "correction_state": {"attempt_count": 1}}, "memory")

        log = structlog.get_logger("test_ctx_prop")
        log.info("node_done")

        captured = capsys.readouterr()
        assert "node_done" in captured.err
        assert "sess-xyz" in captured.err
        assert "memory" in captured.err
        assert "complete" in captured.err
        os.environ.pop("LOG_FORMAT", None)
