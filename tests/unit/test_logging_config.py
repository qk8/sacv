"""
tests/unit/test_logging_config.py
==================================
Tests for the structlog configuration in logging_config.py.
"""
from __future__ import annotations

import json
import logging
import os
import sys

import pytest

import structlog

from sacv.logging_config import configure_logging


class TestConfigureLogging:

    def setup_method(self):
        """Save and restore env vars around each test."""
        self._saved_format = os.environ.get("LOG_FORMAT")
        self._saved_level = os.environ.get("LOG_LEVEL")

    def teardown_method(self):
        os.environ.pop("LOG_FORMAT", None)
        os.environ.pop("LOG_LEVEL", None)
        if self._saved_format is not None:
            os.environ["LOG_FORMAT"] = self._saved_format
        if self._saved_level is not None:
            os.environ["LOG_LEVEL"] = self._saved_level

    def test_defaults_to_json_format(self, capsys):
        """Default LOG_FORMAT=json produces JSON lines on stderr."""
        configure_logging()

        log = structlog.get_logger("test_json")
        log.info("hello_world", task_id="t1")

        captured = capsys.readouterr()
        # Output goes to stderr
        assert captured.err.strip()
        record = json.loads(captured.err.strip())
        assert record["event"] == "hello_world"
        assert record["level"] == "info"
        assert record["task_id"] == "t1"
        assert "timestamp" in record

    def test_console_format(self, capsys, monkeypatch):
        """LOG_FORMAT=console produces human-readable output."""
        monkeypatch.setenv("LOG_FORMAT", "console")
        configure_logging()

        log = structlog.get_logger("test_console")
        log.info("hello_console")

        captured = capsys.readouterr()
        assert captured.err.strip()
        # Console format is NOT JSON — should contain the event name
        assert "hello_console" in captured.err

    def test_json_format_env_var(self, capsys):
        """LOG_FORMAT=json explicitly set still produces JSON."""
        os.environ["LOG_FORMAT"] = "json"
        configure_logging()

        log = structlog.get_logger("test_explicit_json")
        log.warning("something_happened")

        captured = capsys.readouterr()
        record = json.loads(captured.err.strip())
        assert record["event"] == "something_happened"
        assert record["level"] == "warning"

    def test_log_level_filtering(self, capsys):
        """DEBUG log level is filtered out when LOG_LEVEL=WARNING."""
        os.environ["LOG_LEVEL"] = "WARNING"
        configure_logging()

        log = structlog.get_logger("test_filter")
        log.debug("should_not_appear")
        log.warning("should_appear")

        captured = capsys.readouterr()
        assert "should_not_appear" not in captured.err
        assert "should_appear" in captured.err

    def test_contextvars_propagate(self, capsys):
        """Context variables bound via structlog propagate to all log records."""
        configure_logging()

        structlog.contextvars.bind_contextvars(task_id="t-123", session_id="sess-abc")
        log = structlog.get_logger("test_ctx")
        log.info("context_test")

        captured = capsys.readouterr()
        record = json.loads(captured.err.strip())
        assert record["task_id"] == "t-123"
        assert record["session_id"] == "sess-abc"

    def test_root_logger_set_to_configured_level(self):
        """Root logger level matches LOG_LEVEL."""
        os.environ["LOG_LEVEL"] = "DEBUG"
        configure_logging()
        assert logging.getLogger().level == logging.DEBUG

        os.environ["LOG_LEVEL"] = "ERROR"
        configure_logging()
        assert logging.getLogger().level == logging.ERROR

    def test_noisy_loggers_silenced(self):
        """Noisy third-party loggers are set to WARNING level."""
        configure_logging()
        for logger_name in ("httpx", "httpcore", "asyncio", "docker", "urllib3"):
            assert logging.getLogger(logger_name).level == logging.WARNING

    def test_multiple_calls_are_idempotent(self, capsys):
        """Calling configure_logging() multiple times does not duplicate handlers."""
        configure_logging()
        configure_logging()

        log = structlog.get_logger("test_idempotent")
        log.info("once")

        captured = capsys.readouterr()
        # Should appear exactly once, not duplicated
        assert captured.err.count("once") == 1
