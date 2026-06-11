"""
tests/unit/test_get_traceparent.py
====================================
Tests for the get_traceparent() helper in sacv.tracing.

TDD checklist:
- [x] Every new function has a test
- [x] Tests cover disabled, no-span, valid-span, and invalid-span scenarios
- [x] Tests verify W3C traceparent format
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from sacv import tracing


@pytest.fixture
def otel_mock():
    """Replace OTel imports with mocks so spans can be created without the SDK."""
    otel_keys = (
        "opentelemetry",
        "opentelemetry.trace",
        "opentelemetry.sdk.trace",
        "opentelemetry.sdk.trace.export",
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
    )

    saved_modules = {}
    for key in otel_keys:
        if key in sys.modules:
            saved_modules[key] = sys.modules[key]

    mock_trace = MagicMock()
    mock_trace.get_tracer.return_value = MagicMock()

    for key in otel_keys:
        if key not in sys.modules:
            if key == "opentelemetry.trace":
                sys.modules[key] = mock_trace
            else:
                sys.modules[key] = MagicMock()

    original_has_otel = tracing._HAS_OTEL
    original_enabled = tracing._ENABLED
    original_tracer = tracing._tracer
    original_trace = getattr(tracing, "trace", None)

    tracing._ENABLED = True
    tracing._HAS_OTEL = True
    tracing._tracer = mock_trace.get_tracer.return_value
    tracing.trace = mock_trace

    yield mock_trace

    # Restore sys.modules
    for key in otel_keys:
        if key in saved_modules:
            sys.modules[key] = saved_modules[key]
        elif key in sys.modules:
            del sys.modules[key]

    tracing._ENABLED = original_enabled
    tracing._HAS_OTEL = original_has_otel
    tracing._tracer = original_tracer
    if original_trace is None:
        if hasattr(tracing, "trace"):
            delattr(tracing, "trace")
    else:
        tracing.trace = original_trace


class TestGetTraceparentDisabled:
    """get_traceparent() returns None when OTel is disabled."""

    def test_returns_none_when_disabled(self):
        """get_traceparent() returns None when OTel is not enabled."""
        assert tracing.get_traceparent() is None


class TestGetTraceparentOutsideSpan:
    """get_traceparent() returns None when not inside a span."""

    def test_returns_none_outside_span(self, otel_mock):
        """get_traceparent() returns None when no active span exists."""
        assert tracing.get_traceparent() is None


class TestGetTraceparentInsideSpan:
    """get_traceparent() returns W3C traceparent string inside a span."""

    def test_returns_w3c_format(self, otel_mock):
        """get_traceparent() returns a valid W3C traceparent string."""
        mock_tracer = otel_mock.get_tracer.return_value
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None,
        )

        # Mock a valid span context
        mock_context = MagicMock()
        mock_context.is_valid = True
        mock_context.trace_id = 0x00000000000000000000000000000042
        mock_context.span_id = 0x0000000000000042
        mock_context.trace_flags = MagicMock()
        mock_context.trace_flags.__format__ = lambda self, fmt: "01"

        mock_current = MagicMock()
        mock_current.get_current_span.return_value.get_span_context.return_value = mock_context

        # Patch via sys.modules since 'trace' is imported inside the _ENABLED block
        mock_trace_mod = sys.modules["opentelemetry.trace"]
        mock_trace_mod.get_current.return_value = mock_current

        with tracing.start_node_span("actor", {"task_id": "T1"}):
            tp = tracing.get_traceparent()

        assert tp is not None
        # W3C format: version-traceid-spanid-flags
        parts = tp.split("-")
        assert len(parts) == 4
        assert parts[0] == "00"  # version
        assert len(parts[1]) == 32  # trace_id (64 bits = 32 hex chars)
        assert len(parts[2]) == 16  # span_id (32 bits = 16 hex chars)
        assert len(parts[3]) == 2  # flags

    def test_returns_none_for_invalid_span_context(self, otel_mock):
        """get_traceparent() returns None when span context is invalid."""
        mock_tracer = otel_mock.get_tracer.return_value
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None,
        )

        # Mock an invalid span context
        mock_context = MagicMock()
        mock_context.is_valid = False

        mock_current = MagicMock()
        mock_current.get_current_span.return_value.get_span_context.return_value = mock_context

        mock_trace_mod = sys.modules["opentelemetry.trace"]
        mock_trace_mod.get_current.return_value = mock_current

        with tracing.start_node_span("actor", {"task_id": "T1"}):
            tp = tracing.get_traceparent()

        assert tp is None

    def test_returns_none_on_exception(self, otel_mock):
        """get_traceparent() returns None when trace module raises."""
        mock_tracer = otel_mock.get_tracer.return_value
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None,
        )

        mock_trace_mod = sys.modules["opentelemetry.trace"]
        mock_trace_mod.get_current.side_effect = RuntimeError("boom")

        with tracing.start_node_span("actor", {"task_id": "T1"}):
            tp = tracing.get_traceparent()

        assert tp is None
