"""
tests/unit/test_span_event.py
==============================
Tests for the span_event() helper in sacv.tracing.

TDD checklist:
- [x] Every new function has a test
- [x] Tests cover disabled, no-span, and active-span scenarios
- [x] Tests use real code (mocks only for OTel SDK)
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


class TestSpanEventDisabled:
    """span_event() is a no-op when OTel is disabled."""

    def test_noop_when_disabled(self):
        """span_event() does not raise when OTel is disabled."""
        # tracing._HAS_OTEL is False by default (no OTel env vars set)
        tracing.span_event("test_event")

    def test_noop_with_attributes_when_disabled(self):
        """span_event() ignores attributes when OTel is disabled."""
        tracing.span_event("test_event", {"key": "val"})


class TestSpanEventOutsideSpan:
    """span_event() is a no-op when not inside start_node_span."""

    def test_noop_outside_span_context(self, otel_mock):
        """span_event() does not raise when called outside start_node_span."""
        tracing.span_event("test_event")

    def test_noop_with_attributes_outside_span(self, otel_mock):
        """span_event() ignores attributes outside span context."""
        tracing.span_event("test_event", {"key": 42})


class TestSpanEventInsideSpan:
    """span_event() records events on the current span."""

    def test_adds_event_to_span(self, otel_mock):
        """span_event() calls span.add_event with the correct name."""
        mock_tracer = otel_mock.get_tracer.return_value
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None,
        )

        with tracing.start_node_span("actor", {"task_id": "T1"}):
            tracing.span_event("generated")

        mock_span.add_event.assert_called_once_with("sacv.generated")

    def test_sets_event_attributes(self, otel_mock):
        """span_event() sets attributes prefixed with sacv.event."""
        mock_tracer = otel_mock.get_tracer.return_value
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None,
        )

        with tracing.start_node_span("actor", {"task_id": "T1"}):
            tracing.span_event("generated", {"files": 3, "lines": 120})

        attr_keys = {c[0][0] for c in mock_span.set_attribute.call_args_list}
        assert "sacv.event.files" in attr_keys
        assert "sacv.event.lines" in attr_keys

    def test_multiple_events_on_same_span(self, otel_mock):
        """Multiple span_event() calls all record on the same span."""
        mock_tracer = otel_mock.get_tracer.return_value
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None,
        )

        with tracing.start_node_span("actor", {"task_id": "T1"}):
            tracing.span_event("first")
            tracing.span_event("second")

        assert mock_span.add_event.call_count == 2
        events = [c[0][0] for c in mock_span.add_event.call_args_list]
        assert "sacv.first" in events
        assert "sacv.second" in events
