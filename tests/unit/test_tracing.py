"""
tests/unit/test_tracing.py
===========================
Tests for the optional OpenTelemetry instrumentation module.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from sacv import tracing


class TestTracingDisabled:
    """Verify no-op behavior when OTel is not enabled and SDK is absent."""

    def test_get_tracer_returns_none_when_disabled(self):
        """get_tracer() returns None when OTel is not enabled."""
        # tracing._HAS_OTEL is False by default (no OTel env vars set)
        assert tracing.get_tracer() is None

    def test_start_node_span_yields_none_when_disabled(self):
        """start_node_span yields None and does not error when OTel is disabled."""
        with tracing.start_node_span("actor", {"task_id": "T1"}) as span:
            assert span is None

    def test_start_node_span_is_context_manager(self):
        """start_node_span can be used as a context manager."""
        with tracing.start_node_span("scout", {}):
            pass  # no exception should be raised


class TestTracingEnabled:
    """Verify correct span creation when OTel is enabled with a mock SDK."""

    @pytest.fixture(autouse=True)
    def _mock_otel_sdk(self):
        """Replace OTel imports with mocks so spans can be created without the SDK."""
        mock_trace = MagicMock()
        mock_provider = MagicMock()
        mock_processor = MagicMock()
        mock_exporter = MagicMock()

        mock_trace.get_tracer.return_value = MagicMock()
        mock_trace.get_tracer.return_value.start_as_current_span = MagicMock()

        sys_modules = {}
        for key in (
            "opentelemetry",
            "opentelemetry.trace",
            "opentelemetry.sdk.trace",
            "opentelemetry.sdk.trace.export",
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
        ):
            parts = key.rsplit(".", 1)
            parent_key = parts[0]
            if parent_key not in sys_modules:
                sys_modules[parent_key] = sys.modules.get(parent_key, None)
            if key not in sys.modules:
                if key == "opentelemetry.trace":
                    sys.modules[key] = mock_trace
                else:
                    sys.modules[key] = MagicMock()

        # Store originals and replace
        original_has_otel = tracing._HAS_OTEL
        original_enabled = tracing._ENABLED
        original_tracer = tracing._tracer

        tracing._ENABLED = True
        tracing._HAS_OTEL = True
        tracing._tracer = mock_trace.get_tracer.return_value

        yield mock_trace

        # Restore
        tracing._ENABLED = original_enabled
        tracing._HAS_OTEL = original_has_otel
        tracing._tracer = original_tracer
        tracing._provider = None
        tracing._provider_add_processor = None

    def test_get_tracer_returns_tracer_when_enabled(self, _mock_otel_sdk):
        """get_tracer() returns the tracer when OTel is enabled."""
        assert tracing.get_tracer() is not None

    def test_start_node_span_creates_span(self, _mock_otel_sdk):
        """start_node_span creates an OTel span with correct name."""
        mock_tracer = _mock_otel_sdk.get_tracer.return_value
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None,
        )

        with tracing.start_node_span("actor", {"task_id": "T1"}) as span:
            assert span is mock_span

        mock_tracer.start_as_current_span.assert_called_once_with("sacv.node.actor")

    def test_start_node_span_sets_state_attributes(self, _mock_otel_sdk):
        """start_node_span sets task_id, session_id, module, phase, attempt attributes."""
        mock_tracer = _mock_otel_sdk.get_tracer.return_value
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None,
        )

        state = {
            "task_id": "T42",
            "session_id": "sess-xyz",
            "module_type": "backend-domain",
            "current_phase": "actor",
            "correction_state": {"attempt_count": 3},
        }
        with tracing.start_node_span("complete", state):
            pass

        calls = [c[0][0] for c in mock_span.set_attribute.call_args_list]
        assert "sacv.task_id" in calls
        assert "sacv.session_id" in calls
        assert "sacv.module" in calls
        assert "sacv.phase" in calls
        assert "sacv.attempt" in calls

    def test_start_node_span_handles_missing_correction_state(self, _mock_otel_sdk):
        """start_node_span defaults attempt to 0 when correction_state is missing."""
        mock_tracer = _mock_otel_sdk.get_tracer.return_value
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span,
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=None,
        )

        with tracing.start_node_span("scout", {"task_id": "T1"}):
            pass

        # Verify attempt was set to 0 (the default)
        attempt_calls = [
            c for c in mock_span.set_attribute.call_args_list
            if c[0][0] == "sacv.attempt"
        ]
        assert len(attempt_calls) == 1
        assert attempt_calls[0][0][1] == 0
