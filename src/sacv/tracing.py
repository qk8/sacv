"""
sacv/tracing.py
===============
Optional OpenTelemetry instrumentation for the SACV workflow.

Activated by setting ``OTEL_EXPORTER_OTLP_ENDPOINT`` or
``SACV_OTEL_ENABLED=true``. When not activated, all functions are no-ops
so no dependency on the OTel SDK is required.

Usage::

    from sacv.tracing import get_tracer, start_node_span
    tracer = get_tracer()
    with start_node_span("actor", state) as span:
        span.set_attribute("attempt", attempt)
        ... do work ...
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

_ENABLED = bool(
    os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    or os.environ.get("SACV_OTEL_ENABLED", "").lower() in ("1", "true", "yes")
)

if _ENABLED:
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        _provider = TracerProvider()
        _provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter())
        )
        trace.set_tracer_provider(_provider)
        _tracer = trace.get_tracer("sacv.workflow", "0.1.0")
        _HAS_OTEL = True
    except ImportError:
        _HAS_OTEL = False
else:
    _tracer = None  # type: ignore[assignment]
    _HAS_OTEL = False


def get_tracer() -> Any:
    """Return the SACV tracer, or None if OTel is disabled."""
    return _tracer if _HAS_OTEL else None


@contextmanager
def start_node_span(
    node_name: str,
    state: dict[str, object],
) -> Generator[Any, None, None]:
    """
    Context manager that wraps a node invocation in an OTel span.

    No-op when OTel is not configured; yields ``None``.
    """
    if not _HAS_OTEL:
        yield None
        return

    correction_state = state.get("correction_state") or {}
    with _tracer.start_as_current_span(f"sacv.node.{node_name}") as span:
        span.set_attribute("sacv.task_id", str(state.get("task_id", "")))
        span.set_attribute("sacv.session_id", str(state.get("session_id", "")))
        span.set_attribute("sacv.module", str(state.get("module_type", "")))
        span.set_attribute("sacv.phase", str(state.get("current_phase", "")))
        span.set_attribute(
            "sacv.attempt",
            int(correction_state.get("attempt_count", 0)),
        )
        yield span
