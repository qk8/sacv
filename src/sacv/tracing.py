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

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
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
    except ImportError as _otel_import_err:
        _HAS_OTEL = False
        logging.getLogger(__name__).warning(
            "OTel enabled via environment variable but opentelemetry packages "
            "are not installed. Tracing is DISABLED. "
            "Install: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc. "
            "Error: %s", _otel_import_err,
        )
else:
    _tracer = None  # type: ignore[assignment]
    _HAS_OTEL = False

# Thread-local current span for span_event() helper
_current_span: ContextVar[Any | None] = ContextVar("current_span", default=None)


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
    token = _current_span.set(None)  # reset before entering
    try:
        with _tracer.start_as_current_span(f"sacv.node.{node_name}") as span:
            _current_span.set(span)
            span.set_attribute("sacv.task_id", str(state.get("task_id", "")))
            span.set_attribute("sacv.session_id", str(state.get("session_id", "")))
            span.set_attribute("sacv.module", str(state.get("module_type", "")))
            span.set_attribute("sacv.phase", str(state.get("current_phase", "")))
            span.set_attribute(
                "sacv.attempt",
                int(correction_state.get("attempt_count", 0)),
            )
            yield span
    finally:
        _current_span.reset(token)  # restore prior context


def span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """
    Record a structured event on the current OTel span.

    Safe to call from any depth — no-op when OTel is disabled or outside
    a ``start_node_span`` context.

    Usage::

        span_event("actor.generated", {"files": 3, "lines": 120})
    """
    span = _current_span.get()
    if span is None:
        return
    if attributes:
        for k, v in attributes.items():
            span.set_attribute(f"sacv.event.{k}", v)
    span.add_event(f"sacv.{name}")


def get_traceparent() -> str | None:
    """
    Return a W3C ``traceparent`` header string for the current span.

    Format: ``{version}-{trace_id}-{span_id}-{trace_flags}``

    Returns ``None`` when OTel is disabled or no active span exists.
    """
    if not _HAS_OTEL:
        return None

    span = _current_span.get()
    if span is None:
        return None

    try:
        ctx = trace.get_current()
        span_context = ctx.get_current_span().get_span_context()
        if not span_context.is_valid:
            return None
        trace_id = format(span_context.trace_id, "032x")
        span_id = format(span_context.span_id, "016x")
        flags = f"{span_context.trace_flags:02x}"
        return f"00-{trace_id}-{span_id}-{flags}"
    except Exception:
        return None
