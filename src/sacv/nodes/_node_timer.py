"""
nodes/_node_timer.py
====================
Context manager for per-node execution timing.

Logs a structured completion event with duration_ms on exit.
When called with a ``state`` dict, also creates an OpenTelemetry span
via :func:`sacv.tracing.start_node_span` (no-op when OTel is disabled).
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

import structlog

from sacv.tracing import start_node_span

log = structlog.get_logger(__name__)


@asynccontextmanager
async def node_timer(
    node_name: str,
    logger: structlog.stdlib.BoundLogger | None = None,
    state: dict[str, object] | None = None,
) -> AsyncGenerator[dict[str, object], None]:
    """
    Async context manager that records start time and logs duration on exit.

    Yields a mutable dict so the caller can add extra fields to the completion log.

    When ``state`` is provided, also wraps execution in an OpenTelemetry span
    (no-op when OTel is not configured).

    Usage:
        async with node_timer("actor", state=state) as timing:
            timing["files_changed"] = len(diffs)
            ... do work ...
        # Logs: actor.timing duration_ms=1234 files_changed=5
    """
    t0 = time.monotonic()
    extra: dict[str, object] = {}
    span: Any = None
    with start_node_span(node_name, state) as span:
        try:
            yield extra
            duration_ms = int((time.monotonic() - t0) * 1000)
            if span is not None:
                span.set_attribute("sacv.duration_ms", duration_ms)
                for k, v in extra.items():
                    span.set_attribute(f"sacv.timing.{k}", v)
            (logger or log).info(f"{node_name}.timing", duration_ms=duration_ms, **extra)
        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            if span is not None:
                span.set_attribute("sacv.duration_ms", duration_ms)
                span.set_attribute("sacv.error", type(exc).__name__)
            (logger or log).error(
                f"{node_name}.timing_error",
                duration_ms=duration_ms,
                exc_info=True,
                **extra,
            )
            raise
