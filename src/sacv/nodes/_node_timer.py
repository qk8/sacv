"""
nodes/_node_timer.py
====================
Context manager for per-node execution timing.

Logs a structured completion event with duration_ms on exit.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog

log = structlog.get_logger(__name__)


@asynccontextmanager
async def node_timer(
    node_name: str,
    logger: structlog.stdlib.BoundLogger | None = None,
) -> AsyncGenerator[dict[str, object], None]:
    """
    Async context manager that records start time and logs duration on exit.

    Yields a mutable dict so the caller can add extra fields to the completion log.

    Usage:
        async with node_timer("actor") as timing:
            timing["files_changed"] = len(diffs)
            ... do work ...
        # Logs: actor.timing duration_ms=1234 files_changed=5
    """
    t0 = time.monotonic()
    extra: dict[str, object] = {}
    try:
        yield extra
        duration_ms = int((time.monotonic() - t0) * 1000)
        (logger or log).info(f"{node_name}.timing", duration_ms=duration_ms, **extra)
    except Exception:
        duration_ms = int((time.monotonic() - t0) * 1000)
        (logger or log).error(
            f"{node_name}.timing_error",
            duration_ms=duration_ms,
            exc_info=True,
            **extra,
        )
        raise
