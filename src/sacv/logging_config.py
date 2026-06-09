"""
sacv/logging_config.py
======================
Centralised structlog configuration for the SACV workflow.

Call configure_logging() once at process startup (in cli.py and in conftest.py).
After that, every structlog.get_logger() call inherits the configured pipeline.

LOG_FORMAT env var:
  "json"  (default in CI/production) -> JSON output suitable for log aggregation
  "console"                          -> Human-readable coloured output for development
LOG_LEVEL env var:
  "DEBUG", "INFO" (default), "WARNING", "ERROR"
"""
from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging() -> None:
    """Configure structlog for the entire process. Call once at startup."""
    log_format = os.environ.get("LOG_FORMAT", "json").lower()
    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # Shared processors run on every log record (structlog and stdlib alike)
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "console":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure Python stdlib logging to route through structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "asyncio", "docker", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
