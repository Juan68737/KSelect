"""
Structured logging via structlog.

KSELECT_LOG_FORMAT env var:
  "json"    -> structlog JSONRenderer (production)
  "console" -> structlog ConsoleRenderer with colors (development, default)

Every search() and query() call emits one structured log event at INFO level.
No PII is logged. Query text is truncated to 200 chars.
"""
from __future__ import annotations

import logging
import os
from typing import Any

_configured = False


def configure_logging() -> None:
    """Configure structlog once per process. Idempotent."""
    global _configured
    if _configured:
        return
    _configured = True

    try:
        import structlog  # type: ignore[import-untyped]

        fmt = os.environ.get("KSELECT_LOG_FORMAT", "console").lower()
        if fmt == "json":
            renderer = structlog.processors.JSONRenderer()
        else:
            renderer = structlog.dev.ConsoleRenderer()

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                renderer,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )
    except ImportError:
        # structlog optional — fall back to stdlib logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


def get_logger(name: str) -> Any:
    """Return a structlog logger if available, else a stdlib logger."""
    try:
        import structlog  # type: ignore[import-untyped]
        configure_logging()
        return structlog.get_logger(name)
    except ImportError:
        return logging.getLogger(name)


def log_search_event(
    logger: Any,
    *,
    query: str,
    mode: str,
    fusion: str,
    cache_hit: bool,
    latency_ms: float,
    chunks_retrieved: int,
    index_drift: float,
) -> None:
    _emit(logger, "search", dict(
        query=query[:200],
        mode=mode,
        fusion=fusion,
        cache_hit=cache_hit,
        latency_ms=round(latency_ms, 2),
        chunks_retrieved=chunks_retrieved,
        index_drift=round(index_drift, 4),
    ))


def log_query_event(
    logger: Any,
    *,
    query: str,
    mode: str,
    fusion: str,
    cache_hit: bool,
    latency_ms: float,
    confidence: float,
    chunks_retrieved: int,
    chunks_in_context: int,
    index_drift: float,
) -> None:
    _emit(logger, "query", dict(
        query=query[:200],
        mode=mode,
        fusion=fusion,
        cache_hit=cache_hit,
        latency_ms=round(latency_ms, 2),
        confidence=round(confidence, 4),
        chunks_retrieved=chunks_retrieved,
        chunks_in_context=chunks_in_context,
        index_drift=round(index_drift, 4),
    ))


def _emit(logger: Any, event: str, fields: dict) -> None:
    try:
        logger.info(event, **fields)
    except Exception:
        logging.getLogger(__name__).info("%s %s", event, fields)
