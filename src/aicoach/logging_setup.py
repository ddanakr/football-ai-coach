"""structlog setup: JSON lines on stderr with the fields `stage`, `match_id` and `event`.

Every long operation must log `duration_ms` itself (CLAUDE.md section 6).
"""

import logging
import sys

import structlog

from aicoach.errors import ConfigError


def configure_logging(level: str = "INFO") -> None:
    """Send structured JSON logs to stderr. Safe to call more than once.

    Raises:
        ConfigError: `level` is not a standard logging level name (case does not matter).
    """
    numeric = logging.getLevelNamesMapping().get(level.strip().upper())
    if numeric is None:
        msg = f"unknown log level {level!r}"
        raise ConfigError(msg)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def stage_logger(stage: str, match_id: str) -> structlog.typing.FilteringBoundLogger:
    """A logger that carries `stage` and `match_id` in every record."""
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger().bind(
        stage=stage, match_id=match_id
    )
    return logger
