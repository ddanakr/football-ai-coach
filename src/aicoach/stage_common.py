"""Helpers shared by the stages: atomic file writes, status reporting and progress.

The side effects of a stage (disk, `status.json`, logs) go through these functions, so that every
stage fails and reports in the same way.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import structlog

from aicoach.errors import StorageError
from aicoach.status import State, write_status

Logger = structlog.typing.FilteringBoundLogger
_PROGRESS_STEP = 0.02  # write status.json after this much progress


def write_atomic(path: Path, text: str) -> None:
    """Write to a temporary file and rename it, so a failure keeps the old file intact.

    Raises:
        StorageError: the file cannot be written.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError as err:
        tmp.unlink(missing_ok=True)
        msg = f"cannot write {path}"
        raise StorageError(msg) from err


def set_status(
    match_dir: Path,
    stage: str,
    state: State,
    progress: float,
    log: Logger,
    error: str | None = None,
) -> None:
    """Write status.json; a failure (for example the UI holds the file open) is logged and
    never stops the stage, because the status is only a progress report."""
    try:
        write_status(match_dir, stage, state, progress, error=error)
    except StorageError:
        log.warning("status_write_failed", state=state)


@dataclass
class Progress:
    """Writes status.json when progress moved enough; remembers the last value for errors."""

    match_dir: Path
    stage: str
    log: Logger
    last_written: float = 0.0
    value: float = 0.0

    def update(self, value: float) -> None:
        """Record the progress (0..1) and write the status file after a noticeable step."""
        self.value = value
        if value - self.last_written >= _PROGRESS_STEP:
            set_status(self.match_dir, self.stage, "running", value, self.log)
            self.last_written = value


def elapsed_ms(started: float) -> int:
    """Milliseconds since `started` (a `time.perf_counter()` value)."""
    return round((time.perf_counter() - started) * 1000)


def record_failure(
    match_dir: Path, stage: str, progress: float, err: BaseException, log: Logger, started: float
) -> None:
    """Log the failure and mark the stage as failed; a failing status write must not hide the
    original error. The previous result file of the stage is never touched here."""
    reason = str(err) or type(err).__name__
    log.exception("stage_failed", reason=reason, duration_ms=elapsed_ms(started))
    set_status(match_dir, stage, "failed", progress, log, error=reason)


__all__ = [
    "Logger",
    "Progress",
    "elapsed_ms",
    "record_failure",
    "set_status",
    "write_atomic",
]
