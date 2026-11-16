"""`status.json` in the match directory: the state of every stage, for the UI to poll."""

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from aicoach.errors import StorageError
from aicoach.logging_setup import stage_logger

STATUS_FILE = "status.json"
# "waiting": the stage needs a file a person brings (the T-DEED output of the notebook).
# "skipped": an optional step that cannot run here (no Tesseract); `error` tells why.
State = Literal["running", "done", "failed", "waiting", "skipped"]


class StageStatus(BaseModel):
    """State of one stage: `progress` is 0..1, `error` is set when the stage failed.

    `started_at` and `finished_at` are ISO 8601 UTC times and `duration_s` is the length in
    seconds of the last execution; all three are `None` for a stage that did not run
    (skipped, waiting) and in files written before they existed.
    """

    state: State
    progress: float = Field(ge=0, le=1)
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = Field(default=None, ge=0)


class StatusFile(BaseModel):
    """The whole file: one entry per stage name."""

    stages: dict[str, StageStatus] = Field(default_factory=dict)


def utc_now_iso() -> str:
    """Current time as ISO 8601 in UTC with millisecond precision."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def read_status(match_dir: Path) -> StatusFile:
    """Read `status.json`. A missing file gives an empty status; an unreadable or invalid one
    also does, with a warning in the log (the next write replaces it whole, so a broken file
    must not stop a stage)."""
    path = match_dir / STATUS_FILE
    try:
        return StatusFile.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return StatusFile()
    except (OSError, ValidationError, ValueError) as err:
        # The file content is not logged: only where it is and what kind of failure it was.
        stage_logger("status", match_dir.name).warning(
            "status_file_unreadable", path=str(path), reason=type(err).__name__
        )
        return StatusFile()


def write_status(
    match_dir: Path, stage: str, state: State, progress: float, error: str | None = None
) -> None:
    """Set the status of one stage, keeping the other stages. The file is replaced atomically.

    `progress` is clamped to 0..1.

    Raises:
        ValueError: `progress` is NaN.
        StorageError: the file cannot be written.
    """
    if math.isnan(progress):
        msg = "progress must not be NaN"
        raise ValueError(msg)
    status = read_status(match_dir)
    old = status.stages.get(stage)
    new = StageStatus(state=state, progress=min(1.0, max(0.0, progress)), error=error)
    status.stages[stage] = new.model_copy(update=_carried_times(old, state))
    _write(match_dir, status)


def _carried_times(old: StageStatus | None, state: State) -> dict[str, str | float | None]:
    """Times that survive a state change.

    `running` marks the start of real work: the first one stamps `started_at`, progress
    updates keep it. `done` or `failed` after `running` keep that start, so a later timing
    write knows the stage really ran. `done` over `done` is a stage that found its result in
    the cache (a real run writes `running` first): the times of the work that made it stay,
    and if there were none, none appear. Anything else starts without times.
    """
    if state == "running":
        started = old.started_at if old is not None and old.state == "running" else None
        return {"started_at": started or utc_now_iso()}
    if old is None:
        return {}
    if state in ("done", "failed") and old.state == "running":
        return {"started_at": old.started_at}
    if state == "done" and old.state == "done":
        return {
            "started_at": old.started_at,
            "finished_at": old.finished_at,
            "duration_s": old.duration_s,
        }
    return {}


def write_timing(
    match_dir: Path, stage: str, started_at: str, finished_at: str, duration_s: float
) -> bool:
    """Add the timing of the last execution to an existing entry, keeping its state.

    Nothing is written for a stage without an entry, or one that did not run (`skipped`,
    `waiting`): those have no duration.

    Returns:
        Whether the timing was written.

    Raises:
        ValueError: `duration_s` is negative or NaN.
        StorageError: the file cannot be written.
    """
    if math.isnan(duration_s) or duration_s < 0:
        msg = "duration_s must be a non-negative number"
        raise ValueError(msg)
    status = read_status(match_dir)
    entry = status.stages.get(stage)
    if entry is None or entry.state in ("skipped", "waiting"):
        return False
    status.stages[stage] = entry.model_copy(
        update={"started_at": started_at, "finished_at": finished_at, "duration_s": duration_s}
    )
    _write(match_dir, status)
    return True


def _write(match_dir: Path, status: StatusFile) -> None:
    path = match_dir / STATUS_FILE
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(status.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as err:
        tmp.unlink(missing_ok=True)
        msg = f"cannot write {path}"
        raise StorageError(msg) from err
