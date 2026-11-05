"""`status.json` in the match directory: the state of every stage, for the UI to poll."""

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from aicoach.errors import StorageError
from aicoach.logging_setup import stage_logger

STATUS_FILE = "status.json"
State = Literal["running", "done", "failed"]


class StageStatus(BaseModel):
    """State of one stage: `progress` is 0..1, `error` is set when the stage failed."""

    state: State
    progress: float = Field(ge=0, le=1)
    error: str | None = None


class StatusFile(BaseModel):
    """The whole file: one entry per stage name."""

    stages: dict[str, StageStatus] = Field(default_factory=dict)


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
    status.stages[stage] = StageStatus(
        state=state, progress=min(1.0, max(0.0, progress)), error=error
    )
    path = match_dir / STATUS_FILE
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(status.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as err:
        tmp.unlink(missing_ok=True)
        msg = f"cannot write {path}"
        raise StorageError(msg) from err
