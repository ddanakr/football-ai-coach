"""Schemas for the evaluation inputs: ground-truth file and predictions."""

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from aicoach.domain.events import EventType
from aicoach.errors import EvalError

# Video seconds: finite and non-negative. Strict, so that "12" or true is not read as a time.
VideoTime = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class HalfBounds(_Frozen):
    """Start and end of a half in video seconds; `None` when not marked."""

    start: VideoTime | None = None
    end: VideoTime | None = None

    @model_validator(mode="after")
    def _start_not_after_end(self) -> "HalfBounds":
        if self.start is not None and self.end is not None and self.start > self.end:
            msg = "start must not be greater than end"
            raise ValueError(msg)
        return self


class GroundTruthEvent(_Frozen):
    """One ground-truth event as written by the labeling tool."""

    type: EventType
    t_v: VideoTime
    half: Literal[1, 2]
    clock: str | None = None
    team: Literal["home", "away"] | None = None
    visibility: bool = True
    replay: bool = False
    annotator: str = ""
    note: str = ""


class GroundTruth(_Frozen):
    """Ground-truth file: video name, half bounds and events."""

    video: str
    halves: dict[Literal["1", "2"], HalfBounds] = Field(default_factory=dict)
    events: tuple[GroundTruthEvent, ...]


class Prediction(_Frozen):
    """One system prediction. Predictions are immutable (CLAUDE.md section 5, item 3).

    `half` is optional; predictions without it are dropped when evaluating a single half.
    `status` is free text with the default `"auto"`; only `"disallowed"` changes evaluation.
    """

    type: EventType
    t_v: VideoTime
    score: Annotated[float, Field(allow_inf_nan=False)] | None = None
    status: str = "auto"
    half: Literal[1, 2] | None = None


def _format_validation_error(path: Path, err: ValidationError) -> str:
    """Turn a pydantic error into short `location: message` lines naming the file."""
    lines = [
        f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
        for item in err.errors()
    ]
    return f"{path}: invalid ground-truth file:\n  " + "\n  ".join(lines)


def load_ground_truth(path: Path) -> GroundTruth:
    """Read and validate a ground-truth JSON file.

    Args:
        path: Path to the labeling tool's JSON file.

    Returns:
        The validated, immutable ground truth.

    Raises:
        EvalError: The file cannot be read, is not valid JSON, or does not match the schema.
            The message names the file and the offending field (for example `events.1.t_v`).
    """
    try:
        # utf-8-sig tolerates the BOM that Windows editors add.
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as err:
        msg = f"{path}: cannot read ground-truth file: {err}"
        raise EvalError(msg) from err
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        msg = f"{path}: not valid JSON: {err}"
        raise EvalError(msg) from err
    try:
        return GroundTruth.model_validate(data)
    except ValidationError as err:
        raise EvalError(_format_validation_error(path, err)) from err
