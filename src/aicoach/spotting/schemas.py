"""Schema of the file written by T-DEED's `inference.py` and its loader.

Only the JSON file is read; no T-DEED code is imported or copied (GPL-3.0, D-042).
"""

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aicoach.errors import SpottingError


class TDeedLabel(StrEnum):
    """The 17 SoccerNet action-spotting classes, spelled as in T-DEED's `class.txt`."""

    PENALTY = "Penalty"
    KICK_OFF = "Kick-off"
    GOAL = "Goal"
    SUBSTITUTION = "Substitution"
    OFFSIDE = "Offside"
    SHOTS_ON_TARGET = "Shots on target"
    SHOTS_OFF_TARGET = "Shots off target"
    CLEARANCE = "Clearance"
    BALL_OUT_OF_PLAY = "Ball out of play"
    THROW_IN = "Throw-in"
    FOUL = "Foul"
    INDIRECT_FREE_KICK = "Indirect free-kick"
    DIRECT_FREE_KICK = "Direct free-kick"
    CORNER = "Corner"
    YELLOW_CARD = "Yellow card"
    RED_CARD = "Red card"
    YELLOW_RED_CARD = "Yellow->red card"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class TDeedPrediction(_Frozen):
    """One detection: `frame` counts frames of the source video at 25 fps, from 0.

    Strict types: `"12"` or `true` is not read as a frame number or a confidence.
    """

    frame: Annotated[int, Field(ge=0, strict=True)]
    label: TDeedLabel
    confidence: Annotated[float, Field(allow_inf_nan=False, strict=True)]


class TDeedFile(_Frozen):
    """`{"predictions": [...]}` as written by `store_json_inference` in T-DEED."""

    predictions: tuple[TDeedPrediction, ...]


def _format_validation_error(path: Path, err: ValidationError) -> str:
    lines = [
        f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
        for item in err.errors()
    ]
    return f"{path}: invalid T-DEED file:\n  " + "\n  ".join(lines)


def load_tdeed_file(path: Path) -> TDeedFile:
    """Read and validate a T-DEED output file.

    Args:
        path: Path of `results_inference.json` (or a renamed copy).

    Returns:
        The validated predictions, in file order.

    Raises:
        SpottingError: The file cannot be read, is not valid JSON, lacks a field, has a label
            outside the 17 classes, a negative or non-integer frame, or a non-finite
            confidence. The message names the file and the field (for example
            `predictions.3.label`).
    """
    try:
        # utf-8-sig tolerates the BOM that Windows editors add.
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as err:
        msg = f"{path}: cannot read T-DEED file: {err}"
        raise SpottingError(msg) from err
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        msg = f"{path}: not valid JSON: {err}"
        raise SpottingError(msg) from err
    try:
        return TDeedFile.model_validate(data)
    except ValidationError as err:
        raise SpottingError(_format_validation_error(path, err)) from err
