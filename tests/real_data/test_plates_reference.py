"""Check plates.json of a real match against plate times noted by the owner.

Needs the match video and a finished plates stage: set AICOACH_REAL_MATCH_VIDEO to the video
path. The result is read from `plates.json` and the owner's times from `plates_reference.json`
beside the video (outside the repository, never committed). Skipped if either is missing. The
test never reads or stores frames.
"""

import json
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from aicoach.plates.schemas import PlatesResult, PlateType
from aicoach.plates.stage import RESULT_FILE

REFERENCE_FILE = "plates_reference.json"
TOLERANCE_S = 1.0  # sampling is 0.5 s: the noted time may fall just after the last sample


class Check(BaseModel):
    """One noted time: a plate of `type` was on screen at `t_v` (video seconds)."""

    type: PlateType
    t_v: float


class Reference(BaseModel):
    """The reference file: the list of noted times."""

    checks: list[Check]


def _match_dir() -> Path:
    env = os.environ.get("AICOACH_REAL_MATCH_VIDEO")
    if not env:
        pytest.skip("AICOACH_REAL_MATCH_VIDEO is not set")
    return Path(env).parent


@pytest.fixture(scope="module")
def result() -> PlatesResult:
    path = _match_dir() / RESULT_FILE
    if not path.is_file():
        pytest.skip(f"{path} not found: run the plates stage first")
    return PlatesResult.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reference() -> Reference:
    path = _match_dir() / REFERENCE_FILE
    if not path.is_file():
        pytest.skip(f"{path} not found: it holds the owner's noted times")
    return Reference.model_validate(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.real_data
def test_plate_of_the_right_type_is_found_at_every_noted_time(
    result: PlatesResult, reference: Reference
) -> None:
    missing = [
        (check.type, check.t_v)
        for check in reference.checks
        if not any(
            p.type == check.type
            and p.t_first_seen - TOLERANCE_S <= check.t_v <= p.t_last_seen + TOLERANCE_S
            for p in result.plates
        )
    ]

    assert missing == []
