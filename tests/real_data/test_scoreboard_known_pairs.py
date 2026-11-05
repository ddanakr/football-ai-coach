"""Check scoreboard.json of a real match against two clock readings noted by the owner.

Needs the match video and a finished scoreboard stage: set AICOACH_REAL_MATCH_VIDEO to the
video path (the result is read from `scoreboard.json` beside it). Skipped otherwise. The
test never reads or stores frames; nothing from the match enters the repository.
"""

import os
from pathlib import Path

import pytest

from aicoach.scoreboard.config import TimelineParams
from aicoach.scoreboard.mapping import result_clock_at, result_video_time_at
from aicoach.scoreboard.schemas import ScoreboardResult
from aicoach.scoreboard.stage import RESULT_FILE

# (video time in seconds, clock in seconds): 1:40:46 = 69:17 and 1:51:27 = 79:58.
PAIRS = [(1 * 3600 + 40 * 60 + 46, 69 * 60 + 17), (1 * 3600 + 51 * 60 + 27, 79 * 60 + 58)]
TOLERANCE_S = 2.0


@pytest.fixture(scope="module")
def result() -> ScoreboardResult:
    env = os.environ.get("AICOACH_REAL_MATCH_VIDEO")
    if not env:
        pytest.skip("AICOACH_REAL_MATCH_VIDEO is not set")
    path = Path(env).parent / RESULT_FILE
    if not path.is_file():
        pytest.skip(f"{path} not found: run the scoreboard stage first")
    return ScoreboardResult.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.mark.real_data
@pytest.mark.parametrize(("t_v", "clock_s"), PAIRS)
def test_clock_at_the_noted_video_time_matches_within_two_seconds(
    result: ScoreboardResult, t_v: int, clock_s: int
) -> None:
    limit = TimelineParams().max_extrapolation_s
    got = result_clock_at(result, float(t_v), limit)
    assert got is not None
    assert abs(got - clock_s) <= TOLERANCE_S


@pytest.mark.real_data
@pytest.mark.parametrize(("t_v", "clock_s"), PAIRS)
def test_video_time_of_the_noted_clock_matches_within_two_seconds(
    result: ScoreboardResult, t_v: int, clock_s: int
) -> None:
    limit = TimelineParams().max_extrapolation_s
    got = result_video_time_at(result, float(clock_s), limit, near_t_v=float(t_v))
    assert got is not None
    assert abs(got - t_v) <= TOLERANCE_S
