"""Time codes of `--from` and `--to`, and the segment plan."""

import math

import pytest

from aicoach.errors import SpottingError
from aicoach.tdeed.timecode import (
    Segment,
    format_timecode,
    parse_timecode,
    parse_window,
    plan_segment,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("13:12", 792.0),
        ("00:00", 0.0),
        ("1:00:05", 3605.0),
        ("01:59:59", 7199.0),
        ("792", 792.0),
        (" 10:00 ", 600.0),
        ("120:00", 7200.0),  # minutes may pass 59 without an hour part, as in the match list
    ],
)
def test_parse_timecode_valid_forms_give_seconds(text: str, seconds: float) -> None:
    assert parse_timecode(text) == seconds


@pytest.mark.unit
@pytest.mark.parametrize(
    "text", ["", "abc", "10:60", "-5", "1.5", "1:2:3", "1:75:00", "10:", ":10", "1e3"]
)
def test_parse_timecode_garbage_raises(text: str) -> None:
    with pytest.raises(SpottingError, match="bad time|minutes must"):
        parse_timecode(text)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("seconds", "text"), [(0, "00:00"), (792.9, "13:12"), (3599, "59:59"), (3600, "1:00:00")]
)
def test_format_timecode_rounds_down_and_adds_hours(seconds: float, text: str) -> None:
    assert format_timecode(seconds) == text


@pytest.mark.unit
def test_parse_window_two_time_codes() -> None:
    assert parse_window("13:12-23:12") == (792.0, 1392.0)
    assert parse_window("792-1392") == (792.0, 1392.0)


@pytest.mark.unit
@pytest.mark.parametrize("text", ["13:12", "1-2-3", "20:00-10:00", "10:00-10:00", "a-b", "-"])
def test_parse_window_bad_input_raises(text: str) -> None:
    with pytest.raises(SpottingError, match="bad"):
        parse_window(text)


@pytest.mark.unit
def test_plan_segment_moves_the_start_down_to_the_24_s_grid_and_adds_the_tail() -> None:
    segment = plan_segment(800.0, 1400.0, 7940.0)

    assert segment == Segment(800.0, 792.0, 1400.0, 1460.0)
    assert segment.length_s == 608.0
    assert segment.cut_length_s == 668.0


@pytest.mark.unit
def test_plan_segment_start_already_on_the_grid_is_unchanged() -> None:
    assert plan_segment(792.0, 1392.0, 7940.0).start_s == 792.0
    assert plan_segment(0.0, 100.0, 7940.0).start_s == 0.0


@pytest.mark.unit
def test_plan_segment_missing_bounds_mean_the_whole_video() -> None:
    assert plan_segment(None, None, 500.0) == Segment(0.0, 0.0, 500.0, 500.0)
    assert plan_segment(None, 300.0, 500.0).start_s == 0.0
    assert plan_segment(240.0, None, 500.0).end_s == 500.0


@pytest.mark.unit
def test_plan_segment_end_and_tail_are_cut_at_the_video_end() -> None:
    segment = plan_segment(0.0, 9999.0, 500.0)

    assert (segment.end_s, segment.cut_end_s) == (500.0, 500.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("start", "end", "duration"),
    [
        (0.0, 30.0, 500.0),  # shorter than a minute
        (480.0, 500.0, 500.0),  # the piece left after the grid is under a minute
        (600.0, 700.0, 500.0),  # starts after the video ends
        (50.0, 40.0, 500.0),  # reversed
        (-1.0, 100.0, 500.0),  # negative
        (math.nan, 100.0, 500.0),
        (0.0, math.nan, 500.0),
        (0.0, 100.0, 0.0),  # no usable duration
        (0.0, 100.0, math.nan),
    ],
)
def test_plan_segment_unusable_input_raises(start: float, end: float, duration: float) -> None:
    with pytest.raises(SpottingError):
        plan_segment(start, end, duration)
