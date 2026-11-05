"""Time mapping: offset segments, game clock from video time and back."""

import math

import pytest

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.assemble import RawReading, assemble_result
from aicoach.scoreboard.config import TimelineParams
from aicoach.scoreboard.mapping import (
    build_offset_segments,
    clock_at,
    result_clock_at,
    result_video_time_at,
    video_time_at,
    video_times_at,
)
from aicoach.scoreboard.offsets import detect_jumps
from aicoach.scoreboard.schemas import OffsetSegment, Roi

EX = 30.0
ROI = Roi(x0=0.05, y0=0.08, x1=0.25, y1=0.16)


def _seg(start: float, end: float, offset: float, n: int = 10) -> OffsetSegment:
    return OffsetSegment(
        start=start,
        end=end,
        clock_start_s=round(start - offset),
        clock_end_s=round(end - offset),
        offset_s=offset,
        n_readings=n,
    )


# First half: video 1000..3700 -> clock 0..2700 (offset 1000);
# second half: video 4700..7400 -> clock 2700..5400 (offset 2000).
SEGMENTS = [_seg(1000, 3700, 1000), _seg(4700, 7400, 2000)]


@pytest.mark.unit
def test_build_offset_segments_splits_at_jumps() -> None:
    t = [1000.0, 1001.0, 1002.0, 4700.0, 4701.0]
    smoothed: list[float | None] = [1000.0, 1000.0, 1000.0, 2000.0, 2000.0]
    jumps = detect_jumps(t, smoothed, 2.0)
    segments = build_offset_segments(t, smoothed, jumps)
    assert [(s.start, s.end, s.offset_s, s.n_readings) for s in segments] == [
        (1000.0, 1002.0, 1000.0, 3),
        (4700.0, 4701.0, 2000.0, 2),
    ]
    assert segments[0].clock_start_s == 0
    assert segments[1].clock_end_s == 2701


@pytest.mark.unit
def test_build_offset_segments_without_offsets_returns_nothing() -> None:
    assert build_offset_segments([], [], []) == []
    assert build_offset_segments([0.0, 1.0], [None, None], []) == []


@pytest.mark.unit
def test_build_offset_segments_keeps_a_gap_without_jump_in_one_segment() -> None:
    # The scoreboard is gone for 60 s (a replay), the offset is the same before and after.
    t = [0.0, 1.0, 61.0, 62.0]
    smoothed: list[float | None] = [500.0, 500.0, 500.0, 500.0]
    segments = build_offset_segments(t, smoothed, detect_jumps(t, smoothed, 2.0))
    assert len(segments) == 1


@pytest.mark.unit
def test_clock_at_inside_a_segment_uses_its_offset() -> None:
    assert clock_at(SEGMENTS, 2000.0, EX) == pytest.approx(1000.0)
    assert clock_at(SEGMENTS, 5000.0, EX) == pytest.approx(3000.0)


@pytest.mark.unit
def test_clock_at_extrapolates_only_up_to_the_limit() -> None:
    assert clock_at(SEGMENTS, 3700.0 + 30.0, EX) == pytest.approx(2730.0)
    assert clock_at(SEGMENTS, 3700.0 + 31.0, EX) is None


@pytest.mark.unit
def test_clock_at_the_break_between_halves_is_none() -> None:
    assert clock_at(SEGMENTS, 4200.0, EX) is None


@pytest.mark.unit
def test_clock_at_before_the_first_reading_is_none() -> None:
    assert clock_at(SEGMENTS, 10.0, EX) is None


@pytest.mark.unit
def test_clock_at_without_segments_is_none() -> None:
    assert clock_at([], 100.0, EX) is None


@pytest.mark.unit
def test_video_times_at_gives_one_time_per_matching_segment() -> None:
    assert video_times_at(SEGMENTS, 600.0, EX) == [pytest.approx(1600.0)]
    assert video_times_at(SEGMENTS, 3000.0, EX) == [pytest.approx(5000.0)]


@pytest.mark.unit
def test_video_times_at_overlapping_clock_range_returns_both_candidates() -> None:
    # First half has stoppage time 45:00-47:30, the second half starts again at 45:00.
    segments = [_seg(1000, 3850, 1000), _seg(4700, 7400, 2000)]
    times = video_times_at(segments, 2750.0, EX)
    assert times == [pytest.approx(3750.0), pytest.approx(4750.0)]


@pytest.mark.unit
def test_video_time_at_picks_the_candidate_nearest_to_a_hint() -> None:
    segments = [_seg(1000, 3850, 1000), _seg(4700, 7400, 2000)]
    assert video_time_at(segments, 2750.0, EX, near_t_v=4800.0) == pytest.approx(4750.0)
    assert video_time_at(segments, 2750.0, EX, near_t_v=3700.0) == pytest.approx(3750.0)
    assert video_time_at(segments, 2750.0, EX) == pytest.approx(3750.0)


@pytest.mark.unit
def test_video_time_at_outside_every_segment_is_none() -> None:
    assert video_time_at(SEGMENTS, 9000.0, EX) is None
    assert video_time_at([], 10.0, EX) is None


@pytest.mark.unit
def test_mapping_round_trip_inside_a_segment() -> None:
    for clock in (10.0, 1234.0, 2700.0, 3500.0):
        t = video_time_at(SEGMENTS, clock, EX)
        assert t is not None
        assert clock_at(SEGMENTS, t, EX) == pytest.approx(clock)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_mapping_rejects_non_finite_input(bad: float) -> None:
    with pytest.raises(ScoreboardParseError, match="finite"):
        clock_at(SEGMENTS, bad, EX)
    with pytest.raises(ScoreboardParseError, match="finite"):
        video_times_at(SEGMENTS, bad, EX)


@pytest.mark.unit
def test_result_wrappers_use_the_stored_segments() -> None:
    result = assemble_result(
        "m.mp4",
        "opencv_gpu",
        1.0,
        ROI,
        [float(i) for i in range(40)],
        [RawReading(visible=True, clock_s=i - 5, score=(0, 0), conf=0.9) for i in range(40)],
        TimelineParams(),
        40.0,
    )
    assert result_clock_at(result, 20.0, 30.0) == pytest.approx(15.0)
    assert result_video_time_at(result, 15.0, 30.0) == pytest.approx(20.0)
    assert result_video_time_at(result, 15.0, 30.0, near_t_v=19.0) == pytest.approx(20.0)


@pytest.mark.unit
def test_build_offset_segments_length_mismatch_raises() -> None:
    with pytest.raises(ScoreboardParseError, match="same length"):
        build_offset_segments([0.0, 1.0], [1.0], [])


@pytest.mark.unit
def test_build_offset_segments_repeated_time_at_a_jump_opens_one_segment() -> None:
    t = [0.0, 1.0, 5.0, 5.0, 6.0]
    smoothed: list[float | None] = [10.0, 10.0, 90.0, 90.0, 90.0]
    segments = build_offset_segments(t, smoothed, detect_jumps(t, smoothed, 2.0))
    assert len(segments) == 2


@pytest.mark.unit
def test_clock_at_that_would_be_negative_is_none() -> None:
    segments = [_seg(1000, 3700, 1000)]
    assert clock_at(segments, 990.0, EX) is None
