"""Offset logic: running-clock detection, smoothing, outliers and jumps."""

import pytest

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.config import TimelineParams
from aicoach.scoreboard.offsets import analyze_offsets, detect_jumps, running_mask

PARAMS = TimelineParams()


def _times(n: int, start: float = 0.0) -> list[float]:
    return [start + float(i) for i in range(n)]


def _steady(n: int, offset: int = 1000, start: float = 0.0) -> tuple[list[float], list[int | None]]:
    t = _times(n, start)
    return t, [int(x) - offset for x in t]


@pytest.mark.unit
def test_running_mask_empty_input_returns_empty_list() -> None:
    assert running_mask([], [], PARAMS) == []


@pytest.mark.unit
def test_running_mask_length_mismatch_raises() -> None:
    with pytest.raises(ScoreboardParseError, match="same length"):
        running_mask([0.0, 1.0], [1], PARAMS)


@pytest.mark.unit
def test_running_mask_steady_clock_is_running_everywhere() -> None:
    t, c = _steady(30, start=1100)
    assert all(running_mask(t, c, PARAMS))


@pytest.mark.unit
def test_running_mask_stuck_clock_is_not_running() -> None:
    t = _times(30)
    c: list[int | None] = [2700] * 30  # a frozen 45:00
    assert not any(running_mask(t, c, PARAMS))


@pytest.mark.unit
def test_running_mask_missing_readings_are_never_running() -> None:
    t, c = _steady(30, start=1100)
    c[10] = None
    mask = running_mask(t, c, PARAMS)
    assert mask[10] is False
    assert mask[9] and mask[11]


@pytest.mark.unit
def test_running_mask_single_misread_does_not_break_neighbours() -> None:
    t, c = _steady(40, start=1100)
    c[20] = 99 * 60  # a digit misread
    mask = running_mask(t, c, PARAMS)
    assert mask[19] and mask[21] and mask[15] and mask[25]


@pytest.mark.unit
def test_running_mask_too_few_points_is_not_running() -> None:
    t, c = _steady(3, start=1100)
    assert running_mask(t, c, PARAMS) == [False, False, False]


@pytest.mark.unit
def test_running_mask_clock_that_freezes_midway_stops_being_running() -> None:
    t = _times(60)
    c: list[int | None] = [int(x) if x < 30 else 30 for x in t]
    mask = running_mask(t, c, PARAMS)
    assert mask[10] is True
    assert mask[50] is False


@pytest.mark.unit
def test_analyze_offsets_steady_clock_gives_constant_offset() -> None:
    t, c = _steady(40, offset=1000, start=1100)
    result = analyze_offsets(t, c, PARAMS)
    assert all(o == pytest.approx(1000.0) for o in result.smoothed)
    assert not any(result.outlier)


@pytest.mark.unit
def test_analyze_offsets_misread_is_flagged_and_ignored_by_smoothing() -> None:
    t, c = _steady(40, offset=1000, start=1100)
    c[20] = 99 * 60 + 59
    result = analyze_offsets(t, c, PARAMS)
    assert result.outlier[20] is True
    assert sum(result.outlier) == 1
    assert result.smoothed[20] == pytest.approx(1000.0)


@pytest.mark.unit
def test_analyze_offsets_without_clock_gives_none_everywhere() -> None:
    t = _times(20)
    result = analyze_offsets(t, [None] * 20, PARAMS)
    assert result.smoothed == [None] * 20
    assert not any(result.outlier)
    assert not any(result.running)


@pytest.mark.unit
def test_analyze_offsets_stuck_clock_has_no_offset() -> None:
    t = _times(30)
    result = analyze_offsets(t, [2700] * 30, PARAMS)
    assert result.smoothed == [None] * 30
    assert not any(result.outlier)


@pytest.mark.unit
def test_analyze_offsets_quantisation_flip_stays_below_jump_threshold() -> None:
    t, c = _steady(40, offset=1000, start=1100)
    for i in range(20, 40):  # the clock phase moves by one second
        c[i] = (c[i] or 0) - 1
    result = analyze_offsets(t, c, PARAMS)
    assert detect_jumps(t, result.smoothed, PARAMS.jump_tol_s) == []


@pytest.mark.unit
def test_detect_jumps_constant_offset_returns_nothing() -> None:
    assert detect_jumps(_times(5), [10.0] * 5, 2.0) == []


@pytest.mark.unit
def test_detect_jumps_empty_and_none_only_input_return_nothing() -> None:
    assert detect_jumps([], [], 2.0) == []
    assert detect_jumps(_times(3), [None, None, None], 2.0) == []


@pytest.mark.unit
def test_detect_jumps_records_times_before_and_after_without_correcting() -> None:
    t = [0.0, 1.0, 2.0, 900.0, 901.0]
    s: list[float | None] = [100.0, 100.0, 100.0, 1100.0, 1100.0]
    jumps = detect_jumps(t, s, 2.0)
    assert len(jumps) == 1
    assert (jumps[0].t_before, jumps[0].t_after) == (2.0, 900.0)
    assert (jumps[0].offset_before, jumps[0].offset_after) == (100.0, 1100.0)


@pytest.mark.unit
def test_detect_jumps_skips_samples_without_offset_when_comparing() -> None:
    t = [0.0, 1.0, 2.0, 3.0]
    s: list[float | None] = [100.0, None, None, 110.0]
    jumps = detect_jumps(t, s, 2.0)
    assert [(j.t_before, j.t_after) for j in jumps] == [(0.0, 3.0)]


@pytest.mark.unit
@pytest.mark.parametrize(("delta", "expected"), [(2.0, 0), (2.5, 1), (-2.5, 1), (1.0, 0)])
def test_detect_jumps_threshold_is_exclusive(delta: float, expected: int) -> None:
    jumps = detect_jumps([0.0, 1.0], [100.0, 100.0 + delta], 2.0)
    assert len(jumps) == expected


@pytest.mark.unit
def test_analyze_offsets_real_step_in_offset_is_reported_as_one_jump() -> None:
    # The second half starts 15 minutes later in video time with the same clock rule.
    t1, c1 = _steady(60, offset=1000, start=1100)
    t2, c2 = _steady(60, offset=1000 - 900, start=3000)
    result = analyze_offsets(t1 + t2, c1 + c2, PARAMS)
    jumps = detect_jumps(t1 + t2, result.smoothed, PARAMS.jump_tol_s)
    assert len(jumps) == 1
    assert jumps[0].offset_before == pytest.approx(1000.0)
    assert jumps[0].offset_after == pytest.approx(100.0)
    assert not any(result.outlier)


@pytest.mark.unit
def test_analyze_offsets_is_deterministic() -> None:
    t, c = _steady(50, start=1100)
    c[7] = 5
    assert analyze_offsets(t, c, PARAMS) == analyze_offsets(t, c, PARAMS)


@pytest.mark.unit
def test_running_mask_decreasing_time_raises() -> None:
    with pytest.raises(ScoreboardParseError, match="must not decrease"):
        running_mask([0.0, 2.0, 1.0], [0, 2, 1], PARAMS)


@pytest.mark.unit
def test_running_mask_non_finite_time_raises() -> None:
    with pytest.raises(ScoreboardParseError, match="finite"):
        running_mask([0.0, float("nan")], [0, 1], PARAMS)


@pytest.mark.unit
def test_detect_jumps_length_mismatch_raises() -> None:
    with pytest.raises(ScoreboardParseError, match="same length"):
        detect_jumps([0.0, 1.0], [1.0], 2.0)


@pytest.mark.unit
def test_analyze_offsets_step_of_five_seconds_without_a_time_gap_is_one_jump() -> None:
    # The clock jumps forward by 5 s in the middle of continuous play (no gap in readings).
    t, c = _steady(80, offset=1000, start=1100)
    for i in range(40, 80):
        c[i] = (c[i] or 0) + 5
    result = analyze_offsets(t, c, PARAMS)
    jumps = detect_jumps(t, result.smoothed, PARAMS.jump_tol_s)
    assert len(jumps) == 1
    assert jumps[0].offset_before == pytest.approx(1000.0)
    assert jumps[0].offset_after == pytest.approx(995.0)
    assert not any(result.outlier)


@pytest.mark.unit
def test_analyze_offsets_clock_stopped_for_a_few_seconds_has_no_false_outliers() -> None:
    t, c = _steady(80, offset=1000, start=1100)
    for i in range(40, 80):  # the clock stands still for 10 s, then runs again 10 s behind
        c[i] = (c[i] or 0) - 10
    for i in range(40, 50):
        c[i] = c[39]
    result = analyze_offsets(t, c, PARAMS)
    assert result.smoothed[20] == pytest.approx(1000.0)
    assert result.smoothed[70] == pytest.approx(1010.0)
    assert not any(result.running[43:47])  # the frozen stretch has no offset
    assert not any(result.outlier)
