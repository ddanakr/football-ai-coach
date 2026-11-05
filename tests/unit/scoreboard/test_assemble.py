"""Result assembly on a synthetic match: two halves, a replay, a misread, a goal and a VAR."""

import pytest

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.assemble import RawReading, assemble_result
from aicoach.scoreboard.config import TimelineParams
from aicoach.scoreboard.mapping import clock_at, video_time_at
from aicoach.scoreboard.schemas import Roi, ScoreboardResult

PARAMS = TimelineParams()
ROI = Roi(x0=0.05, y0=0.08, x1=0.25, y1=0.16)
N = 700


def _synthetic() -> tuple[list[float], list[RawReading]]:
    """1 Hz. Half 1 on screen in 10..299 (clock = t - 10), the break, half 2 in 400..699
    (clock = t + 2300). Replay in 150..169 (board hidden). A misread at t=50. Goal at t=100,
    cancelled at t=200; a second goal at t=500."""
    t = [float(i) for i in range(N)]
    readings: list[RawReading] = []
    for tv in t:
        i = int(tv)
        on_screen = (10 <= i < 300 or 400 <= i < 700) and not 150 <= i < 170
        if not on_screen:
            readings.append(RawReading(visible=False))
            continue
        clock = i - 10 if i < 300 else i + 2300
        if i == 50:
            clock = 99 * 60 + 5  # digits misread
        score = (0, 0)
        if 100 <= i < 200:
            score = (1, 0)
        if i >= 500:
            score = (0, 1)
        readings.append(RawReading(visible=True, clock_s=clock, score=score, conf=0.95))
    return t, readings


@pytest.fixture(scope="module")
def result() -> ScoreboardResult:
    t, readings = _synthetic()
    return assemble_result("match.mp4", "opencv_gpu", 1.0, ROI, t, readings, PARAMS, float(N))


@pytest.mark.unit
def test_assemble_result_keeps_one_sample_per_second(result: ScoreboardResult) -> None:
    assert len(result.samples) == N
    assert result.samples[0].visible is False
    assert result.samples[20].clock_s == 10


@pytest.mark.unit
def test_assemble_result_flags_the_misread_and_keeps_the_offset(result: ScoreboardResult) -> None:
    assert result.samples[50].clock_outlier is True
    assert result.samples[50].offset_s == pytest.approx(10.0)
    assert result.stats.n_outliers == 1


@pytest.mark.unit
def test_assemble_result_reports_the_break_as_an_offset_jump(result: ScoreboardResult) -> None:
    [jump] = result.offset_jumps
    assert jump.t_before == 299.0
    assert jump.t_after == 400.0
    assert jump.offset_before == pytest.approx(10.0)
    assert jump.offset_after == pytest.approx(-2300.0)
    assert result.stats.n_jumps == 1


@pytest.mark.unit
def test_assemble_result_segments_and_median_offsets(result: ScoreboardResult) -> None:
    assert [round(s) for s in result.stats.median_offset_by_segment] == [10, -2300]
    assert len(result.offset_segments) == 2


@pytest.mark.unit
def test_assemble_result_score_changes_include_the_var_reversal(result: ScoreboardResult) -> None:
    got = [
        (c.t_old_last_seen, c.t_new_first_seen, c.old, c.new, c.direction)
        for c in result.score_changes
    ]
    assert got == [
        (99.0, 100.0, (0, 0), (1, 0), "up"),
        (199.0, 200.0, (1, 0), (0, 0), "down"),
        (499.0, 500.0, (0, 0), (0, 1), "up"),
    ]


@pytest.mark.unit
def test_assemble_result_visible_segments_bridge_the_replay_gap_only_if_short(
    result: ScoreboardResult,
) -> None:
    got = [(s.start, s.end) for s in result.visible_segments]
    assert got == [(10.0, 149.0), (170.0, 299.0), (400.0, 699.0)]


@pytest.mark.unit
def test_assemble_result_stats_count_visibility_and_unreadable_seconds(
    result: ScoreboardResult,
) -> None:
    assert result.stats.n_samples == N
    assert result.stats.n_visible == 140 + 130 + 300
    assert result.stats.n_unreadable == 0
    assert result.stats.unreadable_share_in_visible == 0.0
    # between the first and the last visible second: 10..699 is 690 s, 570 of them visible
    assert result.stats.visible_share_between_first_and_last == pytest.approx(570 / 690)


@pytest.mark.unit
def test_assemble_result_mapping_works_through_the_replay_and_not_through_the_break(
    result: ScoreboardResult,
) -> None:
    segments = result.offset_segments
    assert clock_at(segments, 160.0, 30.0) == pytest.approx(150.0)  # replay: board hidden
    assert clock_at(segments, 350.0, 30.0) is None  # the break
    assert video_time_at(segments, 2900.0, 30.0) == pytest.approx(600.0)


@pytest.mark.unit
def test_assemble_result_unreadable_seconds_inside_visible_are_counted() -> None:
    t = [float(i) for i in range(30)]
    readings = [RawReading(visible=True, clock_s=i, score=(0, 0), conf=0.9) for i in range(30)]
    readings[5] = RawReading(visible=True, clock_s=None, score=(0, 0))
    readings[6] = RawReading(visible=True, clock_s=6, score=None)
    result = assemble_result("m.mp4", "opencv_gpu", 1.0, ROI, t, readings, PARAMS, 30.0)
    assert result.stats.n_unreadable == 2
    assert result.stats.n_clock_unreadable == 1
    assert result.stats.n_score_unreadable == 1
    assert result.stats.unreadable_share_in_visible == pytest.approx(2 / 30)


@pytest.mark.unit
def test_assemble_result_empty_input_gives_an_empty_result() -> None:
    result = assemble_result("m.mp4", "opencv_gpu", 1.0, ROI, [], [], PARAMS, 0.0)
    assert result.samples == []
    assert result.stats.unreadable_share_in_visible is None
    assert result.stats.visible_share_between_first_and_last is None


@pytest.mark.unit
def test_assemble_result_never_visible_has_no_shares() -> None:
    t = [0.0, 1.0, 2.0]
    result = assemble_result(
        "m.mp4", "opencv_gpu", 1.0, ROI, t, [RawReading(visible=False)] * 3, PARAMS, 3.0
    )
    assert result.visible_segments == []
    assert result.stats.visible_share_between_first_and_last is None


@pytest.mark.unit
def test_assemble_result_hidden_board_drops_stray_values() -> None:
    reading = RawReading(visible=False, clock_s=5, score=(1, 1), conf=0.9, added_visible=True)
    result = assemble_result("m.mp4", "opencv_gpu", 1.0, ROI, [0.0], [reading], PARAMS, 1.0)
    sample = result.samples[0]
    assert (sample.clock_s, sample.score, sample.conf, sample.added_visible) == (
        None,
        None,
        0.0,
        False,
    )


@pytest.mark.unit
def test_assemble_result_rejects_bad_input() -> None:
    with pytest.raises(ScoreboardParseError, match="same length"):
        assemble_result("m.mp4", "opencv_gpu", 1.0, ROI, [0.0], [], PARAMS, 1.0)
    with pytest.raises(ScoreboardParseError, match="positive"):
        assemble_result("m.mp4", "opencv_gpu", 0.0, ROI, [], [], PARAMS, 1.0)


@pytest.mark.unit
def test_assemble_result_round_trips_through_json(result: ScoreboardResult) -> None:
    again = ScoreboardResult.model_validate_json(result.model_dump_json())
    assert again == result


@pytest.mark.unit
@pytest.mark.parametrize("fps", [float("nan"), float("inf"), -1.0])
def test_assemble_result_rejects_non_finite_sample_rate(fps: float) -> None:
    with pytest.raises(ScoreboardParseError, match="positive and finite"):
        assemble_result("m.mp4", "opencv_gpu", fps, ROI, [], [], PARAMS, 1.0)
