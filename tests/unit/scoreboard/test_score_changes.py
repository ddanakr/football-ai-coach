"""Score changes: debounce, VAR reversals, gaps in visibility, garbage readings."""

import pytest

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.score import detect_score_changes, merge_visible_segments

Score = tuple[int, int] | None


def _series(*parts: tuple[int, Score], start: float = 0.0) -> tuple[list[float], list[Score]]:
    """Build 1 Hz readings from (count, score) parts."""
    t: list[float] = []
    s: list[Score] = []
    now = start
    for count, score in parts:
        for _ in range(count):
            t.append(now)
            s.append(score)
            now += 1.0
    return t, s


@pytest.mark.unit
def test_detect_score_changes_empty_input_returns_nothing() -> None:
    assert detect_score_changes([], [], hold_s=3.0, sample_dt=1.0) == []


@pytest.mark.unit
def test_detect_score_changes_unreadable_only_returns_nothing() -> None:
    t, s = _series((10, None))
    assert detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0) == []


@pytest.mark.unit
def test_detect_score_changes_stable_score_returns_nothing() -> None:
    t, s = _series((20, (1, 0)))
    assert detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0) == []


@pytest.mark.unit
def test_detect_score_changes_goal_is_recorded_as_a_window() -> None:
    t, s = _series((10, (0, 0)), (10, (1, 0)), start=100.0)
    [change] = detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0)
    assert change.t_new_first_seen == 110.0
    assert change.t_old_last_seen == 109.0
    assert (change.old, change.new, change.direction) == ((0, 0), (1, 0), "up")


@pytest.mark.unit
def test_detect_score_changes_short_flicker_is_ignored() -> None:
    t, s = _series((10, (0, 0)), (2, (7, 0)), (10, (0, 0)))
    assert detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0) == []


@pytest.mark.unit
@pytest.mark.parametrize(("count", "expected"), [(2, 0), (3, 1), (4, 1)])
def test_detect_score_changes_hold_boundary_is_inclusive(count: int, expected: int) -> None:
    t, s = _series((10, (0, 0)), (count, (1, 0)))
    assert len(detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0)) == expected


@pytest.mark.unit
def test_detect_score_changes_var_reversal_gives_up_then_down() -> None:
    t, s = _series((10, (0, 0)), (20, (1, 0)), (10, (0, 0)))
    changes = detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0)
    assert [c.direction for c in changes] == ["up", "down"]
    assert changes[1].old == (1, 0)
    assert changes[1].new == (0, 0)


@pytest.mark.unit
def test_detect_score_changes_two_quick_goals_give_two_changes() -> None:
    t, s = _series((10, (0, 0)), (5, (1, 0)), (10, (1, 1)))
    changes = detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0)
    assert [c.new for c in changes] == [(1, 0), (1, 1)]


@pytest.mark.unit
def test_detect_score_changes_gap_in_visibility_does_not_break_the_run() -> None:
    t, s = _series((10, (1, 0)), (60, None), (10, (1, 0)))
    assert detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0) == []


@pytest.mark.unit
def test_detect_score_changes_board_hidden_across_the_change_gives_a_wide_window() -> None:
    t, s = _series((10, (0, 0)), (60, None), (10, (1, 0)))
    [change] = detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0)
    assert change.t_old_last_seen == 9.0
    assert change.t_new_first_seen == 70.0


@pytest.mark.unit
def test_detect_score_changes_board_vanishes_in_the_middle_of_confirmation() -> None:
    # The new score is seen for 2 s, the board is hidden for 30 s, then it is seen again:
    # 3 visible seconds in a row in the sequence of readings confirm it.
    t, s = _series((10, (0, 0)), (2, (1, 0)), (30, None), (1, (1, 0)))
    [change] = detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0)
    assert (change.t_old_last_seen, change.t_new_first_seen) == (9.0, 10.0)


@pytest.mark.unit
def test_detect_score_changes_change_right_at_a_segment_boundary() -> None:
    t, s = _series((5, None), (10, (0, 0)), (20, None), (10, (1, 0)))
    [change] = detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0)
    assert (change.t_old_last_seen, change.t_new_first_seen) == (14.0, 35.0)


@pytest.mark.unit
def test_detect_score_changes_hidden_start_is_not_a_change() -> None:
    t, s = _series((40, None), (10, (2, 1)))
    assert detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0) == []


@pytest.mark.unit
def test_detect_score_changes_single_false_reading_after_a_gap_is_ignored() -> None:
    t, s = _series((10, (0, 0)), (30, None), (1, (8, 0)), (30, None), (10, (0, 0)))
    assert detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0) == []


@pytest.mark.unit
def test_detect_score_changes_hold_is_counted_in_visible_seconds_not_wall_time() -> None:
    # Two readings 100 s apart cover a long wall time but are only two visible seconds.
    t = [0.0, 1.0, 2.0, 100.0, 102.0]
    s: list[Score] = [(0, 0), (0, 0), (0, 0), (1, 0), (1, 0)]
    assert detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0) == []


@pytest.mark.unit
def test_detect_score_changes_flicker_inside_new_value_does_not_split_it() -> None:
    t, s = _series((10, (0, 0)), (5, (1, 0)), (1, (1, 8)), (5, (1, 0)))
    changes = detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0)
    assert [c.new for c in changes] == [(1, 0)]


@pytest.mark.unit
def test_detect_score_changes_unconfirmed_value_at_the_end_is_dropped() -> None:
    t, s = _series((10, (0, 0)), (2, (1, 0)))
    assert detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0) == []


@pytest.mark.unit
def test_detect_score_changes_swap_of_both_numbers_is_other() -> None:
    t, s = _series((10, (1, 0)), (10, (0, 1)))
    [change] = detect_score_changes(t, s, hold_s=3.0, sample_dt=1.0)
    assert change.direction == "other"


@pytest.mark.unit
def test_detect_score_changes_length_mismatch_raises() -> None:
    with pytest.raises(ScoreboardParseError, match="same length"):
        detect_score_changes([0.0], [], hold_s=3.0, sample_dt=1.0)


@pytest.mark.unit
def test_detect_score_changes_non_positive_hold_raises() -> None:
    with pytest.raises(ScoreboardParseError, match="positive"):
        detect_score_changes([0.0], [(0, 0)], hold_s=0.0, sample_dt=1.0)


@pytest.mark.unit
def test_merge_visible_segments_bridges_short_gaps_only() -> None:
    t = [float(i) for i in range(30)]
    visible = [True] * 5 + [False] * 2 + [True] * 5 + [False] * 10 + [True] * 8
    segments = merge_visible_segments(t, visible, gap_s=2.0, sample_dt=1.0)
    assert [(s.start, s.end) for s in segments] == [(0.0, 11.0), (22.0, 29.0)]


@pytest.mark.unit
def test_merge_visible_segments_empty_and_never_visible_return_nothing() -> None:
    assert merge_visible_segments([], [], gap_s=2.0, sample_dt=1.0) == []
    assert merge_visible_segments([0.0, 1.0], [False, False], gap_s=2.0, sample_dt=1.0) == []


@pytest.mark.unit
def test_merge_visible_segments_single_sample_is_a_zero_length_segment() -> None:
    [segment] = merge_visible_segments([5.0], [True], gap_s=2.0, sample_dt=1.0)
    assert (segment.start, segment.end) == (5.0, 5.0)


@pytest.mark.unit
def test_merge_visible_segments_length_mismatch_raises() -> None:
    with pytest.raises(ScoreboardParseError, match="same length"):
        merge_visible_segments([0.0], [], gap_s=2.0, sample_dt=1.0)


@pytest.mark.unit
@pytest.mark.parametrize(("hold", "dt"), [(-1.0, 1.0), (3.0, 0.0), (3.0, -1.0)])
def test_detect_score_changes_non_positive_parameters_raise(hold: float, dt: float) -> None:
    with pytest.raises(ScoreboardParseError, match="positive"):
        detect_score_changes([0.0], [(0, 0)], hold_s=hold, sample_dt=dt)
