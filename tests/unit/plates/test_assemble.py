"""Building plates.json from tracks and scoreboard.json (pure logic)."""

import pytest

from aicoach.errors import PlatesError
from aicoach.plates.assemble import assemble_result, half_of, visible_fraction
from aicoach.plates.config import PlatesConfig
from aicoach.plates.layout import PlateLayout
from aicoach.plates.schemas import PlatesResult
from aicoach.plates.tracks import Track
from aicoach.scoreboard.schemas import OffsetSegment
from tests.fixtures.plates_synth import change, scoreboard_result, series


def segment(start: float, end: float) -> OffsetSegment:
    return OffsetSegment(
        start=start, end=end, clock_start_s=0, clock_end_s=1, offset_s=0.0, n_readings=2
    )


HALVES = [segment(792, 3546), segment(4600, 7474)]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("t_v", "expected"),
    [
        (1000.0, 1),
        (3546.0, 1),
        (4000.0, 1),
        (4073.0, 2),  # the middle of the break counts for the second half
        (4100.0, 2),
        (7474.0, 2),
        (500.0, 1),  # within the margin before the first clock
        (400.0, None),  # pre-match
        (7700.0, 2),
        (7800.0, None),  # after the match
    ],
)
def test_half_of_uses_the_widest_gap_between_clock_segments(
    t_v: float, expected: int | None
) -> None:
    assert half_of(t_v, HALVES, 300.0) == expected


@pytest.mark.unit
def test_half_of_fewer_than_two_segments_gives_none() -> None:
    assert half_of(1000.0, [], 300.0) is None
    assert half_of(1000.0, [segment(0, 2000)], 300.0) is None


@pytest.mark.unit
def test_half_of_with_three_segments_breaks_at_the_widest_gap() -> None:
    segments = [segment(0, 1000), segment(1100, 2000), segment(3000, 4000)]

    assert half_of(1500.0, segments, 0.0) == 1
    assert half_of(3500.0, segments, 0.0) == 2


@pytest.mark.unit
def test_half_of_unsorted_segments_are_sorted_first() -> None:
    assert half_of(1000.0, list(reversed(HALVES)), 300.0) == 1


@pytest.mark.unit
def test_visible_fraction_counts_the_visible_samples() -> None:
    times = [float(k) for k in range(10)]
    visible = [k in {2, 3, 4, 5} for k in range(10)]

    assert visible_fraction([2.0, 3.0, 4.5, 8.0], times, visible) == pytest.approx(0.75)


@pytest.mark.unit
def test_visible_fraction_nothing_to_look_up_gives_none() -> None:
    assert visible_fraction([], [0.0, 1.0], [True, True]) is None
    assert visible_fraction([1.0], [], []) is None


@pytest.mark.unit
def test_visible_fraction_times_outside_the_samples_are_skipped() -> None:
    times = [10.0, 11.0, 12.0]
    visible = [True, False, True]

    assert visible_fraction([5.0, 11.0, 50.0], times, visible) == 0.0
    assert visible_fraction([5.0, 50.0], times, visible) is None


@pytest.mark.unit
def test_visible_fraction_single_board_sample() -> None:
    assert visible_fraction([3.0, 3.5], [3.0], [True]) == 1.0


def _result(
    plates: list[Track],
    rejects: list[Track],
    labels: list[int],
    visible: set[int] | None = None,
) -> PlatesResult:
    changes = [
        change((0, 0), (1, 0), 2.0, 3.0),
        change((1, 0), (1, 1), 8.0, 9.0),
    ]
    board = scoreboard_result(
        changes, n_samples=20, segments=[(0.0, 9.0), (11.0, 19.0)], visible=visible
    )
    return assemble_result(
        video="match.mp4",
        decode_mode="opencv_gpu",
        sample_fps=2.0,
        roi=PlateLayout().plate,
        n_samples=40,
        plates=plates,
        rejects=rejects,
        logo_labels=labels,
        board=board,
        scoreboard_version="v1",
        config=PlatesConfig(),
    )


def _plates() -> list[Track]:
    return [
        Track("goal", None, series(5.0, 7.0)),
        Track("card", None, series(10.0, 12.0, kind="card", color="yellow")),
        Track("goal", None, series(14.0, 16.0)),
    ]


@pytest.mark.unit
def test_assemble_result_ties_goal_plates_to_score_changes_and_teams() -> None:
    result = _result(_plates(), [], [0, 1, 1])

    assert [p.score_change for p in result.plates] == [0, None, 1]
    assert [p.team for p in result.plates] == ["home", "away", "away"]
    assert [(c.id, c.team, c.n_plates) for c in result.logo_clusters] == [
        (0, "home", 1),
        (1, "away", 2),
    ]
    assert (result.logo_clusters[0].home_votes, result.logo_clusters[1].away_votes) == (1, 1)
    assert result.stats.warnings == []


@pytest.mark.unit
def test_assemble_result_fills_plate_fields() -> None:
    result = _result(_plates(), [], [0, 1, 1])

    first, card, _ = result.plates
    assert (first.id, first.type, first.t_first_seen, first.t_last_seen) == (
        "plate_001",
        "goal",
        5.0,
        7.0,
    )
    assert card.card_color == "yellow"
    assert card.logo_cluster == 1
    assert first.n_frames == 5
    assert first.conf == 1.0
    assert first.crop == "plate_crops/plate_001.png"
    assert first.logo_crop == "plate_crops/plate_001_logo.png"
    assert [p.half for p in result.plates] == [1, 2, 2]
    assert first.scoreboard_visible_frac == 1.0
    assert result.scoreboard_version == "v1"
    assert result.stats.n_by_type == {"subst": 0, "goal": 2, "card": 1}


@pytest.mark.unit
def test_assemble_result_scoreboard_hidden_during_a_plate_gives_zero_fraction() -> None:
    result = _result(_plates(), [], [0, 1, 1], visible={0, 1, 2, 3, 4})

    assert [p.scoreboard_visible_frac for p in result.plates] == [0.0, 0.0, 0.0]


@pytest.mark.unit
def test_assemble_result_counts_low_confidence_plates() -> None:
    plates = _plates()
    for hit in plates[1].hits:
        object.__setattr__(hit.reading, "conf", 0.5)

    result = _result(plates, [], [0, 1, 1])

    assert result.stats.n_low_conf == 1
    assert result.plates[1].conf == 0.5


@pytest.mark.unit
def test_assemble_result_goal_without_a_score_change_has_no_team() -> None:
    plates = [
        Track("goal", None, series(300.0, 302.0)),
        Track("card", None, series(310.0, 312.0, kind="card", color="yellow")),
    ]

    result = _result(plates, [], [0, 1])

    assert result.plates[0].score_change is None
    assert [p.team for p in result.plates] == [None, None]
    assert any("no goal evidence" in w for w in result.stats.warnings)


@pytest.mark.unit
def test_assemble_result_one_logo_cluster_leaves_teams_empty_and_warns() -> None:
    result = _result(_plates(), [], [0, 0, 0])

    assert [p.team for p in result.plates] == [None, None, None]
    assert "expected 2 logo clusters" in result.stats.warnings[0]


@pytest.mark.unit
def test_assemble_result_no_plates_gives_an_empty_valid_result() -> None:
    result = _result([], [], [])

    assert result.plates == []
    assert result.logo_clusters == []
    assert result.stats.n_by_type == {"subst": 0, "goal": 0, "card": 0}
    assert result.stats.n_candidate_frames == 0
    assert result.stats.warnings == []


@pytest.mark.unit
def test_assemble_result_refused_candidate_passed_as_a_plate_raises() -> None:
    refused = Track(None, "no_marker", series(5.0, 6.0, kind=None, reject="no_marker"))

    with pytest.raises(PlatesError, match="must have a type"):
        _result([refused], [], [0])


@pytest.mark.unit
def test_assemble_result_plate_passed_as_a_refused_candidate_raises() -> None:
    plate = Track("goal", None, series(5.0, 6.0))

    with pytest.raises(PlatesError, match="must have a reason"):
        _result([], [plate], [])


@pytest.mark.unit
def test_assemble_result_lists_rejected_candidates_with_crops() -> None:
    rejects = [
        Track(None, "no_marker", series(20.0, 22.0, kind=None, reject="no_marker")),
        Track(None, "no_panel", series(30.0, 31.0, kind=None, reject="no_panel")),
    ]

    result = _result(_plates(), rejects, [0, 1, 1])

    assert [(r.id, r.reason, r.n_frames) for r in result.rejected] == [
        ("rejected_001", "no_marker", 5),
        ("rejected_002", "no_panel", 3),
    ]
    assert result.rejected[0].crop == "plate_crops/rejected_001.png"
    assert result.stats.n_rejected == 2
    assert result.stats.n_candidate_frames == 15 + 8


@pytest.mark.unit
def test_assemble_result_counts_overlapping_plates_of_different_types() -> None:
    plates = [
        Track("goal", None, series(5.0, 8.0)),
        Track("card", None, series(7.0, 9.0, kind="card", color="yellow")),
    ]

    result = _result(plates, [], [0, 1])

    assert result.stats.n_overlaps == 1
