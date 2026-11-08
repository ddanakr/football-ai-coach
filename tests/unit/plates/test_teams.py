"""Team of a plate from score changes (pure logic)."""

import pytest

from aicoach.errors import PlatesError
from aicoach.plates.schemas import Team
from aicoach.plates.teams import decide_teams, goal_side, match_score_change
from tests.fixtures.plates_synth import change


@pytest.mark.unit
@pytest.mark.parametrize(
    ("old", "new", "side"),
    [
        ((0, 0), (1, 0), "home"),
        ((1, 1), (2, 1), "home"),
        ((1, 0), (1, 1), "away"),
        ((2, 1), (2, 2), "away"),
        ((1, 0), (0, 0), None),  # a goal taken back
        ((0, 0), (1, 1), None),  # both changed: a misread, not one goal
        ((0, 0), (2, 0), None),  # a jump of two
        ((1, 0), (1, 0), None),
    ],
)
def test_goal_side_by_which_number_grew(
    old: tuple[int, int], new: tuple[int, int], side: Team | None
) -> None:
    assert goal_side(change(old, new, 10.0, 11.0)) == side


@pytest.mark.unit
def test_match_score_change_plate_after_a_change_matches_it() -> None:
    changes = [change((0, 0), (1, 0), 100.0, 101.0)]

    assert match_score_change(130.0, changes, 120.0) == 0


@pytest.mark.unit
def test_match_score_change_no_changes_gives_none() -> None:
    assert match_score_change(130.0, [], 120.0) is None


@pytest.mark.unit
def test_match_score_change_change_too_long_ago_gives_none() -> None:
    changes = [change((0, 0), (1, 0), 100.0, 101.0)]

    assert match_score_change(222.0, changes, 120.0) is None
    assert match_score_change(221.0, changes, 120.0) == 0  # exactly at the limit counts


@pytest.mark.unit
def test_match_score_change_change_after_the_plate_does_not_match() -> None:
    changes = [change((0, 0), (1, 0), 200.0, 201.0)]

    assert match_score_change(150.0, changes, 120.0) is None


@pytest.mark.unit
def test_match_score_change_two_changes_in_a_row_takes_the_nearest_before() -> None:
    changes = [change((0, 0), (1, 0), 100.0, 101.0), change((1, 0), (1, 1), 150.0, 151.0)]

    assert match_score_change(170.0, changes, 120.0) == 1
    assert match_score_change(130.0, changes, 120.0) == 0


@pytest.mark.unit
def test_match_score_change_plate_inside_a_hidden_window_matches_it() -> None:
    """The scoreboard was hidden: the new score is seen only after the plate appeared."""
    changes = [change((0, 0), (1, 0), 100.0, 160.0)]

    assert match_score_change(130.0, changes, 120.0) == 0


@pytest.mark.unit
def test_decide_teams_two_clusters_with_evidence_for_each() -> None:
    decision = decide_teams(2, [(0, "home"), (0, "home"), (1, "away")])

    assert decision.teams == {0: "home", 1: "away"}
    assert decision.home_votes == {0: 2, 1: 0}
    assert decision.away_votes == {0: 0, 1: 1}
    assert decision.warnings == ()


@pytest.mark.unit
def test_decide_teams_tied_votes_leave_the_cluster_undecided() -> None:
    decision = decide_teams(2, [(0, "home"), (0, "away"), (1, "away")])

    assert decision.teams == {1: "away"}
    assert any("cluster(s) [0]" in w for w in decision.warnings)


@pytest.mark.unit
def test_decide_teams_no_evidence_decides_nothing_and_warns() -> None:
    decision = decide_teams(2, [])

    assert decision.teams == {}
    assert decision.warnings


@pytest.mark.unit
@pytest.mark.parametrize("n_clusters", [1, 3])
def test_decide_teams_not_two_clusters_decides_nothing(n_clusters: int) -> None:
    decision = decide_teams(n_clusters, [(0, "home")])

    assert decision.teams == {}
    assert "expected 2 logo clusters" in decision.warnings[0]


@pytest.mark.unit
def test_decide_teams_no_clusters_is_not_a_problem() -> None:
    decision = decide_teams(0, [])

    assert decision.teams == {}
    assert decision.warnings == ()


@pytest.mark.unit
@pytest.mark.parametrize("cluster", [-1, 2, 7])
def test_decide_teams_evidence_for_an_unknown_cluster_raises(cluster: int) -> None:
    with pytest.raises(PlatesError, match="logo cluster"):
        decide_teams(2, [(cluster, "home")])


@pytest.mark.unit
def test_match_score_change_zero_lag_allows_only_a_plate_inside_the_window() -> None:
    changes = [change((0, 0), (1, 0), 100.0, 110.0)]

    assert match_score_change(105.0, changes, 0.0) == 0
    assert match_score_change(111.0, changes, 0.0) is None


@pytest.mark.unit
def test_decide_teams_contradicting_evidence_decides_nothing() -> None:
    decision = decide_teams(2, [(0, "home"), (1, "home")])

    assert decision.teams == {}
    assert any("same side" in w for w in decision.warnings)
