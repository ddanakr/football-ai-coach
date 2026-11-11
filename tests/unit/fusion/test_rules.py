"""Fusion rules on synthetic inputs: goals, substitutions, model events, card fouls, halves."""

import pytest

from aicoach.domain.events import EventType
from aicoach.eval.schemas import Prediction
from aicoach.fusion.config import FuseConfig
from aicoach.fusion.rules import NO_MODEL_WARNING, FusionOutput, count_events, fuse_signals
from aicoach.fusion.schemas import Candidate
from aicoach.plates.schemas import Plate
from aicoach.scoreboard.schemas import ScoreChange
from tests.fixtures.factories import make_pred
from tests.fixtures.fusion_synth import HALF_SPANS, plate, plates_result
from tests.fixtures.plates_synth import change, scoreboard_result

CONFIG = FuseConfig()
GOAL_UP = change((0, 0), (1, 0), 199.0, 200.0)
GOAL_TAKEN_BACK = change((1, 0), (0, 0), 259.0, 260.0)


def run(
    changes: list[ScoreChange] | None = None,
    plates: list[Plate] | None = None,
    model: list[Prediction] | None = None,
    config: FuseConfig = CONFIG,
) -> FusionOutput:
    """Fuse with a two-half scoreboard; `model=None` is the "no model" mode."""
    board = scoreboard_result(changes, segments=HALF_SPANS)
    return fuse_signals(board, plates_result(plates or []), model, config)


def of_type(out: FusionOutput, kind: EventType) -> list[Candidate]:
    """The candidates of one type."""
    return [c for c in out.candidates if c.type is kind]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("old", "new", "team"), [((0, 0), (1, 0), "home"), ((1, 0), (1, 1), "away")]
)
def test_goal_score_growth_gives_goal_for_the_side_that_grew(
    old: tuple[int, int], new: tuple[int, int], team: str
) -> None:
    out = run([change(old, new, 199.0, 200.0)], model=[])

    (goal,) = of_type(out, EventType.GOAL)
    assert (goal.t_v, goal.team, goal.score, goal.half) == (200.0, team, 0.8, 1)
    assert goal.source == "scoreboard"
    assert goal.evidence[0].ref == "score_changes[0]"


@pytest.mark.unit
def test_goal_with_plate_in_window_is_confirmed_and_plate_is_not_added_again() -> None:
    out = run([GOAL_UP], [plate("p1", "goal", 230.0, team="home")], model=[])

    (goal,) = of_type(out, EventType.GOAL)
    assert (goal.score, goal.source) == (1.0, "scoreboard+plates")
    assert [e.ref for e in goal.evidence] == ["score_changes[0]", "p1"]
    assert out.warnings == []


@pytest.mark.unit
@pytest.mark.parametrize("plate_t", [321.0, 150.0])
def test_goal_plate_outside_the_window_does_not_confirm(plate_t: float) -> None:
    out = run([GOAL_UP], [plate("p1", "goal", plate_t)], model=[])

    goals = of_type(out, EventType.GOAL)
    assert sorted((g.source, g.score) for g in goals) == [("plates_only", 0.5), ("scoreboard", 0.8)]


@pytest.mark.unit
def test_goal_plate_at_window_edge_120_s_after_confirms() -> None:
    out = run([GOAL_UP], [plate("p1", "goal", 320.0)], model=[])

    (goal,) = of_type(out, EventType.GOAL)
    assert goal.score == 1.0


@pytest.mark.unit
def test_goal_plate_without_score_change_gives_lagged_goal_with_plate_team() -> None:
    out = run(plates=[plate("p1", "goal", 500.0, team="away")], model=[])

    (goal,) = of_type(out, EventType.GOAL)
    assert (goal.t_v, goal.score, goal.team, goal.source) == (470.0, 0.5, "away", "plates_only")
    assert any("p1" in w and "no score change" in w for w in out.warnings)


@pytest.mark.unit
def test_goal_plate_near_the_start_of_the_video_gives_time_not_below_zero() -> None:
    board = scoreboard_result(segments=[(0.0, 1000.0), (1500.0, 2500.0)])

    out = fuse_signals(board, plates_result([plate("p1", "goal", 10.0)]), [], CONFIG)

    assert out.candidates[0].t_v == 0.0


@pytest.mark.unit
def test_score_grew_and_came_back_gives_disallowed_shot_and_no_goal() -> None:
    out = run([GOAL_UP, GOAL_TAKEN_BACK], [plate("p1", "goal", 230.0)], model=[])

    assert of_type(out, EventType.GOAL) == []
    (shot,) = of_type(out, EventType.SHOT)
    assert (shot.status, shot.t_v, shot.team, shot.rule) == (
        "disallowed",
        200.0,
        "home",
        "shot_disallowed",
    )


@pytest.mark.unit
def test_score_fell_without_matching_growth_is_ignored_with_a_warning() -> None:
    out = run([change((2, 0), (1, 0), 199.0, 200.0)], model=[])

    assert out.candidates == []
    assert any("without a matching growth" in w for w in out.warnings)


@pytest.mark.unit
def test_score_changed_in_an_unexpected_way_is_ignored_with_a_warning() -> None:
    out = run([change((0, 0), (3, 0), 199.0, 200.0)], model=[])

    assert out.candidates == []
    assert any("ignored" in w for w in out.warnings)


@pytest.mark.unit
def test_score_growth_by_two_gives_goal_without_team_and_a_warning() -> None:
    jump = ScoreChange(
        t_old_last_seen=199.0, t_new_first_seen=200.0, old=(0, 0), new=(2, 0), direction="up"
    )

    out = run([jump], model=[])

    (goal,) = of_type(out, EventType.GOAL)
    assert goal.team is None
    assert any("team is null" in w for w in out.warnings)


@pytest.mark.unit
def test_two_goals_close_together_get_one_plate_each() -> None:
    changes = [GOAL_UP, change((1, 0), (2, 0), 249.0, 250.0)]
    plates = [plate("p1", "goal", 230.0), plate("p2", "goal", 280.0)]

    out = run(changes, plates, model=[])

    goals = of_type(out, EventType.GOAL)
    assert [(g.t_v, g.score, g.evidence[1].ref) for g in goals] == [
        (200.0, 1.0, "p1"),
        (250.0, 1.0, "p2"),
    ]


@pytest.mark.unit
def test_second_goal_without_its_own_plate_is_not_confirmed_by_the_first_plate() -> None:
    changes = [GOAL_UP, change((1, 0), (2, 0), 249.0, 250.0)]

    out = run(changes, [plate("p1", "goal", 230.0)], model=[])

    assert [g.score for g in of_type(out, EventType.GOAL)] == [1.0, 0.8]


@pytest.mark.unit
def test_plate_team_different_from_scoring_side_warns_and_keeps_scoreboard_team() -> None:
    out = run([GOAL_UP], [plate("p1", "goal", 230.0, team="away")], model=[])

    (goal,) = of_type(out, EventType.GOAL)
    assert goal.team == "home"
    assert any("own goal" in w for w in out.warnings)


@pytest.mark.unit
def test_substitution_plate_gives_subst_at_plate_time_with_plate_confidence() -> None:
    out = run(plates=[plate("p1", "subst", 600.0, team="away", conf=0.93)], model=[])

    (sub,) = of_type(out, EventType.SUBST)
    assert (sub.t_v, sub.team, sub.score, sub.source, sub.half) == (
        600.0,
        "away",
        0.93,
        "plates",
        1,
    )


@pytest.mark.unit
def test_model_goal_and_subst_are_ignored() -> None:
    model = [make_pred("goal", 300.0, score=0.9), make_pred("subst", 400.0, score=0.9)]

    assert run(model=model).candidates == []


@pytest.mark.unit
def test_model_events_below_their_class_threshold_are_dropped() -> None:
    config = FuseConfig(model_thresholds={"shot": 0.5, "corner": 0.2, "foul": 0.2})
    model = [
        make_pred("shot", 300.0, score=0.49),
        make_pred("shot", 400.0, score=0.5),
        make_pred("corner", 500.0, score=0.2),
    ]

    out = run(model=model, config=config)

    assert [(c.type, c.t_v) for c in out.candidates] == [
        (EventType.SHOT, 400.0),
        (EventType.CORNER, 500.0),
    ]


@pytest.mark.unit
def test_default_config_keeps_a_weak_corner_but_not_a_weak_shot_or_foul() -> None:
    model = [
        make_pred("corner", 300.0, score=0.05),
        make_pred("shot", 400.0, score=0.1),
        make_pred("foul", 500.0, score=0.1),
    ]

    out = run(model=model)

    assert [(c.type, c.t_v) for c in out.candidates] == [(EventType.CORNER, 300.0)]


@pytest.mark.unit
def test_model_event_without_score_is_dropped() -> None:
    assert run(model=[make_pred("shot", 300.0)]).candidates == []


@pytest.mark.unit
def test_model_duplicates_of_one_class_within_merge_radius_collapse_to_the_strongest() -> None:
    model = [
        make_pred("shot", 300.0, score=0.4),
        make_pred("shot", 302.0, score=0.7),
        make_pred("shot", 306.0, score=0.3),
    ]

    out = run(model=model)

    assert [(c.t_v, c.score) for c in out.candidates] == [(302.0, 0.7), (306.0, 0.3)]


@pytest.mark.unit
@pytest.mark.parametrize(("shot_t", "kept"), [(205.0, False), (210.0, False), (210.5, True)])
def test_model_shot_near_a_goal_is_dropped_within_radius(shot_t: float, kept: bool) -> None:
    out = run([GOAL_UP], model=[make_pred("shot", shot_t, score=0.9)])

    assert bool(of_type(out, EventType.SHOT)) is kept
    assert [d.reason for d in out.dropped] == ([] if kept else ["shot_near_goal"])


@pytest.mark.unit
def test_model_corner_near_a_goal_is_kept() -> None:
    out = run([GOAL_UP], model=[make_pred("corner", 203.0, score=0.9)])

    assert len(of_type(out, EventType.CORNER)) == 1


@pytest.mark.unit
def test_model_shot_near_a_disallowed_goal_is_dropped_but_the_disallowed_shot_stays() -> None:
    out = run([GOAL_UP, GOAL_TAKEN_BACK], model=[make_pred("shot", 204.0, score=0.9)])

    (shot,) = of_type(out, EventType.SHOT)
    assert shot.status == "disallowed"


@pytest.mark.unit
@pytest.mark.parametrize(("model_score", "expected"), [(0.3, 0.9), (0.95, 0.95)])
def test_card_plate_lifts_a_model_foul_before_it_and_never_lowers_it(
    model_score: float, expected: float
) -> None:
    out = run(
        plates=[plate("p1", "card", 600.0, team="away")],
        model=[make_pred("foul", 590.0, score=model_score)],
    )

    (foul,) = of_type(out, EventType.FOUL)
    assert (foul.t_v, foul.score, foul.team) == (590.0, expected, "away")
    assert (foul.source, foul.rule) == ("model+plates", "foul_plate_boost")
    assert [e.kind for e in foul.evidence] == ["model", "plate"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("foul_t", "lifted"), [(560.0, True), (559.9, False), (605.0, True), (605.1, False)]
)
def test_card_foul_window_edges_are_40_s_before_and_5_s_after(foul_t: float, lifted: bool) -> None:
    out = run(plates=[plate("p1", "card", 600.0)], model=[make_pred("foul", foul_t, score=0.3)])

    sources = sorted(c.source for c in of_type(out, EventType.FOUL))
    assert ("model+plates" in sources) is lifted
    assert len(sources) == (1 if lifted else 2)


@pytest.mark.unit
def test_card_plate_without_a_model_gives_a_foul_lead_s_before_the_plate() -> None:
    config = FuseConfig(card_foul_lead_s=15.0)

    out = run(plates=[plate("p1", "card", 600.0, team="home")], model=None, config=config)

    (foul,) = of_type(out, EventType.FOUL)
    assert (foul.t_v, foul.score, foul.team, foul.source) == (
        600.0 - config.card_foul_lead_s,
        0.5,
        "home",
        "plates_only",
    )
    assert NO_MODEL_WARNING in out.warnings


@pytest.mark.unit
def test_card_plate_with_a_model_but_no_foul_nearby_adds_a_foul_too() -> None:
    out = run(plates=[plate("p1", "card", 600.0)], model=[make_pred("foul", 100.0, score=0.9)])

    assert sorted(c.source for c in of_type(out, EventType.FOUL)) == ["model", "plates_only"]


@pytest.mark.unit
def test_one_model_foul_serves_one_card_plate() -> None:
    out = run(
        plates=[plate("p1", "card", 600.0), plate("p2", "card", 610.0)],
        model=[make_pred("foul", 595.0, score=0.3)],
    )

    assert sorted(c.source for c in of_type(out, EventType.FOUL)) == ["model+plates", "plates_only"]


@pytest.mark.unit
def test_card_plate_picks_the_foul_nearest_to_the_expected_moment() -> None:
    config = FuseConfig(card_foul_lead_s=20.0)
    expected = 600.0 - config.card_foul_lead_s
    near = make_pred("foul", expected + 5.0, score=0.3)
    far = make_pred("foul", expected - 20.0, score=0.9)  # stronger, but farther from `expected`

    out = run(plates=[plate("p1", "card", 600.0)], model=[far, near], config=config)

    lifted = [c for c in of_type(out, EventType.FOUL) if c.source == "model+plates"]
    assert [c.t_v for c in lifted] == [near.t_v]


@pytest.mark.unit
def test_event_in_the_break_is_dropped_with_a_warning_and_has_no_half() -> None:
    out = run(plates=[plate("p1", "subst", 1200.0)], model=[])

    assert out.candidates == []
    assert [(d.type, d.reason) for d in out.dropped] == [(EventType.SUBST, "outside_halves")]
    assert any("subst" in w and "outside the halves" in w for w in out.warnings)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("t", "half"),
    [
        (85.0, 1),
        (1015.0, 1),
        (1485.0, 2),
        (2515.0, 2),
        (84.9, None),
        (1015.1, None),
        (2515.1, None),
    ],
)
def test_half_follows_the_clock_segments_with_margins_and_ends_included(
    t: float, half: int | None
) -> None:
    out = run(plates=[plate("p1", "subst", t)], model=[])

    assert [c.half for c in out.candidates] == ([] if half is None else [half])


@pytest.mark.unit
def test_empty_inputs_give_no_events_and_only_the_no_model_warning() -> None:
    out = run(model=None)

    assert (out.candidates, out.dropped, out.warnings) == ([], [], [NO_MODEL_WARNING])


@pytest.mark.unit
def test_empty_model_list_is_not_the_no_model_mode() -> None:
    assert run(model=[]).warnings == []


@pytest.mark.unit
def test_candidates_are_sorted_by_time_and_counted_by_type_and_half() -> None:
    plates = [plate("p2", "subst", 1600.0), plate("p1", "subst", 600.0)]

    out = run([GOAL_UP], plates, model=[make_pred("corner", 150.0, score=0.5)])

    assert [c.t_v for c in out.candidates] == sorted(c.t_v for c in out.candidates)
    assert count_events(out.candidates) == {
        "goal": {"1": 1},
        "corner": {"1": 1},
        "subst": {"1": 1, "2": 1},
    }


@pytest.mark.unit
def test_fuse_is_deterministic() -> None:
    plates = [plate("p1", "card", 600.0), plate("p2", "goal", 700.0)]
    model = [make_pred("foul", 590.0, score=0.3), make_pred("shot", 300.0, score=0.4)]

    assert run(plates=plates, model=model) == run(plates=plates, model=model)
