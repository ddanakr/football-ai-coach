"""The review layer: which detections become candidates (pure functions)."""

import pytest

from aicoach.domain.events import EventType
from aicoach.errors import SpottingError
from aicoach.eval.schemas import Prediction
from aicoach.fusion.schemas import Half, HalfSpan
from aicoach.review.candidates import (
    LAYER_TYPES,
    LayerBuild,
    build_review_candidates,
    count_candidates,
)
from aicoach.review.schemas import ReviewCandidate
from aicoach.scoreboard.schemas import OffsetSegment

BOUNDS: dict[Half, HalfSpan] = {
    1: HalfSpan(start=0.0, end=1000.0),
    2: HalfSpan(start=1500.0, end=2500.0),
}
SEGMENTS = [
    OffsetSegment(
        start=100.0, end=1000.0, clock_start_s=0, clock_end_s=900, offset_s=100.0, n_readings=10
    )
]
RADIUS = 3.0


def det(kind: EventType, t_v: float, score: float | None = 0.5) -> Prediction:
    return Prediction(type=kind, t_v=t_v, score=score)


def build(
    model: list[Prediction],
    existing: list[tuple[EventType, float]] | None = None,
    radius: float = RADIUS,
) -> LayerBuild:
    return build_review_candidates(
        model,
        existing or [],
        BOUNDS,
        SEGMENTS,
        merge_radius_s=radius,
        max_extrapolation_s=30.0,
    )


def times(layer: LayerBuild) -> list[tuple[str, float]]:
    return [(c.type.value, c.t_v) for c in layer.candidates]


@pytest.mark.unit
def test_every_detection_is_a_candidate_when_the_catalog_is_empty() -> None:
    model = [det(EventType.SHOT, 300.0), det(EventType.SHOT, 200.0), det(EventType.SHOT, 250.0)]

    layer = build(model)

    assert times(layer) == [("shot", 200.0), ("shot", 250.0), ("shot", 300.0)]
    assert [c.id for c in layer.candidates] == ["r:1", "r:2", "r:3"]
    assert (layer.n_model, layer.n_existing, layer.n_merged, layer.n_outside_halves) == (3, 0, 0, 0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "kind", [EventType.GOAL, EventType.SUBST, EventType.FOUL, EventType.CORNER]
)
def test_only_shots_get_candidates(kind: EventType) -> None:
    layer = build([det(kind, 300.0, 0.9), det(EventType.SHOT, 500.0)])

    assert times(layer) == [("shot", 500.0)]
    assert layer.n_model == 1  # the other class is not counted at all
    assert set(LAYER_TYPES) == {EventType.SHOT}


@pytest.mark.unit
def test_a_model_of_other_classes_only_gives_an_empty_layer() -> None:
    assert build([det(EventType.GOAL, 300.0), det(EventType.FOUL, 400.0)]) == LayerBuild(
        [], 0, 0, 0, 0
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("distance", "left_out"), [(0.0, True), (2.9, True), (3.0, True), (3.2, False), (-3.0, True)]
)
def test_detection_in_the_merge_window_of_an_event_of_the_same_type_is_left_out(
    distance: float, left_out: bool
) -> None:
    layer = build([det(EventType.SHOT, 300.0 + distance)], [(EventType.SHOT, 300.0)])

    assert (layer.candidates == []) is left_out
    assert layer.n_existing == (1 if left_out else 0)


@pytest.mark.unit
def test_an_event_of_another_type_does_not_hide_the_shot() -> None:
    layer = build([det(EventType.SHOT, 300.0)], [(EventType.FOUL, 300.0), (EventType.GOAL, 301.0)])

    assert times(layer) == [("shot", 300.0)]


@pytest.mark.unit
def test_duplicates_of_one_type_collapse_to_the_strongest_like_in_fuse() -> None:
    model = [
        det(EventType.SHOT, 300.0, 0.08),
        det(EventType.SHOT, 301.0, 0.30),
        det(EventType.SHOT, 303.5, 0.06),  # 2.5 s from the kept 301.0: merged too
        det(EventType.SHOT, 310.0, 0.07),  # far from the others: stays
    ]

    layer = build(model)

    assert times(layer) == [("shot", 301.0), ("shot", 310.0)]
    assert layer.n_merged == 2


@pytest.mark.unit
def test_detections_in_the_break_are_dropped_and_counted() -> None:
    layer = build([det(EventType.SHOT, 1200.0), det(EventType.SHOT, 1600.0)])

    assert times(layer) == [("shot", 1600.0)]
    assert layer.candidates[0].half == 2
    assert layer.n_outside_halves == 1


@pytest.mark.unit
def test_candidate_carries_half_score_and_game_minute() -> None:
    layer = build([det(EventType.SHOT, 250.0, 0.04), det(EventType.SHOT, 20.0, 0.9)])

    first, second = layer.candidates
    # 250 s of video is 150 s of the clock: minute 3. At 20 s the clock is before 0: unknown.
    assert (first.t_v, first.half, first.minute, first.score) == (20.0, 1, None, 0.9)
    assert (second.t_v, second.half, second.minute, second.score) == (250.0, 1, 3, 0.04)


@pytest.mark.unit
def test_ids_follow_time_order_and_are_the_same_on_every_run() -> None:
    model = [det(EventType.SHOT, 900.0), det(EventType.SHOT, 100.0), det(EventType.SHOT, 500.0)]

    assert build(model) == build(list(reversed(model)))
    assert [c.id for c in build(model).candidates] == ["r:1", "r:2", "r:3"]


@pytest.mark.unit
def test_empty_model_gives_an_empty_layer() -> None:
    layer = build([], [(EventType.SHOT, 10.0)])

    assert layer == LayerBuild([], 0, 0, 0, 0)


@pytest.mark.unit
def test_a_prediction_without_a_score_gets_score_zero() -> None:
    layer = build([det(EventType.SHOT, 300.0, None)])

    assert layer.candidates[0].score == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("radius", [-1.0, float("nan"), float("inf")])
def test_a_bad_radius_is_refused(radius: float) -> None:
    with pytest.raises(SpottingError):
        build([det(EventType.SHOT, 300.0)], radius=radius)


@pytest.mark.unit
def test_count_candidates_groups_by_type_and_half() -> None:
    candidates = [
        ReviewCandidate(id="r:1", type=EventType.SHOT, t_v=10.0, half=1, score=0.1),
        ReviewCandidate(id="r:2", type=EventType.SHOT, t_v=20.0, half=1, score=0.1),
        ReviewCandidate(id="r:3", type=EventType.SHOT, t_v=1600.0, half=2, score=0.1),
        ReviewCandidate(id="r:4", type=EventType.FOUL, t_v=30.0, half=1, score=0.1),
    ]

    assert count_candidates(candidates) == {"shot": {"1": 2, "2": 1}, "foul": {"1": 1}}
    assert count_candidates([]) == {}
