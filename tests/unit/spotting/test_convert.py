"""From T-DEED predictions to our events: time, class mapping, duplicates, threshold, half."""

import math
from typing import Literal

import pytest

from aicoach.domain.events import EventType
from aicoach.errors import EvalError, SpottingError
from aicoach.eval.schemas import HalfBounds, Prediction
from aicoach.spotting.convert import (
    LABEL_TO_EVENT,
    filter_by_confidence,
    merge_duplicates,
    predictions_in_half,
    to_predictions,
)
from aicoach.spotting.schemas import TDeedFile, TDeedLabel, TDeedPrediction
from tests.fixtures.factories import make_pred


def _file(*items: tuple[int, TDeedLabel, float]) -> TDeedFile:
    return TDeedFile(
        predictions=tuple(
            TDeedPrediction(frame=frame, label=label, confidence=conf)
            for frame, label, conf in items
        )
    )


# --- class mapping and time -------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (TDeedLabel.GOAL, EventType.GOAL),
        (TDeedLabel.SHOTS_ON_TARGET, EventType.SHOT),
        (TDeedLabel.SHOTS_OFF_TARGET, EventType.SHOT),
        (TDeedLabel.CORNER, EventType.CORNER),
        (TDeedLabel.FOUL, EventType.FOUL),
        (TDeedLabel.SUBSTITUTION, EventType.SUBST),
    ],
)
def test_label_to_event_maps_the_six_kept_classes(label: TDeedLabel, expected: EventType) -> None:
    assert LABEL_TO_EVENT[label] is expected


@pytest.mark.unit
def test_label_to_event_covers_all_17_labels_and_drops_the_other_11() -> None:
    assert set(LABEL_TO_EVENT) == set(TDeedLabel)
    dropped = [label for label, event in LABEL_TO_EVENT.items() if event is None]
    assert len(dropped) == 11
    for label in (TDeedLabel.PENALTY, TDeedLabel.YELLOW_CARD, TDeedLabel.RED_CARD):
        assert label in dropped


@pytest.mark.unit
def test_to_predictions_converts_frame_to_seconds_at_25_fps() -> None:
    result = to_predictions(_file((250, TDeedLabel.GOAL, 0.7)))

    assert result == [Prediction(type=EventType.GOAL, t_v=10.0, score=0.7, status="auto")]


@pytest.mark.unit
@pytest.mark.parametrize(("fps", "t_v"), [(25.0, 12.0), (50.0, 6.0), (30.0, 10.0)])
def test_to_predictions_fps_parameter_changes_the_time(fps: float, t_v: float) -> None:
    result = to_predictions(_file((300, TDeedLabel.CORNER, 0.5)), fps=fps)

    assert result[0].t_v == pytest.approx(t_v)


@pytest.mark.unit
def test_to_predictions_drops_unmapped_classes_and_keeps_order() -> None:
    result = to_predictions(
        _file(
            (25, TDeedLabel.THROW_IN, 0.9),
            (50, TDeedLabel.FOUL, 0.4),
            (75, TDeedLabel.YELLOW_CARD, 0.9),
            (100, TDeedLabel.SUBSTITUTION, 0.3),
        )
    )

    assert [(p.type, p.t_v) for p in result] == [(EventType.FOUL, 2.0), (EventType.SUBST, 4.0)]


@pytest.mark.unit
def test_to_predictions_empty_file_gives_empty_list() -> None:
    assert to_predictions(TDeedFile(predictions=())) == []


@pytest.mark.unit
@pytest.mark.parametrize("fps", [0.0, -25.0, math.nan, math.inf])
def test_to_predictions_invalid_fps_raises(fps: float) -> None:
    with pytest.raises(SpottingError, match="fps"):
        to_predictions(_file((25, TDeedLabel.GOAL, 0.5)), fps=fps)


@pytest.mark.unit
def test_to_predictions_leaves_half_unset_so_the_bounds_place_it() -> None:
    assert to_predictions(_file((25, TDeedLabel.GOAL, 0.5)))[0].half is None


# --- merging duplicates -----------------------------------------------------------------------


@pytest.mark.unit
def test_merge_duplicates_two_shot_classes_at_one_point_keep_the_more_confident() -> None:
    # "Shots on target" and "Shots off target" at the same frame: a duplicate for us.
    preds = to_predictions(
        _file((500, TDeedLabel.SHOTS_ON_TARGET, 0.3), (500, TDeedLabel.SHOTS_OFF_TARGET, 0.6))
    )

    merged = merge_duplicates(preds)

    assert len(merged) == 1
    assert merged[0].score == 0.6


@pytest.mark.unit
def test_merge_duplicates_keeps_events_farther_apart_than_the_radius() -> None:
    preds = [make_pred("shot", 100.0, score=0.5), make_pred("shot", 103.5, score=0.4)]

    assert len(merge_duplicates(preds, radius_s=3.0)) == 2


@pytest.mark.unit
def test_merge_duplicates_event_exactly_at_the_radius_is_a_duplicate() -> None:
    # Boundary included, the same convention as the matching window in aicoach.eval.
    preds = [make_pred("shot", 100.0, score=0.5), make_pred("shot", 103.0, score=0.9)]

    merged = merge_duplicates(preds, radius_s=3.0)

    assert [(p.t_v, p.score) for p in merged] == [(103.0, 0.9)]


@pytest.mark.unit
def test_merge_duplicates_different_types_in_one_point_both_stay() -> None:
    preds = [make_pred("shot", 100.0, score=0.5), make_pred("corner", 100.0, score=0.4)]

    assert len(merge_duplicates(preds)) == 2


@pytest.mark.unit
def test_merge_duplicates_is_not_transitive_a_chain_keeps_its_far_end() -> None:
    # 0 and 2.5 are close, 2.5 and 5.0 are close, but 0 and 5.0 are not: the strongest event
    # (2.5) suppresses both neighbours only if each is within the radius of it.
    preds = [
        make_pred("foul", 0.0, score=0.5),
        make_pred("foul", 2.5, score=0.9),
        make_pred("foul", 5.0, score=0.5),
    ]

    assert [p.t_v for p in merge_duplicates(preds, radius_s=2.5)] == [2.5]
    # With a smaller radius the strongest event suppresses nobody.
    assert [p.t_v for p in merge_duplicates(preds, radius_s=2.0)] == [0.0, 2.5, 5.0]


@pytest.mark.unit
def test_merge_duplicates_suppressed_event_does_not_suppress_others() -> None:
    # 5.0 is beaten by 2.5; it must not remove 7.0, which is far from the winner.
    preds = [
        make_pred("foul", 2.5, score=0.9),
        make_pred("foul", 5.0, score=0.5),
        make_pred("foul", 7.0, score=0.4),
    ]

    assert [p.t_v for p in merge_duplicates(preds, radius_s=3.0)] == [2.5, 7.0]


@pytest.mark.unit
def test_merge_duplicates_equal_scores_keep_the_earlier_event() -> None:
    preds = [make_pred("shot", 11.0, score=0.5), make_pred("shot", 10.0, score=0.5)]

    assert [p.t_v for p in merge_duplicates(preds)] == [10.0]
    assert [p.t_v for p in merge_duplicates(list(reversed(preds)))] == [10.0]


@pytest.mark.unit
def test_merge_duplicates_returns_events_sorted_by_time() -> None:
    preds = [make_pred("goal", 300.0, score=0.5), make_pred("corner", 100.0, score=0.5)]

    assert [p.t_v for p in merge_duplicates(preds)] == [100.0, 300.0]


@pytest.mark.unit
def test_merge_duplicates_empty_input_and_zero_radius() -> None:
    assert merge_duplicates([]) == []
    twins = [make_pred("shot", 10.0, score=0.5), make_pred("shot", 10.0, score=0.4)]
    assert len(merge_duplicates(twins, radius_s=0.0)) == 1  # identical time is still a duplicate


@pytest.mark.unit
@pytest.mark.parametrize("radius", [-1.0, math.nan, math.inf])
def test_merge_duplicates_invalid_radius_raises(radius: float) -> None:
    with pytest.raises(SpottingError, match="radius_s"):
        merge_duplicates([make_pred("shot", 1.0, score=0.5)], radius_s=radius)


@pytest.mark.unit
def test_merge_duplicates_event_without_score_loses_to_any_scored_event() -> None:
    preds = [make_pred("shot", 10.0, score=None), make_pred("shot", 11.0, score=0.1)]

    assert [p.t_v for p in merge_duplicates(preds)] == [11.0]


# --- confidence threshold ---------------------------------------------------------------------


@pytest.mark.unit
def test_filter_by_confidence_threshold_itself_is_kept() -> None:
    preds = [
        make_pred("goal", 1.0, score=0.1),
        make_pred("goal", 2.0, score=0.1 - 1e-9),
        make_pred("goal", 3.0, score=0.2),
    ]

    assert [p.t_v for p in filter_by_confidence(preds, 0.1)] == [1.0, 3.0]


@pytest.mark.unit
def test_filter_by_confidence_zero_threshold_keeps_everything_scored() -> None:
    preds = [make_pred("goal", 1.0, score=0.0), make_pred("goal", 2.0, score=0.9)]

    assert len(filter_by_confidence(preds, 0.0)) == 2


@pytest.mark.unit
def test_filter_by_confidence_drops_events_without_score() -> None:
    assert filter_by_confidence([make_pred("goal", 1.0, score=None)], 0.0) == []


@pytest.mark.unit
def test_filter_by_confidence_empty_input() -> None:
    assert filter_by_confidence([], 0.5) == []


@pytest.mark.unit
@pytest.mark.parametrize("threshold", [-0.1, math.nan, math.inf])
def test_filter_by_confidence_invalid_threshold_raises(threshold: float) -> None:
    with pytest.raises(SpottingError, match="threshold"):
        filter_by_confidence([], threshold)


@pytest.mark.unit
@pytest.mark.parametrize("threshold", [0.05, 0.3, 0.5, 0.9])
def test_threshold_then_merge_equals_merge_then_threshold(threshold: float) -> None:
    # The pipeline merges once and thresholds afterwards; this is only right if the order
    # does not matter.
    preds = [
        make_pred("shot", 100.0, score=0.9),
        make_pred("shot", 101.5, score=0.35),
        make_pred("shot", 103.0, score=0.5),
        make_pred("shot", 104.8, score=0.06),
        make_pred("shot", 200.0, score=0.4),
        make_pred("foul", 100.0, score=0.2),
    ]

    a = filter_by_confidence(merge_duplicates(preds), threshold)
    b = merge_duplicates(filter_by_confidence(preds, threshold))

    assert a == b


# --- half filter by ground-truth bounds -------------------------------------------------------

BOUNDS: dict[Literal["1", "2"], HalfBounds] = {
    "1": HalfBounds(start=100.0, end=200.0),
    "2": HalfBounds(start=300.0, end=400.0),
}


@pytest.mark.unit
def test_predictions_in_half_keeps_only_events_inside_the_bounds() -> None:
    preds = [
        make_pred("goal", 50.0, score=0.5),  # before the half
        make_pred("goal", 150.0, score=0.5),
        make_pred("goal", 250.0, score=0.5),  # the break
        make_pred("goal", 350.0, score=0.5),  # the other half
    ]

    assert [p.t_v for p in predictions_in_half(preds, 1, BOUNDS)] == [150.0]


@pytest.mark.unit
@pytest.mark.parametrize("t_v", [100.0, 200.0])
def test_predictions_in_half_bounds_are_inclusive(t_v: float) -> None:
    assert len(predictions_in_half([make_pred("goal", t_v, score=0.5)], 1, BOUNDS)) == 1


@pytest.mark.unit
@pytest.mark.parametrize("t_v", [99.99, 200.01])
def test_predictions_in_half_just_outside_is_dropped(t_v: float) -> None:
    assert predictions_in_half([make_pred("goal", t_v, score=0.5)], 1, BOUNDS) == []


@pytest.mark.unit
def test_predictions_in_half_marks_the_half_on_the_result() -> None:
    result = predictions_in_half([make_pred("goal", 150.0, score=0.5)], 1, BOUNDS)

    assert result[0].half == 1


@pytest.mark.unit
def test_predictions_in_half_missing_bounds_raise_eval_error() -> None:
    with pytest.raises(EvalError, match="half 1"):
        predictions_in_half([make_pred("goal", 150.0, score=0.5)], 1, {})
