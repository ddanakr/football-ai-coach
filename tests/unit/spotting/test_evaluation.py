"""Evaluating a T-DEED file on the first half: thresholds, half bounds, duplicates, output."""

import json
import math
from typing import Literal, cast

import pytest

from aicoach.domain.events import EventType
from aicoach.errors import EvalError, SpottingError
from aicoach.eval.schemas import GroundTruth, GroundTruthEvent, HalfBounds
from aicoach.spotting.evaluation import (
    DEFAULT_THRESHOLDS,
    evaluate_tdeed,
    evaluation_to_dict,
    format_evaluation,
)
from aicoach.spotting.schemas import TDeedFile, TDeedLabel, TDeedPrediction
from tests.fixtures.factories import make_gt

FPS = 25


def _frames(seconds: float) -> int:
    return round(seconds * FPS)


def _file(*items: tuple[float, TDeedLabel, float]) -> TDeedFile:
    """A T-DEED file from (seconds, label, confidence)."""
    return TDeedFile(
        predictions=tuple(
            TDeedPrediction(frame=_frames(t), label=label, confidence=conf)
            for t, label, conf in items
        )
    )


def _truth(*events: GroundTruthEvent) -> GroundTruth:
    """Half 1 is 100..2000 s, half 2 is 3000..5000 s."""
    return GroundTruth(
        video="synthetic.mp4",
        halves={
            "1": HalfBounds(start=100.0, end=2000.0),
            "2": HalfBounds(start=3000.0, end=5000.0),
        },
        events=events,
    )


@pytest.mark.unit
def test_default_thresholds_are_the_agreed_five() -> None:
    assert DEFAULT_THRESHOLDS == (0.05, 0.1, 0.2, 0.3, 0.5)


@pytest.mark.unit
def test_evaluate_tdeed_empty_file_gives_zero_recall_and_undefined_precision() -> None:
    truth = _truth(make_gt("goal", 500.0), make_gt("corner", 600.0))

    ev = evaluate_tdeed(TDeedFile(predictions=()), truth, thresholds=(0.1, 0.5))

    assert len(ev.results) == 2
    for result in ev.results:
        assert result.n_predictions == 0
        micro = result.report.result_for(5.0).micro
        assert (micro.tp, micro.fp, micro.fn) == (0, 0, 2)
        assert micro.recall == 0.0
        assert micro.precision is None


@pytest.mark.unit
def test_evaluate_tdeed_matches_a_prediction_to_its_event() -> None:
    truth = _truth(make_gt("goal", 500.0))

    ev = evaluate_tdeed(_file((503.0, TDeedLabel.GOAL, 0.8)), truth, thresholds=(0.1,))

    micro = ev.results[0].report.result_for(5.0).micro
    assert (micro.tp, micro.fp, micro.fn) == (1, 0, 0)
    assert micro.median_error_s == pytest.approx(3.0)


@pytest.mark.unit
def test_evaluate_tdeed_threshold_equal_to_confidence_keeps_the_event() -> None:
    truth = _truth(make_gt("goal", 500.0))
    file = _file((500.0, TDeedLabel.GOAL, 0.2))

    ev = evaluate_tdeed(file, truth, thresholds=(0.1, 0.2, 0.2000001, 0.5))

    assert [r.n_predictions for r in ev.results] == [1, 1, 0, 0]
    assert [r.report.result_for(5.0).micro.tp for r in ev.results] == [1, 1, 0, 0]


@pytest.mark.unit
def test_evaluate_tdeed_events_outside_the_half_are_not_counted_anywhere() -> None:
    truth = _truth(make_gt("goal", 500.0))
    file = _file(
        (50.0, TDeedLabel.GOAL, 0.9),  # before the half
        (500.0, TDeedLabel.GOAL, 0.9),
        (2500.0, TDeedLabel.GOAL, 0.9),  # the break
        (3500.0, TDeedLabel.GOAL, 0.9),  # the second half
    )

    ev = evaluate_tdeed(file, truth, thresholds=(0.1,))

    assert ev.results[0].n_predictions == 1
    micro = ev.results[0].report.result_for(5.0).micro
    assert (micro.tp, micro.fp, micro.fn) == (1, 0, 0)


@pytest.mark.unit
@pytest.mark.parametrize("t_v", [100.0, 2000.0])
def test_evaluate_tdeed_event_on_the_half_bound_belongs_to_the_half(t_v: float) -> None:
    ev = evaluate_tdeed(_file((t_v, TDeedLabel.FOUL, 0.9)), _truth(), thresholds=(0.1,))

    assert ev.results[0].n_predictions == 1


@pytest.mark.unit
def test_evaluate_tdeed_ground_truth_of_the_other_half_does_not_count_as_missed() -> None:
    truth = _truth(make_gt("goal", 500.0, half=1), make_gt("goal", 3500.0, half=2))

    ev = evaluate_tdeed(_file((500.0, TDeedLabel.GOAL, 0.9)), truth, thresholds=(0.1,))

    micro = ev.results[0].report.result_for(5.0).micro
    assert (micro.tp, micro.fn) == (1, 0)
    assert ev.half == 1


@pytest.mark.unit
def test_evaluate_tdeed_two_shot_classes_at_one_point_are_one_prediction() -> None:
    truth = _truth(make_gt("shot", 700.0))
    file = _file(
        (700.0, TDeedLabel.SHOTS_ON_TARGET, 0.4), (700.0, TDeedLabel.SHOTS_OFF_TARGET, 0.6)
    )

    ev = evaluate_tdeed(file, truth, thresholds=(0.1,))

    result = ev.results[0]
    assert result.n_predictions == 1
    micro = result.report.result_for(5.0).micro
    assert (micro.tp, micro.fp, micro.fn) == (1, 0, 0)  # without merging fp would be 1


@pytest.mark.unit
def test_evaluate_tdeed_radius_parameter_controls_the_merge() -> None:
    file = _file((700.0, TDeedLabel.CORNER, 0.4), (704.0, TDeedLabel.CORNER, 0.6))

    wide = evaluate_tdeed(file, _truth(), thresholds=(0.1,), radius_s=5.0)
    narrow = evaluate_tdeed(file, _truth(), thresholds=(0.1,), radius_s=3.0)

    assert wide.results[0].n_predictions == 1
    assert narrow.results[0].n_predictions == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("offset_s", "found_at_5", "found_at_10"),
    [
        (5.0, True, True),
        (-5.0, True, True),
        (5.04, False, True),
        (10.0, False, True),
        (10.04, False, False),
    ],
)
def test_evaluate_tdeed_event_at_the_edge_of_the_window(
    offset_s: float, found_at_5: bool, found_at_10: bool
) -> None:
    truth = _truth(make_gt("foul", 1000.0))

    ev = evaluate_tdeed(_file((1000.0 + offset_s, TDeedLabel.FOUL, 0.9)), truth, thresholds=(0.1,))

    report = ev.results[0].report
    assert (report.result_for(5.0).micro.tp == 1) is found_at_5
    assert (report.result_for(10.0).micro.tp == 1) is found_at_10


@pytest.mark.unit
def test_evaluate_tdeed_per_class_and_error_statistics() -> None:
    truth = _truth(make_gt("goal", 500.0), make_gt("corner", 600.0), make_gt("corner", 800.0))
    file = _file(
        (501.0, TDeedLabel.GOAL, 0.9),
        (604.0, TDeedLabel.CORNER, 0.9),
        (900.0, TDeedLabel.CORNER, 0.9),  # no corner there: a false positive
    )

    ev = evaluate_tdeed(file, truth, thresholds=(0.1,))

    result = ev.results[0].report.result_for(5.0)
    corner = result.per_class[EventType.CORNER]
    assert (corner.tp, corner.fp, corner.fn) == (1, 1, 1)
    assert corner.precision == 0.5
    assert corner.recall == 0.5
    assert corner.median_error_s == pytest.approx(4.0)
    assert result.micro.median_error_s == pytest.approx(2.5)
    assert result.micro.recall_ci is not None


@pytest.mark.unit
def test_evaluate_tdeed_dropped_classes_do_not_make_false_positives() -> None:
    file = _file((500.0, TDeedLabel.YELLOW_CARD, 0.99), (510.0, TDeedLabel.PENALTY, 0.99))

    ev = evaluate_tdeed(file, _truth(), thresholds=(0.1,))

    assert ev.results[0].n_predictions == 0


@pytest.mark.unit
def test_evaluate_tdeed_fps_parameter_rescales_the_time() -> None:
    # 12500 frames: 500 s at 25 fps, 250 s at 50 fps.
    file = TDeedFile(
        predictions=(TDeedPrediction(frame=12500, label=TDeedLabel.GOAL, confidence=0.9),)
    )
    truth = _truth(make_gt("goal", 250.0))

    assert evaluate_tdeed(file, truth, thresholds=(0.1,), fps=50.0).results[0].n_predictions == 1
    micro25 = evaluate_tdeed(file, truth, thresholds=(0.1,)).results[0].report.result_for(5.0).micro
    assert micro25.tp == 0


@pytest.mark.unit
def test_evaluate_tdeed_half_two_is_refused_before_any_work() -> None:
    with pytest.raises(SpottingError, match="reserved"):
        evaluate_tdeed(_file((3500.0, TDeedLabel.GOAL, 0.9)), _truth(), half=2)


@pytest.mark.unit
@pytest.mark.parametrize("half", [0, 3, -1])
def test_evaluate_tdeed_invalid_half_is_refused(half: int) -> None:
    invalid = cast(Literal[1, 2], half)  # an invalid half on purpose
    with pytest.raises(SpottingError, match="half"):
        evaluate_tdeed(TDeedFile(predictions=()), _truth(), half=invalid)


@pytest.mark.unit
@pytest.mark.parametrize(
    "halves",
    [{}, {"1": HalfBounds()}, {"1": HalfBounds(start=100.0)}, {"1": HalfBounds(end=2000.0)}],
)
def test_evaluate_tdeed_missing_half_bounds_raise_even_for_an_empty_file(
    halves: dict[Literal["1", "2"], HalfBounds],
) -> None:
    truth = GroundTruth(video="x.mp4", halves=halves, events=())

    with pytest.raises(EvalError, match="half 1"):
        evaluate_tdeed(TDeedFile(predictions=()), truth)


@pytest.mark.unit
@pytest.mark.parametrize("thresholds", [(), (-0.1,), (math.nan,), (0.2, math.inf)])
def test_evaluate_tdeed_invalid_thresholds_raise(thresholds: tuple[float, ...]) -> None:
    with pytest.raises(SpottingError, match="threshold"):
        evaluate_tdeed(TDeedFile(predictions=()), _truth(), thresholds=thresholds)


@pytest.mark.unit
def test_evaluate_tdeed_result_order_follows_the_thresholds_given() -> None:
    ev = evaluate_tdeed(TDeedFile(predictions=()), _truth(), thresholds=(0.5, 0.1))

    assert [r.threshold for r in ev.results] == [0.5, 0.1]


@pytest.mark.unit
def test_evaluate_tdeed_is_deterministic() -> None:
    truth = _truth(make_gt("goal", 500.0), make_gt("shot", 700.0))
    file = _file((501.0, TDeedLabel.GOAL, 0.9), (800.0, TDeedLabel.SHOTS_ON_TARGET, 0.3))

    assert evaluate_tdeed(file, truth) == evaluate_tdeed(file, truth)


# --- output -----------------------------------------------------------------------------------


@pytest.mark.unit
def test_evaluation_to_dict_is_json_serializable_and_complete() -> None:
    truth = _truth(make_gt("goal", 500.0), make_gt("corner", 600.0))
    ev = evaluate_tdeed(_file((501.0, TDeedLabel.GOAL, 0.9)), truth, thresholds=(0.1, 0.5))

    data = evaluation_to_dict(ev)
    text = json.dumps(data)  # no NaN: json.dumps(allow_nan=False) would raise on one
    json.loads(text)
    json.dumps(data, allow_nan=False)

    assert data["half"] == 1
    assert data["fps"] == 25.0
    assert data["merge_radius_s"] == 3.0
    assert data["tolerances_s"] == [5.0, 10.0]
    assert [t["threshold"] for t in data["thresholds"]] == [0.1, 0.5]
    first = data["thresholds"][0]
    assert first["n_predictions"] == 1
    assert [t["tolerance_s"] for t in first["tolerances"]] == [5.0, 10.0]
    tol = first["tolerances"][0]
    assert set(tol["per_class"]) == {"goal", "shot", "corner", "foul", "subst"}
    assert tol["micro"]["tp"] == 1
    assert tol["per_class"]["goal"]["recall"] == 1.0
    assert tol["per_class"]["foul"]["precision"] is None
    assert tol["micro"]["recall_ci"] is not None


@pytest.mark.unit
def test_evaluation_to_dict_says_nothing_about_the_second_half() -> None:
    truth = _truth(make_gt("goal", 500.0), make_gt("goal", 3500.0, half=2))
    ev = evaluate_tdeed(
        _file((501.0, TDeedLabel.GOAL, 0.9), (3501.0, TDeedLabel.GOAL, 0.9)),
        truth,
        thresholds=(0.1,),
    )

    text = json.dumps(evaluation_to_dict(ev))

    assert "3500" not in text
    assert "3501" not in text
    assert "3000" not in text  # the bound of half 2
    assert evaluation_to_dict(ev)["thresholds"][0]["n_predictions"] == 1


@pytest.mark.unit
def test_format_evaluation_has_a_section_per_threshold_and_a_summary() -> None:
    truth = _truth(make_gt("goal", 500.0))
    ev = evaluate_tdeed(_file((501.0, TDeedLabel.GOAL, 0.9)), truth, thresholds=(0.1, 0.5))

    text = format_evaluation(ev)

    assert "half 1" in text
    assert "0.1" in text
    assert "0.5" in text
    assert "all (micro)" in text
    assert "±5" in text
    assert "±10" in text
    assert "Summary" in text
    assert "half 2" not in text
