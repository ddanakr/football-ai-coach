"""Greedy matching of predictions to ground truth."""

import math
import random

import pytest

from aicoach.domain.events import EventType
from aicoach.errors import EvalError
from aicoach.eval.matching import (
    MatchResult,
    filter_half,
    match_events,
    normalize_predictions,
)
from tests.fixtures.factories import make_gt, make_pred


def _pairs(result: MatchResult) -> set[tuple[int, int]]:
    return {(p.pred_index, p.gt_index) for p in result.pairs}


@pytest.mark.unit
def test_match_events_empty_ground_truth_gives_no_pairs() -> None:
    result = match_events([make_pred("shot", 10.0)], [], tolerance_s=5.0)

    assert result.pairs == ()
    assert result.pred_matched == [False]
    assert result.gt_found == []


@pytest.mark.unit
def test_match_events_empty_predictions_gives_no_pairs() -> None:
    result = match_events([], [make_gt("shot", 10.0)], tolerance_s=5.0)

    assert result.pairs == ()
    assert result.pred_matched == []
    assert result.gt_found == [False]


@pytest.mark.unit
def test_match_events_both_empty_gives_empty_result() -> None:
    result = match_events([], [], tolerance_s=5.0)

    assert result.pairs == ()
    assert result.pred_matched == []
    assert result.gt_found == []


@pytest.mark.unit
def test_match_events_two_predictions_near_one_truth_gives_one_pair() -> None:
    preds = [make_pred("shot", 101.5), make_pred("shot", 99.0)]

    result = match_events(preds, [make_gt("shot", 100.0)], tolerance_s=5.0)

    # The closer prediction (index 1, |dt| = 1.0) wins; the other one is a false positive.
    assert _pairs(result) == {(1, 0)}
    assert result.pred_matched == [False, True]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("t_pred", "expected_found"),
    [
        (105.0, True),  # exactly on the boundary, inclusive
        (95.0, True),
        (105.01, False),  # 0.01 s beyond the boundary
        (94.99, False),
        (100.0, True),
    ],
)
def test_match_events_boundary_of_tolerance_is_inclusive(
    t_pred: float, expected_found: bool
) -> None:
    result = match_events([make_pred("goal", t_pred)], [make_gt("goal", 100.0)], tolerance_s=5.0)

    assert result.gt_found == [expected_found]


@pytest.mark.unit
def test_match_events_boundary_survives_float_rounding() -> None:
    # 1239.56 - 1234.56 is 5.000000000000227 in binary floating point, yet it is a 5 s error.
    result = match_events(
        [make_pred("corner", 1239.56)], [make_gt("corner", 1234.56)], tolerance_s=5.0
    )

    assert result.gt_found == [True]


@pytest.mark.unit
def test_match_events_wrong_class_is_not_matched() -> None:
    result = match_events([make_pred("corner", 100.0)], [make_gt("shot", 100.0)], tolerance_s=5.0)

    assert result.pairs == ()


@pytest.mark.unit
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_match_events_conflict_for_one_truth_nearest_prediction_wins(
    order: tuple[int, int],
) -> None:
    candidates = [make_pred("shot", 103.0), make_pred("shot", 101.0)]
    preds = [candidates[i] for i in order]

    result = match_events(preds, [make_gt("shot", 100.0)], tolerance_s=5.0)

    winner = next(i for i, p in enumerate(preds) if p.t_v == 101.0)
    assert _pairs(result) == {(winner, 0)}


@pytest.mark.unit
def test_match_events_greedy_assigns_globally_smallest_error_first() -> None:
    # gt a=100, b=108. p0=104 is 4 s from both; p1=110 is 2 s from b and 10 s from a.
    # Smallest error first: (p1, b) 2.0, then (p0, a) 4.0 (tie with (p0, b) goes to lower gt index).
    preds = [make_pred("foul", 104.0), make_pred("foul", 110.0)]
    gts = [make_gt("foul", 100.0), make_gt("foul", 108.0)]

    result = match_events(preds, gts, tolerance_s=10.0)

    assert _pairs(result) == {(1, 1), (0, 0)}


@pytest.mark.unit
def test_match_events_equal_errors_resolve_by_lower_prediction_index() -> None:
    preds = [make_pred("shot", 98.0), make_pred("shot", 102.0)]

    result = match_events(preds, [make_gt("shot", 100.0)], tolerance_s=5.0)

    assert _pairs(result) == {(0, 0)}


@pytest.mark.unit
def test_match_events_equal_errors_resolve_by_lower_truth_index() -> None:
    gts = [make_gt("shot", 98.0), make_gt("shot", 102.0)]

    result = match_events([make_pred("shot", 100.0)], gts, tolerance_s=5.0)

    assert _pairs(result) == {(0, 0)}
    assert result.gt_found == [True, False]


@pytest.mark.unit
def test_match_events_one_prediction_near_two_truths_matches_only_one() -> None:
    gts = [make_gt("corner", 100.0), make_gt("corner", 103.0)]

    result = match_events([make_pred("corner", 101.0)], gts, tolerance_s=5.0)

    assert len(result.pairs) == 1
    assert sum(result.gt_found) == 1


@pytest.mark.unit
def test_match_events_input_order_does_not_change_matched_events() -> None:
    gts = [make_gt("shot", 10.0 * i) for i in range(1, 8)]
    preds = [make_pred("shot", 10.0 * i + (i % 3) - 1.0) for i in range(1, 8)]
    expected = {
        (preds[p.pred_index].t_v, gts[p.gt_index].t_v) for p in match_events(preds, gts, 5.0).pairs
    }

    rng = random.Random(7)  # noqa: S311  # deterministic shuffling in a test, not cryptography
    shuffled_preds, shuffled_gts = preds[:], gts[:]
    rng.shuffle(shuffled_preds)
    rng.shuffle(shuffled_gts)
    result = match_events(shuffled_preds, shuffled_gts, 5.0)

    actual = {
        (shuffled_preds[p.pred_index].t_v, shuffled_gts[p.gt_index].t_v) for p in result.pairs
    }
    assert actual == expected


@pytest.mark.unit
def test_normalize_predictions_disallowed_goal_becomes_shot() -> None:
    preds = [
        make_pred("goal", 10.0, status="disallowed"),
        make_pred("goal", 20.0),
        make_pred("shot", 30.0, status="disallowed"),
    ]

    result = normalize_predictions(preds)

    assert [p.type for p in result] == [EventType.SHOT, EventType.GOAL, EventType.SHOT]
    assert preds[0].type is EventType.GOAL  # the input is not mutated


@pytest.mark.unit
def test_match_events_disallowed_goal_matches_shot_truth_not_goal_truth() -> None:
    pred = make_pred("goal", 100.0, status="disallowed")

    as_shot = match_events([pred], [make_gt("shot", 100.0)], tolerance_s=5.0)
    as_goal = match_events([pred], [make_gt("goal", 100.0)], tolerance_s=5.0)

    assert len(as_shot.pairs) == 1
    assert as_goal.pairs == ()


@pytest.mark.unit
@pytest.mark.parametrize("tolerance_s", [-1.0, math.nan, math.inf])
def test_match_events_invalid_tolerance_raises_eval_error(tolerance_s: float) -> None:
    with pytest.raises(EvalError, match="tolerance_s"):
        match_events([], [], tolerance_s=tolerance_s)


@pytest.mark.unit
def test_match_events_zero_tolerance_requires_exact_time() -> None:
    result = match_events(
        [make_pred("shot", 10.0), make_pred("shot", 20.5)],
        [make_gt("shot", 10.0), make_gt("shot", 20.0)],
        tolerance_s=0.0,
    )

    assert _pairs(result) == {(0, 0)}


@pytest.mark.unit
def test_filter_half_keeps_only_requested_half() -> None:
    preds = [make_pred("shot", 1.0, half=1), make_pred("shot", 2.0, half=2), make_pred("shot", 3.0)]
    gts = [make_gt("shot", 1.0, half=1), make_gt("shot", 2.0, half=2)]

    assert [p.t_v for p in filter_half(preds, 2)] == [2.0]
    assert [g.t_v for g in filter_half(gts, 1)] == [1.0]


@pytest.mark.unit
def test_filter_half_unknown_half_raises_eval_error() -> None:
    with pytest.raises(EvalError, match="half"):
        filter_half([make_gt("shot", 1.0)], 3)  # type: ignore[arg-type]  # deliberately invalid
