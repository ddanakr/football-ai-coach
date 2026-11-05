"""Per-class and micro metrics, timing error, half filter."""

import pytest

from aicoach.domain.events import EventType
from aicoach.errors import EvalError
from aicoach.eval.metrics import DEFAULT_TOLERANCES_S, evaluate
from tests.fixtures.factories import make_gt, make_pred

ALL_TYPES = list(EventType)


@pytest.mark.unit
def test_evaluate_both_empty_gives_none_everywhere() -> None:
    report = evaluate([], [])

    for result in report.results:
        for metrics in (*result.per_class.values(), result.micro):
            assert (metrics.tp, metrics.fp, metrics.fn) == (0, 0, 0)
            assert metrics.precision is None
            assert metrics.recall is None
            assert metrics.precision_ci is None
            assert metrics.recall_ci is None
            assert metrics.median_error_s is None
            assert metrics.p90_error_s is None


@pytest.mark.unit
def test_evaluate_empty_ground_truth_gives_none_recall_and_zero_precision() -> None:
    report = evaluate([make_pred("shot", 10.0), make_pred("shot", 50.0)], [])

    micro = report.result_for(5.0).micro
    assert (micro.tp, micro.fp, micro.fn) == (0, 2, 0)
    assert micro.recall is None
    assert micro.recall_ci is None
    assert micro.precision == 0.0
    assert micro.median_error_s is None


@pytest.mark.unit
def test_evaluate_empty_predictions_gives_none_precision_and_zero_recall() -> None:
    report = evaluate([], [make_gt("corner", 10.0), make_gt("shot", 50.0)])

    micro = report.result_for(5.0).micro
    assert (micro.tp, micro.fp, micro.fn) == (0, 0, 2)
    assert micro.precision is None
    assert micro.precision_ci is None
    assert micro.recall == 0.0
    assert micro.p90_error_s is None


@pytest.mark.unit
def test_evaluate_default_tolerances_are_five_and_ten_in_one_call() -> None:
    report = evaluate([], [])

    assert DEFAULT_TOLERANCES_S == (5.0, 10.0)
    assert [r.tolerance_s for r in report.results] == [5.0, 10.0]


@pytest.mark.unit
def test_evaluate_hand_computed_example() -> None:
    # Ground truth (t_v, s):  goal 100 | shot 200 | shot 300 | corner 400 | foul 500
    # Predictions:            goal 103 | shot 196 | shot 308 | shot 600 | corner 405
    #
    # Tolerance 5 s (|dt| <= 5):
    #   goal:   103 vs 100 -> dt 3 -> TP.            tp 1, fp 0, fn 0, P 1,   R 1
    #   shot:   196 vs 200 -> dt 4 -> TP;
    #           308 vs 300 -> dt 8 -> too far;
    #           600 matches nothing.                 tp 1, fp 2, fn 1, P 1/3, R 1/2
    #   corner: 405 vs 400 -> dt 5 (boundary) -> TP. tp 1, fp 0, fn 0, P 1,   R 1
    #   foul:   no predictions.                      tp 0, fp 0, fn 1, P None, R 0
    #   subst:  nothing at all.                      P None, R None
    #   micro:  tp 3, fp 2, fn 2 -> P = 3/5, R = 3/5
    #           errors [3, 4, 5]: median 4, p90 = 4 + 0.8 * (5 - 4) = 4.8
    #
    # Tolerance 10 s: additionally 308 vs 300 (dt 8) is a TP.
    #   shot:   tp 2, fp 1, fn 0, P 2/3, R 1; errors [4, 8]: median 6, p90 = 4 + 0.9 * 4 = 7.6
    #   micro:  tp 4, fp 1, fn 1 -> P = 4/5, R = 4/5
    #           errors [3, 4, 5, 8]: median (4 + 5) / 2 = 4.5, p90 = 5 + 0.7 * 3 = 7.1
    gts = [
        make_gt("goal", 100.0),
        make_gt("shot", 200.0),
        make_gt("shot", 300.0),
        make_gt("corner", 400.0),
        make_gt("foul", 500.0),
    ]
    preds = [
        make_pred("goal", 103.0),
        make_pred("shot", 196.0),
        make_pred("shot", 308.0),
        make_pred("shot", 600.0),
        make_pred("corner", 405.0),
    ]

    report = evaluate(preds, gts)

    tol5 = report.result_for(5.0)
    shot5 = tol5.per_class[EventType.SHOT]
    assert (shot5.tp, shot5.fp, shot5.fn) == (1, 2, 1)
    assert shot5.precision == pytest.approx(1 / 3)
    assert shot5.recall == pytest.approx(1 / 2)
    assert shot5.median_error_s == pytest.approx(4.0)
    goal5 = tol5.per_class[EventType.GOAL]
    assert (goal5.precision, goal5.recall) == (1.0, 1.0)
    assert goal5.median_error_s == pytest.approx(3.0)
    corner5 = tol5.per_class[EventType.CORNER]
    assert (corner5.tp, corner5.fp, corner5.fn) == (1, 0, 0)
    assert corner5.median_error_s == pytest.approx(5.0)
    foul5 = tol5.per_class[EventType.FOUL]
    assert (foul5.tp, foul5.fp, foul5.fn) == (0, 0, 1)
    assert foul5.precision is None
    assert foul5.recall == 0.0
    subst5 = tol5.per_class[EventType.SUBST]
    assert (subst5.precision, subst5.recall) == (None, None)
    micro5 = tol5.micro
    assert (micro5.tp, micro5.fp, micro5.fn) == (3, 2, 2)
    assert micro5.precision == pytest.approx(0.6)
    assert micro5.recall == pytest.approx(0.6)
    assert micro5.median_error_s == pytest.approx(4.0)
    assert micro5.p90_error_s == pytest.approx(4.8)

    tol10 = report.result_for(10.0)
    shot10 = tol10.per_class[EventType.SHOT]
    assert (shot10.tp, shot10.fp, shot10.fn) == (2, 1, 0)
    assert shot10.precision == pytest.approx(2 / 3)
    assert shot10.recall == 1.0
    assert shot10.median_error_s == pytest.approx(6.0)
    assert shot10.p90_error_s == pytest.approx(7.6)
    micro10 = tol10.micro
    assert (micro10.tp, micro10.fp, micro10.fn) == (4, 1, 1)
    assert micro10.precision == pytest.approx(0.8)
    assert micro10.recall == pytest.approx(0.8)
    assert micro10.median_error_s == pytest.approx(4.5)
    assert micro10.p90_error_s == pytest.approx(7.1)


@pytest.mark.unit
def test_evaluate_duplicate_predictions_count_one_tp_and_one_fp() -> None:
    preds = [make_pred("goal", 99.0), make_pred("goal", 101.0)]

    micro = evaluate(preds, [make_gt("goal", 100.0)]).result_for(5.0).micro

    assert (micro.tp, micro.fp, micro.fn) == (1, 1, 0)
    assert micro.precision == 0.5
    assert micro.recall == 1.0


@pytest.mark.unit
def test_evaluate_wrong_class_counts_as_fp_and_fn() -> None:
    micro = evaluate([make_pred("corner", 100.0)], [make_gt("shot", 100.0)]).result_for(5.0).micro

    assert (micro.tp, micro.fp, micro.fn) == (0, 1, 1)


@pytest.mark.unit
def test_evaluate_disallowed_goal_counts_as_shot() -> None:
    preds = [make_pred("goal", 100.0, status="disallowed")]

    result = evaluate(preds, [make_gt("shot", 101.0)]).result_for(5.0)

    assert result.per_class[EventType.SHOT].tp == 1
    assert result.per_class[EventType.GOAL].fp == 0
    assert result.per_class[EventType.GOAL].precision is None


@pytest.mark.unit
def test_evaluate_input_order_does_not_change_metrics() -> None:
    gts = [make_gt("shot", 10.0 * i) for i in range(1, 9)]
    preds = [make_pred("shot", 10.0 * i + 3.0) for i in range(1, 9)]

    forward = evaluate(preds, gts)
    backward = evaluate(preds[::-1], gts[::-1])

    assert forward == backward


@pytest.mark.unit
def test_evaluate_is_deterministic_including_bootstrap() -> None:
    gts = [make_gt("shot", 10.0 * i) for i in range(1, 9)]
    preds = [make_pred("shot", 10.0 * i + (3.0 if i % 2 else 12.0)) for i in range(1, 9)]

    assert evaluate(preds, gts) == evaluate(preds, gts)


@pytest.mark.unit
def test_evaluate_bootstrap_interval_covers_point_estimate() -> None:
    gts = [make_gt("shot", 10.0 * i) for i in range(1, 11)]
    preds = [make_pred("shot", 10.0 * i + (1.0 if i <= 6 else 50.0)) for i in range(1, 11)]

    micro = evaluate(preds, gts).result_for(5.0).micro

    assert micro.recall == pytest.approx(0.6)
    assert micro.recall_ci is not None
    assert micro.recall_ci[0] <= 0.6 <= micro.recall_ci[1]
    assert micro.precision_ci is not None
    assert micro.precision_ci[0] <= 0.6 <= micro.precision_ci[1]


@pytest.mark.unit
def test_evaluate_half_filter_keeps_only_requested_half() -> None:
    gts = [make_gt("shot", 100.0, half=1), make_gt("shot", 3000.0, half=2)]
    preds = [
        make_pred("shot", 101.0, half=1),
        make_pred("shot", 3001.0, half=2),
        make_pred("shot", 3500.0, half=2),
        make_pred("shot", 5000.0),  # unknown half is dropped when filtering
    ]

    both = evaluate(preds, gts).result_for(5.0).micro
    second = evaluate(preds, gts, half=2).result_for(5.0).micro

    assert (both.tp, both.fp, both.fn) == (2, 2, 0)
    assert (second.tp, second.fp, second.fn) == (1, 1, 0)


@pytest.mark.unit
def test_evaluate_half_filter_with_nothing_in_half_gives_none() -> None:
    micro = (
        evaluate([make_pred("shot", 1.0, half=1)], [make_gt("shot", 1.0, half=1)], half=2)
        .result_for(5.0)
        .micro
    )

    assert micro.precision is None
    assert micro.recall is None


@pytest.mark.unit
def test_evaluate_custom_tolerances_are_reported_in_order() -> None:
    report = evaluate([], [], tolerances_s=(30.0, 1.0))

    assert [r.tolerance_s for r in report.results] == [30.0, 1.0]


@pytest.mark.unit
def test_evaluate_empty_tolerances_raise_eval_error() -> None:
    with pytest.raises(EvalError, match="tolerances_s"):
        evaluate([], [], tolerances_s=())


@pytest.mark.unit
def test_report_result_for_unknown_tolerance_raises_eval_error() -> None:
    with pytest.raises(EvalError, match="tolerance"):
        evaluate([], []).result_for(7.0)
