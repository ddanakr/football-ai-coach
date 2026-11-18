"""The comparison of two T-DEED output files."""

import math

import pytest

from aicoach.errors import SpottingError
from aicoach.fusion.config import FuseConfig
from aicoach.spotting.schemas import TDeedFile, TDeedLabel, TDeedPrediction
from aicoach.tdeed.compare import compare_tdeed, filter_by_fuse_thresholds, format_comparison


def _file(*items: tuple[float, TDeedLabel, float]) -> TDeedFile:
    """`(seconds, label, confidence)` as a file at 25 fps."""
    return TDeedFile(
        predictions=tuple(
            TDeedPrediction(frame=round(t * 25), label=label, confidence=c) for t, label, c in items
        )
    )


FOUL, CORNER, GOAL = TDeedLabel.FOUL, TDeedLabel.CORNER, TDeedLabel.GOAL
BASE = (
    (100.0, FOUL, 0.50),
    (112.0, CORNER, 0.30),
    (130.0, FOUL, 0.20),
)


@pytest.mark.unit
def test_compare_identical_files_is_the_same() -> None:
    result = compare_tdeed(_file(*BASE), _file(*BASE), window_s=(0, 300), edge_margin_s=0)

    assert result.same
    assert result.differences() == []
    assert format_comparison(result).splitlines()[-1] == "совпадает"


@pytest.mark.unit
def test_compare_different_count_is_reported_per_class() -> None:
    other = _file(*BASE[:2])

    result = compare_tdeed(_file(*BASE), other, window_s=(0, 300), edge_margin_s=0)

    assert not result.same
    assert result.differences() == ["Foul: 2 против 1 срабатываний"]
    assert format_comparison(result).splitlines()[-1].startswith("различия: Foul")


@pytest.mark.unit
def test_compare_time_shift_beyond_the_tolerance_breaks_the_pairs() -> None:
    shifted = _file((100.48, FOUL, 0.50), (112.0, CORNER, 0.30), (130.0, FOUL, 0.20))

    result = compare_tdeed(_file(*BASE), shifted, window_s=(0, 300), edge_margin_s=0)

    assert not result.same
    assert result.differences() == ["Foul: 1 без пары по времени"]


@pytest.mark.unit
def test_compare_time_shift_inside_the_tolerance_is_a_pair() -> None:
    near = _file((100.04, FOUL, 0.50), (112.0, CORNER, 0.30), (130.0, FOUL, 0.20))

    assert compare_tdeed(_file(*BASE), near, window_s=(0, 300), edge_margin_s=0).same


@pytest.mark.unit
def test_compare_tolerance_is_inclusive() -> None:
    a, b = _file((100.0, FOUL, 0.5)), _file((100.08, FOUL, 0.51))

    assert compare_tdeed(
        a, b, window_s=(0, 300), edge_margin_s=0, time_tol_s=0.08, score_tol=0.01
    ).same
    assert not compare_tdeed(a, b, window_s=(0, 300), edge_margin_s=0, time_tol_s=0.04).same


@pytest.mark.unit
def test_compare_score_difference_beyond_the_tolerance_is_reported_with_the_largest() -> None:
    changed = _file((100.0, FOUL, 0.52), (112.0, CORNER, 0.30), (130.0, FOUL, 0.20))

    result = compare_tdeed(_file(*BASE), changed, window_s=(0, 300), edge_margin_s=0)

    assert not result.same
    foul = next(c for c in result.classes if c.label == "Foul")
    assert foul.score_differs == 1
    assert foul.max_score_diff == pytest.approx(0.02)
    assert "Foul: 1 с разницей оценки больше допуска (наибольшая 0.0200)" in result.differences()


@pytest.mark.unit
def test_compare_score_difference_inside_the_tolerance_is_the_same() -> None:
    changed = _file((100.0, FOUL, 0.505), (112.0, CORNER, 0.30), (130.0, FOUL, 0.20))

    assert compare_tdeed(_file(*BASE), changed, window_s=(0, 300), edge_margin_s=0).same


@pytest.mark.unit
def test_compare_window_and_edge_margin_leave_out_the_ends() -> None:
    a = _file(*BASE)
    b = _file((112.0, CORNER, 0.30))  # the foul at 100 s and at 130 s are not in b

    inside = compare_tdeed(a, b, window_s=(95, 135), edge_margin_s=10)

    assert inside.window_s == (105.0, 125.0)
    assert inside.same  # only the corner at 112 s lies in 105-125


@pytest.mark.unit
def test_compare_default_window_is_the_overlap_of_the_two_files() -> None:
    a = _file((10.0, FOUL, 0.5), (100.0, FOUL, 0.5), (150.0, FOUL, 0.5), (200.0, FOUL, 0.5))
    b = _file((100.0, FOUL, 0.5), (150.0, FOUL, 0.5))  # spans 100-150 s

    result = compare_tdeed(a, b, edge_margin_s=0)

    assert result.window_s == (100.0, 150.0)
    assert result.same


@pytest.mark.unit
def test_compare_empty_files() -> None:
    both_empty = compare_tdeed(TDeedFile(predictions=()), TDeedFile(predictions=()))
    one_empty = compare_tdeed(_file(*BASE), TDeedFile(predictions=()), edge_margin_s=0)

    assert both_empty.same
    assert both_empty.classes == []
    assert not one_empty.same


@pytest.mark.unit
def test_compare_each_prediction_pairs_once() -> None:
    a = _file((100.0, FOUL, 0.5), (100.04, FOUL, 0.5))
    b = _file((100.0, FOUL, 0.5))

    result = compare_tdeed(a, b, window_s=(0, 300), edge_margin_s=0)

    assert not result.same
    assert result.classes[0].matched == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_tol_s": -0.1},
        {"score_tol": math.nan},
        {"edge_margin_s": -1.0},
        {"edge_margin_s": math.inf},
        {"fps": 0.0},
        {"edge_margin_s": 200.0, "window_s": (0.0, 300.0)},  # leaves no window
    ],
)
def test_compare_bad_parameters_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(SpottingError):
        compare_tdeed(_file(*BASE), _file(*BASE), **kwargs)  # type: ignore[arg-type]  # a bad value on purpose


@pytest.mark.unit
def test_format_comparison_shows_the_window_and_one_row_per_class() -> None:
    result = compare_tdeed(_file(*BASE), _file(*BASE), window_s=(60, 300), edge_margin_s=10)

    text = format_comparison(result)

    assert "окно сравнения 01:10-04:50" in text
    assert "Foul" in text
    assert "Corner" in text


@pytest.mark.unit
def test_filter_by_fuse_thresholds_keeps_shots_corners_fouls_at_or_above_their_threshold() -> None:
    data = _file(
        (10.0, TDeedLabel.SHOTS_ON_TARGET, 0.2),  # shot threshold 0.2, included
        (11.0, TDeedLabel.SHOTS_OFF_TARGET, 0.19),  # below
        (12.0, CORNER, 0.05),  # corner threshold 0.05, included
        (13.0, CORNER, 0.04),
        (14.0, FOUL, 0.5),
        (15.0, FOUL, 0.1),
        (16.0, GOAL, 0.99),  # fuse takes goals from the scoreboard, not from the model
        (17.0, TDeedLabel.SUBSTITUTION, 0.99),
        (18.0, TDeedLabel.THROW_IN, 0.99),  # no counterpart
    )

    kept = filter_by_fuse_thresholds(data)

    assert [p.frame / 25 for p in kept.predictions] == [10.0, 12.0, 14.0]


@pytest.mark.unit
def test_filter_by_fuse_thresholds_uses_the_given_config_and_handles_empty() -> None:
    config = FuseConfig(model_thresholds={"shot": 0.9, "corner": 0.9, "foul": 0.0})

    kept = filter_by_fuse_thresholds(_file((1.0, FOUL, 0.0), (2.0, CORNER, 0.5)), config)

    assert [p.label for p in kept.predictions] == [FOUL]
    assert filter_by_fuse_thresholds(TDeedFile(predictions=())).predictions == ()
