"""Percentile with linear interpolation (numpy default)."""

import math
import statistics

import pytest

from aicoach.errors import EvalError
from aicoach.eval.stats import percentile, percentile_nonempty


@pytest.mark.unit
def test_percentile_empty_returns_none() -> None:
    assert percentile([], 50.0) is None


@pytest.mark.unit
def test_percentile_single_value_returns_it_for_any_q() -> None:
    assert percentile([4.2], 0.0) == 4.2
    assert percentile([4.2], 90.0) == 4.2
    assert percentile([4.2], 100.0) == 4.2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        ([1.0, 2.0, 3.0, 4.0], 50.0, 2.5),  # pos = 0.5 * 3 = 1.5 -> (2 + 3) / 2
        ([3.0, 1.0, 2.0], 50.0, 2.0),  # input order does not matter
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 90.0, 9.1),  # pos = 8.1
        ([3.0, 4.0, 5.0], 90.0, 4.8),  # pos = 1.8 -> 4 + 0.8 * 1
        ([1.0, 9.0], 0.0, 1.0),
        ([1.0, 9.0], 100.0, 9.0),
    ],
)
def test_percentile_matches_hand_computed_values(
    values: list[float], q: float, expected: float
) -> None:
    result = percentile(values, q)

    assert result == pytest.approx(expected)


@pytest.mark.unit
def test_percentile_agrees_with_stdlib_inclusive_quantiles() -> None:
    values = [0.3, 7.1, 2.2, 9.9, 4.4, 1.0, 5.5, 8.8, 6.6, 3.3, 12.0]
    deciles = statistics.quantiles(values, n=10, method="inclusive")

    assert percentile(values, 90.0) == pytest.approx(deciles[8])
    assert percentile(values, 10.0) == pytest.approx(deciles[0])


@pytest.mark.unit
@pytest.mark.parametrize("q", [-0.1, 100.1, math.nan])
def test_percentile_q_out_of_range_raises_eval_error(q: float) -> None:
    with pytest.raises(EvalError, match="percentile"):
        percentile([1.0, 2.0], q)


@pytest.mark.unit
def test_percentile_nan_value_raises_eval_error() -> None:
    with pytest.raises(EvalError, match="NaN"):
        percentile([1.0, math.nan], 50.0)


@pytest.mark.unit
def test_percentile_nonempty_empty_input_raises_eval_error() -> None:
    with pytest.raises(EvalError, match="empty"):
        percentile_nonempty([], 50.0)
