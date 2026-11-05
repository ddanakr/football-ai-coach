"""Percentile bootstrap over match flags."""

import pytest

from aicoach.errors import EvalError
from aicoach.eval.bootstrap import bootstrap_ci


@pytest.mark.unit
def test_bootstrap_ci_empty_flags_returns_none() -> None:
    assert bootstrap_ci([]) is None


@pytest.mark.unit
@pytest.mark.parametrize("flag", [True, False])
def test_bootstrap_ci_constant_flags_give_degenerate_interval(flag: bool) -> None:
    expected = 1.0 if flag else 0.0

    assert bootstrap_ci([flag] * 6) == (expected, expected)


@pytest.mark.unit
def test_bootstrap_ci_same_seed_gives_identical_result() -> None:
    flags = [True, False, True, True, False, True, False, True]

    assert bootstrap_ci(flags, seed=3) == bootstrap_ci(flags, seed=3)


@pytest.mark.unit
def test_bootstrap_ci_different_seed_changes_result() -> None:
    # Shares are discrete, so two seeds may coincide by chance; across several they must not.
    flags = [i % 3 != 0 for i in range(60)]

    intervals = {bootstrap_ci(flags, seed=seed) for seed in range(6)}

    assert len(intervals) > 1


@pytest.mark.unit
def test_bootstrap_ci_default_call_is_deterministic_across_calls() -> None:
    flags = [True, False, False, True, True]

    assert bootstrap_ci(flags) == bootstrap_ci(flags)


@pytest.mark.unit
def test_bootstrap_ci_interval_is_ordered_inside_unit_range_and_covers_estimate() -> None:
    flags = [True] * 7 + [False] * 3

    interval = bootstrap_ci(flags)

    assert interval is not None
    low, high = interval
    assert 0.0 <= low <= 0.7 <= high <= 1.0


@pytest.mark.unit
def test_bootstrap_ci_single_flag_is_degenerate() -> None:
    assert bootstrap_ci([True]) == (1.0, 1.0)


@pytest.mark.unit
def test_bootstrap_ci_wider_for_fewer_events() -> None:
    small = bootstrap_ci([True, False, True, False])
    large = bootstrap_ci([True, False] * 50)

    assert small is not None
    assert large is not None
    assert (small[1] - small[0]) > (large[1] - large[0])


@pytest.mark.unit
@pytest.mark.parametrize("n_boot", [0, -5])
def test_bootstrap_ci_non_positive_iterations_raise_eval_error(n_boot: int) -> None:
    with pytest.raises(EvalError, match="n_boot"):
        bootstrap_ci([True, False], n_boot=n_boot)
