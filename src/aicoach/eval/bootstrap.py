"""Percentile bootstrap for recall and precision."""

import random
from collections.abc import Sequence

from aicoach.errors import EvalError
from aicoach.eval.stats import percentile_nonempty

DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 0
_LOWER_Q = 2.5
_UPPER_Q = 97.5


def bootstrap_ci(
    flags: Sequence[bool],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float] | None:
    """95% bootstrap interval for the share of `True` flags.

    Resamples the flags with replacement `n_boot` times and takes the 2.5th and 97.5th
    percentiles of the resampled shares. The flags are already-computed "found" (for recall,
    one per ground-truth event) or "correct" (for precision, one per prediction) marks, so
    the matching is not repeated.

    Args:
        flags: One flag per ground-truth event or per prediction.
        n_boot: Number of resamples, at least 1.
        seed: Seed of the private random generator; the same input always gives the same result.

    Returns:
        `(low, high)`, or `None` when `flags` is empty.

    Raises:
        EvalError: `n_boot` is less than 1.
    """
    if n_boot < 1:
        msg = f"n_boot must be at least 1, got {n_boot}"
        raise EvalError(msg)
    if not flags:
        return None
    n = len(flags)
    hits = [int(flag) for flag in flags]
    # Non-cryptographic resampling. random() is used directly because its sequence for a
    # given seed is guaranteed stable across Python versions, unlike choices().
    rng = random.Random(seed)  # noqa: S311
    shares = [sum(hits[int(rng.random() * n)] for _ in range(n)) / n for _ in range(n_boot)]
    return (percentile_nonempty(shares, _LOWER_Q), percentile_nonempty(shares, _UPPER_Q))
