"""Small statistics helpers (no numpy)."""

import math
from collections.abc import Sequence

from aicoach.errors import EvalError


def percentile_nonempty(values: Sequence[float], q: float) -> float:
    """Percentile with linear interpolation, the same as numpy's default method.

    The value at sorted position `q / 100 * (n - 1)` is interpolated between its neighbours.

    Args:
        values: Samples in any order, at least one.
        q: Percentile in [0, 100].

    Returns:
        The percentile.

    Raises:
        EvalError: `q` is outside [0, 100] or NaN, `values` is empty or contains NaN.
    """
    if not 0.0 <= q <= 100.0:  # also False for NaN
        msg = f"percentile q must be in [0, 100], got {q}"
        raise EvalError(msg)
    if not values:
        msg = "percentile of an empty sequence is undefined"
        raise EvalError(msg)
    if any(math.isnan(v) for v in values):
        msg = "percentile input contains NaN"
        raise EvalError(msg)
    ordered = sorted(values)
    position = q / 100.0 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def percentile(values: Sequence[float], q: float) -> float | None:
    """Like `percentile_nonempty`, but returns `None` for an empty input.

    Raises:
        EvalError: For a non-empty input, `q` is outside [0, 100] or NaN, or `values`
            contains NaN.
    """
    if not values:
        return None
    return percentile_nonempty(values, q)
