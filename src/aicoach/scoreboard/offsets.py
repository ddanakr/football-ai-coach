"""Offset between video time and match clock: `offset = t_v - clock_s`.

Pure functions over plain lists. The clock may be frozen (before the match, in the break) or
misread; only readings of a running clock define the offset, and jumps are reported, not fixed.
"""

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from statistics import median_low

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.config import TimelineParams
from aicoach.scoreboard.schemas import OffsetJump


@dataclass(frozen=True)
class OffsetAnalysis:
    """Per-sample results: `raw` offset, whether the clock `running`, the `smoothed` offset
    (None where the clock is not running) and the `outlier` flag of a misread."""

    raw: list[float | None]
    running: list[bool]
    smoothed: list[float | None]
    outlier: list[bool]


def _check_input(t_v: list[float], clock_s: list[int | None]) -> None:
    if len(t_v) != len(clock_s):
        msg = "t_v and clock_s must have the same length"
        raise ScoreboardParseError(msg)
    if not all(math.isfinite(tv) for tv in t_v):
        msg = "t_v must be finite"
        raise ScoreboardParseError(msg)
    if any(b < a for a, b in zip(t_v, t_v[1:], strict=False)):
        msg = "t_v must not decrease"
        raise ScoreboardParseError(msg)


def _raw_offsets(t_v: list[float], clock_s: list[int | None]) -> list[float | None]:
    return [None if c is None else tv - c for tv, c in zip(t_v, clock_s, strict=True)]


def _window(times: list[float], centre: float, half_width: float) -> tuple[int, int]:
    return bisect_left(times, centre - half_width), bisect_right(times, centre + half_width)


def running_mask(t_v: list[float], clock_s: list[int | None], params: TimelineParams) -> list[bool]:
    """Mark samples where the clock runs: the offset does not drift in a window around them.

    The window's earlier and later halves are compared by their medians, so a frozen clock
    (offset grows with time) is rejected while a single misread digit changes nothing.
    A reading with too few neighbours is not running.
    """
    _check_input(t_v, clock_s)
    pairs = [(i, t_v[i] - c) for i, c in enumerate(clock_s) if c is not None]
    idx = [i for i, _ in pairs]
    times = [t_v[i] for i in idx]
    values = [o for _, o in pairs]
    mask = [False] * len(t_v)
    for k, i in enumerate(idx):
        lo, hi = _window(times, times[k], params.run_window_s / 2)
        window = values[lo:hi]
        if len(window) < params.run_min_points:
            continue
        mid = len(window) // 2
        drift = median_low(window[mid:]) - median_low(window[:mid])
        mask[i] = abs(drift) <= params.run_tol_s
    return mask


def analyze_offsets(
    t_v: list[float], clock_s: list[int | None], params: TimelineParams
) -> OffsetAnalysis:
    """Compute running flags, the smoothed offset and the outlier flags for every sample."""
    running = running_mask(t_v, clock_s, params)
    raw = _raw_offsets(t_v, clock_s)
    idx = [i for i, ok in enumerate(running) if ok]
    times = [t_v[i] for i in idx]
    values = [raw[i] for i in idx]
    smoothed: list[float | None] = [None] * len(t_v)
    outlier = [False] * len(t_v)
    for k, i in enumerate(idx):
        lo, hi = _window(times, times[k], params.offset_window_s / 2)
        window = [v for v in values[lo:hi] if v is not None]
        centre = median_low(window)
        smoothed[i] = centre
        own = raw[i]
        outlier[i] = own is not None and abs(own - centre) > params.outlier_tol_s
    return OffsetAnalysis(raw=raw, running=running, smoothed=smoothed, outlier=outlier)


def detect_jumps(
    t_v: list[float], smoothed: list[float | None], jump_tol_s: float
) -> list[OffsetJump]:
    """List changes of the smoothed offset larger than `jump_tol_s` between neighbouring
    readings (samples without an offset are skipped). Nothing is corrected.

    Steps of at most `jump_tol_s` are not reported (a clock that stopped for a few seconds
    leaves such a step); the median smoothing also hides them.

    Raises:
        ScoreboardParseError: lengths differ.
    """
    if len(t_v) != len(smoothed):
        msg = "t_v and smoothed must have the same length"
        raise ScoreboardParseError(msg)
    jumps: list[OffsetJump] = []
    prev: tuple[float, float] | None = None
    for tv, off in zip(t_v, smoothed, strict=True):
        if off is None:
            continue
        if prev is not None and abs(off - prev[1]) > jump_tol_s:
            jumps.append(
                OffsetJump(t_before=prev[0], t_after=tv, offset_before=prev[1], offset_after=off)
            )
        prev = (tv, off)
    return jumps
