"""Mapping between video time `t_v` and the match clock, built from offset segments.

Inside a segment the offset is constant, so the mapping also holds where the scoreboard is
hidden (a replay). It never crosses a jump of the offset: the break between the halves is not
interpolated, and extrapolation beyond a segment is limited to `max_extrapolation_s`.
"""

import math
from collections.abc import Sequence
from statistics import median_low

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.schemas import OffsetJump, OffsetSegment, ScoreboardResult


def build_offset_segments(
    t_v: list[float], smoothed: list[float | None], jumps: list[OffsetJump]
) -> list[OffsetSegment]:
    """Group samples that have a smoothed offset into segments split at the recorded jumps.

    The offset of a segment is the median of its smoothed offsets, so inside a segment the
    mapping is exact only up to about the jump threshold.

    Raises:
        ScoreboardParseError: lengths differ.
    """
    if len(t_v) != len(smoothed):
        msg = "t_v and smoothed must have the same length"
        raise ScoreboardParseError(msg)
    pending = {jump.t_after for jump in jumps}
    groups: list[list[tuple[float, float]]] = []
    for tv, off in zip(t_v, smoothed, strict=True):
        if off is None:
            continue
        if not groups or tv in pending:
            pending.discard(tv)  # a repeated time must not open a second segment
            groups.append([])
        groups[-1].append((tv, off))
    return [_segment(group) for group in groups]


def _segment(group: list[tuple[float, float]]) -> OffsetSegment:
    offset = median_low([off for _, off in group])
    start, end = group[0][0], group[-1][0]
    return OffsetSegment(
        start=start,
        end=end,
        clock_start_s=round(start - offset),
        clock_end_s=round(end - offset),
        offset_s=offset,
        n_readings=len(group),
    )


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        msg = f"{name} must be finite, got {value}"
        raise ScoreboardParseError(msg)


def clock_at(
    segments: Sequence[OffsetSegment], t_v: float, max_extrapolation_s: float
) -> float | None:
    """Match clock (seconds) at video time `t_v`, or None if no segment is close enough or
    the extrapolated clock would be negative (before the start of the match).

    Raises:
        ScoreboardParseError: `t_v` is not finite.
    """
    _require_finite(t_v, "t_v")
    best: tuple[float, OffsetSegment] | None = None
    for seg in segments:
        distance = max(seg.start - t_v, t_v - seg.end, 0.0)
        if distance <= max_extrapolation_s and (best is None or distance < best[0]):
            best = (distance, seg)
    if best is None:
        return None
    clock = t_v - best[1].offset_s
    return clock if clock >= 0 else None


def video_times_at(
    segments: Sequence[OffsetSegment], clock_s: float, max_extrapolation_s: float
) -> list[float]:
    """Video times at which the match clock showed `clock_s`: one per segment that covers it.

    More than one candidate appears where the clock ranges of two halves overlap (stoppage
    time of the first half and the start of the second).

    Raises:
        ScoreboardParseError: `clock_s` is not finite.
    """
    _require_finite(clock_s, "clock_s")
    return sorted(
        clock_s + seg.offset_s
        for seg in segments
        if seg.clock_start_s - max_extrapolation_s
        <= clock_s
        <= seg.clock_end_s + max_extrapolation_s
    )


def video_time_at(
    segments: Sequence[OffsetSegment],
    clock_s: float,
    max_extrapolation_s: float,
    near_t_v: float | None = None,
) -> float | None:
    """Video time of the match clock value `clock_s`; with several candidates takes the one
    nearest to `near_t_v` (e.g. the time of a plate), or the earliest if no hint is given."""
    candidates = video_times_at(segments, clock_s, max_extrapolation_s)
    if not candidates:
        return None
    if near_t_v is None:
        return candidates[0]
    return min(candidates, key=lambda t: abs(t - near_t_v))


def result_clock_at(
    result: ScoreboardResult, t_v: float, max_extrapolation_s: float
) -> float | None:
    """`clock_at` over the segments stored in a `ScoreboardResult`."""
    return clock_at(result.offset_segments, t_v, max_extrapolation_s)


def result_video_time_at(
    result: ScoreboardResult,
    clock_s: float,
    max_extrapolation_s: float,
    near_t_v: float | None = None,
) -> float | None:
    """`video_time_at` over the segments stored in a `ScoreboardResult`."""
    return video_time_at(result.offset_segments, clock_s, max_extrapolation_s, near_t_v)
