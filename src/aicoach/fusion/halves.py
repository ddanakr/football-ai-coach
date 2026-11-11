"""Half bounds from the clock segments of `scoreboard.json` (never from the ground truth)."""

from collections.abc import Mapping, Sequence

from aicoach.errors import FusionError
from aicoach.fusion.schemas import Half, HalfSpan
from aicoach.scoreboard.schemas import OffsetSegment


def half_bounds(
    segments: Sequence[OffsetSegment], start_margin_s: float, end_margin_s: float
) -> dict[Half, HalfSpan]:
    """Video-time bounds of both halves.

    The break is the widest gap between two clock segments (as in `plates.assemble.half_of`);
    segments before it belong to half 1, after it to half 2. A half starts `start_margin_s`
    before its first clock reading (never before 0) and ends `end_margin_s` after its last one.

    Raises:
        FusionError: fewer than two clock segments, so the halves cannot be told apart.
    """
    if len(segments) < 2:
        msg = (
            f"scoreboard.json has {len(segments)} clock segment(s); two are needed to find the "
            "break between the halves: check the scoreboard stage"
        )
        raise FusionError(msg)
    ordered = sorted(segments, key=lambda s: s.start)
    _, widest = max((ordered[i + 1].start - ordered[i].end, i) for i in range(len(ordered) - 1))
    first, second = ordered[: widest + 1], ordered[widest + 1 :]
    return {
        1: HalfSpan(
            start=max(0.0, first[0].start - start_margin_s), end=first[-1].end + end_margin_s
        ),
        2: HalfSpan(
            start=max(0.0, second[0].start - start_margin_s), end=second[-1].end + end_margin_s
        ),
    }


def half_at(t_v: float, bounds: Mapping[Half, HalfSpan]) -> Half | None:
    """The half whose bounds contain `t_v` (ends included), or None for the break and the studio."""
    for half in (1, 2):
        span = bounds[half]
        if span.start <= t_v <= span.end:
            return half
    return None
