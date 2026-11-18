"""Time codes of the command line and the segment of a video that the model is run on."""

import math
import re
from dataclasses import dataclass

from aicoach.errors import SpottingError
from aicoach.tdeed.config import MIN_SEGMENT_S, SEGMENT_GRID_S, SEGMENT_TAIL_S

_CLOCK = re.compile(r"(?:(\d{1,3}):)?(\d{1,4}):([0-5]\d)")
_SECONDS = re.compile(r"\d{1,6}")


def parse_timecode(text: str) -> float:
    """Seconds of `мм:сс`, `чч:мм:сс` or a plain number of seconds (digits only).

    Raises:
        SpottingError: The text is none of these (a sign, a fraction, seconds of 60 or more).
    """
    value = text.strip()
    if _SECONDS.fullmatch(value):
        return float(int(value))
    found = _CLOCK.fullmatch(value)
    if found is None:
        msg = f"bad time {text!r}: use mm:ss, hh:mm:ss or whole seconds"
        raise SpottingError(msg)
    hours, minutes, seconds = found.groups()
    if hours is not None and int(minutes) > 59:
        msg = f"bad time {text!r}: minutes must be 00-59 after hours"
        raise SpottingError(msg)
    return float(int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds))


def format_timecode(seconds: float) -> str:
    """`ч:мм:сс` for an hour or more, otherwise `мм:сс` (seconds are rounded down)."""
    whole = int(math.floor(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_window(text: str) -> tuple[float, float]:
    """`from-to` as two time codes (`13:12-23:12`); `from` must be below `to`.

    Raises:
        SpottingError: Not two time codes with a `-` between, or the order is wrong.
    """
    parts = text.split("-")
    if len(parts) != 2:
        msg = f"bad window {text!r}: use from-to, for example 13:12-23:12"
        raise SpottingError(msg)
    start, end = parse_timecode(parts[0]), parse_timecode(parts[1])
    if start >= end:
        msg = f"bad window {text!r}: the start must be before the end"
        raise SpottingError(msg)
    return start, end


@dataclass(frozen=True)
class Segment:
    """A piece of the video: `start_s` is on the window grid of the model, `end_s` is the end of
    the piece the user asked for; the video is cut `SEGMENT_TAIL_S` beyond it (`cut_end_s`)."""

    requested_start_s: float
    start_s: float
    end_s: float
    cut_end_s: float

    @property
    def length_s(self) -> float:
        """Length of the part that is predicted and reported."""
        return self.end_s - self.start_s

    @property
    def cut_length_s(self) -> float:
        """Length of the piece that is cut from the source."""
        return self.cut_end_s - self.start_s


def plan_segment(start_s: float | None, end_s: float | None, duration_s: float) -> Segment:
    """The piece of the video for `--from` and `--to`.

    The start is moved down to a multiple of 24 s: the model samples every 12th frame and moves its
    window by 50 samples, so a piece that starts there has the same sampling and the same windows
    as the whole match (apart from the first one, which has no earlier frames). A missing `--from`
    is the start of the video, a missing `--to` its end.

    Raises:
        SpottingError: The piece is shorter than a minute, lies outside the video or has a
            non-finite bound.
    """
    if duration_s <= 0 or not math.isfinite(duration_s):
        msg = "the video has no usable duration"
        raise SpottingError(msg)
    requested = 0.0 if start_s is None else start_s
    end = duration_s if end_s is None else min(end_s, duration_s)
    if not (math.isfinite(requested) and math.isfinite(end)) or requested < 0:
        msg = "the segment bounds must be finite and not negative"
        raise SpottingError(msg)
    start = math.floor(requested / SEGMENT_GRID_S) * SEGMENT_GRID_S
    if end - start < MIN_SEGMENT_S:
        msg = (
            f"the segment {format_timecode(start)}-{format_timecode(end)} is shorter than "
            f"{MIN_SEGMENT_S:g} s (the video is {format_timecode(duration_s)} long)"
        )
        raise SpottingError(msg)
    return Segment(requested, start, end, min(end + SEGMENT_TAIL_S, duration_s))
