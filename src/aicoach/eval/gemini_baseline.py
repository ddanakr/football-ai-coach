"""Reader of the Gemini baseline file: a header line, then a JSON array of `type`, `t`, `note`.

The baseline is the "ask a video LLM where the events are" comparison row (D-007 notes).
`t` is video time as `HH:MM:SS` (or `MM:SS`). `note` is free text that may name people, so it is
never read into the result.
"""

import json
import re
from dataclasses import dataclass, field

from aicoach.domain.events import EventType
from aicoach.errors import EvalError
from aicoach.eval.schemas import Prediction

_TIME = re.compile(r"^(?:(\d{1,3}):)?([0-5]?\d):([0-5]\d)$")


@dataclass(frozen=True)
class GeminiBaseline:
    """Parsed baseline: its header line, the events and how many entries were skipped.

    `skipped` maps a reason (`not_an_object`, `unknown_type`, `bad_time`) to a count.
    """

    header: str
    events: tuple[Prediction, ...]
    skipped: dict[str, int] = field(default_factory=dict)


def parse_clock_time(text: object) -> float | None:
    """`HH:MM:SS` or `MM:SS` to seconds; `None` for anything else."""
    if not isinstance(text, str):
        return None
    match = _TIME.match(text.strip())
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600.0 + int(minutes) * 60.0 + int(seconds)


def _entry(item: object) -> tuple[Prediction | None, str]:
    if not isinstance(item, dict):
        return None, "not_an_object"
    raw_type = item.get("type")
    try:
        event_type = EventType(raw_type) if isinstance(raw_type, str) else None
    except ValueError:
        event_type = None
    if event_type is None:
        return None, "unknown_type"
    seconds = parse_clock_time(item.get("t"))
    if seconds is None:
        return None, "bad_time"
    return Prediction(type=event_type, t_v=seconds), ""


def parse_gemini_baseline(text: str) -> GeminiBaseline:
    """Parse the baseline file text.

    A bad entry (not an object, unknown type, unreadable time) is skipped and counted; an empty
    array is a valid empty baseline.

    Raises:
        EvalError: There is no header line, or the rest is not a JSON array.
    """
    header, _, body = text.lstrip("﻿").partition("\n")
    if not header.strip() or not body.strip():
        msg = "the Gemini baseline needs a header line and a JSON array after it"
        raise EvalError(msg)
    try:
        items = json.loads(body)
    except json.JSONDecodeError as err:
        msg = f"the Gemini baseline is not a JSON array after the header: {err.msg}"
        raise EvalError(msg) from err
    if not isinstance(items, list):
        msg = "the Gemini baseline must be a JSON array"
        raise EvalError(msg)
    events: list[Prediction] = []
    skipped: dict[str, int] = {}
    for item in items:
        event, reason = _entry(item)
        if event is None:
            skipped[reason] = skipped.get(reason, 0) + 1
        else:
            events.append(event)
    return GeminiBaseline(header=header.strip(), events=tuple(events), skipped=skipped)
