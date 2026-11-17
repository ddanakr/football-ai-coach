"""Pure logic of the manual edits (D12b): time parsing, checks, and the edit to write.

No Streamlit, no I/O, no SQL. The page turns what the user typed into an `EditPlan` here, and
`aicoach.ui.data` hands the plan to the storage layer. The halves come from the fusion stage
(`aicoach.fusion.halves.half_at`), so an edit follows the same bounds as the events it edits.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from aicoach.domain.events import EventType
from aicoach.fusion.halves import half_at
from aicoach.fusion.schemas import Half, HalfSpan
from aicoach.i18n import ru
from aicoach.storage import CurrentEvent
from aicoach.ui.logic import format_mmss

_MMSS = re.compile(r"(\d{1,4}):([0-5]\d)", re.ASCII)  # ASCII: no Arabic-Indic digits


class EditInputError(ValueError):
    """What the user typed cannot be written as an edit; the message is Russian, for the user."""


@dataclass(frozen=True)
class TimeBounds:
    """Where an event may stand: the two halves and the end of the video, in video seconds."""

    halves: Mapping[Half, HalfSpan]
    video_duration_s: float


@dataclass(frozen=True)
class EditPlan:
    """One edit ready for the store. `message` tells the user what was done, in Russian."""

    action: Literal["add", "move", "retype", "delete"]
    message: str
    target: str | None = None
    type: EventType | None = None
    t_v: float | None = None


def parse_mmss(text: str) -> float:
    """Video time in seconds from `mm:ss` text (`12:05`; minutes may exceed 59: `125:03`).

    Raises:
        EditInputError: Not `mm:ss`: empty, negative, seconds above 59, other characters.
    """
    found = _MMSS.fullmatch(text.strip())
    if found is None:
        raise EditInputError(ru.ERR_TIME_FORMAT)
    return float(int(found.group(1)) * 60 + int(found.group(2)))


def check_time(t_v: float, bounds: TimeBounds) -> Half:
    """The half that contains `t_v`; refuses a time past the end of the video or outside halves.

    Raises:
        EditInputError: `t_v` is past the end of the video or in the break or the studio.
    """
    if not math.isfinite(t_v) or t_v < 0:
        raise EditInputError(ru.ERR_TIME_FORMAT)
    if t_v > bounds.video_duration_s:
        raise EditInputError(
            ru.ERR_TIME_AFTER_END.format(
                time=format_mmss(t_v), end=format_mmss(bounds.video_duration_s)
            )
        )
    half = half_at(t_v, bounds.halves)
    if half is None:
        spans = "; ".join(
            ru.HALF_SPAN.format(
                half=h,
                start=format_mmss(bounds.halves[h].start),
                end=format_mmss(bounds.halves[h].end),
            )
            for h in (1, 2)
        )
        raise EditInputError(ru.ERR_TIME_OUTSIDE_HALVES.format(time=format_mmss(t_v), spans=spans))
    return half


def plan_move(event: CurrentEvent, text: str, bounds: TimeBounds) -> EditPlan:
    """Move an event to the time typed as `mm:ss`.

    Raises:
        EditInputError: Bad format, time outside the halves or past the end, or the same second
            the event already has (the list shows whole seconds, so that is not a move).
    """
    t_v = parse_mmss(text)
    half = check_time(t_v, bounds)
    if t_v == math.floor(event.t_v):
        raise EditInputError(ru.ERR_TIME_UNCHANGED)
    message = ru.EDIT_MOVED.format(
        type=ru.TYPE_LABELS[event.type], time=format_mmss(t_v), half=half
    )
    return EditPlan("move", message, target=event.event_id, t_v=t_v)


def plan_retype(event: CurrentEvent, new_type: EventType) -> EditPlan:
    """Change the category of an event.

    Raises:
        EditInputError: The event already has this category.
    """
    if new_type == event.type:
        raise EditInputError(ru.ERR_TYPE_UNCHANGED)
    message = ru.EDIT_RETYPED.format(old=ru.TYPE_LABELS[event.type], new=ru.TYPE_LABELS[new_type])
    return EditPlan("retype", message, target=event.event_id, type=new_type)


def plan_delete(event: CurrentEvent) -> EditPlan:
    """Hide an event (the store keeps it; the last edit can be undone)."""
    message = ru.EDIT_DELETED.format(type=ru.TYPE_LABELS[event.type], time=format_mmss(event.t_v))
    return EditPlan("delete", message, target=event.event_id)


def plan_add_at(event_type: EventType, t_v: float, bounds: TimeBounds) -> EditPlan:
    """A new event of the given category at an exact video time (a candidate of the review).

    Raises:
        EditInputError: The time is outside the halves or past the end.
    """
    half = check_time(t_v, bounds)
    message = ru.EDIT_ADDED.format(
        type=ru.TYPE_LABELS[event_type], time=format_mmss(t_v), half=half
    )
    return EditPlan("add", message, type=event_type, t_v=t_v)


def plan_add(event_type: EventType, text: str, bounds: TimeBounds) -> EditPlan:
    """A new event of the given category at the time typed as `mm:ss`.

    Raises:
        EditInputError: Bad format, time outside the halves or past the end.
    """
    return plan_add_at(event_type, parse_mmss(text), bounds)


def with_derived_half(event: CurrentEvent, bounds: TimeBounds | None) -> CurrentEvent:
    """The event with its half taken from its current time, if a user touched it.

    The store keeps no half for edits: a moved prediction still carries the half of its first
    time, and an added event has none. For those the half is derived from `t_v` by the fusion
    bounds. Untouched predictions keep the half the fusion stage wrote. Without bounds nothing
    changes.
    """
    if bounds is None or (event.origin != "user" and event.n_edits == 0):
        return event
    return event.model_copy(update={"half": half_at(event.t_v, bounds.halves)})
