"""Pure logic of the event viewer: filters, time and label formatting, YouTube links.

No Streamlit and no I/O here, so it is tested on its own and the agent can reuse
`youtube_link` later. `t_v` is video seconds from the start of the file (D-018).
"""

import math
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from aicoach.domain.events import EventType
from aicoach.i18n import ru
from aicoach.storage import CurrentEvent
from aicoach.youtube import WATCH_URL, validate_video_id

# The player starts a little before the event so that the action itself is not cut off (D-054).
PLAYER_PRE_ROLL_S = 5.0

HalfFilter = Literal[1, 2] | None


def _whole_seconds(t_v: float) -> int:
    if not math.isfinite(t_v) or t_v < 0:
        msg = f"t_v must be a finite non-negative number of seconds, got {t_v}"
        raise ValueError(msg)
    return math.floor(t_v)


def format_mmss(t_v: float) -> str:
    """Video time as `mm:ss` from the start of the file; minutes may exceed 59 (`125:03`).

    Seconds are cut, not rounded, so the text never runs ahead of the player.

    Raises:
        ValueError: `t_v` is negative, NaN or infinite.
    """
    minutes, seconds = divmod(_whole_seconds(t_v), 60)
    return f"{minutes:02d}:{seconds:02d}"


def player_start(t_v: float, pre_roll_s: float = PLAYER_PRE_ROLL_S) -> int:
    """Whole second where the player starts: `t_v - pre_roll_s`, not before 0.

    Raises:
        ValueError: `t_v` or `pre_roll_s` is negative, NaN or infinite.
    """
    pre_roll = _whole_seconds(pre_roll_s)
    return max(0, _whole_seconds(t_v) - pre_roll)


def youtube_link(video_id: str, t_v: float, pre_roll_s: float = PLAYER_PRE_ROLL_S) -> str:
    """Link to the moment of an event on YouTube, `pre_roll_s` seconds before it.

    Raises:
        ConfigError: Bad video id.
        ValueError: `t_v` or `pre_roll_s` is negative, NaN or infinite.
    """
    start = player_start(t_v, pre_roll_s)
    return f"{WATCH_URL.format(video_id=validate_video_id(video_id))}&t={start}s"


@dataclass(frozen=True)
class EventFilter:
    """What the user wants to see. `types` empty shows nothing; `half=None` shows both halves."""

    types: frozenset[EventType]
    half: HalfFilter = None
    min_score: float = 0.0


def filter_events(events: Iterable[CurrentEvent], flt: EventFilter) -> list[CurrentEvent]:
    """Keep events of the chosen types and half with `score >= min_score`, in input order.

    Events without a score (plates, user additions) pass the score threshold, as in the
    store (D-027). A chosen half hides events whose half is unknown.
    """
    return [
        e
        for e in events
        if e.type in flt.types
        and (flt.half is None or e.half == flt.half)
        and (e.score is None or e.score >= flt.min_score)
    ]


def available_types(events: Iterable[CurrentEvent]) -> list[EventType]:
    """The categories present in `events`, in the fixed order of the taxonomy."""
    present = {e.type for e in events}
    return [t for t in EventType if t in present]


def type_label(event_type: EventType) -> str:
    """Russian name of the category."""
    return ru.TYPE_LABELS[event_type]


@dataclass(frozen=True)
class EventRow:
    """One event of the list, ready to show. Texts are Russian; `t_v` is kept for the player."""

    event_id: str
    t_v: float
    time_text: str
    type_label: str
    half_text: str
    score: float | None
    score_text: str
    source_text: str
    team_text: str
    status_text: str
    edit_text: str = ""

    @property
    def type_text(self) -> str:
        """Category with the status note in brackets, if there is one."""
        return f"{self.type_label} ({self.status_text})" if self.status_text else self.type_label


def edit_mark(event: CurrentEvent) -> str:
    """Text for the "edit" column: added by the user, edited by the user, or empty."""
    if event.origin == "user":
        return ru.MARK_ADDED
    return ru.MARK_EDITED if event.n_edits else ""


def to_row(event: CurrentEvent) -> EventRow:
    """Format an event for the list; missing fields show as a dash."""
    return EventRow(
        event_id=event.event_id,
        t_v=event.t_v,
        time_text=format_mmss(event.t_v),
        type_label=type_label(event.type),
        half_text=ru.UNKNOWN if event.half is None else str(event.half),
        score=event.score,
        score_text=ru.UNKNOWN if event.score is None else f"{event.score:.2f}",
        source_text=ru.SOURCE_LABELS.get(event.source, event.source)
        if event.source
        else ru.UNKNOWN,
        team_text=ru.TEAM_LABELS[event.team] if event.team else ru.UNKNOWN,
        status_text=""
        if event.status == "auto"
        else ru.STATUS_LABELS.get(event.status, event.status),
        edit_text=edit_mark(event),
    )


def to_rows(events: Collection[CurrentEvent]) -> list[EventRow]:
    """`to_row` for every event, order kept."""
    return [to_row(e) for e in events]


def _blank(text: str) -> str:
    return "" if text == ru.UNKNOWN else text


def table_records(rows: Sequence[EventRow]) -> list[dict[str, str | float | None]]:
    """Rows of the events table, one dict per event; unknown values are empty.

    Keys are the Russian column names in display order: time, type, half, confidence, source,
    team, edit mark. The order of the records is the order of `rows`, so a selected table
    index is an index into `rows`.
    """
    return [
        {
            ru.COLUMN_TIME: row.time_text,
            ru.COLUMN_TYPE: row.type_text,
            ru.COLUMN_HALF: _blank(row.half_text),
            ru.COLUMN_SCORE: row.score,
            ru.COLUMN_SOURCE: _blank(row.source_text),
            ru.COLUMN_TEAM: _blank(row.team_text),
            ru.COLUMN_EDIT: row.edit_text,
        }
        for row in rows
    ]


def selected_row(rows: Sequence[EventRow], selected: Sequence[int]) -> EventRow | None:
    """The event of the table row the user picked, or `None` when nothing valid is picked.

    `selected` holds indices into `rows` as the table reports them. The table allows one row;
    if there are several, the first one counts. Indices that are negative or past the end (a
    table that changed after the pick) give `None`.
    """
    if not selected:
        return None
    index = selected[0]
    return rows[index] if 0 <= index < len(rows) else None
