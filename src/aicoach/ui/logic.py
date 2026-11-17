"""Pure logic of the event viewer: filters, time and label formatting, YouTube links.

No Streamlit and no I/O here, so it is tested on its own and the agent can reuse
`youtube_link` later. `t_v` is video seconds from the start of the file (D-018).
"""

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from aicoach.domain.events import EventType
from aicoach.fusion.schemas import Candidate
from aicoach.i18n import ru
from aicoach.plate_names.schemas import PlateNames, PlayerRead
from aicoach.storage import CurrentEvent
from aicoach.youtube import WATCH_URL, validate_video_id

# The player starts a little before the event so that the action itself is not cut off (D-054).
PLAYER_PRE_ROLL_S = 5.0

HalfFilter = Literal[1, 2] | None
# Event key under which the player text is stored: the type and `t_v` the fusion stage gave it.
PlayerKey = tuple[EventType, float]
_PLATE_TIME_TOL_S = 1e-6  # the same plate has the same first-seen time in both files
# The plate type whose names belong to an event type (a card plate gives a foul).
_PLATE_OF_EVENT = {EventType.GOAL: "goal", EventType.FOUL: "card", EventType.SUBST: "subst"}


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
    player_text: str = ""
    corner_text: str = ""

    @property
    def type_text(self) -> str:
        """Category with the status note in brackets, if there is one."""
        return f"{self.type_label} ({self.status_text})" if self.status_text else self.type_label


def edit_mark(event: CurrentEvent) -> str:
    """Text for the "edit" column: added by the user, edited by the user, or empty."""
    if event.origin == "user":
        return ru.MARK_ADDED
    return ru.MARK_EDITED if event.n_edits else ""


def _name_text(read: PlayerRead) -> str:
    return read.name or ru.PLAYER_UNRECOGNIZED


def player_text(names: PlateNames) -> str:
    """The "player" cell of an event from the names read on its plate.

    A goal or a card plate gives the name; a substitution gives "came on X; left Y". A name that
    is not checked against a lineup carries a note (once, at the end); a name that could not be
    read or found in the lineup shows as "not recognized".
    """
    if names.type == "subst":
        verbs = {"in": ru.PLAYER_WENT_ON, "out": ru.PLAYER_WENT_OFF}
        text = "; ".join(f"{verbs.get(p.role, '')} {_name_text(p)}".strip() for p in names.players)
    else:
        text = "; ".join(_name_text(p) for p in names.players)
    unchecked = any(p.status == "unchecked" for p in names.players)
    return text + ru.PLAYER_UNCHECKED_MARK if unchecked else text


def player_texts(
    candidates: Iterable[Candidate], names: Iterable[PlateNames]
) -> dict[PlayerKey, str]:
    """The player text of every event that was built from a plate with read names.

    The key is the type and time of the candidate, which the database row has as well (the
    fusion stage stored them). The first plate of the right kind among the candidate's
    evidence counts.
    """
    by_plate = {n.plate_id: n for n in names}
    texts: dict[PlayerKey, str] = {}
    for candidate in candidates:
        for evidence in candidate.evidence:
            found = by_plate.get(evidence.ref) if evidence.kind == "plate" else None
            # The time check keeps names of an older `plates.json` (plate ids are renumbered when
            # the stage runs again) from landing on the wrong plate.
            if (
                found is not None
                and found.type == _PLATE_OF_EVENT.get(candidate.type)
                and math.isclose(found.t_v, evidence.t_v, abs_tol=_PLATE_TIME_TOL_S)
            ):
                texts[(candidate.type, candidate.t_v)] = player_text(found)
                break
    return texts


def to_row(
    event: CurrentEvent,
    players: Mapping[PlayerKey, str] | None = None,
    corners: Mapping[str, str] | None = None,
) -> EventRow:
    """Format an event for the list; missing fields show as a dash.

    `players` gives the player texts by (type, `t_v`); an event added or moved by the user has
    another key, so it gets no name, and one added by the user never does. `corners` gives the
    text of the corner tags by event id (D-078).
    """
    player = (
        "" if event.origin == "user" or not players else players.get((event.type, event.t_v), "")
    )
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
        player_text=player,
        corner_text=(corners or {}).get(event.event_id, ""),
    )


def to_rows(
    events: Collection[CurrentEvent],
    players: Mapping[PlayerKey, str] | None = None,
    corners: Mapping[str, str] | None = None,
) -> list[EventRow]:
    """`to_row` for every event, order kept."""
    return [to_row(e, players, corners) for e in events]


def _blank(text: str) -> str:
    return "" if text == ru.UNKNOWN else text


def table_records(rows: Sequence[EventRow]) -> list[dict[str, str | float | None]]:
    """Rows of the events table, one dict per event; unknown values are empty.

    Keys are the Russian column names in display order: time, type, half, confidence, source,
    team, player, edit mark. The order of the records is the order of `rows`, so a selected table
    index is an index into `rows`.
    """
    with_corners = any(row.corner_text for row in rows)
    records: list[dict[str, str | float | None]] = [
        {
            ru.COLUMN_TIME: row.time_text,
            ru.COLUMN_TYPE: row.type_text,
            ru.COLUMN_HALF: _blank(row.half_text),
            ru.COLUMN_SCORE: row.score,
            ru.COLUMN_SOURCE: _blank(row.source_text),
            ru.COLUMN_TEAM: _blank(row.team_text),
            ru.COLUMN_PLAYER: row.player_text,
            ru.COLUMN_EDIT: row.edit_text,
        }
        for row in rows
    ]
    if with_corners:  # the column appears only when some corner has tags (D-078)
        for record, row in zip(records, rows, strict=True):
            record[ru.COLUMN_CORNER] = row.corner_text
    return records


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
