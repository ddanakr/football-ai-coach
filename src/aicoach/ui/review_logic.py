"""Pure logic of the review mode: which rows to show, the progress, the next row (D-075).

No Streamlit and no I/O. A row is a current event or a candidate of the review layer; the
journal state (`aicoach.review.log.ReviewState`) says which rows are already decided.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal

from aicoach.agent.context import game_minute, minute_text
from aicoach.domain.events import EventType
from aicoach.i18n import ru
from aicoach.review.log import ReviewState
from aicoach.review.schemas import ReviewCandidate
from aicoach.scoreboard.mapping import clock_at
from aicoach.scoreboard.schemas import OffsetSegment
from aicoach.storage import CurrentEvent
from aicoach.ui.logic import EventFilter

RowKind = Literal["event", "candidate"]
_KIND_ORDER = {"event": 0, "candidate": 1}


@dataclass(frozen=True)
class ReviewRow:
    """One row of the review list. Texts are Russian, `t_v` is kept for the player."""

    row_id: str
    kind: RowKind
    type: EventType
    t_v: float
    half: int | None
    minute_text: str | None
    score: float | None
    source_text: str
    team_text: str
    decided: bool


def _event_row(
    event: CurrentEvent,
    state: ReviewState,
    segments: Sequence[OffsetSegment],
    max_extrapolation_s: float,
) -> ReviewRow:
    minute = game_minute(clock_at(segments, event.t_v, max_extrapolation_s))
    return ReviewRow(
        row_id=event.event_id,
        kind="event",
        type=event.type,
        t_v=event.t_v,
        half=event.half,
        minute_text=minute_text(minute, event.half),
        score=event.score,
        source_text=ru.SOURCE_LABELS.get(event.source, event.source) if event.source else "",
        team_text=ru.TEAM_LABELS[event.team] if event.team else "",
        decided=event.event_id in state.decided,
    )


def _candidate_row(candidate: ReviewCandidate, state: ReviewState) -> ReviewRow:
    return ReviewRow(
        row_id=candidate.id,
        kind="candidate",
        type=candidate.type,
        t_v=candidate.t_v,
        half=candidate.half,
        minute_text=minute_text(candidate.minute, candidate.half),
        score=candidate.score,
        source_text=ru.REVIEW_CANDIDATE_MARK,
        team_text="",
        decided=candidate.id in state.decided,
    )


def build_review_rows(
    events: Sequence[CurrentEvent],
    candidates: Sequence[ReviewCandidate],
    state: ReviewState,
    segments: Sequence[OffsetSegment],
    max_extrapolation_s: float,
    *,
    show_all: bool,
) -> list[ReviewRow]:
    """The rows to show, sorted by time (an event before a candidate at the same second).

    By default: events of the model and the broadcast graphics that are not confirmed yet, and
    candidates not decided yet; events the reviewer added are not shown (they are decided by
    being added). With `show_all` every event is shown, and every candidate except an accepted
    one (the event it created stands for it); decided rows carry `decided=True`.
    """
    rows = [
        _event_row(e, state, segments, max_extrapolation_s)
        for e in events
        if show_all or e.origin == "prediction"
    ]
    rows += [_candidate_row(c, state) for c in candidates if c.id not in state.accepted]
    if not show_all:
        rows = [r for r in rows if not r.decided]
    return sorted(rows, key=lambda r: (r.t_v, _KIND_ORDER[r.kind], r.row_id))


def filter_review_rows(rows: Sequence[ReviewRow], flt: EventFilter) -> list[ReviewRow]:
    """Keep rows of the chosen types and half with `score >= min_score` (rows without a score
    pass the threshold, as in `filter_events`)."""
    return [
        r
        for r in rows
        if r.type in flt.types
        and (flt.half is None or r.half == flt.half)
        and (r.score is None or r.score >= flt.min_score)
    ]


def review_progress(
    events: Sequence[CurrentEvent],
    candidates: Sequence[ReviewCandidate],
    state: ReviewState,
    deleted: Collection[str],
) -> tuple[int, int]:
    """`(checked, total)` rows of the review.

    The rows are all candidates, the events of the model and the graphics that are in the
    catalog, and those the reviewer deleted (`deleted`: ids of the journal, `p:N` only counts).
    Events added by the reviewer are not rows of the review.
    """
    live = {e.event_id for e in events}
    total = (
        {c.id for c in candidates}
        | {e.event_id for e in events if e.origin == "prediction"}
        | {i for i in deleted if i.startswith("p:") and i not in live}
    )
    return len(total & state.decided), len(total)


def next_row_id(ids: Sequence[str], current: str | None) -> str | None:
    """The row after `current` in `ids` (the order on the screen).

    The first row when nothing is chosen or `current` is not in the list; None when `current`
    is the last row or the list is empty.
    """
    if current in ids:
        position = ids.index(current) + 1
        return ids[position] if position < len(ids) else None
    return ids[0] if ids else None
