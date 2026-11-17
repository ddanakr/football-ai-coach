"""The review journal `review_log.jsonl`: what the reviewer decided about each row (D-075).

One JSON line per decision: UTC time, the id of the row, the action, and the time and type of the
row (so that a time window can be measured without the database). Append only. The journal is
separate from the `edits` journal of the store: confirming an event and rejecting a candidate
change no event, so they are written here only; adding a candidate and deleting an event also go
through `edits` (`ui/review_panel.py`).
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from aicoach.domain.events import EventType
from aicoach.errors import StorageError

LOG_FILE = "review_log.jsonl"
Action = Literal["accept", "reject", "confirm", "delete"]
# accept and reject decide a candidate (`r:N`); confirm and delete decide an event.
CANDIDATE_ACTIONS: frozenset[Action] = frozenset({"accept", "reject"})
VideoTime = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ReviewLogEntry(BaseModel):
    """One decision. `event_id` is the event that an `accept` created (`e:N`)."""

    model_config = ConfigDict(frozen=True)

    ts: AwareDatetime
    id: str = Field(pattern=r"^[rpe]:[0-9]+$")
    action: Action
    t_v: VideoTime
    type: EventType
    event_id: str | None = Field(default=None, pattern=r"^e:[0-9]+$")

    @model_validator(mode="after")
    def _id_fits_action(self) -> Self:
        candidate_row = self.id.startswith("r:")
        if candidate_row != (self.action in CANDIDATE_ACTIONS):
            msg = f"action {self.action!r} does not fit the row {self.id!r}"
            raise ValueError(msg)
        if (self.event_id is not None) != (self.action == "accept"):
            msg = "event_id goes with the action 'accept' and with no other"
            raise ValueError(msg)
        return self


def make_entry(
    row_id: str,
    action: Action,
    t_v: float,
    event_type: EventType,
    *,
    event_id: str | None = None,
    now: datetime | None = None,
) -> ReviewLogEntry:
    """A decision stamped with the current UTC time (or `now`, for tests).

    Raises:
        ValueError: The row id, the action and the event id do not fit each other.
    """
    return ReviewLogEntry(
        ts=now or datetime.now(UTC),
        id=row_id,
        action=action,
        t_v=t_v,
        type=event_type,
        event_id=event_id,
    )


def append_entry(match_dir: Path, entry: ReviewLogEntry) -> None:
    """Append one line to the journal of the match.

    Raises:
        StorageError: The file cannot be written.
    """
    path = match_dir / LOG_FILE
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")
    except OSError as err:
        msg = f"cannot write {path}: {type(err).__name__}"
        raise StorageError(msg) from err


def read_log(match_dir: Path) -> list[ReviewLogEntry]:
    """All decisions in file order; a missing file is an empty journal.

    Raises:
        StorageError: The file cannot be read or has a line that is not a valid decision (the
            message names the line number, not its content).
    """
    path = match_dir / LOG_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError) as err:
        msg = f"cannot read {path}: {type(err).__name__}"
        raise StorageError(msg) from err
    entries: list[ReviewLogEntry] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(ReviewLogEntry.model_validate_json(line))
        except ValidationError as err:
            msg = f"{path}: line {number} is not a valid decision"
            raise StorageError(msg) from err
    return entries


@dataclass(frozen=True)
class ReviewState:
    """What the journal says about the rows now.

    `decided` are the ids that need no more attention; `accepted` the candidates whose created
    event is still in the catalog (the event row stands for them).
    """

    decided: frozenset[str]
    accepted: frozenset[str]


def review_state(entries: Iterable[ReviewLogEntry], live_event_ids: Iterable[str]) -> ReviewState:
    """Fold the journal into the current state; the last decision on a row counts.

    A rejection and a confirmation always stand. An accepted candidate stands while the event it
    created is alive, so undoing the addition brings the candidate back. A deletion stands while
    the event is gone, so undoing it brings the event back for review.
    """
    live = set(live_event_ids)
    latest: dict[str, ReviewLogEntry] = {}
    for entry in entries:
        latest[entry.id] = entry
    decided: set[str] = set()
    accepted: set[str] = set()
    for row_id, entry in latest.items():
        if entry.action in {"reject", "confirm"}:
            decided.add(row_id)
        elif entry.action == "accept" and entry.event_id in live:
            decided.add(row_id)
            accepted.add(row_id)
        elif entry.action == "delete" and row_id not in live:
            decided.add(row_id)
    return ReviewState(frozenset(decided), frozenset(accepted))


def entries_for_layer(
    entries: Iterable[ReviewLogEntry], candidates: Iterable[tuple[str, EventType, float]]
) -> list[ReviewLogEntry]:
    """The journal without the decisions on candidates that no longer exist.

    Ids `r:N` follow the time order of the layer, so a rebuilt layer (another T-DEED file, other
    fusion parameters) may give `r:5` to another detection. A decision on a candidate stands only
    if the layer still has a candidate with that id, type and time; decisions on events are kept.

    Args:
        entries: The journal in file order.
        candidates: `(id, type, t_v)` of the candidates of the current layer.
    """
    current = {(row_id, kind, t_v) for row_id, kind, t_v in candidates}
    return [e for e in entries if not e.id.startswith("r:") or (e.id, e.type, e.t_v) in current]


def deleted_ids(entries: Sequence[ReviewLogEntry]) -> frozenset[str]:
    """Ids of events whose last decision is a deletion."""
    latest: dict[str, Action] = {}
    for entry in entries:
        latest[entry.id] = entry.action
    return frozenset(row_id for row_id, action in latest.items() if action == "delete")
