"""Export of the current events as a ground-truth file (FR-9).

The file has the layout that `aicoach.eval.schemas.load_ground_truth` reads and that
`data_labels/match01_ground_truth.json` has, so what the analyst checked in the viewer can serve as
a label file for `eval` and `spotting-eval`. The events are the current ones (predictions with the
user's edits applied, deleted ones hidden). The store, `fuse` and the predictions are only read.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog

from aicoach.agent.context import EventFacts, load_context
from aicoach.domain.events import EventType
from aicoach.errors import StorageError
from aicoach.eval.schemas import GroundTruth, GroundTruthEvent, HalfBounds
from aicoach.fusion.schemas import CandidatesResult
from aicoach.fusion.stage import RESULT_FILE as CANDIDATES_FILE
from aicoach.review.log import ReviewLogEntry, read_log
from aicoach.stage_common import write_atomic

# Who is named as the annotator of an exported event: the system for its own predictions, the
# viewer for the events a user added.
ANNOTATOR_SYSTEM = "aicoach"
ANNOTATOR_USER = "ui"
DEFAULT_VIDEO_NAME = "src.mp4"

logger = structlog.get_logger()


@dataclass(frozen=True)
class LabelExport:
    """The ground truth and what was left out: events whose half is unknown (a ground-truth event
    needs one), and, with `reviewed_only`, the number of events that were not reviewed."""

    ground_truth: GroundTruth
    skipped_no_half: int = 0
    skipped_unreviewed: int = 0

    def counts_by_type(self) -> dict[str, int]:
        """Number of exported events per category (only the categories that occur)."""
        return dict(Counter(e.type.value for e in self.ground_truth.events))


ConfirmedRow = tuple[str, EventType, float]


def confirmed_rows(entries: Iterable[ReviewLogEntry]) -> frozenset[ConfirmedRow]:
    """`(id, type, t_v)` of the events whose last decision in the review journal is a confirmation.

    The type and the time go with the id because the journal lies in the match directory and
    the database may be another file with its own numbering of `p:N`: an id alone could confirm
    a different event.
    """
    latest: dict[str, ReviewLogEntry] = {}
    for entry in entries:
        latest[entry.id] = entry
    return frozenset(
        (row_id, entry.type, entry.t_v)
        for row_id, entry in latest.items()
        if entry.action == "confirm"
    )


def is_reviewed(facts: EventFacts, confirmed: frozenset[ConfirmedRow]) -> bool:
    """A person has looked at the event: added it, moved it, changed its type, verified it, or
    confirmed it in the review mode."""
    event = facts.event
    return (
        event.origin == "user"
        or event.verified
        or event.n_edits > 0
        or (event.event_id, event.type, event.t_v) in confirmed
    )


def read_half_bounds(match_dir: Path) -> dict[Literal["1", "2"], HalfBounds]:
    """The half bounds the fusion stage used (`candidates.json`); empty when the file is missing
    or invalid (the file is then exported without `halves`)."""
    try:
        text = (match_dir / CANDIDATES_FILE).read_text(encoding="utf-8")
        bounds = CandidatesResult.model_validate_json(text).half_bounds
    except (OSError, ValueError) as err:
        logger.warning(
            "labels_no_half_bounds", stage="export", reason=type(err).__name__, path=str(match_dir)
        )
        return {}
    return {key: HalfBounds(start=span.start, end=span.end) for key, span in bounds.items()}


def _label(facts: EventFacts, half: Literal[1, 2]) -> GroundTruthEvent:
    event = facts.event
    return GroundTruthEvent(
        type=event.type,
        t_v=event.t_v,
        half=half,
        clock=facts.minute_text,
        team=event.team,
        annotator=ANNOTATOR_USER if event.origin == "user" else ANNOTATOR_SYSTEM,
        note=f"source={event.source or 'user'}; status={event.status}",
    )


def build_ground_truth(
    facts: Sequence[EventFacts],
    *,
    video: str,
    halves: Mapping[Literal["1", "2"], HalfBounds],
    reviewed_only: bool = False,
    confirmed: frozenset[ConfirmedRow] = frozenset(),
) -> LabelExport:
    """Turn current events into a ground-truth file (pure).

    Args:
        facts: The current events with their half and game minute (`MatchContext.events`).
        video: Name of the video file for the `video` field.
        halves: Half bounds for the `halves` field.
        reviewed_only: Keep only the events a person has looked at (see `is_reviewed`).
        confirmed: Rows confirmed in the review journal (see `confirmed_rows`).
    """
    events: list[GroundTruthEvent] = []
    no_half = unreviewed = 0
    for item in facts:
        if reviewed_only and not is_reviewed(item, confirmed):
            unreviewed += 1
        elif item.event.half is None:
            no_half += 1
        else:
            events.append(_label(item, item.event.half))
    events.sort(key=lambda e: (e.t_v, e.type.value))
    truth = GroundTruth(video=video, halves=dict(halves), events=tuple(events))
    return LabelExport(truth, no_half, unreviewed)


def export_labels(
    db_path: Path,
    match_id: str,
    match_dir: Path,
    *,
    video_name: str = DEFAULT_VIDEO_NAME,
    reviewed_only: bool = False,
) -> LabelExport:
    """Read the current state of the match and build its ground-truth file.

    Raises:
        StorageError: The database is missing, SQLite failed, or (with `reviewed_only`) the
            review journal cannot be read.
    """
    context = load_context(db_path, match_id, match_dir)
    confirmed = confirmed_rows(read_log(match_dir)) if reviewed_only else frozenset()
    result = build_ground_truth(
        context.events,
        video=video_name,
        halves=read_half_bounds(match_dir),
        reviewed_only=reviewed_only,
        confirmed=confirmed,
    )
    if result.skipped_no_half:
        logger.warning("labels_events_without_half", stage="export", count=result.skipped_no_half)
    return result


def ground_truth_json(truth: GroundTruth) -> str:
    """The file text: indented JSON, non-ASCII letters kept as they are."""
    return truth.model_dump_json(indent=2) + "\n"


def write_labels(path: Path, truth: GroundTruth) -> None:
    """Write the file atomically, creating its directory.

    Raises:
        StorageError: The file or its directory cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        msg = f"cannot create the directory of {path}: {err}"
        raise StorageError(msg) from err
    write_atomic(path, ground_truth_json(truth))
