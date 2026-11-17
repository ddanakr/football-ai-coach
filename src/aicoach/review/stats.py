"""`review-stats`: how much of a video window the reviewer went through, from the journal (D-075).

The measurement for the time experiment: the rows of the review in a window of video time (the
frozen predictions and the candidates), the decisions made on rows of that window, the time from
the first to the last of them, and the number of each action. Pure core, thin reading of files.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from aicoach.errors import ConfigError, EvalError, StorageError
from aicoach.review.log import ReviewLogEntry, read_log
from aicoach.review.stage import load_review_candidates
from aicoach.storage import connect, get_latest_predictions
from aicoach.ui.edits import EditInputError, parse_mmss

_ACTIONS = ("accept", "reject", "confirm", "delete")


@dataclass(frozen=True)
class ReviewStats:
    """The measurement of one window (ends included).

    `rows` are the review rows in the window, `rows_decided` the distinct rows with at least one
    decision, `decisions` all journal lines of the window; `seconds` runs from the first to the
    last of them (0 for one line, None for none).
    """

    start_s: float
    end_s: float
    rows: int
    rows_decided: int
    decisions: int
    seconds: float | None
    accepted: int
    rejected: int
    confirmed: int
    deleted: int


def parse_window(start_text: str, end_text: str) -> tuple[float, float]:
    """The window `[start, end]` in video seconds from two `mm:ss` texts.

    Raises:
        ConfigError: A text is not `mm:ss`, or the start is after the end.
    """
    try:
        start, end = parse_mmss(start_text), parse_mmss(end_text)
    except EditInputError as err:
        msg = "the window needs mm:ss times (--from and --to), e.g. 12:30"
        raise ConfigError(msg) from err
    if start > end:
        msg = f"--from {start_text} is after --to {end_text}"
        raise ConfigError(msg)
    return start, end


def review_stats(
    entries: Sequence[ReviewLogEntry], row_times: Iterable[float], start_s: float, end_s: float
) -> ReviewStats:
    """Measure the window `[start_s, end_s]` (ends included).

    Args:
        entries: The journal in file order.
        row_times: Video time of every row of the review (candidates and frozen predictions).
        start_s: Start of the window, video seconds.
        end_s: End of the window.

    Raises:
        EvalError: The window is not a finite non-negative range with `start_s <= end_s`.
    """
    if not (math.isfinite(start_s) and math.isfinite(end_s)) or not 0 <= start_s <= end_s:
        msg = f"the window must be 0 <= start <= end, got {start_s} and {end_s}"
        raise EvalError(msg)
    inside = [e for e in entries if start_s <= e.t_v <= end_s]
    stamps = sorted(e.ts for e in inside)
    count = {action: sum(e.action == action for e in inside) for action in _ACTIONS}
    return ReviewStats(
        start_s=start_s,
        end_s=end_s,
        rows=sum(start_s <= t <= end_s for t in row_times),
        rows_decided=len({e.id for e in inside}),
        decisions=len(inside),
        seconds=(stamps[-1] - stamps[0]).total_seconds() if stamps else None,
        accepted=count["accept"],
        rejected=count["reject"],
        confirmed=count["confirm"],
        deleted=count["delete"],
    )


def format_stats(stats: ReviewStats) -> str:
    """The measurement as `key: value` lines."""
    seconds = "none" if stats.seconds is None else f"{stats.seconds:.1f}"
    return (
        f"window: {stats.start_s:.0f}-{stats.end_s:.0f} s of video\n"
        f"rows in the window: {stats.rows}\n"
        f"rows with a decision: {stats.rows_decided}\n"
        f"decisions (journal lines): {stats.decisions}\n"
        f"seconds from the first to the last decision: {seconds}\n"
        f"added (candidates accepted): {stats.accepted}\n"
        f"rejected (candidates): {stats.rejected}\n"
        f"confirmed (events): {stats.confirmed}\n"
        f"deleted (events): {stats.deleted}\n"
    )


def run_review_stats(
    match_dir: Path, match_id: str, db_path: Path, start_s: float, end_s: float
) -> ReviewStats:
    """Read the journal, the layer and the frozen predictions of the match and measure a window.

    Raises:
        StorageError: The database or the journal cannot be read.
        EvalError: The window is not valid.
    """
    if not db_path.is_file():
        msg = f"database {db_path} does not exist"
        raise StorageError(msg)
    layer = load_review_candidates(match_dir)
    with connect(db_path) as conn:
        predictions = get_latest_predictions(conn, match_id)
    times = [p.t_v for p in predictions] + (
        [] if layer is None else [c.t_v for c in layer.candidates]
    )
    return review_stats(read_log(match_dir), times, start_s, end_s)
