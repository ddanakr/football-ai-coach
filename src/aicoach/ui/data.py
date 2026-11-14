"""Reading and writing for the viewer, through the storage layer only (no SQL here)."""

from functools import lru_cache
from pathlib import Path

import structlog
from pydantic import ValidationError

from aicoach.errors import StorageError
from aicoach.fusion.schemas import CandidatesResult
from aicoach.fusion.stage import RESULT_FILE as CANDIDATES_FILE
from aicoach.plate_names.schemas import PlateNamesResult
from aicoach.plate_names.stage import RESULT_FILE as PLATE_NAMES_FILE
from aicoach.scoreboard.schemas import ScoreboardResult
from aicoach.scoreboard.stage import RESULT_FILE as SCOREBOARD_FILE
from aicoach.storage import (
    CurrentEvent,
    EditRecord,
    add_edit,
    connect,
    count_effective_edits,
    get_current_events,
    undo_last_edit,
)
from aicoach.ui.edits import EditPlan, TimeBounds
from aicoach.ui.logic import PlayerKey, player_texts

# Who the journal names as the author of an edit made in the viewer.
EDIT_AUTHOR = "ui"

logger = structlog.get_logger()


def _require_db(db_path: Path) -> None:
    """A missing file is refused: opening it would create an empty database."""
    if not db_path.is_file():
        msg = f"database {db_path} does not exist"
        raise StorageError(msg)


def load_events(db_path: Path, match_id: str) -> list[CurrentEvent]:
    """Current events of the match (deleted ones hidden), ordered by time.

    Nothing is written. A missing file is refused here, because opening it would create an
    empty database.

    Raises:
        StorageError: The file does not exist, is not a database of this project, or SQLite
            failed.
    """
    _require_db(db_path)
    with connect(db_path) as conn:
        return get_current_events(conn, match_id)


def count_edits(db_path: Path, match_id: str) -> int:
    """Number of edits of the match that are in force (what "undo" can still take back).

    Raises:
        StorageError: The file does not exist or SQLite failed.
    """
    _require_db(db_path)
    with connect(db_path) as conn:
        return count_effective_edits(conn, match_id)


def apply_edit(db_path: Path, match_id: str, plan: EditPlan) -> EditRecord:
    """Append the planned edit to the journal; predictions are not touched.

    Raises:
        StorageError: The file does not exist, the target is not an event of the latest run
            (it may have been replaced by a new fusion run), or SQLite failed.
    """
    _require_db(db_path)
    with connect(db_path) as conn:
        return add_edit(
            conn,
            match_id,
            plan.action,
            author=EDIT_AUTHOR,
            target=plan.target,
            type=plan.type,
            t_v=plan.t_v,
        )


def undo_last(db_path: Path, match_id: str) -> EditRecord:
    """Cancel the newest edit in force with an `undo` record.

    Raises:
        StorageError: The file does not exist, there is no edit to undo, or SQLite failed.
    """
    _require_db(db_path)
    with connect(db_path) as conn:
        return undo_last_edit(conn, match_id, EDIT_AUTHOR)


@lru_cache(maxsize=4)
def _read_bounds(
    candidates: Path,
    board: Path,
    stamp: tuple[int, int],  # noqa: ARG001 - only part of the cache key
) -> TimeBounds | None:
    """Parse the two files; `stamp` (their modification times) only keys the cache."""
    try:
        halves = CandidatesResult.model_validate_json(candidates.read_text(encoding="utf-8"))
        duration = ScoreboardResult.model_validate_json(
            board.read_text(encoding="utf-8")
        ).stats.video_duration_s
    except (OSError, ValidationError) as err:
        logger.warning(
            "time_bounds_unavailable",
            stage="ui",
            reason=type(err).__name__,
            path=str(candidates.parent),
        )
        return None
    return TimeBounds(
        halves={1: halves.half_bounds["1"], 2: halves.half_bounds["2"]}, video_duration_s=duration
    )


def load_time_bounds(match_dir: Path) -> TimeBounds | None:
    """Half bounds (from `candidates.json`, as the fusion stage applied them) and video length
    (from `scoreboard.json`); `None` if either file is missing or invalid.

    The result is cached until one of the files changes, because the page reruns on every
    click and `scoreboard.json` is large.
    """
    candidates, board = match_dir / CANDIDATES_FILE, match_dir / SCOREBOARD_FILE
    try:
        stamp = (candidates.stat().st_mtime_ns, board.stat().st_mtime_ns)
    except OSError:
        logger.warning("time_bounds_unavailable", stage="ui", reason="missing", path=str(match_dir))
        return None
    return _read_bounds(candidates, board, stamp)


@lru_cache(maxsize=4)
def _read_players(
    candidates: Path,
    names: Path,
    stamp: tuple[int, int],  # noqa: ARG001 - only part of the cache key
) -> dict[PlayerKey, str]:
    """Parse the two files; `stamp` (their modification times) only keys the cache."""
    try:
        events = CandidatesResult.model_validate_json(candidates.read_text(encoding="utf-8"))
        read = PlateNamesResult.model_validate_json(names.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as err:
        logger.warning(
            "player_names_unavailable",
            stage="ui",
            reason=type(err).__name__,
            path=str(candidates.parent),
        )
        return {}
    return player_texts(events.candidates, read.names)


def load_player_texts(match_dir: Path) -> dict[PlayerKey, str]:
    """The player texts of the match's events (`candidates.json` joined with
    `plate_names.json` by the plate id); empty when either file is missing or invalid.

    The names are optional: a match without them (no Tesseract, step not run) simply has an
    empty "player" column. The result is cached until one of the files changes.
    """
    candidates, names = match_dir / CANDIDATES_FILE, match_dir / PLATE_NAMES_FILE
    try:
        stamp = (candidates.stat().st_mtime_ns, names.stat().st_mtime_ns)
    except OSError:
        return {}
    return _read_players(candidates, names, stamp)
