"""Reads and writes of the event store: predictions, the edits journal, current events.

Nothing here updates or deletes a row: new facts are appended, and the current state is
derived by the view `events_current` (see `aicoach.storage.schema`).
"""

import math
import sqlite3
from collections.abc import Collection, Sequence
from typing import Literal

from pydantic import ValidationError

from aicoach.domain.events import EventType
from aicoach.errors import StorageError
from aicoach.storage.db import transaction, translate_errors
from aicoach.storage.models import CurrentEvent, EditRecord, NewEdit, NewPrediction

_INSERT_PREDICTION = """
INSERT INTO predictions (match_id, type, t_v, half, score, status, source, team, run_id)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
RETURNING id
"""
_INSERT_EDIT = """
INSERT INTO edits (match_id, action, target, type, t_v, undo_of, author)
VALUES (?, ?, ?, ?, ?, ?, ?)
RETURNING *
"""


def add_predictions(
    conn: sqlite3.Connection,
    match_id: str,
    run_id: str,
    predictions: Sequence[NewPrediction],
) -> list[int]:
    """Store the predictions of one run in one transaction: all of them or none.

    A run is the complete set of events of the match and is written by the fusion stage
    only. The new run replaces the previous one in `events_current` (D-028); the older
    predictions stay in the table. An empty batch writes nothing, so it does not start a run.

    Args:
        conn: Open connection from `connect`.
        match_id: Match the predictions belong to.
        run_id: Identifier of the run that produced them; new for this match.
        predictions: The whole run; may be empty.

    Returns:
        The ids of the new predictions in input order (the event id is `p:<id>`).

    Raises:
        StorageError: Empty `match_id` or `run_id`, `run_id` already used for this match,
            or the database rejected a row; in that case nothing from the batch is stored.
    """
    if not match_id or not run_id:
        msg = "match_id and run_id must not be empty"
        raise StorageError(msg)
    if not predictions:
        return []
    ids: list[int] = []
    with transaction(conn), translate_errors("cannot store predictions"):
        # Inside the write transaction, so two writers cannot both pass this check.
        if conn.execute(
            "SELECT 1 FROM predictions WHERE match_id = ? AND run_id = ? LIMIT 1",
            (match_id, run_id),
        ).fetchone():
            msg = f"run {run_id!r} already has predictions for match {match_id!r}"
            raise StorageError(msg)
        for p in predictions:
            row = conn.execute(
                _INSERT_PREDICTION,
                (
                    match_id,
                    str(p.type),  # StrEnum -> its value; plain str passes through
                    p.t_v,
                    p.half,
                    p.score,
                    p.status,
                    p.source,
                    p.team,
                    run_id,
                ),
            ).fetchone()
            ids.append(int(row["id"]))
    return ids


def latest_run_id(conn: sqlite3.Connection, match_id: str) -> str | None:
    """Return the `run_id` of the latest run of a match, or `None` if it has no predictions.

    The latest run is the one of the prediction with the largest id; it is the run that
    `events_current` shows.

    Raises:
        StorageError: The database failed.
    """
    with translate_errors("cannot read the latest run"):
        row = conn.execute(
            "SELECT run_id FROM latest_runs WHERE match_id = ?", (match_id,)
        ).fetchone()
    return None if row is None else str(row["run_id"])


def _require_live_event(conn: sqlite3.Connection, match_id: str, target: str) -> None:
    """Check that `target` is an existing, not deleted event of the match."""
    row = conn.execute(
        "SELECT deleted FROM events_current WHERE match_id = ? AND event_id = ?",
        (match_id, target),
    ).fetchone()
    if row is None:
        msg = f"event {target!r} does not exist in match {match_id!r}"
        raise StorageError(msg)
    if row["deleted"]:
        msg = f"event {target!r} is deleted; undo the delete first"
        raise StorageError(msg)


def add_edit(
    conn: sqlite3.Connection,
    match_id: str,
    action: Literal["add", "move", "retype", "delete", "verify"],
    *,
    author: str,
    target: str | None = None,
    type: EventType | None = None,
    t_v: float | None = None,
) -> EditRecord:
    """Append one edit to the journal.

    Required fields per action: `add` needs `type` and `t_v` (the new event gets the id
    `e:<edit id>`, see `EditRecord.event_id`); `move` needs `target` and `t_v`; `retype`
    needs `target` and `type`; `delete` and `verify` need `target`. A `target` is `p:<id>`
    or `e:<id>` and must be an existing event of this match that is not deleted. Undoing is
    a separate function, `undo_edit`.

    Raises:
        StorageError: Wrong or missing fields, an invalid time, an unknown or deleted target
            (including a target in another match), or a database failure.
    """
    try:
        edit = NewEdit(
            match_id=match_id, action=action, author=author, target=target, type=type, t_v=t_v
        )
    except ValidationError as err:
        msg = f"invalid edit: {err}"
        raise StorageError(msg) from err
    with transaction(conn), translate_errors("cannot store the edit"):
        if edit.target is not None:
            _require_live_event(conn, edit.match_id, edit.target)
        row = conn.execute(
            _INSERT_EDIT,
            (
                edit.match_id,
                edit.action,
                edit.target,
                edit.type.value if edit.type else None,
                edit.t_v,
                None,
                edit.author,
            ),
        ).fetchone()
    return EditRecord.model_validate(dict(row))


def undo_edit(conn: sqlite3.Connection, edit_id: int, author: str) -> EditRecord:
    """Cancel an edit by appending an `undo` record; the cancelled edit stays in the journal.

    Undoing a `delete` brings the event back; undoing an `add` removes the added event.

    Returns:
        The new `undo` record.

    Raises:
        StorageError: Empty `author`, unknown `edit_id`, the edit is itself an `undo`, or it
            was already undone.
    """
    if not author:
        msg = "author must not be empty"
        raise StorageError(msg)
    with transaction(conn), translate_errors("cannot undo the edit"):
        edit = conn.execute(
            "SELECT match_id, action FROM edits WHERE id = ?", (edit_id,)
        ).fetchone()
        if edit is None:
            msg = f"edit {edit_id} does not exist"
            raise StorageError(msg)
        if edit["action"] == "undo":
            msg = f"edit {edit_id} is an undo and cannot be undone"
            raise StorageError(msg)
        if conn.execute("SELECT 1 FROM edits WHERE undo_of = ?", (edit_id,)).fetchone():
            msg = f"edit {edit_id} is already undone"
            raise StorageError(msg)
        row = conn.execute(
            _INSERT_EDIT, (edit["match_id"], "undo", None, None, None, edit_id, author)
        ).fetchone()
    return EditRecord.model_validate(dict(row))


def get_current_events(
    conn: sqlite3.Connection,
    match_id: str,
    *,
    types: Collection[EventType] | None = None,
    include_deleted: bool = False,
    min_score: float | None = None,
) -> list[CurrentEvent]:
    """Read the current events of one match, ordered by time.

    Args:
        conn: Open connection from `connect`.
        match_id: Match to read; an unknown match gives an empty list.
        types: Keep only these classes; `None` keeps all, an empty collection keeps none.
        include_deleted: Also return events hidden by a delete (`deleted=True`).
        min_score: Keep events with `score >= min_score`. Events without a score (user
            additions, plate events) are kept: the threshold is for model scores only.

    Raises:
        StorageError: `min_score` is NaN or infinite, or the database failed.
    """
    clauses = ["match_id = ?"]
    params: list[str | float] = [match_id]
    if types is not None:
        if not types:
            return []
        clauses.append(f"type IN ({', '.join('?' * len(types))})")
        try:
            params.extend(EventType(t).value for t in types)
        except ValueError as err:
            msg = f"unknown event type in filter: {err}"
            raise StorageError(msg) from err
    if not include_deleted:
        clauses.append("deleted = 0")
    if min_score is not None:
        if not math.isfinite(min_score):
            msg = f"min_score must be finite, got {min_score}"
            raise StorageError(msg)
        clauses.append("(score IS NULL OR score >= ?)")
        params.append(min_score)
    # Only fixed fragments are joined here; every value is bound through `params`.
    where = " AND ".join(clauses)
    sql = (
        f"SELECT * FROM events_current WHERE {where} "  # noqa: S608
        "ORDER BY t_v, CAST(substr(event_id, 3) AS INTEGER), event_id"
    )
    with translate_errors("cannot read events"):
        rows = conn.execute(sql, params).fetchall()
    return [CurrentEvent.model_validate(dict(row)) for row in rows]


def count_effective_edits(conn: sqlite3.Connection, match_id: str) -> int:
    """Count the edits of a match that are in force: the "edits per match" metric.

    Edits cancelled by an undo and the `undo` records themselves are not counted, nor are
    edits that belong to an added event whose `add` was undone, nor edits of predictions
    from earlier runs (they are not applied, D-028). `add` edits count in every run.
    """
    with translate_errors("cannot count edits"):
        row = conn.execute(
            """
            SELECT COUNT(*) FROM effective_edits AS x
            WHERE x.match_id = :m
              AND (x.action = 'add'
                   OR x.target IN (SELECT event_id FROM events_current WHERE match_id = :m))
            """,
            {"m": match_id},
        ).fetchone()
    return int(row[0])
