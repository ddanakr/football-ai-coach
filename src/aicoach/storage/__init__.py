"""SQLite event store: immutable predictions, an append-only edits journal, current view."""

from aicoach.storage.db import connect, init_db, transaction
from aicoach.storage.models import CurrentEvent, EditRecord, NewEdit, NewPrediction
from aicoach.storage.repository import (
    add_edit,
    add_predictions,
    count_effective_edits,
    get_current_events,
    latest_run_id,
    undo_edit,
    undo_last_edit,
)

__all__ = [
    "CurrentEvent",
    "EditRecord",
    "NewEdit",
    "NewPrediction",
    "add_edit",
    "add_predictions",
    "connect",
    "count_effective_edits",
    "get_current_events",
    "init_db",
    "latest_run_id",
    "transaction",
    "undo_edit",
    "undo_last_edit",
]
