"""Connections and schema setup for the event store (stdlib `sqlite3` only)."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import structlog

from aicoach.errors import StorageError
from aicoach.storage.schema import SCHEMA_VERSION, STATEMENTS, VIEWS

BUSY_TIMEOUT_MS = 5000
logger = structlog.get_logger()


@contextmanager
def translate_errors(action: str) -> Iterator[None]:
    """Re-raise `sqlite3.Error` as `StorageError`, keeping the original as the cause."""
    try:
        yield
    except sqlite3.Error as err:
        msg = f"{action}: {err}"
        raise StorageError(msg) from err


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Open the database in WAL mode and close it when the block ends.

    The connection is in autocommit mode (transactions are explicit, see `transaction`),
    has foreign keys on, waits up to 5 s on a locked database, and returns `sqlite3.Row`.

    Raises:
        StorageError: The file cannot be opened or is not a SQLite database.
    """
    with translate_errors(f"cannot open {path}"):
        conn = sqlite3.connect(path, isolation_level=None, timeout=BUSY_TIMEOUT_MS / 1000)
    try:
        with translate_errors(f"cannot configure {path}"):
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA recursive_triggers = ON")  # REPLACE INTO must hit the triggers
        yield conn
    finally:
        conn.close()


def _rollback(conn: sqlite3.Connection) -> None:
    """Roll back if a transaction is open; never raise, so the original error is not hidden."""
    if conn.in_transaction:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            # The connection is unusable anyway; the caller re-raises the original error.
            logger.warning("rollback_failed", stage="storage")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Run the block in one write transaction: commit on success, roll back on any error.

    `BEGIN IMMEDIATE` takes the write lock up front, so checks done inside the block cannot
    be invalidated by another writer before the commit.

    Raises:
        StorageError: A transaction is already open on this connection, or SQLite failed.
    """
    if conn.in_transaction:
        msg = "a transaction is already open on this connection"
        raise StorageError(msg)
    with translate_errors("cannot start a transaction"):
        conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        _rollback(conn)
        raise
    try:
        with translate_errors("cannot commit"):
            conn.execute("COMMIT")
    except StorageError:
        _rollback(conn)  # a failed COMMIT leaves the transaction open
        raise


def init_db(path: Path) -> None:
    """Create the database file and schema if absent; safe to call again.

    Views are recreated on every call, so each call takes the write lock briefly. Call it
    once at the start of a stage or an app session, not on every UI rerun, and before any
    reader uses `events_current` (a database never passed through here may hold old views).

    Raises:
        StorageError: The file is not a SQLite database, or its schema version is not the
            one this code expects.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as conn:
        with translate_errors(f"cannot read schema version of {path}"):
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version not in (0, SCHEMA_VERSION):
            msg = f"{path}: schema version {version}, expected {SCHEMA_VERSION}"
            raise StorageError(msg)
        with transaction(conn), translate_errors(f"cannot create schema in {path}"):
            for statement in STATEMENTS:
                conn.execute(statement)
            for name in reversed(VIEWS):  # dependants first
                conn.execute(f"DROP VIEW IF EXISTS {name}")
            for definition in VIEWS.values():
                conn.execute(definition)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
