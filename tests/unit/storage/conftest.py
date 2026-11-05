"""Fixtures for the event store tests: a temporary database and an open connection."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from aicoach.storage import connect, init_db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Path of an initialized, empty database in a temporary directory."""
    path = tmp_path / "events.db"
    init_db(path)
    return path


@pytest.fixture
def conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    """An open connection to the temporary database."""
    with connect(db_path) as connection:
        yield connection
