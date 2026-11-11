"""Reading the events for the viewer, through the storage layer only."""

from pathlib import Path

from aicoach.errors import StorageError
from aicoach.storage import CurrentEvent, connect, get_current_events


def load_events(db_path: Path, match_id: str) -> list[CurrentEvent]:
    """Current events of the match (deleted ones hidden), ordered by time.

    The database is opened for reading only in effect: nothing is written. A missing file is
    refused here, because opening it would create an empty database.

    Raises:
        StorageError: The file does not exist, is not a database of this project, or SQLite
            failed.
    """
    if not db_path.is_file():
        msg = f"database {db_path} does not exist"
        raise StorageError(msg)
    with connect(db_path) as conn:
        return get_current_events(conn, match_id)
