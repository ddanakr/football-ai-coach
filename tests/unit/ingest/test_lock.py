"""One ingest per match directory: the pid lock file."""

import os
from pathlib import Path

import pytest

from aicoach.errors import IngestError
from aicoach.ingest.lock import LOCK_FILE, ingest_lock, pid_alive, read_lock_pid, running_pid


def alive(_pid: int) -> bool:
    return True


def dead(_pid: int) -> bool:
    return False


@pytest.mark.unit
def test_ingest_lock_writes_the_pid_and_removes_the_file_at_the_end(tmp_path: Path) -> None:
    with ingest_lock(tmp_path, pid=4321, is_alive=alive):
        assert read_lock_pid(tmp_path) == 4321
    assert not (tmp_path / LOCK_FILE).exists()


@pytest.mark.unit
def test_ingest_lock_is_released_after_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), ingest_lock(tmp_path, pid=1, is_alive=alive):
        raise RuntimeError
    assert not (tmp_path / LOCK_FILE).exists()


@pytest.mark.unit
def test_ingest_lock_held_by_a_live_process_refuses(tmp_path: Path) -> None:
    (tmp_path / LOCK_FILE).write_text("777")

    with pytest.raises(IngestError, match="already being processed.*777"):  # noqa: SIM117
        with ingest_lock(tmp_path, pid=1, is_alive=alive):
            pass
    assert read_lock_pid(tmp_path) == 777  # the other process keeps its lock


@pytest.mark.unit
@pytest.mark.parametrize("content", ["777", "", "garbage"])
def test_ingest_lock_left_by_a_dead_process_is_taken_over(tmp_path: Path, content: str) -> None:
    (tmp_path / LOCK_FILE).write_text(content)

    with ingest_lock(tmp_path, pid=5, is_alive=dead if content else alive):
        assert read_lock_pid(tmp_path) == 5
    assert not (tmp_path / LOCK_FILE).exists()


@pytest.mark.unit
def test_ingest_lock_does_not_remove_a_lock_someone_else_took(tmp_path: Path) -> None:
    with ingest_lock(tmp_path, pid=5, is_alive=alive):
        (tmp_path / LOCK_FILE).write_text("6")
    assert read_lock_pid(tmp_path) == 6


@pytest.mark.unit
def test_ingest_lock_in_a_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="cannot create the lock"):  # noqa: SIM117
        with ingest_lock(tmp_path / "nope", pid=1, is_alive=alive):
            pass


@pytest.mark.unit
def test_running_pid_reports_only_live_holders(tmp_path: Path) -> None:
    assert running_pid(tmp_path, alive) is None
    (tmp_path / LOCK_FILE).write_text("12")
    assert running_pid(tmp_path, alive) == 12
    assert running_pid(tmp_path, dead) is None


@pytest.mark.unit
def test_pid_alive_own_process_is_alive_and_impossible_ids_are_not() -> None:
    assert pid_alive(os.getpid())
    assert not pid_alive(0)
    assert not pid_alive(-3)
    assert not pid_alive(2**31 - 1)
