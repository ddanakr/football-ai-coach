"""Tests of the helpers shared by the stages."""

import time
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from aicoach.errors import StorageError
from aicoach.logging_setup import stage_logger
from aicoach.stage_common import Progress, elapsed_ms, set_status, write_atomic
from aicoach.status import read_status


@pytest.mark.unit
def test_write_atomic_replaces_the_file_and_leaves_no_temporary(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_text("old", encoding="utf-8")

    write_atomic(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


@pytest.mark.unit
def test_write_atomic_failing_rename_raises_storage_error_and_keeps_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "result.json"
    target.write_text("old", encoding="utf-8")

    original = Path.replace

    def failing_replace(self: Path, dest: Path) -> Path:
        if self.name == "result.json.tmp":
            msg = "disk full"
            raise OSError(msg)
        return original(self, dest)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(StorageError):
        write_atomic(target, "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


@pytest.mark.unit
def test_write_atomic_missing_directory_raises_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError):
        write_atomic(tmp_path / "missing_dir" / "result.json", "new")


@pytest.mark.unit
def test_progress_writes_status_only_after_a_noticeable_step(tmp_path: Path) -> None:
    log = stage_logger("demo", "m")
    progress = Progress(tmp_path, "demo", log)

    progress.update(0.001)
    assert "demo" not in read_status(tmp_path).stages
    progress.update(0.5)

    entry = read_status(tmp_path).stages["demo"]
    assert (entry.state, entry.progress) == ("running", 0.5)
    assert progress.value == 0.5


@pytest.mark.unit
def test_set_status_failing_write_is_logged_and_not_raised(tmp_path: Path) -> None:
    log = stage_logger("demo", "m")

    with capture_logs() as logs:
        set_status(tmp_path / "no_such_dir", "demo", "running", 0.1, log)

    assert [entry["event"] for entry in logs] == ["status_write_failed"]


@pytest.mark.unit
def test_elapsed_ms_is_a_non_negative_int() -> None:
    value = elapsed_ms(time.perf_counter())

    assert isinstance(value, int)
    assert value >= 0
