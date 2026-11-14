"""Background ingest from the viewer: job planning, command, launch, step list, phase."""

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from aicoach.errors import IngestError
from aicoach.i18n import ru
from aicoach.ingest.lock import LOCK_FILE
from aicoach.ingest.stage import STAGE, STEPS
from aicoach.status import StageStatus, StatusFile
from aicoach.ui.ingest_jobs import (
    LOG_FILE,
    IngestJob,
    build_ingest_command,
    busy_pid,
    job_phase,
    launch,
    plan_job,
    step_rows,
)

VID = "abcDEF_-123"  # invented id


@pytest.mark.unit
def test_plan_job_puts_the_match_under_the_video_id(tmp_path: Path) -> None:
    job = plan_job(f"https://youtu.be/{VID}?t=9", tmp_path, tmp_path / "tpl")

    assert job == IngestJob(VID, tmp_path / VID, tmp_path / "tpl")
    assert job.log_path == tmp_path / VID / LOG_FILE


@pytest.mark.unit
@pytest.mark.parametrize("url", ["", "https://example.com/watch?v=abcDEF_-123", "abc"])
def test_plan_job_bad_link_raises_and_creates_nothing(tmp_path: Path, url: str) -> None:
    with pytest.raises(IngestError):
        plan_job(url, tmp_path / "data")
    assert not (tmp_path / "data").exists()


@pytest.mark.unit
def test_view_args_open_the_database_and_video_of_the_match(tmp_path: Path) -> None:
    args = IngestJob(VID, tmp_path / VID).view_args()

    assert args.db == tmp_path / VID / "aicoach.db"
    assert (args.match_id, args.youtube_id) == (VID, VID)
    assert args.match_dir == tmp_path / VID
    assert args.video == tmp_path / VID / "src.mp4"


@pytest.mark.unit
def test_build_ingest_command_uses_the_plain_link_and_the_templates(tmp_path: Path) -> None:
    job = IngestJob(VID, tmp_path / VID, tmp_path / "tpl")

    command = build_ingest_command(job, python="py")

    assert command == [
        "py",
        "-m",
        "aicoach.cli",
        "ingest",
        f"https://www.youtube.com/watch?v={VID}",
        "--data-root",
        str(tmp_path),
        "--templates",
        str(tmp_path / "tpl"),
    ]
    assert "--templates" not in build_ingest_command(IngestJob(VID, tmp_path / VID))


@pytest.mark.unit
def test_launch_appends_the_output_to_the_log_and_does_not_wait(tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    # Any: Popen keyword arguments are mixed.
    def spawn(args: list[str], **kwargs: Any) -> str:
        kwargs["stdout"].write(b"started\n")
        calls.append((args, kwargs))
        return "process"

    job = IngestJob(VID, tmp_path / "data" / VID)
    (tmp_path / "data" / VID).mkdir(parents=True)
    job.log_path.write_bytes(b"old\n")

    assert launch(job, spawn) == "process"

    args, kwargs = calls[0]
    assert args[3] == "ingest"
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.STDOUT
    assert "creationflags" in kwargs or "start_new_session" in kwargs
    assert job.log_path.read_bytes() == b"old\nstarted\n"


@pytest.mark.unit
def test_launch_spawn_failure_raises_oserror(tmp_path: Path) -> None:
    def spawn(_args: list[str], **_kwargs: object) -> object:
        raise FileNotFoundError

    with pytest.raises(OSError):
        launch(IngestJob(VID, tmp_path / VID), spawn)


@pytest.mark.unit
def test_busy_pid_sees_a_live_lock_holder_only(tmp_path: Path) -> None:
    job = IngestJob(VID, tmp_path)
    assert busy_pid(job) is None
    (tmp_path / LOCK_FILE).write_text(str(os.getpid()))
    assert busy_pid(job) == os.getpid()
    assert busy_pid(job, lambda _pid: False) is None


@pytest.mark.unit
def test_step_rows_follow_the_steps_and_mark_missing_ones_pending() -> None:
    status = StatusFile(
        stages={
            "download": StageStatus(state="done", progress=1.0),
            "scoreboard": StageStatus(state="running", progress=0.5),
            "tdeed": StageStatus(state="waiting", progress=0.0),
            "fuse": StageStatus(state="failed", progress=0.0, error="boom"),
            "run": StageStatus(state="done", progress=1.0),  # not an ingest step
        }
    )

    rows = step_rows(status)

    assert [r.label for r in rows] == [ru.INGEST_STEP_LABELS[s] for s in STEPS]
    assert [r.state for r in rows] == [
        ru.INGEST_STATE_LABELS[s]
        for s in ("done", "pending", "running", "pending", "waiting", "failed")
    ]
    assert rows[2].progress == 0.5
    assert rows[5].error == "boom"


@pytest.mark.unit
def test_step_rows_of_an_empty_status_are_all_pending() -> None:
    assert {r.state for r in step_rows(StatusFile())} == {ru.INGEST_STATE_LABELS["pending"]}


def status_of(state: str | None) -> StatusFile:
    if state is None:
        return StatusFile()
    return StatusFile(stages={STAGE: StageStatus.model_validate({"state": state, "progress": 0})})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("state", "alive", "phase"),
    [
        (None, True, "running"),
        ("done", True, "running"),  # a new run of a finished match before its first status
        ("running", True, "running"),
        ("running", False, "gone"),
        (None, False, "gone"),
        ("waiting", False, "waiting"),
        ("done", False, "done"),
        ("failed", False, "failed"),
    ],
)
def test_job_phase_from_status_and_process(state: str | None, alive: bool, phase: str) -> None:
    assert job_phase(status_of(state), alive) == phase
