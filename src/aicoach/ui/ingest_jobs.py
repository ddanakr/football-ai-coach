"""Background ingest for the viewer: start `cli ingest` as a separate process and read its state.

The page never waits for the ingest: the process writes `status.json` and a log file in the
match directory, and the page reads them every few seconds. A second start of the same match
is refused while its process runs (the ingest lock, D-058).
"""

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal

from aicoach.fusion.stage import DB_FILE
from aicoach.i18n import ru
from aicoach.ingest.download import VIDEO_FILE
from aicoach.ingest.lock import AliveCheck, pid_alive, running_pid
from aicoach.ingest.stage import STAGE, STEPS, match_directory
from aicoach.ingest.url import canonical_url, parse_youtube_url
from aicoach.status import StatusFile
from aicoach.ui.launch import AppArgs

LOG_FILE = "ingest.log"
Phase = Literal["running", "waiting", "done", "failed", "gone"]


# `subprocess.Popen` or a fake. Any: Popen's keyword arguments are many and mixed, and only
# `pid` and `poll()` of the result are used.
Spawn = Callable[..., Any]


@dataclass(frozen=True)
class IngestJob:
    """One match to ingest: where it goes and where its log is."""

    video_id: str
    match_dir: Path
    templates: Path | None = None

    @property
    def log_path(self) -> Path:
        """The process output (`ingest.log` in the match directory)."""
        return self.match_dir / LOG_FILE

    def view_args(self) -> AppArgs:
        """Arguments that open this match in the viewer."""
        return AppArgs(
            db=self.match_dir / DB_FILE,
            match_id=self.match_dir.name,
            youtube_id=self.video_id,
            match_dir=self.match_dir,
            video=self.match_dir / VIDEO_FILE,
        )


@dataclass(frozen=True)
class StepRow:
    """One line of the step list in the sidebar."""

    label: str
    state: str
    progress: float
    error: str | None


def plan_job(url: str, data_root: Path, templates: Path | None = None) -> IngestJob:
    """The job for a link: video id and match directory (`<data_root>/<video id>`).

    Raises:
        IngestError: The link is not a YouTube video link.
    """
    video_id = parse_youtube_url(url)
    return IngestJob(video_id, match_directory(data_root, video_id), templates)


def build_ingest_command(job: IngestJob, python: str | None = None) -> list[str]:
    """`python -m aicoach.cli ingest <plain link> --data-root <dir> [--templates <dir>]`."""
    command = [
        python or sys.executable,
        "-m",
        "aicoach.cli",
        "ingest",
        canonical_url(job.video_id),
        "--data-root",
        str(job.match_dir.parent),
    ]
    if job.templates is not None:
        command += ["--templates", str(job.templates)]
    return command


def busy_pid(job: IngestJob, is_alive: AliveCheck = pid_alive) -> int | None:
    """The id of an ingest process already working on this match, if any."""
    return running_pid(job.match_dir, is_alive)


def _detach_options() -> dict[str, Any]:
    """Keep the ingest running when the viewer's console gets Ctrl+C or is closed.

    Any: the options differ by platform and go straight to Popen.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def launch(job: IngestJob, spawn: Spawn | None = None, python: str | None = None) -> Any:
    """Start the ingest process with its output appended to `ingest.log`; returns the process
    (`subprocess.Popen`, or what a fake returns).

    Raises:
        OSError: The directory, the log file or the process cannot be created.
    """
    job.match_dir.mkdir(parents=True, exist_ok=True)
    log: IO[bytes]
    with job.log_path.open("ab") as log:
        # Our own fixed arguments, no shell; the link was rebuilt from the checked id.
        return (spawn or subprocess.Popen)(
            build_ingest_command(job, python),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            **_detach_options(),
        )


def step_rows(status: StatusFile) -> list[StepRow]:
    """The ingest steps in order, with their state from `status.json` ("pending" if absent)."""
    rows = []
    for name in STEPS:
        entry = status.stages.get(name)
        state = "pending" if entry is None else entry.state
        rows.append(
            StepRow(
                label=ru.INGEST_STEP_LABELS[name],
                state=ru.INGEST_STATE_LABELS[state],
                progress=0.0 if entry is None else entry.progress,
                error=None if entry is None else entry.error,
            )
        )
    return rows


def job_phase(status: StatusFile, process_alive: bool) -> Phase:
    """What the job is doing, from the `ingest` entry and whether its process still runs.

    While the process runs the job is "running", whatever the file still says from an earlier
    ingest. Without a process, a missing or `running` entry means it ended without reporting
    (killed, or failed before its first status): "gone", the log tells why.
    """
    if process_alive:
        return "running"
    entry = status.stages.get(STAGE)
    if entry is None or entry.state == "running":
        return "gone"
    return entry.state
