"""The `ingest` command: a YouTube link to events in the database (FR-1, FR-2).

Steps, each with its own entry in `status.json` of the match directory: `download` (yt-dlp,
frame size check), `audio` (16 kHz mono WAV), `scoreboard`, `plates`, `plate_names` (player names
read from the plate crops, skipped when Tesseract is missing), `transcribe` (the commentary of
the whole match, skipped without a key of the provider), `tdeed` (the model output: run here when
a video card is there, otherwise brought from the notebook), `fuse` and `review_candidates` (the
extra model shots for the review mode, D-075), `teams` (team names and kit colours, D-078) and
`corner_tags` (the tags of every corner from its clip, D-078); the last three are optional: a
failure is recorded and the ingest goes on. The `ingest` entry tells the whole state. Every step
skips what is ready, so a repeated ingest continues where the last one stopped. Without the T-DEED
file and without a video card the command stops with the `tdeed` step `waiting` (D-058, D-080).
"""

import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from aicoach.clips import Probe, Runner
from aicoach.errors import AicoachError, IngestError, SpottingError
from aicoach.fusion.schemas import CandidatesResult
from aicoach.ingest.download import VIDEO_FILE, Downloader, YtDlpDownloader
from aicoach.ingest.lock import ingest_lock
from aicoach.ingest.media import AUDIO_FILE, check_frame_size, extract_audio
from aicoach.ingest.url import canonical_url, parse_youtube_url
from aicoach.logging_setup import stage_logger
from aicoach.pipeline import Stages, run_step
from aicoach.plates.config import PlatesConfig
from aicoach.scoreboard.config import ScoreboardConfig
from aicoach.scoreboard.stage import TEMPLATES_DIR
from aicoach.scoreboard.video import probe_video
from aicoach.spotting.schemas import load_tdeed_file
from aicoach.stage_common import Logger, Progress, elapsed_ms, record_failure, set_status
from aicoach.tdeed.device import cuda_available

STAGE = "ingest"
STEPS = (
    "download",
    "audio",
    "scoreboard",
    "plates",
    "plate_names",
    "transcribe",
    "tdeed",
    "fuse",
    "review_candidates",
    "teams",
    "corner_tags",
)
TDEED_STEP = "tdeed"
TDEED_PATTERN = "tdeed_*.json"
EXIT_WAITING_TDEED = 3  # exit code of the command while the T-DEED file is missing
_MATCH_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}[A-Za-z0-9_-]|[A-Za-z0-9]")
_WINDOWS_RESERVED = re.compile(r"(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?", re.IGNORECASE)
_T = TypeVar("_T")


@dataclass(frozen=True)
class IngestTools:
    """Outside effects of the command; the real ones by default, fakes in tests."""

    downloader: Downloader = field(default_factory=YtDlpDownloader)
    ffmpeg: Runner | None = None  # None: run the real FFmpeg
    probe: Probe = probe_video
    stages: Stages = field(default_factory=Stages)
    cuda_available: Callable[[], bool] = cuda_available  # False: wait for the notebook's file


@dataclass(frozen=True)
class IngestResult:
    """Where the match lies and how far it got; `fusion` is None while T-DEED is missing."""

    video_id: str
    match_id: str
    match_dir: Path
    fusion: CandidatesResult | None

    @property
    def waiting_for_tdeed(self) -> bool:
        """True when the command stopped before fusion for want of the T-DEED file."""
        return self.fusion is None


def match_directory(data_root: Path, video_id: str, match_id: str | None = None) -> Path:
    """`<data_root>/<match_id>`; without a match id the (already checked) video id names it.

    A video id may start with `-` or `_`, so the default needs no extra check. An explicit
    match id must be a plain name: it must not escape the data root, end with a dot or be a
    reserved Windows device name.

    Raises:
        IngestError: The match id is not a plain name.
    """
    if not match_id:
        return data_root / video_id
    if not _MATCH_ID.fullmatch(match_id) or _WINDOWS_RESERVED.fullmatch(match_id):
        msg = (
            "match id: 1-64 letters, digits, '_', '-' or '.', starting with a letter or digit, "
            "not ending with '.', not a reserved name such as NUL or COM1"
        )
        raise IngestError(msg)
    return data_root / match_id


def find_tdeed_file(match_dir: Path) -> Path | None:
    """The T-DEED output in the match directory (`tdeed_*.json`); None when there is none.

    Raises:
        IngestError: More than one such file: which one to use is the user's choice.
    """
    found = sorted(p for p in match_dir.glob(TDEED_PATTERN) if p.is_file())
    if len(found) > 1:
        names = ", ".join(p.name for p in found)
        msg = f"several T-DEED files in {match_dir} ({names}): keep exactly one"
        raise IngestError(msg)
    return found[0] if found else None


def ensure_templates(match_dir: Path, source: Path | None) -> None:
    """Make sure the scoreboard digit templates are in the match directory.

    They are cut from a broadcast by calibration (D-030); a new match of the same broadcast
    design reuses those of an earlier match (`source`), copied once.

    Raises:
        IngestError: No templates here and no usable `source`.
    """
    target = match_dir / TEMPLATES_DIR
    if target.is_dir():
        return
    if source is None:
        msg = (
            f"no scoreboard templates in {target}: pass --templates <scoreboard_templates of a "
            "calibrated match of the same broadcast>, or calibrate this match "
            "(scoreboard-calibrate)"
        )
        raise IngestError(msg)
    if not (source / "calibration.json").is_file():
        msg = f"{source} is not a scoreboard_templates directory (no calibration.json)"
        raise IngestError(msg)
    partial = target.with_name(target.name + ".part")
    try:
        shutil.rmtree(partial, ignore_errors=True)
        shutil.copytree(source, partial)
        partial.replace(target)
    except OSError as err:
        shutil.rmtree(partial, ignore_errors=True)
        msg = f"cannot copy the templates to {target}: {err}"
        raise IngestError(msg) from err


def _own_stage(
    match_dir: Path, name: str, log: Logger, call: Callable[[], _T], *, reuse: bool = False
) -> _T:
    """Run a step that has no stage module of its own and keep its `status.json` entry.

    `reuse` says the step will find its result ready: `running` is then not written, so a
    `done` entry keeps the times of the work that made the result, and any other entry gets
    `done` without times (the check is not the work).
    """
    started = time.perf_counter()
    if not reuse:
        set_status(match_dir, name, "running", 0.0, log)
    try:
        result = call()
    except BaseException as err:
        record_failure(match_dir, name, 0.0, err, log, started)
        raise
    set_status(match_dir, name, "done", 1.0, log)
    log.info("step_done", step=name, duration_ms=elapsed_ms(started))
    return result


def _download(video_id: str, match_dir: Path, tools: IngestTools, log: Logger) -> Path:
    """Download unless `src.mp4` is there (it appears only after a complete merge), then check
    the frame size, also of a file downloaded before."""
    video = match_dir / VIDEO_FILE
    if video.is_file():
        log.info("download_skipped", reason="src.mp4 exists")
    else:
        progress = Progress(match_dir, "download", log)
        tools.downloader.download(canonical_url(video_id), match_dir, progress.update)
    try:
        info = tools.probe(video)
    except AicoachError as err:
        msg = (
            f"cannot read {video} ({err}); if the file is broken or cut short, "
            "delete it and run again"
        )
        raise IngestError(msg) from err
    check_frame_size(info)
    return video


def _audio(video: Path, match_dir: Path, tools: IngestTools, log: Logger) -> Path:
    audio = match_dir / AUDIO_FILE
    if audio.is_file():
        log.info("audio_skipped", reason="audio.wav exists")
        return audio
    return extract_audio(video, audio, tools.ffmpeg)


def _find_tdeed(match_dir: Path, log: Logger) -> Path | None:
    """Look for the T-DEED file; "waiting" is not an error, so it is not logged as a failure."""
    found = find_tdeed_file(match_dir)
    log.info("tdeed_lookup", found=found is not None)
    return found


def _check_tdeed(path: Path) -> Path:
    """A broken T-DEED file is an ingest error with the file named, not a silent skip."""
    try:
        load_tdeed_file(path)
    except SpottingError as err:
        msg = f"the T-DEED file is not valid: {err}"
        raise IngestError(msg) from err
    return path


@dataclass(frozen=True)
class _Job:
    video_id: str
    match_id: str
    match_dir: Path
    templates: Path | None
    hw_accel: bool
    tools: IngestTools
    log: Logger

    def step(self, name: str, call: Callable[[], _T], *, timed: bool = True) -> _T:
        """One step under the `ingest` entry; its progress is the share of finished steps."""
        index = STEPS.index(name) if name in STEPS else 0  # "templates" comes before all steps
        return run_step(
            self.match_dir, STAGE, name, index / len(STEPS), call, self.log, timed=timed
        )


def _optional(job: _Job, name: str, call: Callable[[], object]) -> None:
    """A step after the fusion that the events do not depend on: the layer of extra model shots
    for the review mode (D-075), the team names and kits and the corner tags (D-078).

    The events are in the database already, so a failure here must not stop the ingest: it is
    logged and recorded in the entry of the step in `status.json`, and the ingest goes on.
    Ctrl+C and other interrupts are not caught: they pass through (the caller marks the ingest
    as failed).
    """
    try:
        call()
    except Exception as err:  # an optional step: the reason is logged and recorded, not hidden
        reason = str(err) or type(err).__name__
        job.log.warning("optional_step_failed", step=name, reason=reason)
        set_status(job.match_dir, name, "failed", 0.0, job.log, error=reason)


def _run(job: _Job) -> IngestResult:
    d, log, tools = job.match_dir, job.log, job.tools
    job.step("templates", lambda: ensure_templates(d, job.templates))
    video = job.step(
        "download",
        lambda: _own_stage(
            d,
            "download",
            log,
            lambda: _download(job.video_id, d, tools, log),
            reuse=(d / VIDEO_FILE).is_file(),
        ),
    )
    job.step(
        "audio",
        lambda: _own_stage(
            d, "audio", log, lambda: _audio(video, d, tools, log), reuse=(d / AUDIO_FILE).is_file()
        ),
    )
    job.step(
        "scoreboard",
        lambda: tools.stages.scoreboard(
            video, job.match_id, ScoreboardConfig(hw_accel=job.hw_accel), force=False
        ),
    )
    job.step(
        "plates",
        lambda: tools.stages.plates(
            video, job.match_id, PlatesConfig(hw_accel=job.hw_accel), force=False
        ),
    )
    # Optional: without Tesseract the step records "skipped" and the ingest goes on. It comes
    # before the T-DEED wait, so the names are there while the model output is still missing.
    job.step("plate_names", lambda: tools.stages.plate_names(d, job.match_id, force=False))
    # Optional too: without a key the step records "skipped". A failure (service, budget) stops the
    # ingest loudly; the answers of the chunks done are kept, and the next run goes on from them.
    job.step("transcribe", lambda: tools.stages.transcribe(d, job.match_id, force=False))
    # The lookup is a step of its own, so a failure (two files) is recorded; "waiting" below
    # replaces its "done" entry.
    tdeed = job.step(
        TDEED_STEP,
        lambda: _own_stage(d, TDEED_STEP, log, lambda: _find_tdeed(d, log)),
        timed=False,
    )
    # With a video card here the model runs now (D-080); without one the notebook brings the file.
    ran_here = False
    if tdeed is None and tools.cuda_available():
        job.step(TDEED_STEP, lambda: tools.stages.tdeed(d, job.match_id, device="auto"))
        tdeed, ran_here = _find_tdeed(d, log), True
    if tdeed is None:
        progress = STEPS.index(TDEED_STEP) / len(STEPS)
        set_status(d, TDEED_STEP, "waiting", 0.0, log)
        set_status(d, STAGE, "waiting", progress, log)
        log.warning("waiting_for_tdeed", pattern=TDEED_PATTERN)
        return IngestResult(job.video_id, job.match_id, d, None)
    # `reuse` after a run here: the times the model run just wrote stay in the entry.
    job.step(
        TDEED_STEP,
        lambda: _own_stage(d, TDEED_STEP, log, lambda: _check_tdeed(tdeed), reuse=ran_here),
        timed=False,
    )
    fusion = job.step(
        "fuse",
        lambda: tools.stages.fusion(d, job.match_id, tdeed, None, force=False, discard_edits=False),
    )
    stages = tools.stages
    job.step(
        "review_candidates",
        lambda: _optional(
            job,
            "review_candidates",
            lambda: stages.review_candidates(d, job.match_id, tdeed, force=False),
        ),
    )
    # Paid steps last: the free ones are done by then. Without a key of the video model both are
    # recorded as skipped, not failed.
    job.step(
        "teams",
        lambda: _optional(job, "teams", lambda: stages.teams(d, job.match_id, force=False)),
    )
    job.step(
        "corner_tags",
        lambda: _optional(
            job, "corner_tags", lambda: stages.corner_tags(d, job.match_id, force=False)
        ),
    )
    set_status(d, STAGE, "done", 1.0, log)
    return IngestResult(job.video_id, job.match_id, d, fusion)


def run_ingest(
    url: str,
    data_root: Path,
    *,
    match_id: str | None = None,
    templates: Path | None = None,
    hw_accel: bool = True,
    tools: IngestTools | None = None,
) -> IngestResult:
    """Download a match from YouTube and run all stages up to the events in the database.

    Args:
        url: YouTube link of the match (watch, youtu.be or live form).
        data_root: Directory that holds one directory per match.
        match_id: Name of the match directory and id in the database; default the video id.
        templates: `scoreboard_templates` of a calibrated match of the same broadcast, copied
            when the match directory has none.
        hw_accel: Decode on the video card when there is one.
        tools: Downloader, FFmpeg runner, probe and stage functions (injected in tests).

    Returns:
        The result; `waiting_for_tdeed` is True when the T-DEED file is still missing.

    Raises:
        IngestError: Bad link (checked before anything is written), another ingest of the same
            match is running, no templates, YouTube refused, wrong frame size, no audio.
        AicoachError: A later stage failed; the message names it.
    """
    video_id = parse_youtube_url(url)
    match_dir = match_directory(data_root, video_id, match_id)
    if templates is None and not (match_dir / TEMPLATES_DIR).is_dir():
        ensure_templates(match_dir, None)  # raises the "no templates" error, creates nothing
    try:
        match_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        msg = f"cannot create the match directory {match_dir}: {err}"
        raise IngestError(msg) from err
    job_id = match_id or video_id
    log = stage_logger(STAGE, job_id)
    started = time.perf_counter()
    with ingest_lock(match_dir):
        job = _Job(video_id, job_id, match_dir, templates, hw_accel, tools or IngestTools(), log)
        result = _run(job)
    log.info("ingest_finished", waiting=result.waiting_for_tdeed, duration_ms=elapsed_ms(started))
    return result
