"""The `run` command: scoreboard, plates and fusion one after another.

The model (T-DEED) is not part of it: it runs separately in Colab and its output file is passed
in. Every stage keeps its own cache, so a repeated run only recomputes what changed. A failing
stage stops the run; `status.json` gets a `run` entry that names the stage and the reason.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from aicoach.corner_tags.schemas import CornerTagsFile
from aicoach.corner_tags.stage import run_corner_tags
from aicoach.errors import AicoachError
from aicoach.fusion.config import FuseConfig
from aicoach.fusion.schemas import CandidatesResult
from aicoach.fusion.stage import run_fuse
from aicoach.logging_setup import stage_logger
from aicoach.plate_names.schemas import PlateNamesResult
from aicoach.plate_names.stage import run_plate_names
from aicoach.plates.config import PlatesConfig
from aicoach.plates.schemas import PlatesResult
from aicoach.plates.stage import run_plates
from aicoach.review.schemas import ReviewCandidatesResult
from aicoach.review.stage import run_review_candidates
from aicoach.scoreboard.config import ScoreboardConfig
from aicoach.scoreboard.schemas import ScoreboardResult
from aicoach.scoreboard.stage import run_scoreboard
from aicoach.speech.stage import run_transcribe
from aicoach.speech.transcript import TranscriptResult
from aicoach.spotting.schemas import load_tdeed_file
from aicoach.stage_common import Logger, elapsed_ms, finish_timing, set_status
from aicoach.status import utc_now_iso
from aicoach.tdeed.stage import TdeedResult, run_tdeed
from aicoach.teams.stage import run_teams
from aicoach.teams.store import TeamsResult

STAGE = "run"
_STEPS = ("scoreboard", "plates", "fuse")
_T = TypeVar("_T")


@dataclass(frozen=True)
class Stages:
    """The stage functions; the real ones by default, fakes in tests (`run` uses three of them,
    `ingest` also `plate_names`, `transcribe`, `tdeed`, `review_candidates`, `teams` and
    `corner_tags`).

    Typed as `Callable[..., X]` on purpose: the stages take different arguments and the
    pipeline calls them by keyword, so a precise type would only repeat their signatures.
    """

    scoreboard: Callable[..., ScoreboardResult] = run_scoreboard
    plates: Callable[..., PlatesResult] = run_plates
    fusion: Callable[..., CandidatesResult] = run_fuse
    plate_names: Callable[..., PlateNamesResult | None] = run_plate_names
    transcribe: Callable[..., TranscriptResult | None] = run_transcribe
    review_candidates: Callable[..., ReviewCandidatesResult] = run_review_candidates
    teams: Callable[..., TeamsResult] = run_teams
    corner_tags: Callable[..., CornerTagsFile | None] = run_corner_tags
    tdeed: Callable[..., TdeedResult] = run_tdeed


@dataclass(frozen=True)
class PipelineResult:
    """The results of the three stages."""

    scoreboard: ScoreboardResult
    plates: PlatesResult
    fusion: CandidatesResult


def run_step(
    match_dir: Path,
    entry: str,
    name: str,
    progress: float,
    call: Callable[[], _T],
    log: Logger,
    *,
    timed: bool = True,
) -> _T:
    """Run one stage under the `entry` record of `status.json` (`run`, `ingest`).

    The entry is set to `running` with `progress` first. After the call, success or failure,
    the start, the end (UTC) and the duration of the step are added to the `name` entry of
    `status.json` (a step that wrote no entry, or only `skipped` or `waiting`, gets none).
    `timed=False` leaves the step without times (a lookup is not work worth timing).
    A failure names the stage and is raised again: our own errors keep their type, any other
    exception becomes an `AicoachError` (Ctrl+C and other interrupts pass through unchanged).
    """
    set_status(match_dir, entry, "running", progress, log)
    started_at, started = utc_now_iso(), time.perf_counter()
    try:
        result = call()
    except BaseException as err:  # any failure, even Ctrl+C, must not leave "running" behind
        reason = f"stage {name} failed: {str(err) or type(err).__name__}"
        log.exception("pipeline_failed", failed_stage=name, reason=reason)
        set_status(match_dir, entry, "failed", progress, log, error=reason)
        if timed:
            finish_timing(match_dir, name, started_at, started, log)
        if isinstance(err, AicoachError):
            err.args = (reason,)  # same object and type, the message names the stage
            raise
        if isinstance(err, Exception):
            # A foreign error (OpenCV, OS) must reach the command line as a handled error too.
            raise AicoachError(reason) from err
        raise
    if timed:
        finish_timing(match_dir, name, started_at, started, log)
    return result


def _step(match_dir: Path, index: int, call: Callable[[], _T], log: Logger) -> _T:
    """One stage of `run`, reported in the `run` entry of `status.json`."""
    return run_step(match_dir, STAGE, _STEPS[index], index / len(_STEPS), call, log)


def run_pipeline(
    video: Path,
    match_id: str,
    *,
    predictions_path: Path | None = None,
    fuse_config: FuseConfig | None = None,
    force: bool = False,
    hw_accel: bool = True,
    discard_edits: bool = False,
    stages: Stages | None = None,
) -> PipelineResult:
    """Run scoreboard, plates and fusion in this order and stop at the first failure.

    Args:
        video: The match video; all results go to its directory.
        match_id: Match id for logs and the database.
        predictions_path: T-DEED output for the fusion stage; None means "no model" mode.
        fuse_config: Fusion parameters; default `FuseConfig()`.
        force: Recompute every stage (reads the video twice, slow); otherwise caches decide.
        hw_accel: Decode on the video card when there is one (`--cpu` turns it off).
        discard_edits: Let the fusion stage start a new run although the match has user edits.
        stages: The stage functions (injected in tests).

    Raises:
        SpottingError: `predictions_path` is missing or broken (checked before any stage).
        AicoachError: the first failing stage's error, with the stage name in the message.
    """
    if predictions_path is not None:
        load_tdeed_file(predictions_path)  # a bad file fails now, not after two long stages
    stages = stages or Stages()
    log = stage_logger(STAGE, match_id)
    match_dir = video.parent
    started = time.perf_counter()
    board = _step(
        match_dir,
        0,
        lambda: stages.scoreboard(
            video, match_id, ScoreboardConfig(hw_accel=hw_accel), force=force
        ),
        log,
    )
    plates = _step(
        match_dir,
        1,
        lambda: stages.plates(video, match_id, PlatesConfig(hw_accel=hw_accel), force=force),
        log,
    )
    fusion = _step(
        match_dir,
        2,
        lambda: stages.fusion(
            match_dir,
            match_id,
            predictions_path,
            fuse_config,
            force=force,
            discard_edits=discard_edits,
        ),
        log,
    )
    set_status(match_dir, STAGE, "done", 1.0, log)
    log.info("pipeline_done", duration_ms=elapsed_ms(started))
    return PipelineResult(board, plates, fusion)
