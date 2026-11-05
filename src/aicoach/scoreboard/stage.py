"""The scoreboard stage: read the video once, write `scoreboard.json` and `status.json`.

Everything is written next to the video (the match directory). Digit templates and the board
reference live in `scoreboard_templates/` of that directory and are never part of the repository.
The side effects (video, disk, status, logs) are here; the logic is in the pure modules.
"""

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import structlog
from pydantic import TypeAdapter, ValidationError

from aicoach.errors import ConfigError, ScoreboardParseError, StorageError
from aicoach.logging_setup import stage_logger
from aicoach.scoreboard.assemble import RawReading, assemble_result
from aicoach.scoreboard.board import load_reference
from aicoach.scoreboard.calibrate import (
    CalibrationReport,
    apply_labels,
    build_clusters,
    collect,
    read_calibrated_mode,
    save_board_reference,
    save_workdir,
)
from aicoach.scoreboard.config import ScoreboardConfig
from aicoach.scoreboard.glyphs import TemplateSet, templates_from_dir
from aicoach.scoreboard.imaging import BOARD_ROI, ImageParams
from aicoach.scoreboard.reader import ReaderContext, read_frame
from aicoach.scoreboard.schemas import DecodeMode, ScoreboardResult
from aicoach.scoreboard.video import (
    Frame,
    SampledReader,
    VideoInfo,
    frame_step,
    probe_video,
    sample_count,
)
from aicoach.status import State, write_status

Logger = structlog.typing.FilteringBoundLogger
STAGE = "scoreboard"
RESULT_FILE = "scoreboard.json"
TEMPLATES_DIR = "scoreboard_templates"
DEFAULT_CALIBRATION_STEP_S = 5.0
_PROGRESS_STEP = 0.02  # write status.json after this much progress
_LABELS = TypeAdapter(dict[str, dict[str, str]])


class FrameSource(Protocol):
    """What the stage needs from a frame reader (the real one is `SampledReader`)."""

    frames_decoded: int
    decode_mode: DecodeMode

    def __iter__(self) -> Iterator[tuple[int, Frame]]:
        """Yield `(sample_index, frame)`."""
        ...


SourceFactory = Callable[[Path, int, bool], FrameSource]


def duration_warning(info: VideoInfo, frames_decoded: int, tol_s: float) -> str | None:
    """A warning text if the decodable length differs from the metadata by more than `tol_s`."""
    diff_s = (frames_decoded - info.frame_count) / info.fps
    if abs(diff_s) <= tol_s:
        return None
    return (
        f"metadata says {info.frame_count} frames ({info.duration_s:.1f} s) but "
        f"{frames_decoded} decoded: the file is damaged or truncated (difference {diff_s:+.1f} s)"
    )


def load_context(templates_dir: Path, params: ImageParams) -> tuple[ReaderContext, DecodeMode]:
    """Load the board reference, the digit templates and the decode mode of the calibration.

    Raises:
        ScoreboardParseError: something is missing; the message says how to calibrate.
    """
    mode = read_calibrated_mode(templates_dir)
    reference = load_reference(templates_dir)
    score = templates_from_dir(templates_dir / "score", params)
    clock = templates_from_dir(templates_dir / "clock", params)
    added: TemplateSet | None = None
    if any((templates_dir / "added").glob("*.png")):
        added = templates_from_dir(templates_dir / "added", params)
    return ReaderContext(reference, score, clock, params, added), mode


def cache_key(video: Path, templates_dir: Path, config: ScoreboardConfig, mode: DecodeMode) -> str:
    """A hash of everything that changes the result: the video file, the parameters, the
    decode mode and the contents of the templates directory."""
    digest = hashlib.sha256()
    stat = video.stat()
    digest.update(f"{video.name}|{stat.st_size}|{stat.st_mtime_ns}|{mode}".encode())
    digest.update(config.model_dump_json().encode())
    for path in sorted(templates_dir.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(templates_dir).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_cached(path: Path, key: str, log: Logger) -> ScoreboardResult | None:
    """The stored result if it parses and was made with the same inputs, otherwise None."""
    try:
        result = ScoreboardResult.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValidationError, ValueError) as err:
        log.warning("cache_unreadable", path=str(path), reason=type(err).__name__)
        return None
    return result if result.cache_key == key else None


def _write_atomic(path: Path, text: str) -> None:
    """Write to a temporary file and rename it, so a failure keeps the old file intact."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError as err:
        tmp.unlink(missing_ok=True)
        msg = f"cannot write {path}"
        raise StorageError(msg) from err


@dataclass
class Progress:
    """Writes status.json when progress moved enough; remembers the last value for errors."""

    match_dir: Path
    log: Logger
    last_written: float = 0.0
    value: float = 0.0

    def update(self, value: float) -> None:
        self.value = value
        if value - self.last_written >= _PROGRESS_STEP:
            _set_status(self.match_dir, "running", value, self.log)
            self.last_written = value


def _set_status(
    match_dir: Path, state: State, progress: float, log: Logger, error: str | None = None
) -> None:
    """Write status.json; a failure (for example the UI holds the file open) is logged and
    never stops the stage, because the status is only a progress report."""
    try:
        write_status(match_dir, STAGE, state, progress, error=error)
    except StorageError:
        log.warning("status_write_failed", state=state)


def read_samples(
    source: FrameSource,
    info: VideoInfo,
    step: int,
    ctx: ReaderContext,
    expected_mode: DecodeMode,
    progress: Progress,
) -> tuple[list[float], list[RawReading]]:
    """Read every sampled frame. `t_v` of sample `k` is `k * step / fps` (frame index / fps).

    Raises:
        ScoreboardParseError: the video is decoded in another mode than the calibration was
            made in (colours differ slightly between modes).
    """
    total = max(1, sample_count(info.frame_count, step))
    t_v: list[float] = []
    readings: list[RawReading] = []
    for k, frame in source:
        if not t_v and source.decode_mode != expected_mode:
            msg = (
                f"decode mode {source.decode_mode} differs from the calibration "
                f"({expected_mode}); calibrate again on this machine or use the same mode"
            )
            raise ScoreboardParseError(msg)
        t_v.append(k * step / info.fps)
        readings.append(read_frame(frame, ctx))
        progress.update(min(1.0, (k + 1) / total))
    return t_v, readings


def run_scoreboard(
    video: Path,
    match_id: str,
    config: ScoreboardConfig | None = None,
    force: bool = False,
    source_factory: SourceFactory = SampledReader,
) -> ScoreboardResult:
    """Run the stage on `video`; the result goes to `scoreboard.json` beside it.

    A valid earlier result (same video, parameters, decode mode and templates) is returned
    without reading the video unless `force` is set. Progress and failure go to `status.json`.

    On any failure `status.json` says `failed` with the reason and the error is raised again;
    an earlier `scoreboard.json` stays as it was.

    Raises:
        ScoreboardParseError: no templates (the message says how to calibrate), decode mode
            differs from the calibration, or the video cannot be read.
    """
    config = config or ScoreboardConfig()
    log = stage_logger(STAGE, match_id)
    match_dir = video.parent
    progress = Progress(match_dir, log)
    started = time.perf_counter()
    try:
        result, from_cache = _run(video, match_dir, config, force, source_factory, progress, log)
    except BaseException as err:  # any failure, even Ctrl+C, must not leave "running" behind
        _record_failure(match_dir, progress.value, err, log, started)
        raise
    _set_status(match_dir, "done", 1.0, log)
    event = "stage_skipped" if from_cache else "stage_done"
    log.info(event, n_samples=result.stats.n_samples, duration_ms=_ms(started))
    return result


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _record_failure(
    match_dir: Path, progress: float, err: BaseException, log: Logger, started: float
) -> None:
    """Log the failure and mark the stage as failed; a failing status write must not hide the
    original error. The previous `scoreboard.json` is never touched here."""
    reason = str(err) or type(err).__name__
    log.exception("stage_failed", reason=reason, duration_ms=_ms(started))
    _set_status(match_dir, "failed", progress, log, error=reason)


def _run(
    video: Path,
    match_dir: Path,
    config: ScoreboardConfig,
    force: bool,
    source_factory: SourceFactory,
    progress: Progress,
    log: Logger,
) -> tuple[ScoreboardResult, bool]:
    """Returns the result and whether it came from the cache."""
    info = probe_video(video)
    templates_dir = match_dir / TEMPLATES_DIR
    ctx, mode = load_context(templates_dir, config.image)
    key = cache_key(video, templates_dir, config, mode)
    out_path = match_dir / RESULT_FILE
    if not force:
        cached = _load_cached(out_path, key, log)
        if cached is not None:
            return cached, True
    _set_status(match_dir, "running", 0.0, log)
    step = frame_step(info.fps, config.sample_fps)
    source = source_factory(video, step, config.hw_accel)
    t_v, readings = read_samples(source, info, step, ctx, mode, progress)
    warning = duration_warning(info, source.frames_decoded, config.duration_tol_s)
    if warning is not None:
        log.warning("duration_mismatch", detail=warning)
    result = assemble_result(
        video.name,
        source.decode_mode,
        info.fps / step,
        BOARD_ROI,
        t_v,
        readings,
        config.timeline,
        info.duration_s,
        warning,
    ).model_copy(update={"cache_key": key})
    _write_atomic(out_path, result.model_dump_json())
    return result, False


@dataclass(frozen=True)
class CalibrationSummary:
    """What a calibration pass found."""

    n_frames: int
    n_board_frames: int
    decode_mode: DecodeMode
    clusters: dict[str, int]


def run_calibration(
    video: Path,
    workdir: Path,
    match_id: str,
    config: ScoreboardConfig | None = None,
    step_s: float = DEFAULT_CALIBRATION_STEP_S,
    source_factory: SourceFactory = SampledReader,
) -> CalibrationSummary:
    """Calibration pass: sample the video every `step_s` seconds, save the board reference in
    the match directory, and write clusters and contact sheets to `workdir` for labelling.

    Raises:
        ScoreboardParseError: the video cannot be read or no frame shows the scoreboard.
    """
    if not (step_s > 0 and math.isfinite(step_s)):
        msg = f"calibration step must be a positive number of seconds, got {step_s}"
        raise ConfigError(msg)
    config = config or ScoreboardConfig()
    log = stage_logger(STAGE, match_id)
    started = time.perf_counter()
    info = probe_video(video)
    source = source_factory(video, frame_step(info.fps, 1.0 / step_s), config.hw_accel)
    sample = collect((frame for _, frame in source), config.image)
    save_board_reference(sample, config.image, video.parent / TEMPLATES_DIR)
    clusters = build_clusters(sample, config.image)
    save_workdir(workdir, sample, clusters, source.decode_mode, video.name)
    log.info(
        "calibration_collected",
        n_frames=sample.n_frames,
        n_board_frames=sample.n_board_frames,
        decode_mode=source.decode_mode,
        duration_ms=_ms(started),
    )
    return CalibrationSummary(
        n_frames=sample.n_frames,
        n_board_frames=sample.n_board_frames,
        decode_mode=source.decode_mode,
        clusters={kind: len(c) for kind, c in clusters.items()},
    )


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    """Read a labels file: `{"clock": {"0": "colon", "1": "0"}, "score": {...}, "added": {...}}`.

    Raises:
        ScoreboardParseError: the file is missing, is not JSON or has the wrong shape.
    """
    try:
        return _LABELS.validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, json.JSONDecodeError) as err:
        msg = f"cannot read labels file {path}: {type(err).__name__}"
        raise ScoreboardParseError(msg) from err


def run_labelling(video: Path, workdir: Path, labels_file: Path) -> CalibrationReport:
    """Turn the labelled clusters of `workdir` into templates in the match directory."""
    return apply_labels(workdir, load_labels(labels_file), video.parent / TEMPLATES_DIR)
