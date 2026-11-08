"""The plates stage: read the video once, find the lower-third plates, write `plates.json`.

Needs `scoreboard.json` of the same match directory (the score changes tell which logo is the
home team, and the scoreboard's decode mode must be the one used here). Everything is written
next to the video: `plates.json`, `status.json` and `plate_crops/` (a crop of every plate, its
logo, and every refused candidate; never part of the repository). Names on the plates are not
read here. Side effects live in this module, the logic in the pure modules.
"""

import hashlib
import shutil
import time
from pathlib import Path

import cv2
from pydantic import ValidationError

from aicoach.errors import PlatesError, StorageError
from aicoach.logging_setup import stage_logger
from aicoach.plates.assemble import CROPS_DIR, assemble_result
from aicoach.plates.config import PlatesConfig
from aicoach.plates.detect import read_frame
from aicoach.plates.logos import cluster_logos, logo_vector
from aicoach.plates.schemas import PlatesResult
from aicoach.plates.tracks import Hit, Track, build_tracks
from aicoach.scoreboard.board import crop_fraction
from aicoach.scoreboard.schemas import DecodeMode, ScoreboardResult
from aicoach.scoreboard.stage import (
    RESULT_FILE as SCOREBOARD_FILE,
)
from aicoach.scoreboard.stage import FrameSource, SourceFactory, duration_warning
from aicoach.scoreboard.video import (
    Frame,
    SampledReader,
    VideoInfo,
    frame_step,
    probe_video,
    sample_count,
)
from aicoach.stage_common import (
    Logger,
    Progress,
    elapsed_ms,
    record_failure,
    set_status,
    write_atomic,
)

STAGE = "plates"
RESULT_FILE = "plates.json"


def load_scoreboard(match_dir: Path) -> tuple[ScoreboardResult, str]:
    """Read `scoreboard.json` of the match and a version string (a hash of the file).

    Raises:
        PlatesError: the file is missing or invalid; the message says to run the scoreboard stage.
    """
    path = match_dir / SCOREBOARD_FILE
    hint = "run first: python -m aicoach.cli scoreboard --video <file>"
    try:
        raw = path.read_bytes()
    except FileNotFoundError as err:
        msg = f"{path} not found: {hint}"
        raise PlatesError(msg) from err
    except OSError as err:
        msg = f"cannot read {path}: {type(err).__name__}"
        raise PlatesError(msg) from err
    try:
        board = ScoreboardResult.model_validate_json(raw)
    except ValidationError as err:
        msg = f"{path} is not a valid scoreboard result ({hint})"
        raise PlatesError(msg) from err
    return board, hashlib.sha256(raw).hexdigest()


def cache_key(video: Path, scoreboard_version: str, mode: DecodeMode, config: PlatesConfig) -> str:
    """A hash of everything that changes the result: the video file, the version of
    scoreboard.json, the decode mode and the parameters (including the layout)."""
    digest = hashlib.sha256()
    stat = video.stat()
    digest.update(f"{video.name}|{stat.st_size}|{stat.st_mtime_ns}|{mode}".encode())
    digest.update(scoreboard_version.encode())
    digest.update(config.model_dump_json().encode())
    return digest.hexdigest()


def _load_cached(match_dir: Path, key: str, log: Logger) -> PlatesResult | None:
    """The stored result if it parses, was made with the same inputs and its crops exist."""
    path = match_dir / RESULT_FILE
    try:
        result = PlatesResult.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValidationError, ValueError) as err:
        log.warning("cache_unreadable", path=str(path), reason=type(err).__name__)
        return None
    if result.cache_key != key:
        return None
    files = [p.crop for p in result.plates] + [p.logo_crop for p in result.plates]
    files += [r.crop for r in result.rejected]
    if not all((match_dir / name).is_file() for name in files):
        log.warning("cache_crops_missing", path=str(match_dir / CROPS_DIR))
        return None
    return result


def _encode(image: Frame) -> bytes:
    """A PNG of `image`.

    Raises:
        PlatesError: the image cannot be encoded (for example it has no pixels).
    """
    try:
        ok, buffer = cv2.imencode(".png", image)
    except cv2.error as err:
        msg = "cannot encode a crop as PNG"
        raise PlatesError(msg) from err
    if not ok:
        msg = "cannot encode a crop as PNG"
        raise PlatesError(msg)
    return bytes(buffer.tobytes())


def collect_hits(
    source: FrameSource,
    info: VideoInfo,
    step: int,
    config: PlatesConfig,
    expected_mode: DecodeMode,
    progress: Progress,
) -> tuple[list[Hit], dict[int, bytes], int]:
    """Read every sampled frame; keep the candidates and a PNG of the plate region of each.

    `t_v` of sample `k` is `k * step / fps` (frame index / fps), as in the scoreboard stage.
    Returns the hits, the PNGs by sample index and the number of samples read.

    Raises:
        PlatesError: the video is decoded in another mode than scoreboard.json was made in.
    """
    total = max(1, sample_count(info.frame_count, step))
    hits: list[Hit] = []
    crops: dict[int, bytes] = {}
    n_samples = 0
    for k, frame in source:
        if n_samples == 0 and source.decode_mode != expected_mode:
            msg = (
                f"decode mode {source.decode_mode} differs from scoreboard.json "
                f"({expected_mode}): run both stages in the same mode"
            )
            raise PlatesError(msg)
        n_samples += 1
        reading = read_frame(frame, config.layout, config.detect)
        if reading.is_candidate:
            hits.append(Hit(k, k * step / info.fps, reading))
            crops[k] = _encode(crop_fraction(frame, config.layout.plate))
        progress.update(min(1.0, (k + 1) / total))
    return hits, crops, n_samples


def write_crops(match_dir: Path, files: dict[str, bytes]) -> None:
    """Replace `plate_crops/` with `files` (name -> PNG bytes).

    The new folder is built beside the old one and swapped in; if the swap fails the old folder
    is put back, so a failure never leaves the match directory without crops it had before.

    Raises:
        StorageError: the folder cannot be written.
    """
    target = match_dir / CROPS_DIR
    tmp = match_dir / (CROPS_DIR + ".tmp")
    old = match_dir / (CROPS_DIR + ".old")
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for name, data in files.items():
            (tmp / name).write_bytes(data)
        _swap_dirs(tmp, target, old)
    except OSError as err:
        shutil.rmtree(tmp, ignore_errors=True)
        msg = f"cannot write {target}"
        raise StorageError(msg) from err


def _swap_dirs(new: Path, target: Path, old: Path) -> None:
    """Move `target` aside to `old`, put `new` in its place, then drop `old`; on a failure put
    `target` back and raise the error again."""
    shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        target.replace(old)
    try:
        new.replace(target)
    except OSError:
        if old.exists():
            old.replace(target)
        raise
    shutil.rmtree(old, ignore_errors=True)


def _logo_of(track: Track) -> Frame:
    """The logo crop of the middle hit of a plate."""
    logo = track.mid_hit.reading.logo
    if logo is None:  # unreachable: every candidate keeps its logo crop
        msg = "plate without a logo crop"
        raise PlatesError(msg)
    return logo


def _crop_files(
    plates: list[Track], rejects: list[Track], crops: dict[int, bytes], result: PlatesResult
) -> dict[str, bytes]:
    """PNG bytes by file name for every plate (crop and logo) and refused candidate."""
    files: dict[str, bytes] = {}
    for track, plate in zip(plates, result.plates, strict=True):
        files[Path(plate.crop).name] = crops[track.mid_hit.k]
        files[Path(plate.logo_crop).name] = _encode(_logo_of(track))
    for track, rejected in zip(rejects, result.rejected, strict=True):
        files[Path(rejected.crop).name] = crops[track.mid_hit.k]
    return files


def _logo_labels(plates: list[Track], max_dist: float) -> list[int]:
    return cluster_logos([logo_vector(_logo_of(t)) for t in plates], max_dist)


def _build_result(
    video: Path,
    source: FrameSource,
    info: VideoInfo,
    step: int,
    config: PlatesConfig,
    found: tuple[list[Hit], int],
    board: ScoreboardResult,
    version: str,
    key: str,
) -> tuple[PlatesResult, list[Track], list[Track]]:
    """Tracks, logo clusters and the result for the hits of one pass."""
    hits, n_samples = found
    warning = duration_warning(info, source.frames_decoded, config.duration_tol_s)
    plates, rejects = build_tracks(hits, config.merge_gap_s, config.split_diff, config.min_hits)
    result = assemble_result(
        video=video.name,
        decode_mode=source.decode_mode,
        sample_fps=info.fps / step,
        roi=config.layout.plate,
        n_samples=n_samples,
        plates=plates,
        rejects=rejects,
        logo_labels=_logo_labels(plates, config.logo_max_dist),
        board=board,
        scoreboard_version=version,
        config=config,
        duration_warning=warning,
    ).model_copy(update={"cache_key": key})
    return result, plates, rejects


def _run(
    video: Path,
    match_dir: Path,
    config: PlatesConfig,
    force: bool,
    source_factory: SourceFactory,
    progress: Progress,
    log: Logger,
) -> tuple[PlatesResult, bool]:
    """Returns the result and whether it came from the cache."""
    info = probe_video(video)
    board, version = load_scoreboard(match_dir)
    key = cache_key(video, version, board.decode_mode, config)
    if not force:
        cached = _load_cached(match_dir, key, log)
        if cached is not None:
            return cached, True
    set_status(match_dir, STAGE, "running", 0.0, log)
    step = frame_step(info.fps, config.sample_fps)
    source = source_factory(video, step, config.hw_accel)
    hits, crops, n_samples = collect_hits(source, info, step, config, board.decode_mode, progress)
    result, plates, rejects = _build_result(
        video, source, info, step, config, (hits, n_samples), board, version, key
    )
    if result.stats.duration_warning is not None:
        log.warning("duration_mismatch", detail=result.stats.duration_warning)
    for text in result.stats.warnings:
        log.warning("teams_warning", detail=text)
    if result.stats.n_overlaps:
        log.warning("plates_overlap", n_overlaps=result.stats.n_overlaps)
    write_crops(match_dir, _crop_files(plates, rejects, crops, result))
    write_atomic(match_dir / RESULT_FILE, result.model_dump_json())
    return result, False


def run_plates(
    video: Path,
    match_id: str,
    config: PlatesConfig | None = None,
    force: bool = False,
    source_factory: SourceFactory = SampledReader,
) -> PlatesResult:
    """Run the stage on `video`; the result goes to `plates.json` beside it.

    A valid earlier result (same video, scoreboard.json, decode mode and parameters, crops
    present) is returned without reading the video unless `force` is set. Progress and failure
    go to `status.json`; on a failure the error is raised again and an earlier `plates.json`
    stays as it was (the crops are replaced just before it, so after a failure between the two
    writes the next run notices the mismatch through the cache check and recomputes).

    Raises:
        PlatesError: no or invalid scoreboard.json, or the decode mode differs from it.
        ScoreboardParseError: the video cannot be read.
    """
    config = config or PlatesConfig()
    log = stage_logger(STAGE, match_id)
    match_dir = video.parent
    progress = Progress(match_dir, STAGE, log)
    started = time.perf_counter()
    try:
        result, from_cache = _run(video, match_dir, config, force, source_factory, progress, log)
    except BaseException as err:  # any failure, even Ctrl+C, must not leave "running" behind
        record_failure(match_dir, STAGE, progress.value, err, log, started)
        raise
    set_status(match_dir, STAGE, "done", 1.0, log)
    event = "stage_skipped" if from_cache else "stage_done"
    log.info(
        event,
        n_samples=result.stats.n_samples,
        n_plates=len(result.plates),
        duration_ms=elapsed_ms(started),
    )
    return result
