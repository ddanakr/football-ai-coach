"""The `review-candidates` stage: `candidates.json`, `scoreboard.json` and the T-DEED file in the
match directory give `review_candidates.json` (D-075).

Disk, status and logs are here; the rules are in `aicoach.review.candidates` (pure). The stage
reads what the fusion stage wrote and changes none of it: no database access, no new run.
"""

import hashlib
import time
from pathlib import Path

from pydantic import ValidationError

from aicoach.errors import FusionError
from aicoach.fusion.schemas import CandidatesResult
from aicoach.fusion.stage import RESULT_FILE as CANDIDATES_FILE
from aicoach.logging_setup import stage_logger
from aicoach.review.candidates import build_review_candidates, count_candidates
from aicoach.review.schemas import SCHEMA_VERSION, ReviewCandidatesResult
from aicoach.scoreboard.config import ScoreboardConfig
from aicoach.scoreboard.schemas import ScoreboardResult
from aicoach.scoreboard.stage import RESULT_FILE as SCOREBOARD_FILE
from aicoach.spotting.convert import to_predictions
from aicoach.spotting.schemas import load_tdeed_file
from aicoach.stage_common import Logger, elapsed_ms, record_failure, set_status, write_atomic

STAGE = "review_candidates"
RESULT_FILE = "review_candidates.json"
# Bump when the rule of the layer changes, so that an older file is not reused.
LAYER_VERSION = 2  # 2: only shots get candidates


def _read(path: Path, hint: str) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as err:
        msg = f"{path} not found: {hint}"
        raise FusionError(msg) from err
    except OSError as err:
        msg = f"cannot read {path}: {type(err).__name__}"
        raise FusionError(msg) from err


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def cache_key(events_version: str, model_version: str, scoreboard_version: str) -> str:
    """A hash of everything that changes the layer: the three input files and the versions of
    the file schema and of the rule."""
    digest = hashlib.sha256()
    for part in (f"{SCHEMA_VERSION}.{LAYER_VERSION}", events_version, model_version):
        digest.update(part.encode() + b"|")
    digest.update(scoreboard_version.encode())
    return digest.hexdigest()


def load_review_candidates(match_dir: Path) -> ReviewCandidatesResult | None:
    """The stored layer of the match; None when there is no valid file.

    Never raises: the viewer shows the catalog without the layer if the file is missing or
    broken. A missing file is normal (the stage did not run); a broken one is logged.
    """
    try:
        return ReviewCandidatesResult.model_validate_json(
            (match_dir / RESULT_FILE).read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return None
    except (OSError, ValidationError, ValueError) as err:
        stage_logger(STAGE, match_dir.name).warning(
            "review_candidates_unreadable", reason=type(err).__name__
        )
        return None


def _build(
    match_dir: Path, match_id: str, predictions_path: Path, log: Logger, *, force: bool
) -> tuple[ReviewCandidatesResult, bool]:
    """Returns the result and whether it came from the cache."""
    events_raw = _read(
        match_dir / CANDIDATES_FILE,
        "run first: python -m aicoach.cli fuse --video <file> --predictions <T-DEED file>",
    )
    board_raw = _read(match_dir / SCOREBOARD_FILE, "run first: python -m aicoach.cli scoreboard")
    model_raw = _read(predictions_path, "the T-DEED file")
    key = cache_key(_sha(events_raw), _sha(model_raw), _sha(board_raw))
    if not force:
        stored = load_review_candidates(match_dir)
        if stored is not None and stored.cache_key == key and stored.match_id == match_id:
            return stored, True
    try:
        events = CandidatesResult.model_validate_json(events_raw)
        board = ScoreboardResult.model_validate_json(board_raw)
    except ValidationError as err:
        msg = f"{match_dir}: candidates.json or scoreboard.json is not valid"
        raise FusionError(msg) from err
    if events.model_version != _sha(model_raw):
        msg = (
            f"{match_dir / CANDIDATES_FILE} was made from another T-DEED file than "
            f"{predictions_path.name}: run the fuse stage again with this file"
        )
        raise FusionError(msg)
    set_status(match_dir, STAGE, "running", 0.0, log)
    model = to_predictions(load_tdeed_file(predictions_path), events.config.model_fps)
    build = build_review_candidates(
        model,
        [(c.type, c.t_v) for c in events.candidates],
        {1: events.half_bounds["1"], 2: events.half_bounds["2"]},
        board.offset_segments,
        merge_radius_s=events.config.merge_radius_s,
        max_extrapolation_s=ScoreboardConfig().timeline.max_extrapolation_s,
    )
    result = ReviewCandidatesResult(
        match_id=match_id,
        cache_key=key,
        events_version=_sha(events_raw),
        model_version=_sha(model_raw),
        scoreboard_version=_sha(board_raw),
        model_file=predictions_path.name,
        merge_radius_s=events.config.merge_radius_s,
        n_model=build.n_model,
        n_existing=build.n_existing,
        n_merged=build.n_merged,
        n_outside_halves=build.n_outside_halves,
        half_bounds=events.half_bounds,
        candidates=build.candidates,
        counts=count_candidates(build.candidates),
    )
    write_atomic(match_dir / RESULT_FILE, result.model_dump_json(indent=2))
    return result, False


def run_review_candidates(
    match_dir: Path, match_id: str, predictions_path: Path, *, force: bool = False
) -> ReviewCandidatesResult:
    """Build `review_candidates.json` in `match_dir`; a valid earlier file made from the same
    three inputs is returned as it is unless `force` is set.

    Args:
        match_dir: Directory with `candidates.json` and `scoreboard.json`.
        match_id: Match id for the file and the logs.
        predictions_path: The T-DEED file that the fusion stage used (checked by its hash).
        force: Build again even if the file is up to date.

    Raises:
        FusionError: An input is missing or invalid, or `candidates.json` was made from another
            T-DEED file.
        SpottingError: The T-DEED file is broken.
        StorageError: The result file cannot be written.
    """
    log = stage_logger(STAGE, match_id)
    started = time.perf_counter()
    try:
        result, from_cache = _build(match_dir, match_id, predictions_path, log, force=force)
    except BaseException as err:  # any failure, even Ctrl+C, must not leave "running" behind
        record_failure(match_dir, STAGE, 0.0, err, log, started)
        raise
    set_status(match_dir, STAGE, "done", 1.0, log)
    log.info(
        "stage_skipped" if from_cache else "stage_done",
        n_candidates=len(result.candidates),
        n_existing=result.n_existing,
        n_merged=result.n_merged,
        duration_ms=elapsed_ms(started),
    )
    return result
