"""The fusion stage: read `scoreboard.json`, `plates.json` and an optional T-DEED file, apply the
rules, write the events to SQLite and `candidates.json`.

Everything lives in the match directory (the video's directory): `candidates.json`,
`status.json` and the database `aicoach.db`. The rules are in `aicoach.fusion.rules` (pure);
disk, database, status and logs are here.
"""

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from aicoach.errors import FusionError, StorageError
from aicoach.eval.schemas import Prediction
from aicoach.fusion.config import FuseConfig
from aicoach.fusion.rules import RULES_VERSION, count_events, fuse_signals
from aicoach.fusion.schemas import SCHEMA_VERSION, Candidate, CandidatesResult, Source
from aicoach.logging_setup import stage_logger
from aicoach.plates.schemas import PlatesResult
from aicoach.plates.stage import RESULT_FILE as PLATES_FILE
from aicoach.scoreboard.schemas import ScoreboardResult
from aicoach.scoreboard.stage import RESULT_FILE as SCOREBOARD_FILE
from aicoach.spotting.convert import to_predictions
from aicoach.spotting.schemas import load_tdeed_file
from aicoach.stage_common import Logger, elapsed_ms, record_failure, set_status, write_atomic
from aicoach.storage import (
    NewPrediction,
    add_predictions,
    connect,
    count_effective_edits,
    init_db,
    latest_run_id,
)
from aicoach.storage.models import Source as DbSource

STAGE = "fuse"
RESULT_FILE = "candidates.json"
DB_FILE = "aicoach.db"
_Model = TypeVar("_Model", bound=BaseModel)


def db_source(source: Source) -> DbSource:
    """The `source` column of `predictions` for a candidate.

    The table knows `plates`, `model`, `speech` and `merge` only (schema version 1, not changed
    for this stage). Events from the broadcast graphics (plates and scoreboard) are `plates`, a
    model event is `model`, a model event confirmed by a plate is `merge`. The finer label stays
    in `candidates.json`.
    """
    if source == "model":
        return "model"
    if source == "model+plates":
        return "merge"
    return "plates"


def to_new_prediction(candidate: Candidate) -> NewPrediction:
    """The row to store for a candidate."""
    return NewPrediction(
        type=candidate.type,
        t_v=candidate.t_v,
        source=db_source(candidate.source),
        half=candidate.half,
        score=candidate.score,
        status=candidate.status,
        team=candidate.team,
    )


def new_run_id() -> str:
    """A run id unique for a match: the UTC time to the microsecond."""
    return "fuse-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")


def _read_input(path: Path, model: type[_Model], hint: str) -> tuple[_Model, str]:
    """Read and validate one input file; returns it and the sha256 of its bytes.

    Raises:
        FusionError: the file is missing, unreadable or not valid; the message says what to run.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError as err:
        msg = f"{path} not found: {hint}"
        raise FusionError(msg) from err
    except OSError as err:
        msg = f"cannot read {path}: {type(err).__name__}"
        raise FusionError(msg) from err
    try:
        return model.model_validate_json(raw), hashlib.sha256(raw).hexdigest()
    except ValidationError as err:
        msg = f"{path} is not a valid {model.__name__} ({hint})"
        raise FusionError(msg) from err


def cache_key(
    scoreboard_version: str, plates_version: str, model_version: str | None, config: FuseConfig
) -> str:
    """A hash of everything that changes the result: the three input files, the parameters, the
    schema version of `candidates.json` and the version of the rules."""
    digest = hashlib.sha256()
    for part in (
        f"{SCHEMA_VERSION}.{RULES_VERSION}",
        scoreboard_version,
        plates_version,
        model_version or "none",
    ):
        digest.update(part.encode() + b"|")
    digest.update(config.model_dump_json().encode())
    return digest.hexdigest()


def _stored_run(db_path: Path, match_id: str, log: Logger) -> str | None:
    """The latest run of the match in the database; None if there is no usable database."""
    if not db_path.is_file():  # do not create an empty database just to look into it
        return None
    try:
        with connect(db_path) as conn:
            return latest_run_id(conn, match_id)
    except StorageError as err:
        log.warning("database_unreadable", path=str(db_path), reason=str(err))
        return None


def _load_cached(
    match_dir: Path, match_id: str, key: str, db_path: Path, log: Logger
) -> CandidatesResult | None:
    """The stored result if it was made from the same inputs and its run is still the latest
    run of the match in the database (so a deleted database is filled again)."""
    path = match_dir / RESULT_FILE
    try:
        result = CandidatesResult.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValidationError, ValueError) as err:
        log.warning("cache_unreadable", path=str(path), reason=type(err).__name__)
        return None
    if result.cache_key != key or result.match_id != match_id:
        return None
    if result.run_id is not None and _stored_run(db_path, match_id, log) != result.run_id:
        log.warning("cache_run_missing", run_id=result.run_id)
        return None
    return result


def _store(
    db_path: Path,
    match_id: str,
    run_id: str,
    candidates: list[Candidate],
    discard_edits: bool,
    log: Logger,
) -> str | None:
    """Write the run to the database; returns its id, or None if there was nothing to write.

    Raises:
        FusionError: the match has user edits and `discard_edits` is not set: a new run would
            hide them (D-028).
        StorageError: the database failed.
    """
    if not candidates and not db_path.is_file():
        return None
    init_db(db_path)
    with connect(db_path) as conn:
        if not candidates:
            # An empty run is not stored (D-028), so it hides nothing and needs no edit check.
            previous = latest_run_id(conn, match_id)
            if previous is not None:
                log.warning("empty_run_not_stored", previous_run=previous)
            return None
        n_edits = count_effective_edits(conn, match_id)
        if n_edits and not discard_edits:
            msg = (
                f"match {match_id!r} has {n_edits} user edit(s) in {db_path}: a new run would "
                "hide the edits of its events from the current view (D-028). Export the checked "
                "labels first, then run again with --discard-edits"
            )
            raise FusionError(msg)
        add_predictions(conn, match_id, run_id, [to_new_prediction(c) for c in candidates])
    if n_edits:
        log.warning("edits_hidden", n_edits=n_edits)
    return run_id


@dataclass(frozen=True)
class _Inputs:
    """The validated input files with the sha256 of each (None: no model file)."""

    board: ScoreboardResult
    board_version: str
    plates: PlatesResult
    plates_version: str
    model: list[Prediction] | None
    model_version: str | None


def _load_inputs(match_dir: Path, predictions_path: Path | None, config: FuseConfig) -> _Inputs:
    """Read `scoreboard.json`, `plates.json` and the optional T-DEED file.

    Raises:
        FusionError: a file is missing or invalid, or `plates.json` was made from another
            `scoreboard.json` (its teams would not fit the scoreboard).
        SpottingError: the T-DEED file is broken.
    """
    board, board_v = _read_input(
        match_dir / SCOREBOARD_FILE,
        ScoreboardResult,
        "run first: python -m aicoach.cli scoreboard --video <file>",
    )
    plates, plates_v = _read_input(
        match_dir / PLATES_FILE,
        PlatesResult,
        "run first: python -m aicoach.cli plates --video <file>",
    )
    if plates.scoreboard_version != board_v:
        msg = (
            f"{match_dir / PLATES_FILE} was made from another scoreboard.json: "
            "run the plates stage again (python -m aicoach.cli plates --video <file>)"
        )
        raise FusionError(msg)
    if predictions_path is None:
        return _Inputs(board, board_v, plates, plates_v, None, None)
    model = to_predictions(load_tdeed_file(predictions_path), config.model_fps)
    try:
        model_v = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    except OSError as err:
        msg = f"cannot read {predictions_path}: {type(err).__name__}"
        raise FusionError(msg) from err
    return _Inputs(board, board_v, plates, plates_v, model, model_v)


@dataclass(frozen=True)
class _Options:
    """The options of one run that are not inputs."""

    predictions_path: Path | None
    config: FuseConfig
    force: bool
    discard_edits: bool
    db_path: Path
    run_id: str | None


def _run(
    match_dir: Path,
    match_id: str,
    options: _Options,
    log: Logger,
) -> tuple[CandidatesResult, bool]:
    """Returns the result and whether it came from the cache."""
    config = options.config
    inputs = _load_inputs(match_dir, options.predictions_path, config)
    key = cache_key(inputs.board_version, inputs.plates_version, inputs.model_version, config)
    if not options.force:
        cached = _load_cached(match_dir, match_id, key, options.db_path, log)
        if cached is not None:
            return cached, True
    set_status(match_dir, STAGE, "running", 0.0, log)
    out = fuse_signals(inputs.board, inputs.plates, inputs.model, config)
    for text in out.warnings:
        log.warning("fuse_warning", detail=text)
    stored = _store(
        options.db_path,
        match_id,
        options.run_id or new_run_id(),
        out.candidates,
        options.discard_edits,
        log,
    )
    result = CandidatesResult(
        match_id=match_id,
        cache_key=key,
        run_id=stored,
        mode="no_model" if inputs.model is None else "with_model",
        scoreboard_version=inputs.board_version,
        plates_version=inputs.plates_version,
        model_version=inputs.model_version,
        model_file=None if options.predictions_path is None else options.predictions_path.name,
        config=config,
        half_bounds={"1": out.half_bounds[1], "2": out.half_bounds[2]},
        candidates=out.candidates,
        dropped=out.dropped,
        warnings=out.warnings,
        counts=count_events(out.candidates),
    )
    write_atomic(match_dir / RESULT_FILE, result.model_dump_json(indent=2))
    return result, False


def run_fuse(
    match_dir: Path,
    match_id: str,
    predictions_path: Path | None = None,
    config: FuseConfig | None = None,
    *,
    force: bool = False,
    discard_edits: bool = False,
    db_path: Path | None = None,
    run_id: str | None = None,
) -> CandidatesResult:
    """Run the stage in `match_dir`; events go to the database, details to `candidates.json`.

    A valid earlier result (same three input files and parameters, and its run is still the
    latest in the database) is returned without writing anything unless `force` is set. A new
    run replaces the previous one in the current view (D-028), so if the match already has user
    edits the stage refuses unless `discard_edits` is set. Progress and failure go to
    `status.json`; on a failure the error is raised again and earlier files stay as they were.

    Args:
        match_dir: Directory with `scoreboard.json` and `plates.json` (the video's directory).
        match_id: Match id for the database and the logs.
        predictions_path: T-DEED output; None runs the "no model" mode with a warning.
        config: Rule parameters; default `FuseConfig()`.
        force: Recompute even if the result is up to date.
        discard_edits: Allow a new run although the match has user edits.
        db_path: Database file; default `aicoach.db` in `match_dir`.
        run_id: Id of the new run (default: made from the clock); must be new for the match.

    Raises:
        FusionError: an input is missing, invalid or stale, there are no half bounds, or user
            edits would be hidden.
        SpottingError: the T-DEED file is broken.
        StorageError: the database or an output file cannot be written.
    """
    config = config or FuseConfig()
    log = stage_logger(STAGE, match_id)
    started = time.perf_counter()
    options = _Options(
        predictions_path, config, force, discard_edits, db_path or match_dir / DB_FILE, run_id
    )
    try:
        result, from_cache = _run(match_dir, match_id, options, log)
    except BaseException as err:  # any failure, even Ctrl+C, must not leave "running" behind
        record_failure(match_dir, STAGE, 0.0, err, log, started)
        raise
    set_status(match_dir, STAGE, "done", 1.0, log)
    log.info(
        "stage_skipped" if from_cache else "stage_done",
        n_events=len(result.candidates),
        n_dropped=len(result.dropped),
        mode=result.mode,
        duration_ms=elapsed_ms(started),
    )
    return result
