"""The plate-names stage: read the player names from the plate crops, write `plate_names.json`.

Input: `plates.json` and the crops `plate_crops/` of the match directory (the video is not
read), and an optional `lineups.csv` there. Output: `plate_names.json` and the state in
`status.json`. The names do not go to `predictions` or `edits`: the viewer joins them to the
events by the plate id. Without Tesseract the stage is "skipped", not failed, so the ingest of
a match goes on (D-059).
"""

import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from aicoach.errors import PlatesError
from aicoach.i18n import ru
from aicoach.logging_setup import stage_logger
from aicoach.plate_names.assemble import build_plate_names, count_names
from aicoach.plate_names.lineup import LINEUP_FILE, Player, parse_lineup
from aicoach.plate_names.reader import (
    Gray,
    NameReader,
    OcrUnavailableError,
    TesseractReader,
    prepare_line,
)
from aicoach.plate_names.schemas import (
    RULES_VERSION,
    SCHEMA_VERSION,
    PlateNames,
    PlateNamesConfig,
    PlateNamesResult,
    Role,
)
from aicoach.plates.schemas import Plate, PlatesResult
from aicoach.plates.stage import RESULT_FILE as PLATES_FILE
from aicoach.scoreboard.board import crop_fraction
from aicoach.scoreboard.schemas import Roi
from aicoach.stage_common import (
    Logger,
    Progress,
    elapsed_ms,
    record_failure,
    set_status,
    write_atomic,
)

STAGE = "plate_names"
RESULT_FILE = "plate_names.json"
_ROI_TOLERANCE = 1e-6


def load_plates(match_dir: Path) -> tuple[PlatesResult, str]:
    """Read `plates.json` of the match and a version string (a hash of the file).

    Raises:
        PlatesError: the file is missing or invalid; the message says to run the plates stage.
    """
    path = match_dir / PLATES_FILE
    hint = "run first: python -m aicoach.cli plates --video <file>"
    try:
        raw = path.read_bytes()
    except FileNotFoundError as err:
        msg = f"{path} not found: {hint}"
        raise PlatesError(msg) from err
    except OSError as err:
        msg = f"cannot read {path}: {type(err).__name__}"
        raise PlatesError(msg) from err
    try:
        plates = PlatesResult.model_validate_json(raw)
    except ValidationError as err:
        msg = f"{path} is not a valid plates result ({hint})"
        raise PlatesError(msg) from err
    return plates, hashlib.sha256(raw).hexdigest()


def _check_region(plates: PlatesResult, config: PlateNamesConfig) -> None:
    """The text lines are placed inside the crop of the plate region of the layout; crops cut
    from another region would be read at the wrong places."""
    have, want = plates.roi, config.layout.plate
    pairs = ((have.x0, want.x0), (have.y0, want.y0), (have.x1, want.x1), (have.y1, want.y1))
    if any(abs(a - b) > _ROI_TOLERANCE for a, b in pairs):
        msg = "plates.json was made with another plate region than the name layout expects"
        raise PlatesError(msg)


def load_lineup_file(match_dir: Path, lineup_path: Path | None) -> tuple[list[Player], str] | None:
    """The lineup and a hash of the file; None when there is none.

    Without `lineup_path` the file `lineups.csv` of the match directory is used if it exists.

    Raises:
        PlatesError: the file cannot be read or is invalid.
    """
    path = lineup_path or match_dir / LINEUP_FILE
    if lineup_path is None and not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError as err:
        msg = f"cannot read {path}: {type(err).__name__}"
        raise PlatesError(msg) from err
    return parse_lineup(data), hashlib.sha256(data).hexdigest()


def cache_key(plates_version: str, lineup_version: str | None, config: PlateNamesConfig) -> str:
    """A hash of everything that changes the result: `plates.json`, the lineup file, the
    parameters and the version of the rules."""
    digest = hashlib.sha256()
    digest.update(f"{plates_version}|{lineup_version}|{RULES_VERSION}|{SCHEMA_VERSION}".encode())
    digest.update(config.model_dump_json().encode())
    return digest.hexdigest()


def _load_cached(match_dir: Path, key: str, log: Logger) -> PlateNamesResult | None:
    path = match_dir / RESULT_FILE
    try:
        result = PlateNamesResult.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValidationError, ValueError) as err:
        log.warning("cache_unreadable", path=str(path), reason=type(err).__name__)
        return None
    return result if result.cache_key == key else None


def _read_crop(match_dir: Path, plate: Plate) -> Gray:
    """The saved crop of the plate as a grey image.

    Raises:
        PlatesError: the crop is outside the match directory, missing or not an image.
    """
    path = (match_dir / plate.crop).resolve()
    if not path.is_relative_to(match_dir.resolve()):
        msg = f"crop of {plate.id} points outside the match directory"
        raise PlatesError(msg)
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as err:
        msg = f"cannot read the crop of {plate.id} ({plate.crop}): run the plates stage again"
        raise PlatesError(msg) from err
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        msg = f"the crop of {plate.id} ({plate.crop}) is not an image"
        raise PlatesError(msg)
    return np.asarray(image, dtype=np.uint8)


def read_plate(
    plate: Plate, crop: Gray, reader: NameReader, config: PlateNamesConfig
) -> dict[Role, str]:
    """Read the text lines of one plate: one for a goal or a card, two for a substitution
    (came on first, then left)."""
    layout = config.layout
    regions: dict[Role, Roi] = (
        {"in": layout.went_on, "out": layout.went_off}
        if plate.type == "subst"
        else {"player": layout.single}
    )
    return {
        role: reader.read(prepare_line(crop_fraction(crop, region), config.scale, config.pad_px))
        for role, region in regions.items()
    }


def _warnings(names: list[PlateNames], has_lineup: bool) -> list[str]:
    if not has_lineup:
        return []
    return [
        f"{n.plate_id}: side unknown, names not checked against the lineup"
        for n in names
        if n.side is None
    ]


def _run(
    match_dir: Path,
    lineup_path: Path | None,
    config: PlateNamesConfig,
    force: bool,
    reader: NameReader | None,
    progress: Progress,
    log: Logger,
) -> tuple[PlateNamesResult, bool]:
    """Returns the result and whether it came from the cache."""
    plates, plates_version = load_plates(match_dir)
    _check_region(plates, config)
    loaded = load_lineup_file(match_dir, lineup_path)
    lineup = None if loaded is None else loaded[0]
    key = cache_key(plates_version, None if loaded is None else loaded[1], config)
    if not force:
        cached = _load_cached(match_dir, key, log)
        if cached is not None:
            return cached, True
    set_status(match_dir, STAGE, "running", 0.0, log)
    reader = reader or TesseractReader.create()
    names: list[PlateNames] = []
    for i, plate in enumerate(plates.plates):
        texts = read_plate(plate, _read_crop(match_dir, plate), reader, config)
        names.append(build_plate_names(plate, texts, lineup, config.min_similarity))
        progress.update((i + 1) / len(plates.plates))
    result = PlateNamesResult(
        cache_key=key,
        plates_version=plates_version,
        lineup=None if loaded is None else (lineup_path or Path(LINEUP_FILE)).name,
        lineup_version=None if loaded is None else loaded[1],
        config=config,
        names=names,
        stats=count_names(names),
        warnings=_warnings(names, lineup is not None),
    )
    for text in result.warnings:
        log.warning("names_warning", detail=text)
    write_atomic(match_dir / RESULT_FILE, result.model_dump_json(indent=2))
    return result, False


def run_plate_names(
    match_dir: Path,
    match_id: str,
    *,
    lineup_path: Path | None = None,
    config: PlateNamesConfig | None = None,
    force: bool = False,
    reader: NameReader | None = None,
) -> PlateNamesResult | None:
    """Read the names on the plates of the match in `match_dir`.

    A result made from the same `plates.json`, lineup file and parameters is returned without
    reading anything (also without Tesseract) unless `force` is set. Without a usable Tesseract
    (not installed, or no `kaz` or `rus` language data) the step is recorded as `skipped` with
    the reason and None is returned; no other stage is affected.

    Args:
        match_dir: Directory with `plates.json`, `plate_crops/` and optionally `lineups.csv`.
        match_id: Match id for the logs.
        lineup_path: Lineup file; default `lineups.csv` in `match_dir` when it exists.
        config: Parameters; default `PlateNamesConfig()`.
        force: Read again even if an up-to-date result exists.
        reader: The text reader (a fake in tests); default Tesseract.

    Raises:
        PlatesError: no or invalid `plates.json`, a missing crop, an invalid lineup file, or
            Tesseract failing on a line.
    """
    config = config or PlateNamesConfig()
    log = stage_logger(STAGE, match_id)
    progress = Progress(match_dir, STAGE, log)
    started = time.perf_counter()
    try:
        result, from_cache = _run(match_dir, lineup_path, config, force, reader, progress, log)
    except OcrUnavailableError as err:
        reason = ru.PLATE_NAMES_SKIPPED.format(reason=err)
        log.warning("stage_skipped_no_ocr", reason=str(err))
        set_status(match_dir, STAGE, "skipped", 1.0, log, error=reason)
        return None
    except BaseException as err:  # any failure, even Ctrl+C, must not leave "running" behind
        record_failure(match_dir, STAGE, progress.value, err, log, started)
        raise
    set_status(match_dir, STAGE, "done", 1.0, log)
    log.info(
        "stage_skipped" if from_cache else "stage_done",
        n_plates=result.stats.n_plates,
        n_matched=result.stats.n_matched,
        duration_ms=elapsed_ms(started),
    )
    return result
