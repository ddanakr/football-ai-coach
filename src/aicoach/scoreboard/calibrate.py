"""Calibration from the broadcast itself: board reference, glyph clusters, labelled templates.

Two steps. `collect` reads sampled frames, saves the board reference and cuts every glyph of the
clock, the score and the added-time block; similar glyphs are clustered and written as contact
sheets. A person looks at the sheets and writes a labels file (cluster id -> `0`..`9`, `colon`,
`plus`). `apply_labels` turns the labelled clusters into template files in the match directory.
Nothing from the broadcast is stored in the repository.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationError

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.board import (
    board_blue_fraction,
    build_reference,
    crop_fraction,
    red_fraction,
    save_reference,
)
from aicoach.scoreboard.glyphs import (
    ALL_LABELS,
    COLON_LABEL,
    DIGIT_LABELS,
    PLUS_LABEL,
    Patch,
    cluster_patches,
    patch_to_png,
    save_template,
    segment_glyphs,
    write_png,
)
from aicoach.scoreboard.imaging import (
    ADDED_ROI,
    BOARD_ROI,
    CLOCK_ROI,
    SCORE_AWAY_ROI,
    SCORE_HOME_ROI,
    ImageParams,
)
from aicoach.scoreboard.schemas import DecodeMode, Roi
from aicoach.scoreboard.video import Frame

KINDS = ("score", "clock", "added")
REQUIRED_KINDS = ("score", "clock")
CALIBRATION_FILE = "calibration.json"
_FIELD_ROIS: dict[str, tuple[Roi, ...]] = {
    "score": (SCORE_HOME_ROI, SCORE_AWAY_ROI),
    "clock": (CLOCK_ROI,),
    "added": (ADDED_ROI,),
}
_EXEMPLARS_PER_CLUSTER = 3
_MAX_REFERENCE_CROPS = 300
_SHEET_ROWS = 40  # clusters shown per sheet; the biggest come first


@dataclass
class CalibrationSample:
    """Board crops and glyph patches gathered from the sampled frames."""

    n_frames: int = 0
    n_board_frames: int = 0
    board_crops: list[Frame] = field(default_factory=list)
    patches: dict[str, list[Patch]] = field(default_factory=lambda: {k: [] for k in KINDS})


def collect(frames: Iterable[Frame], params: ImageParams) -> CalibrationSample:
    """Gather board crops and glyph patches from frames where the board is on screen.

    The board is recognised here only by the colour of its blue blocks (there is no reference
    yet); advertising boards are cyan and do not match.
    """
    sample = CalibrationSample()
    for frame in frames:
        sample.n_frames += 1
        board = crop_fraction(frame, BOARD_ROI)
        if board_blue_fraction(board) < params.candidate_blue_min:
            continue
        sample.n_board_frames += 1
        sample.board_crops.append(board.copy())
        for kind, rois in _FIELD_ROIS.items():
            field_crops = [crop_fraction(frame, roi) for roi in rois]
            if kind == "added" and red_fraction(field_crops[0]) < params.red_block_min_share:
                continue
            for crop in field_crops:
                sample.patches[kind].extend(g.patch for g in segment_glyphs(crop, params))
    return sample


def build_clusters(sample: CalibrationSample, params: ImageParams) -> dict[str, list[list[int]]]:
    """Cluster the glyph patches of every kind (biggest cluster first)."""
    return {
        kind: cluster_patches(sample.patches[kind], params.cluster_similarity) for kind in KINDS
    }


def render_sheet(
    patches: list[Patch], clusters: list[list[int]], max_clusters: int = _SHEET_ROWS
) -> NDArray[np.uint8]:
    """A contact sheet: one row per cluster (first, middle and last member, magnified) with
    the cluster id and size written on the left."""
    rows: list[NDArray[np.uint8]] = []
    for cid, members in enumerate(clusters[:max_clusters]):
        cells = [
            cv2.resize(patch_to_png(patches[m]), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            for m in (members[0], members[len(members) // 2], members[-1])
        ]
        row = cv2.copyMakeBorder(np.hstack(cells), 0, 0, 90, 0, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(
            row, f"#{cid} n={len(members)}", (2, row.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, 255, 1,
        )  # fmt: skip
        rows.append(np.asarray(row, dtype=np.uint8))
    if not rows:
        return np.zeros((8, 8), dtype=np.uint8)
    return np.vstack(rows)


class _ClustersFile(BaseModel):
    """`clusters.json` of the workdir."""

    model_config = ConfigDict(extra="ignore")
    decode_mode: DecodeMode
    video: str
    n_frames: int
    n_board_frames: int
    clusters: dict[str, list[list[int]]]


class _CalibrationRecord(BaseModel):
    """`calibration.json` of the templates directory."""

    model_config = ConfigDict(extra="ignore")
    decode_mode: DecodeMode
    video: str
    templates: dict[str, int]


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def save_workdir(
    workdir: Path,
    sample: CalibrationSample,
    clusters: dict[str, list[list[int]]],
    decode_mode: DecodeMode,
    video_name: str,
) -> None:
    """Write patches, clusters, sheets and metadata so that labelling needs no video pass.

    Raises:
        ScoreboardParseError: an image cannot be written.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        if sample.patches[kind]:
            np.save(workdir / f"patches_{kind}.npy", np.stack(sample.patches[kind]))
        write_png(workdir / f"sheet_{kind}.png", render_sheet(sample.patches[kind], clusters[kind]))
    meta = _ClustersFile(
        decode_mode=decode_mode,
        video=video_name,
        n_frames=sample.n_frames,
        n_board_frames=sample.n_board_frames,
        clusters=clusters,
    )
    _write_text_atomic(workdir / "clusters.json", meta.model_dump_json(indent=2))


def save_board_reference(
    sample: CalibrationSample, params: ImageParams, templates_dir: Path
) -> Path:
    """Build the board reference from an even subsample of the board crops and save it.

    Raises:
        ScoreboardParseError: no frame with the board was found.
    """
    crops = sample.board_crops
    stride = max(1, len(crops) // _MAX_REFERENCE_CROPS)
    reference = build_reference(crops[::stride], params.static_std)
    return save_reference(reference, templates_dir)


@dataclass(frozen=True)
class CalibrationReport:
    """Result of `apply_labels`: how many templates per kind and which labels are missing."""

    templates: dict[str, int]
    missing: dict[str, list[str]]


def _expected_labels(kind: str) -> list[str]:
    if kind == "clock":
        return [*DIGIT_LABELS, COLON_LABEL]
    if kind == "added":
        return [PLUS_LABEL, *DIGIT_LABELS[1:]]
    return list(DIGIT_LABELS)


def _check_labels(kind: str, labels: dict[str, str], n_clusters: int) -> None:
    for cid, label in labels.items():
        if not (cid.isascii() and cid.isdecimal()) or int(cid) >= n_clusters:
            msg = f"labels for {kind}: unknown cluster id {cid!r} (there are {n_clusters})"
            raise ScoreboardParseError(msg)
        if label not in ALL_LABELS:
            msg = f"labels for {kind}: unknown label {label!r}; use 0-9, colon or plus"
            raise ScoreboardParseError(msg)


def _read_clusters(workdir: Path) -> _ClustersFile:
    path = workdir / "clusters.json"
    try:
        return _ClustersFile.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as err:
        msg = (
            f"no usable calibration data in {workdir} ({type(err).__name__}); "
            "run scoreboard-calibrate without --labels first"
        )
        raise ScoreboardParseError(msg) from err


def _plan_templates(
    workdir: Path, meta: _ClustersFile, labels: dict[str, dict[str, str]]
) -> dict[str, list[tuple[str, int, Patch]]]:
    """Validate the labels and load every patch to be written; touches no template file.

    Raises:
        ScoreboardParseError: unknown label or cluster id, a required kind without labels, or
            a patches file that is missing.
    """
    for kind in KINDS:
        _check_labels(kind, labels.get(kind, {}), len(meta.clusters.get(kind, [])))
    for kind in REQUIRED_KINDS:
        if not labels.get(kind):
            msg = f"no templates for {kind}: label at least one cluster in the {kind} sheet"
            raise ScoreboardParseError(msg)
    plan: dict[str, list[tuple[str, int, Patch]]] = {kind: [] for kind in KINDS}
    for kind in KINDS:
        if not labels.get(kind):
            continue
        try:
            patches = np.load(workdir / f"patches_{kind}.npy")
        except (OSError, ValueError) as err:
            msg = f"calibration patches for {kind} are missing or broken in {workdir}"
            raise ScoreboardParseError(msg) from err
        for cid, label in sorted(labels[kind].items(), key=lambda kv: int(kv[0])):
            members = meta.clusters[kind][int(cid)]
            picks = sorted({members[0], members[len(members) // 2], members[-1]})
            plan[kind] += [
                (label, int(cid) * 10 + k, patches[m])
                for k, m in enumerate(picks[:_EXEMPLARS_PER_CLUSTER])
            ]
    return plan


def apply_labels(
    workdir: Path, labels: dict[str, dict[str, str]], templates_dir: Path
) -> CalibrationReport:
    """Write template files for the labelled clusters and the calibration record.

    `labels[kind][cluster id] = label`; clusters without a label are ignored. Everything is
    validated and loaded before the first file is touched, so a mistake leaves the old
    templates as they were. Old templates of a kind are then replaced. The record stores the
    decoding mode of the calibration frames.

    Raises:
        ScoreboardParseError: the workdir is incomplete, a label or cluster id is unknown, or
            `score` or `clock` ended up without templates.
    """
    meta = _read_clusters(workdir)
    plan = _plan_templates(workdir, meta, labels)
    counts: dict[str, int] = {}
    for kind in KINDS:
        kind_dir = templates_dir / kind
        for old in kind_dir.glob("*.png"):
            old.unlink()
        for label, index, patch in plan[kind]:
            save_template(kind_dir, label, index, patch)
        counts[kind] = len(plan[kind])
    record = _CalibrationRecord(decode_mode=meta.decode_mode, video=meta.video, templates=counts)
    templates_dir.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(templates_dir / CALIBRATION_FILE, record.model_dump_json(indent=2))
    missing = {
        kind: [x for x in _expected_labels(kind) if x not in set(labels.get(kind, {}).values())]
        for kind in KINDS
    }
    return CalibrationReport(templates=counts, missing=missing)


def read_calibrated_mode(templates_dir: Path) -> DecodeMode:
    """Decoding mode recorded at calibration time.

    Raises:
        ScoreboardParseError: the record is missing or broken (the message says how to fix it).
    """
    path = templates_dir / CALIBRATION_FILE
    hint = "Calibrate first: python -m aicoach.cli scoreboard-calibrate --video <file>"
    if not path.is_file():
        msg = f"calibration record not found: {path}. {hint}"
        raise ScoreboardParseError(msg)
    try:
        record = _CalibrationRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as err:
        msg = f"broken calibration record: {path}. {hint}"
        raise ScoreboardParseError(msg) from err
    return record.decode_mode
