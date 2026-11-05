"""Digits of the scoreboard: segmentation into glyphs and matching with labelled templates.

The text is white on a dark blue block. A field is cut into glyphs by empty pixel columns, every
glyph is resized to a fixed patch and compared with the templates by normalised correlation.
Templates are cut from the broadcast itself (calibration) and are not stored in git.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.imaging import ImageParams
from aicoach.scoreboard.video import Frame

Patch = NDArray[np.float32]
DIGIT_LABELS = tuple("0123456789")
COLON_LABEL = "colon"
PLUS_LABEL = "plus"
ALL_LABELS = (*DIGIT_LABELS, COLON_LABEL, PLUS_LABEL)
_SYMBOLS = {COLON_LABEL: ":", PLUS_LABEL: "+"}
_EPS = 1e-9
_CLOCK_RE = re.compile(r"^(\d{1,3}):(\d{2})$")
_SCORE_RE = re.compile(r"^\d{1,2}$")
_ADDED_RE = re.compile(r"^\+(\d{1,2})$")


@dataclass(frozen=True, eq=False)
class Glyph:
    """One segmented symbol: its column range in the field and its normalised patch."""

    x0: int
    x1: int
    patch: Patch


@dataclass(frozen=True)
class Match:
    """Best template of a glyph: its label, correlation and the margin over the runner-up."""

    label: str | None
    score: float
    margin: float


@dataclass(frozen=True)
class FieldReading:
    """Text read from one field; `text` is None if any glyph was rejected, `conf` is the
    lowest glyph correlation (0.0 for an unreadable field)."""

    text: str | None
    conf: float


def _darkest_channel(crop: Frame) -> NDArray[np.uint8]:
    # np.min is typed as "array or scalar"; for a 3-D input it is always an array.
    return np.asarray(np.min(crop, axis=2), dtype=np.uint8)


def text_mask(crop: Frame, threshold: int) -> NDArray[np.bool_]:
    """Pixels of white text: bright in every channel (the darkest channel is above threshold)."""
    return np.asarray(_darkest_channel(crop) > threshold, dtype=np.bool_)


def _normalise(gray: NDArray[np.uint8], width: int, height: int) -> Patch:
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    resized -= resized.mean()
    norm = float(np.sqrt((resized * resized).sum()))
    return resized / norm if norm > _EPS else np.zeros_like(resized)


def segment_glyphs(crop: Frame, params: ImageParams) -> list[Glyph]:
    """Cut a field into glyphs: groups of columns that contain text pixels.

    Groups with fewer than `min_glyph_pixels` text pixels are noise. A colon (two dots) is one
    group because its dots share their columns.
    """
    if crop.size == 0:
        return []
    mask = text_mask(crop, params.text_threshold)
    gray = _darkest_channel(crop)
    columns = np.asarray(mask.any(axis=0), dtype=np.bool_)
    glyphs: list[Glyph] = []
    x = 0
    width = mask.shape[1]
    while x < width:
        if not columns[x]:
            x += 1
            continue
        start = x
        while x < width and columns[x]:
            x += 1
        part = mask[:, start:x]
        if int(part.sum()) < params.min_glyph_pixels:
            continue
        rows = np.flatnonzero(part.any(axis=1))
        tight = gray[rows[0] : rows[-1] + 1, start:x]
        glyphs.append(Glyph(start, x, _normalise(tight, params.patch_width, params.patch_height)))
    return glyphs


@dataclass(frozen=True, eq=False)
class TemplateSet:
    """Labelled patches of one field kind; several exemplars per label are allowed."""

    labels: list[str]
    patches: Patch  # (n, patch_height, patch_width)

    def __len__(self) -> int:
        return len(self.labels)


def match_glyph(glyph: Glyph, templates: TemplateSet, params: ImageParams) -> Match:
    """Compare a glyph with every template; the label is None if the match is weak or ambiguous.

    The best score per label is used, so many exemplars of one digit do not count as rivals.
    """
    if len(templates) == 0:
        return Match(None, 0.0, 0.0)
    scores = (templates.patches * glyph.patch).sum(axis=(1, 2))
    best: dict[str, float] = {}
    for label, score in zip(templates.labels, scores.tolist(), strict=True):
        best[label] = max(score, best.get(label, -1.0))
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    top_label, top = ranked[0]
    margin = top - ranked[1][1] if len(ranked) > 1 else top
    accepted = top >= params.min_match and margin >= params.min_margin
    return Match(top_label if accepted else None, top, margin)


def read_field(crop: Frame, templates: TemplateSet, params: ImageParams) -> FieldReading:
    """Read the text of a field. Any rejected glyph makes the whole field unreadable."""
    glyphs = segment_glyphs(crop, params)
    if not glyphs:
        return FieldReading(None, 0.0)
    chars: list[str] = []
    conf = 1.0
    for glyph in glyphs:
        match = match_glyph(glyph, templates, params)
        if match.label is None:
            return FieldReading(None, 0.0)
        chars.append(_SYMBOLS.get(match.label, match.label))
        conf = min(conf, match.score)
    return FieldReading("".join(chars), conf)


def parse_clock_text(text: str | None, max_minutes: int) -> int | None:
    """`"69:17"` becomes 4157 seconds; anything that is not MM:SS with a sane value is None."""
    if text is None:
        return None
    found = _CLOCK_RE.match(text)
    if found is None:
        return None
    minutes, seconds = int(found.group(1)), int(found.group(2))
    if seconds >= 60 or minutes > max_minutes:
        return None
    return minutes * 60 + seconds


def parse_score_text(text: str | None, max_score: int) -> int | None:
    """One team's goals from its field; None for garbage or an implausible number."""
    if text is None or _SCORE_RE.match(text) is None:
        return None
    value = int(text)
    return value if value <= max_score else None


def parse_added_text(text: str | None) -> int | None:
    """`"+3"` becomes 3; anything else is None."""
    if text is None:
        return None
    found = _ADDED_RE.match(text)
    return None if found is None else int(found.group(1))


_CALIBRATE_HINT = "Calibrate first: python -m aicoach.cli scoreboard-calibrate --video <file>"


def read_gray(path: Path) -> NDArray[np.uint8]:
    """Read a PNG as grayscale through bytes (works for any path, including non-ASCII ones).

    Raises:
        ScoreboardParseError: the file cannot be read or is not an image.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as err:
        msg = f"cannot read image: {path}"
        raise ScoreboardParseError(msg) from err
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None
    if gray is None:
        msg = f"cannot read image: {path}"
        raise ScoreboardParseError(msg)
    return np.asarray(gray, dtype=np.uint8)


def write_png(path: Path, image: NDArray[np.uint8]) -> None:
    """Write an image as PNG through bytes and check that it reached the disk.

    Raises:
        ScoreboardParseError: the image cannot be encoded or the file cannot be written.
    """
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        msg = f"cannot encode image for {path}"
        raise ScoreboardParseError(msg)
    try:
        path.write_bytes(encoded.tobytes())
    except OSError as err:
        msg = f"cannot write image: {path}"
        raise ScoreboardParseError(msg) from err


def templates_from_dir(directory: Path, params: ImageParams) -> TemplateSet:
    """Load `<label>__<n>.png` files of one field kind (grayscale patches, resized to the
    patch size of `params`).

    Raises:
        ScoreboardParseError: the directory is missing or empty, or a file has an unknown label
            in its name or is not an image. The message says how to calibrate.
    """
    if not directory.is_dir():
        msg = f"template directory not found: {directory}. {_CALIBRATE_HINT}"
        raise ScoreboardParseError(msg)
    labels: list[str] = []
    patches: list[Patch] = []
    for path in sorted(directory.glob("*.png")):
        label = path.stem.split("__")[0]
        if label not in ALL_LABELS:
            msg = f"unexpected template label in file name: {path.name}"
            raise ScoreboardParseError(msg)
        labels.append(label)
        patches.append(_normalise(read_gray(path), params.patch_width, params.patch_height))
    if not labels:
        msg = f"no templates in {directory}. {_CALIBRATE_HINT}"
        raise ScoreboardParseError(msg)
    return TemplateSet(labels, np.stack(patches))


def patch_to_png(patch: Patch) -> NDArray[np.uint8]:
    """Render a normalised patch as an 8-bit image (for template files and contact sheets)."""
    low, high = float(patch.min()), float(patch.max())
    if high - low < _EPS:
        return np.zeros(patch.shape, dtype=np.uint8)
    return np.asarray(((patch - low) / (high - low) * 255).round(), dtype=np.uint8)


def save_template(directory: Path, label: str, index: int, patch: Patch) -> Path:
    """Write one labelled template; returns the file path.

    Raises:
        ScoreboardParseError: unknown label.
    """
    if label not in ALL_LABELS:
        msg = f"unknown template label: {label!r}"
        raise ScoreboardParseError(msg)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{label}__{index}.png"
    write_png(path, patch_to_png(patch))
    return path


def cluster_patches(patches: Sequence[Patch], similarity: float) -> list[list[int]]:
    """Greedy clustering: a patch joins the first cluster whose first member it matches
    with correlation `>= similarity`, otherwise it starts a new cluster. Returns index lists,
    biggest cluster first (ties by first index), so the result is deterministic."""
    leaders: list[Patch] = []
    members: list[list[int]] = []
    for i, patch in enumerate(patches):
        for k, leader in enumerate(leaders):
            if float((leader * patch).sum()) >= similarity:
                members[k].append(i)
                break
        else:
            leaders.append(patch)
            members.append([i])
    return sorted(members, key=lambda m: (-len(m), m[0]))
