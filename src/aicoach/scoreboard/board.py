"""Scoreboard visibility: compare the board region with a reference built from the video.

The reference is the per-pixel median of frames where the board is on screen, together with a
mask of the pixels that do not change (frames, team boxes, logo). Digits and the picture behind
are not in the mask. Visibility is the normalised correlation over the masked pixels.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.schemas import Roi
from aicoach.scoreboard.video import Frame

REFERENCE_FILE = "board_reference.npz"
_MIN_MASK_SHARE = 0.10  # a mask thinner than this cannot tell the board from the picture
_EPS = 1e-9


@dataclass(frozen=True, eq=False)
class BoardReference:
    """Median board image (H, W, 3, float32) and the mask of its static pixels (H, W)."""

    pixels: NDArray[np.float32]
    mask: NDArray[np.bool_]


def crop_fraction(frame: Frame, roi: Roi) -> Frame:
    """Cut a region given in fractions of the frame size; the result has at least one pixel.

    Raises:
        ScoreboardParseError: the frame itself has no pixels.
    """
    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        msg = "frame has no pixels"
        raise ScoreboardParseError(msg)
    x0 = min(int(round(roi.x0 * width)), width - 1)
    y0 = min(int(round(roi.y0 * height)), height - 1)
    x1 = max(int(round(roi.x1 * width)), x0 + 1)
    y1 = max(int(round(roi.y1 * height)), y0 + 1)
    return frame[y0:y1, x0:x1]


def board_blue_fraction(crop: Frame) -> float:
    """Share of pixels with the dark blue of the score and clock blocks.

    Used only to choose frames for the reference: advertising boards are cyan and do not match.
    """
    if crop.size == 0:
        return 0.0
    blue = crop[..., 0].astype(np.int16)
    green = crop[..., 1].astype(np.int16)
    red = crop[..., 2].astype(np.int16)
    hit = (red < 40) & (green > 40) & (green < 110) & (blue > 100) & (blue < 200)
    return float(hit.mean())


def red_fraction(crop: Frame) -> float:
    """Share of strongly red pixels: the "+N" block of the added time."""
    if crop.size == 0:
        return 0.0
    blue = crop[..., 0].astype(np.int16)
    green = crop[..., 1].astype(np.int16)
    red = crop[..., 2].astype(np.int16)
    return float(((red > 150) & (green < 70) & (blue < 70)).mean())


def build_reference(crops: Sequence[Frame], static_std: float) -> BoardReference:
    """Median image and static-pixel mask of board crops that all have the same size.

    Raises:
        ScoreboardParseError: no crops, different sizes, or too few static pixels.
    """
    if not crops:
        msg = (
            "no frames with a visible scoreboard to build a reference: check the video and "
            "the scoreboard region in scoreboard/imaging.py"
        )
        raise ScoreboardParseError(msg)
    shape = crops[0].shape
    if any(c.shape != shape for c in crops):
        msg = "reference crops must have the same size"
        raise ScoreboardParseError(msg)
    stack = np.stack(crops).astype(np.float32)
    pixels = np.median(stack, axis=0).astype(np.float32)
    mask = stack.std(axis=0).max(axis=-1) < static_std
    if float(mask.mean()) < _MIN_MASK_SHARE:
        msg = "reference has too few static pixels: the crops do not show one scoreboard"
        raise ScoreboardParseError(msg)
    return BoardReference(pixels=pixels, mask=mask)


def similarity(crop: Frame, reference: BoardReference) -> float:
    """Normalised correlation of the crop with the reference over the static pixels, in [-1, 1].

    A crop of another size is resized first. Returns 0.0 if either side has no contrast.
    """
    height, width = reference.mask.shape
    if crop.size == 0:
        return 0.0
    if crop.shape[:2] != (height, width):
        crop = np.asarray(
            cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA), dtype=np.uint8
        )
    x = crop[reference.mask].astype(np.float64)
    y = reference.pixels[reference.mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom < _EPS:
        return 0.0
    return float((x * y).sum() / denom)


def save_reference(reference: BoardReference, directory: Path) -> Path:
    """Write the reference into the templates directory and return the file path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REFERENCE_FILE
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, pixels=reference.pixels, mask=reference.mask)
    tmp.replace(path)
    return path


def load_reference(directory: Path) -> BoardReference:
    """Read the reference from the templates directory.

    Raises:
        ScoreboardParseError: the file is missing or broken; the message says how to create it.
    """
    path = directory / REFERENCE_FILE
    if not path.is_file():
        msg = (
            f"scoreboard reference not found: {path}. "
            "Calibrate first: python -m aicoach.cli scoreboard-calibrate --video <file>"
        )
        raise ScoreboardParseError(msg)
    try:
        with np.load(path) as data:
            return BoardReference(
                pixels=data["pixels"].astype(np.float32), mask=data["mask"].astype(np.bool_)
            )
    except (OSError, ValueError, KeyError) as err:
        msg = f"broken scoreboard reference: {path}"
        raise ScoreboardParseError(msg) from err
