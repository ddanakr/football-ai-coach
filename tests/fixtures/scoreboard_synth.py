"""Synthetic broadcast frames with a scoreboard drawn by OpenCV (no real footage in tests)."""

import cv2
import numpy as np

from aicoach.scoreboard.board import crop_fraction
from aicoach.scoreboard.glyphs import Patch, TemplateSet, segment_glyphs
from aicoach.scoreboard.imaging import CLOCK_ROI, SCORE_HOME_ROI, ImageParams
from aicoach.scoreboard.video import Frame

WIDTH = 1280
HEIGHT = 720
BLUE = (150, 72, 4)  # BGR of the dark blue blocks
WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def grass(seed: int = 0, width: int = WIDTH, height: int = HEIGHT) -> Frame:
    """A noisy green picture standing in for the pitch."""
    rng = np.random.default_rng(seed)
    base = np.array([40, 120, 60], dtype=np.int16)
    noise = rng.integers(-25, 26, size=(height, width, 3), dtype=np.int16)
    return np.asarray(np.clip(base + noise, 0, 255), dtype=np.uint8)


def _centered_text(img: Frame, text: str, cx: int, cy: int, scale: float, thickness: int) -> None:
    (w, h), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    cv2.putText(img, text, (cx - w // 2, cy + h // 2), _FONT, scale, WHITE, thickness, cv2.LINE_8)


def _draw_clock(img: Frame, clock: str) -> None:
    """Digits right-aligned to the left of the colon and left-aligned to its right; the colon
    is two 2x2 dots, as on the real board."""
    left, _, right = clock.partition(":")
    (lw, _), _ = cv2.getTextSize(left, _FONT, 0.5, 1)
    cv2.putText(img, left, (189 - lw, 106), _FONT, 0.5, WHITE, 1, cv2.LINE_8)
    cv2.putText(img, right, (196, 106), _FONT, 0.5, WHITE, 1, cv2.LINE_8)
    cv2.rectangle(img, (191, 99), (192, 100), WHITE, -1)
    cv2.rectangle(img, (191, 104), (192, 105), WHITE, -1)


def draw_board(
    frame: Frame,
    score: tuple[int, int] = (0, 0),
    clock: str = "00:00",
    added: str | None = None,
) -> Frame:
    """Draw the scoreboard at its calibrated place (1280x720 geometry) and return the frame.

    `added` draws a red block with the text to the right of the clock ("+3").
    """
    cv2.rectangle(frame, (67, 59), (143, 86), (235, 235, 235), -1)
    cv2.rectangle(frame, (144, 59), (240, 86), BLUE, -1)
    cv2.rectangle(frame, (241, 59), (317, 86), (240, 240, 240), -1)
    cv2.rectangle(frame, (178, 58), (207, 87), WHITE, -1)
    cv2.rectangle(frame, (157, 86), (227, 112), BLUE, -1)
    cv2.putText(frame, "AKT", (90, 78), _FONT, 0.5, (20, 20, 20), 1, cv2.LINE_8)
    _centered_text(frame, str(score[0]), 161, 73, 0.8, 2)
    _centered_text(frame, str(score[1]), 224, 73, 0.8, 2)
    _draw_clock(frame, clock)
    if added is not None:
        cv2.rectangle(frame, (228, 90), (252, 110), (30, 30, 200), -1)
        _centered_text(frame, added, 240, 100, 0.4, 1)
    return frame


def board_frame(
    score: tuple[int, int] = (0, 0), clock: str = "00:00", seed: int = 0, added: str | None = None
) -> Frame:
    """Grass with the scoreboard on top."""
    return draw_board(grass(seed), score, clock, added)


def _labels_of(text: str) -> list[str]:
    return ["colon" if ch == ":" else ch for ch in text]


def synthetic_templates(params: ImageParams) -> tuple[TemplateSet, TemplateSet]:
    """Score and clock template sets labelled from the known drawn text (all digits and colon)."""
    score_labels: list[str] = []
    score_patches: list[Patch] = []
    for digit in range(10):
        glyphs = segment_glyphs(crop_fraction(board_frame((digit, 0)), SCORE_HOME_ROI), params)
        assert len(glyphs) == 1
        score_labels.append(str(digit))
        score_patches.append(glyphs[0].patch)
    clock_labels: list[str] = []
    clock_patches: list[Patch] = []
    for text in ("01:23", "45:67", "89:01"):
        glyphs = segment_glyphs(crop_fraction(board_frame(clock=text), CLOCK_ROI), params)
        labels = _labels_of(text)
        assert len(glyphs) == len(labels)
        clock_labels += labels
        clock_patches += [g.patch for g in glyphs]
    return (
        TemplateSet(score_labels, np.stack(score_patches)),
        TemplateSet(clock_labels, np.stack(clock_patches)),
    )
