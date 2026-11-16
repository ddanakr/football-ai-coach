"""Synthetic broadcast frames with a lower-third plate, drawn with OpenCV (no real footage).

Plates are drawn with the regions of `PlateLayout`, so the frames work at any size.
"""

from typing import Literal

import cv2
import numpy as np

from aicoach.plates.detect import FrameReading
from aicoach.plates.layout import PlateLayout
from aicoach.plates.schemas import CardColor, PlateType, RejectReason
from aicoach.plates.tracks import Hit
from aicoach.scoreboard.schemas import (
    OffsetSegment,
    Roi,
    Sample,
    ScoreboardResult,
    ScoreChange,
    Stats,
    VisibleSegment,
)
from aicoach.scoreboard.video import Frame

PlateKind = Literal["subst", "goal", "card", "card_red", "coach", "round_dots"]
WIDTH = 640
HEIGHT = 360
BLUE = (163, 76, 8)  # BGR, the block's blue
WHITE = (240, 240, 240)
GREY = (205, 205, 205)
YELLOW = (40, 220, 250)
ORANGE = (20, 115, 235)
GREEN = (60, 200, 40)
RED = (30, 30, 220)
HOME_LOGO = (40, 40, 220)  # a red logo
AWAY_LOGO = (30, 220, 240)  # a yellow logo


def box(frame: Frame, roi: Roi) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) of a region on `frame`."""
    h, w = frame.shape[:2]
    return round(roi.x0 * w), round(roi.y0 * h), round(roi.x1 * w), round(roi.y1 * h)


def _fill(frame: Frame, roi: Roi, colour: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box(frame, roi)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), colour, thickness=-1)


def triangle(frame: Frame, roi: Roi, colour: tuple[int, int, int], up: bool) -> None:
    x0, y0, x1, y1 = box(frame, roi)
    mx = (x0 + x1) // 2
    pad = max(1, (y1 - y0) // 6)
    top, bottom = y0 + pad, y1 - pad - 1
    points = (
        [(mx, top), (x0 + 1, bottom), (x1 - 2, bottom)]
        if up
        else [
            (mx, bottom),
            (x0 + 1, top),
            (x1 - 2, top),
        ]
    )
    cv2.fillPoly(frame, [np.array(points, dtype=np.int32)], colour)


def _dot(frame: Frame, roi: Roi, colour: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box(frame, roi)
    radius = max(2, min(x1 - x0, y1 - y0) // 2 - 1)
    cv2.circle(frame, ((x0 + x1) // 2, (y0 + y1) // 2), radius, colour, thickness=-1)


def background(width: int = WIDTH, height: int = HEIGHT, seed: int = 0) -> Frame:
    """A grass-like noisy background."""
    rng = np.random.default_rng(seed)
    base = np.array([60, 140, 70], dtype=np.int16)
    noise = rng.integers(-12, 13, size=(height, width, 3), dtype=np.int16)
    return np.asarray(np.clip(base + noise, 0, 255), dtype=np.uint8)


def _text_bars(frame: Frame, layout: PlateLayout, text_seed: int) -> None:
    """Dark bars in the panel standing for text; the seed changes the picture (another name)."""
    x0, y0, x1, y1 = box(frame, layout.panel)
    rng = np.random.default_rng(text_seed)
    for row in range(2):
        top = y0 + 2 + row * ((y1 - y0) // 2)
        start = x0 + int(rng.integers(2, max(3, (x1 - x0) // 4)))
        end = start + int(rng.integers((x1 - x0) // 4, (x1 - x0) // 2))
        cv2.rectangle(frame, (start, top), (end, top + max(1, (y1 - y0) // 6)), (30, 30, 30), -1)


def draw_plate(
    frame: Frame,
    kind: PlateKind,
    logo: tuple[int, int, int] = HOME_LOGO,
    text_seed: int = 0,
    layout: PlateLayout | None = None,
) -> Frame:
    """Draw a plate on `frame` (in place) and return it.

    `coach` is a plate without any type marker, `card_red` has a red header, `round_dots`
    has round green and red dots where a substitution has triangles.
    """
    layout = layout or PlateLayout()
    bx0, by0, bx1, by1 = box(frame, layout.block)
    _, _, tab_x1, _ = box(frame, layout.tab)
    cv2.rectangle(frame, (bx1, by0), (tab_x1, by1 - 1), WHITE, thickness=-1)
    x0, y0, x1, y1 = box(frame, layout.header)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), GREY, thickness=-1)
    _fill(frame, layout.block, BLUE)
    _dot(frame, layout.logo, logo)
    _text_bars(frame, layout, text_seed)
    if kind == "goal":
        _fill(frame, layout.tag, ORANGE)
    elif kind in ("card", "card_red"):
        _fill(frame, layout.header, YELLOW if kind == "card" else RED)
        if kind == "card":
            _fill(frame, layout.tab, YELLOW)
    elif kind == "subst":
        triangle(frame, layout.arrow_up, GREEN, up=True)
        triangle(frame, layout.arrow_down, RED, up=False)
    elif kind == "round_dots":
        _dot(frame, layout.arrow_up, GREEN)
        _dot(frame, layout.arrow_down, RED)
    return frame


def plate_frame(
    kind: PlateKind | None,
    logo: tuple[int, int, int] = HOME_LOGO,
    text_seed: int = 0,
    seed: int = 0,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> Frame:
    """A background frame, with a plate of `kind` if given."""
    frame = background(width, height, seed)
    if kind is not None:
        draw_plate(frame, kind, logo, text_seed)
    return frame


def blue_object_frame(seed: int = 0) -> Frame:
    """A blue box where the logo block is, with no panel beside it (an advertising board)."""
    frame = background(seed=seed)
    _fill(frame, PlateLayout().block, BLUE)
    return frame


def scoreboard_result(
    changes: list[ScoreChange] | None = None,
    n_samples: int = 12,
    visible: set[int] | None = None,
    segments: list[tuple[float, float]] | None = None,
    decode_mode: Literal["opencv_gpu", "opencv_cpu"] = "opencv_gpu",
) -> ScoreboardResult:
    """A minimal scoreboard.json with one sample per second.

    `visible` lists the sample indices where the board is visible (default: all).
    """
    on = set(range(n_samples)) if visible is None else visible
    samples = [Sample(t_v=float(k), visible=k in on) for k in range(n_samples)]
    spans = segments if segments is not None else [(0.0, float(n_samples - 1))]
    offsets = [
        OffsetSegment(
            start=a, end=b, clock_start_s=0, clock_end_s=int(b - a), offset_s=a, n_readings=2
        )
        for a, b in spans
    ]
    stats = Stats(
        n_samples=n_samples,
        n_visible=len(on),
        n_unreadable=0,
        n_clock_unreadable=0,
        n_score_unreadable=0,
        n_outliers=0,
        n_running=0,
        n_jumps=0,
        video_duration_s=float(n_samples),
        visible_share_between_first_and_last=None,
        unreadable_share_in_visible=None,
        median_offset_by_segment=[],
    )
    return ScoreboardResult(
        video="match.mp4",
        decode_mode=decode_mode,
        sample_fps=1.0,
        roi=Roi(x0=0.05, y0=0.08, x1=0.25, y1=0.15),
        samples=samples,
        visible_segments=[VisibleSegment(start=a, end=b) for a, b in spans],
        score_changes=changes or [],
        offset_jumps=[],
        offset_segments=offsets,
        stats=stats,
    )


def change(old: tuple[int, int], new: tuple[int, int], t_old: float, t_new: float) -> ScoreChange:
    """A score change window."""
    direction: Literal["up", "down", "other"] = (
        "up" if sum(new) == sum(old) + 1 else "down" if sum(new) < sum(old) else "other"
    )
    return ScoreChange(
        t_old_last_seen=t_old, t_new_first_seen=t_new, old=old, new=new, direction=direction
    )


def hit(
    t: float,
    kind: PlateType | None = "goal",
    reject: RejectReason | None = None,
    sig: float = 0.0,
    color: CardColor | None = None,
) -> Hit:
    """A hit at `t`; the panel signature is a flat picture of grey level `sig`."""
    reading = FrameReading(
        kind=kind,
        card_color=color,
        reject=reject,
        conf=1.0,
        anchor=0.9,
        signature=np.full((8, 54), sig, dtype=np.float32),
    )
    return Hit(round(t * 2), t, reading)


def series(
    start: float,
    end: float,
    kind: PlateType | None = "goal",
    reject: RejectReason | None = None,
    sig: float = 0.0,
    color: CardColor | None = None,
) -> list[Hit]:
    """Hits every 0.5 s from `start` to `end`."""
    count = round((end - start) * 2) + 1
    return [hit(start + i * 0.5, kind, reject, sig, color) for i in range(count)]
