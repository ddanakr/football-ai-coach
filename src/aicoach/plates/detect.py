"""Find a plate in one frame by colour and shape and tell its type.

All plates share a dark blue square with the team logo at a fixed place and a light panel next to
it. The type comes from what sits in the header: an orange minute tag (goal), a yellow header with
a yellow tab (card), a green and a red triangle (substitution). A plate-like frame without any of
them (a coach or referee plate) is rejected, and so is a blue object without a panel. Nothing
here reads text.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from aicoach.plates.config import DetectParams
from aicoach.plates.layout import PlateLayout
from aicoach.plates.schemas import CardColor, PlateType, RejectReason
from aicoach.scoreboard.board import crop_fraction
from aicoach.scoreboard.video import Frame

_SIGNATURE_SIZE = (54, 8)  # width, height of the downsized panel picture
_OTHER_SATURATION = 150
_OTHER_VALUE = 120
_YELLOW_HUE = (18, 38)  # OpenCV hue units (0..179), excluded from "another colour"

Mask = NDArray[np.bool_]
MaskFn = Callable[[NDArray[np.int16], NDArray[np.int16], NDArray[np.int16]], Mask]


@dataclass(frozen=True, eq=False)
class FrameReading:
    """What one frame shows. `kind` is set for an accepted plate, `reject` for a candidate that
    was refused, both None if the frame has no candidate. `signature` is a small picture of the
    panel (to tell two plates of one type apart), `logo` the logo crop; both only for candidates.
    """

    kind: PlateType | None
    card_color: CardColor | None
    reject: RejectReason | None
    conf: float
    anchor: float
    signature: NDArray[np.float32] | None = None
    logo: Frame | None = None

    @property
    def is_candidate(self) -> bool:
        """True if the frame has a plate or a refused plate-like object."""
        return self.kind is not None or self.reject is not None


def _blue(r: NDArray[np.int16], g: NDArray[np.int16], b: NDArray[np.int16]) -> Mask:
    """The dark blue of the logo block (the same blue as the scoreboard's)."""
    return (r < 40) & (g > 40) & (g < 110) & (b > 100) & (b < 200)


def _yellow(r: NDArray[np.int16], g: NDArray[np.int16], b: NDArray[np.int16]) -> Mask:
    return (r > 200) & (g > 180) & (b < 90)


def _orange(r: NDArray[np.int16], g: NDArray[np.int16], b: NDArray[np.int16]) -> Mask:
    return (r > 200) & (g > 90) & (g < 170) & (b < 70)


def _green(r: NDArray[np.int16], g: NDArray[np.int16], b: NDArray[np.int16]) -> Mask:
    return (g > 170) & (r < 110) & (b < 120) & (g - r > 80)


def _red(r: NDArray[np.int16], g: NDArray[np.int16], b: NDArray[np.int16]) -> Mask:
    return (r > 180) & (g < 90) & (b < 90)


def share(crop: Frame, mask_fn: MaskFn) -> float:
    """Share of pixels of a BGR crop for which `mask_fn(r, g, b)` is true; 0.0 for no pixels."""
    if crop.size == 0:
        return 0.0
    pixels = crop.astype(np.int16)
    return float(mask_fn(pixels[..., 2], pixels[..., 1], pixels[..., 0]).mean())


def halves_count(crop: Frame, mask_fn: MaskFn) -> tuple[int, int]:
    """Pixels of the mask in the upper and in the lower half of a BGR crop."""
    if crop.size == 0:
        return 0, 0
    pixels = crop.astype(np.int16)
    mask = mask_fn(pixels[..., 2], pixels[..., 1], pixels[..., 0])
    middle = mask.shape[0] // 2
    return int(mask[:middle].sum()), int(mask[middle:].sum())


def ring_blue_share(block: Frame, inset: float) -> float:
    """Share of the plate's blue in the frame (ring) of the logo block.

    The logo covers the middle of the block, so only the ring is looked at.
    """
    height, width = block.shape[:2]
    if height == 0 or width == 0:
        return 0.0
    margin = max(1, round(inset * min(height, width)))
    ring = np.ones((height, width), dtype=np.bool_)
    ring[margin : height - margin, margin : width - margin] = False
    if not ring.any():
        return 0.0
    pixels = block[ring].astype(np.int16)
    return float(_blue(pixels[:, 2], pixels[:, 1], pixels[:, 0]).mean())


def light_share(panel: Frame, level: int) -> float:
    """Share of light pixels (darkest channel above `level`) in the panel."""
    if panel.size == 0:
        return 0.0
    return float((panel.min(axis=2) > level).mean())


def other_header_share(header: Frame) -> float:
    """Share of strongly coloured pixels that are not yellow: a header of another colour."""
    if header.size == 0:
        return 0.0
    hsv = cv2.cvtColor(header, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    strong = (sat > _OTHER_SATURATION) & (val > _OTHER_VALUE)
    yellow = (hue >= _YELLOW_HUE[0]) & (hue <= _YELLOW_HUE[1])
    return float((strong & ~yellow).mean())


def panel_signature(panel: Frame) -> NDArray[np.float32]:
    """A small grey picture of the panel in 0..1. Stable within one plate, different between
    plates (the text changes)."""
    if panel.size == 0:
        return np.zeros((_SIGNATURE_SIZE[1], _SIGNATURE_SIZE[0]), dtype=np.float32)
    grey = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, _SIGNATURE_SIZE, interpolation=cv2.INTER_AREA)
    return np.asarray(small, dtype=np.float32) / 255.0


def signature_distance(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Mean absolute difference of two signatures (0..1)."""
    return float(np.abs(a - b).mean())


def _scaled(value: float, ref: float) -> float:
    return min(1.0, value / ref)


def _arrows_conf(frame: Frame, layout: PlateLayout, params: DetectParams) -> float | None:
    """Confidence of a green up-triangle over a red down-triangle, None if they are not there.

    Both colours must cover enough of their regions, and the shapes must point the right way
    (a round green or red dot is not an arrow).
    """
    up_crop = crop_fraction(frame, layout.arrow_up)
    down_crop = crop_fraction(frame, layout.arrow_down)
    up, down = share(up_crop, _green), share(down_crop, _red)
    if up < params.min_arrow or down < params.min_arrow:
        return None
    up_top, up_bottom = halves_count(up_crop, _green)
    down_top, down_bottom = halves_count(down_crop, _red)
    if up_bottom < params.arrow_balance * up_top or down_top < params.arrow_balance * down_bottom:
        return None
    return min(_scaled(up, params.arrow_ref), _scaled(down, params.arrow_ref))


def classify_header(
    frame: Frame, layout: PlateLayout, params: DetectParams
) -> tuple[PlateType | None, CardColor | None, float]:
    """The type from the header markers, the card colour and the marker's confidence (0..1).

    Goal is checked first, then card, then substitution, then a header of another colour.
    """
    tag = share(crop_fraction(frame, layout.tag), _orange)
    if tag >= params.min_tag:
        return "goal", None, _scaled(tag, params.tag_ref)
    header = share(crop_fraction(frame, layout.header), _yellow)
    tab = share(crop_fraction(frame, layout.tab), _yellow)
    if header >= params.min_header_yellow and tab >= params.min_tab:
        conf = min(_scaled(header, params.header_yellow_ref), _scaled(tab, params.tab_ref))
        return "card", "yellow", conf
    arrows = _arrows_conf(frame, layout, params)
    if arrows is not None:
        return "subst", None, arrows
    other = other_header_share(crop_fraction(frame, layout.header))
    if other >= params.min_header_other:
        return "card", "other", _scaled(other, params.header_other_ref)
    return None, None, 0.0


def read_frame(frame: Frame, layout: PlateLayout, params: DetectParams) -> FrameReading:
    """Read one frame.

    A frame is a candidate if the ring of the logo block is blue enough. A candidate without a
    light panel is rejected as `no_panel`, one without a type marker as `no_marker`. The
    confidence of a plate is the smallest of the scaled block, panel and marker shares.

    Raises:
        ScoreboardParseError: the frame has no pixels.
    """
    block = crop_fraction(frame, layout.block)
    anchor = ring_blue_share(block, layout.ring_inset)
    if anchor < params.min_anchor:
        return FrameReading(None, None, None, 0.0, anchor)
    panel_crop = crop_fraction(frame, layout.panel)
    panel = light_share(panel_crop, params.panel_light)
    signature = panel_signature(panel_crop)
    logo = crop_fraction(frame, layout.logo).copy()
    if panel < params.min_panel:
        return FrameReading(None, None, "no_panel", 0.0, anchor, signature, logo)
    kind, color, marker_conf = classify_header(frame, layout, params)
    if kind is None:
        return FrameReading(None, None, "no_marker", 0.0, anchor, signature, logo)
    conf = min(_scaled(anchor, params.anchor_ref), _scaled(panel, params.panel_ref), marker_conf)
    return FrameReading(kind, color, None, conf, anchor, signature, logo)
