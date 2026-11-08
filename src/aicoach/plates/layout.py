"""Where the lower-third plate and its parts are on the frame (fractions of the frame size).

Calibrated on one KFF match (1280x720); every pixel position of the plate graphics enters the
code only here. The blue logo block has the same position and size in all plates seen so far,
the white panel is wider for substitutions than for the other plates, so the plate region
covers the wider one. Other broadcasts or seasons may differ: recalibrate.
"""

from pydantic import BaseModel, ConfigDict

from aicoach.scoreboard.schemas import Roi

_REF_W = 1280
_REF_H = 720


def _px(x0: int, y0: int, x1: int, y1: int) -> Roi:
    """A region given in pixels of a 1280x720 frame, stored as fractions."""
    return Roi(x0=x0 / _REF_W, y0=y0 / _REF_H, x1=x1 / _REF_W, y1=y1 / _REF_H)


class PlateLayout(BaseModel):
    """Regions of the plate (all in frame fractions)."""

    model_config = ConfigDict(frozen=True)

    # The whole plate with a margin: this crop is saved for reading the names later.
    plate: Roi = _px(60, 580, 480, 665)
    # Dark blue square with the team logo at the left end of every plate.
    block: Roi = _px(68, 589, 134, 655)
    # Light panel under the header (white in every plate type).
    panel: Roi = _px(141, 619, 411, 653)
    # Header strip: yellow for a card, light grey for a goal or a coach plate.
    header: Roi = _px(141, 592, 251, 613)
    # Orange minute tag of a goal plate ("NN' MIN").
    tag: Roi = _px(236, 593, 314, 612)
    # Yellow tab at the right end of a card plate.
    tab: Roi = _px(405, 591, 427, 613)
    # Green (went on) and red (went off) triangles of a substitution plate.
    arrow_up: Roi = _px(139, 596, 163, 615)
    arrow_down: Roi = _px(139, 628, 163, 647)
    # The logo inside the block, without the blue frame around it.
    logo: Roi = _px(76, 597, 126, 647)
    # The block's frame (ring) is this share of the block's size wide.
    ring_inset: float = 0.12
