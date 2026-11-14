"""Where the line of text is inside a saved plate crop (fractions of the crop).

The crop is the plate region of `plates/layout.py` (frame pixels 60,580 to 480,665 of a 1280x720
frame). The text lines were measured on the same match as that layout (KFF, 2026 season); another
broadcast may need other numbers. A goal or card plate has one line "number NAME SURNAME"; a
substitution plate has two, the upper one next to the green arrow (the player who came on) and
the lower one next to the red arrow (who left). The number is read with the name and cut later.
"""

from pydantic import BaseModel, ConfigDict

from aicoach.plates.layout import PlateLayout
from aicoach.scoreboard.schemas import Roi

_REF_W = 1280
_REF_H = 720
_PLATE = PlateLayout().plate


def _in_crop(x0: int, y0: int, x1: int, y1: int) -> Roi:
    """A region given in frame pixels, stored as fractions of the plate crop."""
    origin_x, origin_y = _PLATE.x0 * _REF_W, _PLATE.y0 * _REF_H
    width, height = (_PLATE.x1 - _PLATE.x0) * _REF_W, (_PLATE.y1 - _PLATE.y0) * _REF_H
    return Roi(
        x0=(x0 - origin_x) / width,
        y0=(y0 - origin_y) / height,
        x1=(x1 - origin_x) / width,
        y1=(y1 - origin_y) / height,
    )


class NameLayout(BaseModel):
    """Text lines of the plate crop (fractions of the crop)."""

    model_config = ConfigDict(frozen=True)

    # The plate region of the frame that the crops were cut from; `plates.json` must agree.
    plate: Roi = _PLATE
    # Goal and card plates: the line in the light panel, right of the logo block.
    single: Roi = _in_crop(143, 621, 412, 651)
    # Substitution plates: the line of the player who came on (green arrow) and of the one
    # who left (red arrow); the arrows themselves are left out.
    went_on: Roi = _in_crop(170, 594, 466, 618)
    went_off: Roi = _in_crop(170, 626, 466, 650)
