"""Inputs for the plate-names tests: a match directory with `plates.json` and crops, a lineup
with invented players, and a reader that returns prepared texts. No real names, no real video."""

from pathlib import Path

import cv2
import numpy as np

from aicoach.plate_names.layout import NameLayout
from aicoach.plate_names.reader import Gray
from aicoach.plates.schemas import Plate, PlateType, Team
from tests.fixtures.fusion_synth import plate, plates_result

LINEUP_CSV = """team,side,number,name,position,started,minute_on,minute_off
Север,home,9,Лебедев Роман,FW,1,,
Север,home,5,Мирзоев Тимур,MF,1,,70
Север,home,14,Стрельцов Антон Павлович,DF,1,,
Юг,away,10,Нурланов Айдар,MF,1,,
Юг,away,7,Лебедев Артур,FW,1,,
Юг,away,21,Қосанов Ерлан,DF,0,60,
"""
CROP_SHAPE = (85, 420)  # the size of a plate crop in a 1280x720 frame


class FakeReader:
    """Returns the prepared texts one by one and keeps the images it was given."""

    def __init__(self, *texts: str) -> None:
        self._texts = list(texts)
        self.images: list[Gray] = []

    def read(self, image: Gray) -> str:
        """The next prepared text."""
        self.images.append(image)
        if not self._texts:
            msg = "the fake reader has no more texts"
            raise AssertionError(msg)
        return self._texts.pop(0)


def make_plate(
    plate_id: str, kind: PlateType, t: float, *, team: Team | None = "home", half: int = 1
) -> Plate:
    """A plate of `kind` first seen at `t` (half 1 or 2)."""
    base = plate(plate_id, kind, t, team=team)
    return base.model_copy(update={"half": half})


def gray_crop() -> Gray:
    """A plain light crop of the plate size."""
    return np.full(CROP_SHAPE, 230, dtype=np.uint8)


def write_crop(match_dir: Path, plate_: Plate, image: Gray | None = None) -> None:
    """Save the crop of a plate where `plates.json` says it is."""
    path = match_dir / plate_.crop
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", gray_crop() if image is None else image)
    assert ok
    path.write_bytes(buffer.tobytes())


def write_match(
    match_dir: Path, plates: list[Plate], lineup: str | bytes | None = None, crops: bool = True
) -> None:
    """`plates.json` (with the plate region the name layout expects), crops and a lineup."""
    match_dir.mkdir(parents=True, exist_ok=True)
    result = plates_result(plates).model_copy(update={"roi": NameLayout().plate})
    (match_dir / "plates.json").write_text(result.model_dump_json(), encoding="utf-8")
    if crops:
        for p in plates:
            write_crop(match_dir, p)
    if lineup is not None:
        data = lineup.encode("utf-8") if isinstance(lineup, str) else lineup
        (match_dir / "lineups.csv").write_bytes(data)
