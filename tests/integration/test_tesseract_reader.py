"""The real Tesseract on a synthetic plate crop. Skipped when the program or its English data
is missing (the project reads `kaz+rus`, but a drawn Latin line needs no Kazakh data)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from aicoach.plate_names.layout import NameLayout
from aicoach.plate_names.reader import OcrUnavailableError, TesseractReader, prepare_line
from aicoach.plate_names.stage import run_plate_names
from aicoach.scoreboard.board import crop_fraction
from tests.fixtures.plate_names_synth import CROP_SHAPE, make_plate, write_crop, write_match


def english_reader() -> TesseractReader:
    try:
        return TesseractReader.create(languages=("eng",))
    except OcrUnavailableError as err:
        pytest.skip(f"Tesseract is not usable here: {err}")


def crop_with_text(text: str) -> np.ndarray:
    """A light plate crop with a dark line of `text` where a goal plate has its name."""
    crop = np.full(CROP_SHAPE, 235, dtype=np.uint8)
    region = NameLayout().single
    height, width = CROP_SHAPE
    x, y = int(region.x0 * width) + 4, int(region.y1 * height) - 8
    cv2.putText(crop, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 20, 2, cv2.LINE_AA)
    return crop


@pytest.mark.integration
def test_tesseract_reads_a_prepared_line_of_a_synthetic_crop() -> None:
    reader = english_reader()
    line = crop_fraction(crop_with_text("9 SMITH ROMAN"), NameLayout().single)

    text = reader.read(prepare_line(line, scale=3, pad_px=20))

    assert "SMITH" in text.upper()
    assert "ROMAN" in text.upper()


@pytest.mark.integration
def test_plate_names_stage_with_the_real_tesseract_on_a_synthetic_plate(tmp_path: Path) -> None:
    reader = english_reader()
    plate = make_plate("plate_001", "goal", 100.0)
    write_match(tmp_path, [plate], crops=False)
    write_crop(tmp_path, plate, crop_with_text("9 SMITH ROMAN"))

    result = run_plate_names(tmp_path, "m", reader=reader)

    assert result is not None
    (player,) = result.names[0].players
    assert player.status == "unchecked"
    assert player.name is not None
    assert player.name.upper() == "SMITH ROMAN"
