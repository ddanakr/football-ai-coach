"""Tests of the picture preparation and of the Tesseract wrapper (the program is replaced by
fakes: no unit test needs it installed)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytesseract
import pytest

from aicoach.errors import PlatesError
from aicoach.plate_names.reader import (
    OcrUnavailableError,
    TesseractReader,
    find_tesseract,
    prepare_line,
)


@pytest.mark.unit
def test_prepare_line_enlarges_binarizes_and_pads() -> None:
    line = np.full((20, 100), 230, dtype=np.uint8)
    line[8:12, 10:60] = 30  # dark text on a light panel

    out = prepare_line(line, scale=3, pad_px=20)

    assert out.shape == (20 * 3 + 40, 100 * 3 + 40)
    assert out.dtype == np.uint8
    assert set(np.unique(out)) == {0, 255}
    assert (out[:20] == 255).all()  # the border is white
    assert (out == 0).any()  # the text stays black


@pytest.mark.unit
def test_prepare_line_without_padding_keeps_the_scaled_size() -> None:
    out = prepare_line(np.full((10, 30), 200, dtype=np.uint8), scale=2, pad_px=0)

    assert out.shape == (20, 60)


@pytest.mark.unit
@pytest.mark.parametrize(
    "image",
    [np.zeros((0, 10), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8)],
)
def test_prepare_line_empty_or_colour_image_raises(image: Any) -> None:
    with pytest.raises(PlatesError, match="empty or colour"):
        prepare_line(image, scale=3, pad_px=20)


@pytest.mark.unit
def test_find_tesseract_prefers_the_one_in_path(tmp_path: Path) -> None:
    default = tmp_path / "tesseract.exe"
    default.write_text("x")

    found = find_tesseract(lambda _name: "C:/tools/tesseract.exe", [default])

    assert found == Path("C:/tools/tesseract.exe")


@pytest.mark.unit
def test_find_tesseract_falls_back_to_the_default_folder(tmp_path: Path) -> None:
    missing, present = tmp_path / "a" / "tesseract.exe", tmp_path / "tesseract.exe"
    present.write_text("x")

    assert find_tesseract(lambda _name: None, [missing, present]) == present


@pytest.mark.unit
def test_find_tesseract_not_installed_is_none(tmp_path: Path) -> None:
    assert find_tesseract(lambda _name: None, [tmp_path / "tesseract.exe"]) is None
    assert find_tesseract(lambda _name: None, []) is None


class _FakePytesseract:
    """Stands in for the pytesseract functions the wrapper calls."""

    def __init__(self, languages: list[str] | Exception, text: str | Exception = "") -> None:
        self.languages, self.text = languages, text
        self.calls: list[dict[str, Any]] = []

    def get_languages(self, config: str = "") -> list[str]:  # noqa: ARG002
        if isinstance(self.languages, Exception):
            raise self.languages
        return self.languages

    def image_to_string(self, image: Any, **kwargs: Any) -> str:  # noqa: ARG002
        self.calls.append(kwargs)
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


def _patch(monkeypatch: pytest.MonkeyPatch, fake: _FakePytesseract) -> None:
    monkeypatch.setattr(pytesseract, "get_languages", fake.get_languages)
    monkeypatch.setattr(pytesseract, "image_to_string", fake.image_to_string)


@pytest.mark.unit
def test_create_without_tesseract_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aicoach.plate_names.reader.find_tesseract", lambda: None)

    with pytest.raises(OcrUnavailableError, match="not found"):
        TesseractReader.create()


@pytest.mark.unit
def test_create_missing_language_is_named(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch, _FakePytesseract(["eng", "rus"]))

    with pytest.raises(OcrUnavailableError, match="kaz"):
        TesseractReader.create(tmp_path / "tesseract.exe")


@pytest.mark.unit
@pytest.mark.parametrize("error", [OSError("no such file"), RuntimeError("boom")])
def test_create_program_that_does_not_run_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Exception
) -> None:
    _patch(monkeypatch, _FakePytesseract(error))

    with pytest.raises(OcrUnavailableError, match="does not run"):
        TesseractReader.create(tmp_path / "tesseract.exe")


@pytest.mark.unit
def test_read_uses_the_kazakh_and_russian_models_in_the_one_line_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakePytesseract(["eng", "kaz", "rus"], text="  9 ИВАНОВ \n")
    _patch(monkeypatch, fake)
    reader = TesseractReader.create(tmp_path / "tesseract.exe")

    text = reader.read(np.zeros((5, 5), dtype=np.uint8))

    assert text == "9 ИВАНОВ"
    assert fake.calls[0]["lang"] == "kaz+rus"
    assert fake.calls[0]["config"] == "--psm 7"
    assert fake.calls[0]["timeout"] > 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "error",
    [pytesseract.TesseractError(1, "bad image"), RuntimeError("Tesseract process timeout")],
)
def test_read_failure_is_a_plates_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Exception
) -> None:
    _patch(monkeypatch, _FakePytesseract(["kaz", "rus"], text=error))
    reader = TesseractReader.create(tmp_path / "tesseract.exe")

    with pytest.raises(PlatesError, match="tesseract failed"):
        reader.read(np.zeros((5, 5), dtype=np.uint8))
