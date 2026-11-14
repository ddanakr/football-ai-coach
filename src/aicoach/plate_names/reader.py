"""Reading one line of text from an image: the preparation of the picture and Tesseract.

`prepare_line` is pure. `TesseractReader` is the only place that calls the program; tests use a
fake with the same `read` method (`NameReader`). Tesseract is an outside program, not a Python
package, so it may be missing: that is reported as `OcrUnavailableError`, which the stage turns
into a skipped step (D-059).
"""

import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt
import pytesseract

from aicoach.errors import PlatesError

Gray = npt.NDArray[np.uint8]
WhichFunc = Callable[[str], str | None]
# Where the Windows installer puts Tesseract when it is not in PATH.
DEFAULT_TESSERACT_PATHS: tuple[Path, ...] = (
    Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
    Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
)
LANGUAGES = ("kaz", "rus")  # both are needed: Kazakh letters in the names, Russian ones too
_TIMEOUT_S = 30


class OcrUnavailableError(PlatesError):
    """Tesseract cannot be used: not installed, or a needed language is missing."""


class NameReader(Protocol):
    """Reads the text of one prepared line (the real one is Tesseract)."""

    def read(self, image: Gray) -> str:
        """The text of the picture, as the engine returns it."""
        ...


def prepare_line(gray: Gray, scale: int, pad_px: int) -> Gray:
    """The picture Tesseract reads best: grey, enlarged `scale` times (cubic), black and white
    by Otsu's threshold, with a white border of `pad_px` pixels.

    Chosen on the first half of the match only (D-059): at 2x one letter of a long name broke,
    3x and 4x gave the same text.

    Raises:
        PlatesError: the image has no pixels or is not a 2-dimensional grey image.
    """
    if gray.ndim != 2 or gray.size == 0:
        msg = "cannot read a line from an empty or colour image"
        raise PlatesError(msg)
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    padded = cv2.copyMakeBorder(
        binary, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_CONSTANT, value=255
    )
    return np.asarray(padded, dtype=np.uint8)


def find_tesseract(
    which: WhichFunc = shutil.which, defaults: Iterable[Path] = DEFAULT_TESSERACT_PATHS
) -> Path | None:
    """The Tesseract program: the one in PATH, else a default install folder; None if absent."""
    found = which("tesseract")
    if found:
        return Path(found)
    return next((p for p in defaults if p.is_file()), None)


class TesseractReader:
    """Tesseract in the one-line mode (`--psm 7`) with the languages `kaz+rus`."""

    def __init__(self, command: Path, languages: Iterable[str] = LANGUAGES, psm: int = 7) -> None:
        self._languages = "+".join(languages)
        self._psm = psm
        pytesseract.pytesseract.tesseract_cmd = str(command)

    @classmethod
    def create(
        cls, command: Path | None = None, languages: Iterable[str] = LANGUAGES
    ) -> "TesseractReader":
        """A reader for the Tesseract found in PATH or in the default folder.

        Raises:
            OcrUnavailableError: no Tesseract, it does not start, or a language is missing.
        """
        languages = tuple(languages)
        command = command or find_tesseract()
        if command is None:
            msg = "tesseract not found (not in PATH, not in the default install folder)"
            raise OcrUnavailableError(msg)
        pytesseract.pytesseract.tesseract_cmd = str(command)
        try:
            installed = set(pytesseract.get_languages(config=""))
        except (pytesseract.TesseractError, OSError, RuntimeError) as err:
            msg = f"tesseract at {command} does not run: {type(err).__name__}"
            raise OcrUnavailableError(msg) from err
        absent = [lang for lang in languages if lang not in installed]
        if absent:
            msg = f"tesseract has no language data for: {', '.join(absent)}"
            raise OcrUnavailableError(msg)
        return cls(command, languages)

    def read(self, image: Gray) -> str:
        """The text of a prepared line, stripped.

        Raises:
            PlatesError: Tesseract failed or ran longer than the timeout.
        """
        try:
            text: str = pytesseract.image_to_string(
                image, lang=self._languages, config=f"--psm {self._psm}", timeout=_TIMEOUT_S
            )
        except (pytesseract.TesseractError, OSError, RuntimeError) as err:
            msg = f"tesseract failed on a line: {type(err).__name__}"
            raise PlatesError(msg) from err
        return text.strip()
