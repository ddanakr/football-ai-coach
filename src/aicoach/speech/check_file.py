"""The listener's check file (`speech_check.txt`).

Format: a header line `HH:MM:SS - HH:MM:SS <language>` (video time; the language is `kz`, `ru` or
`mixed`; spaces around the dash are optional) followed by lines with the words or phrases the
listener heard, up to the next header. Blank lines are skipped, and lines before the first
header are ignored (they belong to no fragment).
"""

import re
from dataclasses import dataclass
from pathlib import Path

from aicoach.errors import TranscriptionError

CHECK_FILE = "speech_check.txt"
_HEADER = re.compile(r"^\s*(\d+):(\d{2}):(\d{2})\s*-\s*(\d+):(\d{2}):(\d{2})\s*(\S*)\s*$")
_LANGUAGES = {"kz": "kk", "kk": "kk", "ru": "ru", "mixed": "mixed"}
BOM = chr(0xFEFF)  # a byte order mark from an editor
_MAX_MINUTE_SECOND = 59
_SECONDS_PER_HOUR = 3600
_SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class CheckFragment:
    """One fragment: video time bounds, the language the listener noted, and what was heard."""

    index: int
    start_s: float
    end_s: float
    language: str
    phrases: tuple[str, ...]

    @property
    def length_s(self) -> float:
        """Length of the fragment in seconds."""
        return self.end_s - self.start_s


def _seconds(hours: str, minutes: str, seconds: str, line: int) -> float:
    if int(minutes) > _MAX_MINUTE_SECOND or int(seconds) > _MAX_MINUTE_SECOND:
        msg = f"line {line}: minutes and seconds must be below 60"
        raise TranscriptionError(msg)
    return float(int(hours) * _SECONDS_PER_HOUR + int(minutes) * _SECONDS_PER_MINUTE + int(seconds))


def parse_check_text(text: str) -> list[CheckFragment]:
    """Fragments of a check file text, in file order (numbered from 1).

    Raises:
        TranscriptionError: A header has a bad time, an end not after the start, or an unknown
            language; or the text has no fragment at all.
    """
    fragments: list[CheckFragment] = []
    header: tuple[float, float, str] | None = None
    phrases: list[str] = []

    def close() -> None:
        if header is not None:
            fragments.append(
                CheckFragment(len(fragments) + 1, header[0], header[1], header[2], tuple(phrases))
            )

    for number, raw in enumerate(text.removeprefix(BOM).splitlines(), start=1):
        match = _HEADER.match(raw)
        if match:
            close()
            phrases = []
            start = _seconds(match[1], match[2], match[3], number)
            end = _seconds(match[4], match[5], match[6], number)
            language = _LANGUAGES.get(match.group(7).casefold())
            if end <= start or language is None:
                msg = f"line {number}: bad header (end before start or unknown language)"
                raise TranscriptionError(msg)
            header = (start, end, language)
        elif header is not None and raw.strip():
            phrases.append(raw.strip())
    close()
    if not fragments:
        msg = "no fragment header found in the check file"
        raise TranscriptionError(msg)
    return fragments


def load_check_file(path: Path) -> list[CheckFragment]:
    """Read and parse a check file.

    Raises:
        TranscriptionError: The file cannot be read or has no valid fragment.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        msg = f"cannot read the check file {path.name}"
        raise TranscriptionError(msg) from err
    return parse_check_text(text)
