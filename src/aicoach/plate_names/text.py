"""Text of a plate name: the cleaned text shown to the user. Pure functions.

The plate writes a name as "number given-names SURNAME" in capitals; the lineup writes
"Surname Given-names". Comparing them is `aicoach.namematch` (it ignores case, order and the
Kazakh letters).
"""

import re

_LEADING_NUMBER = re.compile(r"^\s*\d+\s*")
_NOT_NAME = re.compile(r"[^\w\s'-]|[\d_]", re.UNICODE)
MIN_LETTERS = 2  # a text with fewer letters is noise, not a name


def clean_text(raw: str) -> str:
    """The name part of an OCR line: the leading shirt number and stray symbols are cut, the
    words are written with a capital letter. `""` when fewer than two letters remain."""
    without_number = _LEADING_NUMBER.sub("", raw)
    kept = " ".join(_NOT_NAME.sub(" ", without_number).split())
    if sum(ch.isalpha() for ch in kept) < MIN_LETTERS:
        return ""
    return " ".join(
        "-".join(part.capitalize() for part in word.split("-")) for word in kept.split()
    )
