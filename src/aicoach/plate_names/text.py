"""Text of a name: the comparison key and the cleaned text shown to the user. Pure functions.

The plate writes a name as "number given-names SURNAME" in capitals; the lineup writes
"Surname Given-names". The comparison key therefore ignores case, order and the Kazakh letters
(a protocol may spell «Кудайбергенов» where the plate has «ҚҰДАЙБЕРГЕНОВ»).
"""

import re
import unicodedata

# Kazakh letters and the yo, folded to the Russian base letter.
_FOLD = str.maketrans(
    {
        "ё": "е",
        "ә": "а",
        "ғ": "г",
        "қ": "к",
        "ң": "н",
        "ө": "о",
        "ұ": "у",
        "ү": "у",
        "һ": "х",
        "і": "и",
        "й": "и",
    }
)
# Latin letters that look like Cyrillic ones (OCR and typing mix them up).
_HOMOGLYPHS = str.maketrans(
    {
        "a": "а",
        "b": "в",
        "c": "с",
        "e": "е",
        "h": "н",
        "i": "и",
        "k": "к",
        "m": "м",
        "o": "о",
        "p": "р",
        "t": "т",
        "x": "х",
        "y": "у",
    }
)
_LEADING_NUMBER = re.compile(r"^\s*\d+\s*")
_NOT_NAME = re.compile(r"[^\w\s'-]|[\d_]", re.UNICODE)
_SEPARATORS = re.compile(r"[\s'-]+")
MIN_LETTERS = 2  # a text with fewer letters is noise, not a name


def name_key(text: str) -> str:
    """Comparison key: lower case, Kazakh letters and Latin look-alikes folded to Russian,
    digits and punctuation dropped, hyphens and apostrophes read as spaces, one space between
    words. An empty or letterless text gives `""`."""
    folded = unicodedata.normalize("NFKC", text).casefold().translate(_FOLD).translate(_HOMOGLYPHS)
    words = _SEPARATORS.sub(" ", _NOT_NAME.sub(" ", folded))
    return " ".join(words.split())


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


def sorted_key(text: str) -> str:
    """`name_key` with the words in alphabetical order, so that "Given SURNAME" equals
    "Surname Given"."""
    return " ".join(sorted(name_key(text).split()))
