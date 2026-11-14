"""Fuzzy comparison of names as users write them (cases, Kazakh letters, word order).

Built on `aicoach.plate_names.text.name_key`, the same key the plate names use, so a query
"Заруцкого" finds "Заруцкий Александр" and "ӘНУАРБЕКОВ" finds "Ануарбеков".
"""

from aicoach.plate_names.text import name_key

_MIN_STEM = 5  # shorter stems join different surnames (Петров, Петренко)


def words(text: str) -> list[str]:
    """The words of `text` as comparison keys (lower case, folded letters, no punctuation)."""
    return name_key(text).split()


def same_word(a: str, b: str) -> bool:
    """Two keys of words are one word, ignoring the case ending ("заруцкого" ~ "заруцкий").

    Words are the same when equal or when they share a stem of at least five letters that
    leaves at most two letters of the shorter word out; shorter words must be equal.
    """
    if a == b:
        return True
    shorter = min(len(a), len(b))
    stem = max(_MIN_STEM, shorter - 2)
    return shorter >= _MIN_STEM and a[:stem] == b[:stem]


def name_matches(query: str, full_name: str) -> bool:
    """Every word of `query` is a word of `full_name` (any order, endings ignored).

    An empty query (no letters) matches nothing.
    """
    wanted = words(query)
    have = words(full_name)
    return bool(wanted) and all(any(same_word(w, h) for h in have) for w in wanted)
