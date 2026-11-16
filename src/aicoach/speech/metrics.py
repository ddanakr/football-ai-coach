"""How many of the listener's words the transcript contains (pure functions).

Normalization: lower case, `ё` to `е`, punctuation and digits dropped; Kazakh letters are kept
(unlike the plate-name key, which folds them). A heard word counts as found when the transcript
has the same word or a word with a common stem of at least five letters (the rule of
`aicoach.agent.names.same_word`: case endings do not matter, short words must be equal).
"""

import re
import unicodedata
from dataclasses import dataclass

from aicoach.agent.names import same_word

_LETTERS = re.compile(r"[^\W\d_]+", re.UNICODE)


def normalize_words(text: str) -> list[str]:
    """The words of `text`, normalized (see the module note); no letters gives `[]`."""
    folded = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return _LETTERS.findall(folded)


@dataclass(frozen=True)
class Coverage:
    """Heard words found in a transcript: counts and the missing words (normalized)."""

    found: int
    total: int
    missing: tuple[str, ...]

    @property
    def share(self) -> float | None:
        """Found over total; `None` when nothing was heard (no division by zero)."""
        return self.found / self.total if self.total else None


def coverage(heard_phrases: list[str], transcript: str) -> Coverage:
    """Score `transcript` against the heard phrases (every word of every phrase counts once)."""
    heard = [w for phrase in heard_phrases for w in normalize_words(phrase)]
    have = set(normalize_words(transcript))
    missing = tuple(w for w in heard if not any(same_word(w, h) for h in have))
    return Coverage(len(heard) - len(missing), len(heard), missing)
