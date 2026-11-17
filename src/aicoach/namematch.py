"""Matching of player names, one module for plates, speech and the agent (D-070).

Everything compares *folded* forms: lower case, `ё` to `е`, the Kazakh letters to the Russian base
letter (`ә→а ғ→г қ→к ң→н ө→о ұ→у ү→у һ→х і→и`), `й` to `и` and Latin look-alikes to Cyrillic, so
the plate «ҚҰДАЙБЕРГЕНОВ», the protocol «Кудайбергенов» and the ASR «Кудайбергенова» meet.

Two kinds of question are answered here:

- `find_player`: a *whole* read name (a plate line) against the lineup; fuzzy comparison of the
  folded words in any order, a threshold, no guess on a tie (D-059);
- `find_mentions`: single words of a text (commentary) against the surnames and the aliases
  (nicknames) of the lineup; a case ending does not matter ("Заруцкого" is "Заруцкий").

The lineup writes a name as "Surname Given-names" (D-059), so the surname is the first word.
A match is a reference to the player (`side` and `number`) with the spelling of the lineup; the
word as it was read or said stays in `source`.
"""

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from aicoach.plate_names.lineup import Player, Side

# Kazakh letters, the yo and the short i, folded to the Russian base letter.
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
_NOT_NAME = re.compile(r"[^\w\s'-]|[\d_]", re.UNICODE)
_SEPARATORS = re.compile(r"[\s'-]+")
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)

_MAX_ENDING = 3  # letters a case ending may add to the stem
_MIN_STEM = 5  # shorter stems join different surnames (Петров, Петренко)
_MIN_FUZZY = 5  # a word shorter than this must be equal: "Иван" is not "Ивана" by typo
_DIGITS = 3  # similarity is stored with three decimals
# The fuzzy threshold chosen for the plate names on the first half of one match (D-059); the speech
# stage and the agent use the same one.
DEFAULT_MIN_SIMILARITY = 0.88

Via = Literal["name", "alias"]


def name_key(text: str) -> str:
    """Comparison key: lower case, Kazakh letters and Latin look-alikes folded to Russian,
    digits and punctuation dropped, hyphens and apostrophes read as spaces, one space between
    words. An empty or letterless text gives `""`."""
    folded = unicodedata.normalize("NFKC", text).casefold().translate(_FOLD).translate(_HOMOGLYPHS)
    words = _SEPARATORS.sub(" ", _NOT_NAME.sub(" ", folded))
    return " ".join(words.split())


def sorted_key(text: str) -> str:
    """`name_key` with the words in alphabetical order, so that "Given SURNAME" equals
    "Surname Given"."""
    return " ".join(sorted(name_key(text).split()))


def words(text: str) -> list[str]:
    """The words of `text` as comparison keys (lower case, folded letters, no punctuation)."""
    return name_key(text).split()


def similarity(read: str, candidate: str) -> float:
    """How alike two names are, 0..1; the order of the words does not matter."""
    a, b = sorted_key(read), sorted_key(candidate)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def same_word(a: str, b: str) -> bool:
    """Two keys of words are one word, ignoring the case ending ("заруцкого" ~ "заруцкий").

    Words are the same when equal or when they share a stem of at least five letters that
    leaves at most two letters of the shorter word out; shorter words must be equal.
    """
    if a == b:
        return True
    shorter = min(len(a), len(b))
    stem = max(_MIN_STEM, shorter - 2)
    # A case ending is a few letters: Петров is not Петровский.
    longer_by_ending = max(len(a), len(b)) - stem <= _MAX_ENDING
    return shorter >= _MIN_STEM and a[:stem] == b[:stem] and longer_by_ending


def name_matches(query: str, full_name: str) -> bool:
    """Every word of `query` is a word of `full_name` (any order, endings ignored).

    An empty query (no letters) matches nothing.
    """
    wanted = words(query)
    have = words(full_name)
    return bool(wanted) and all(any(same_word(w, h) for h in have) for w in wanted)


@dataclass(frozen=True)
class PlayerRef:
    """A player of the lineup: the side, the shirt number (None when the lineup has none) and
    the spelling of the lineup."""

    side: Side
    number: int | None
    name: str


@dataclass(frozen=True)
class NameMatch:
    """A player found for a text: `similarity` (0..1), `source` the text as it was read or said,
    `via` whether the lineup name or an alias matched."""

    player: Player
    similarity: float
    source: str
    via: Via = "name"

    @property
    def ref(self) -> PlayerRef:
        """The reference to the player with the lineup spelling."""
        return PlayerRef(self.player.side, self.player.number, self.player.name)


def _candidates(player: Player) -> list[tuple[str, Via]]:
    return [(player.name, "name"), *((alias, "alias") for alias in player.aliases)]


def find_player(
    read: str, players: list[Player], side: Side | None, min_similarity: float
) -> tuple[NameMatch | None, float | None]:
    """The player of `side` whose name or alias is most like the whole text `read`.

    Returns the match (None when the best similarity is below `min_similarity`, when two
    different players tie for the best place, or when `side` has no players) and the best
    similarity seen (None when nothing was compared). `side=None` looks at every player; a
    given side never looks at the other one. Nothing is guessed: a doubtful text stays
    unrecognized.
    """
    scored: list[tuple[float, Player, Via]] = []
    for player in players:
        if side is not None and player.side != side:
            continue
        best_score, via = max(
            ((similarity(read, text), way) for text, way in _candidates(player)),
            key=lambda item: item[0],
        )
        scored.append((best_score, player, via))
    scored.sort(key=lambda item: -item[0])
    if not scored:
        return None, None
    best_score, best, via = scored[0]
    if best_score < min_similarity:
        return None, best_score
    if len(scored) > 1 and scored[1][0] == best_score and scored[1][1].name != best.name:
        return None, best_score
    return NameMatch(best, best_score, read, via), best_score


@dataclass(frozen=True)
class _Form:
    """One way to call a player: the folded words of the surname or of an alias."""

    player: Player
    key: tuple[str, ...]
    via: Via


def _forms(players: list[Player], side: Side | None) -> list[_Form]:
    forms: list[_Form] = []
    for player in players:
        if side is not None and player.side != side:
            continue
        surname = words(player.name)[:1]
        if surname:
            forms.append(_Form(player, tuple(surname), "name"))
        forms.extend(
            _Form(player, tuple(key), "alias") for a in player.aliases if (key := words(a))
        )
    return forms


def _word_score(token: str, form_word: str, min_similarity: float) -> float | None:
    """The similarity of two folded words when they are one word, else None."""
    if token == form_word:
        return 1.0
    if min(len(token), len(form_word)) < _MIN_FUZZY:
        return None
    ratio = SequenceMatcher(None, token, form_word, autojunk=False).ratio()
    return ratio if same_word(token, form_word) or ratio >= min_similarity else None


def _phrase_score(keys: list[str], form: _Form, min_similarity: float) -> float | None:
    """The mean similarity of the tokens `keys` to the words of `form`, None if any differs."""
    scores = [_word_score(k, w, min_similarity) for k, w in zip(keys, form.key, strict=True)]
    if any(s is None for s in scores):
        return None
    return sum(s for s in scores if s is not None) / len(scores)


def find_mentions(
    text: str, players: list[Player], min_similarity: float, side: Side | None = None
) -> list[NameMatch]:
    """The players a free text (commentary) speaks of, in the order of the text.

    A word is a mention when it is the surname (the first word of the lineup name) or an alias of
    a player: equal, with another case ending, or within `min_similarity` for words of five letters
    and more. An alias of several words must be said in full. Where two different players fit the
    same words equally well (two players of one surname) nothing is reported: it is not guessed.
    A longer alias wins over a surname inside it.

    Args:
        text: The text; any case, punctuation is ignored.
        players: The lineup.
        min_similarity: The fuzzy threshold of a single word (0..1).
        side: Only players of this side; `None` for both.
    """
    tokens = _TOKEN.findall(text)
    keys = [name_key(t) for t in tokens]
    forms = _forms(players, side)
    found: list[NameMatch] = []
    i = 0
    while i < len(tokens):
        best_length = 0
        fitting: list[tuple[float, _Form]] = []
        for form in forms:
            length = len(form.key)
            if i + length > len(tokens) or length < best_length:
                continue
            score = _phrase_score(keys[i : i + length], form, min_similarity)
            if score is None:
                continue
            if length > best_length:
                best_length, fitting = length, []
            fitting.append((score, form))
        if not fitting:
            i += 1
            continue
        top = max(score for score, _ in fitting)
        winners = {(f.player.side, f.player.number, f.player.name) for s, f in fitting if s == top}
        if len(winners) == 1:
            score, form = next((s, f) for s, f in fitting if s == top)
            source = " ".join(tokens[i : i + best_length])
            found.append(NameMatch(form.player, round(score, _DIGITS), source, form.via))
        i += best_length
    return found
