"""From timed words to utterance segments with a language label (pure functions).

The transcription model labels words (start, end, speaker) but not languages. A segment is a run
of words of one speaker; it ends at a speaker change, at a pause of `MAX_GAP_S`, or at a
sentence end once it is `SOFT_MAX_S` long (commentary is nearly continuous, so pauses alone
would give one segment per minute). The language is guessed from the letters: a Kazakh-only
letter makes the segment `kk`; Cyrillic without them is `ru`; no letters is `unknown`. Kazakh
speech without those letters reads as `ru`: the label is a script heuristic, not a detection.
"""

import re
from dataclasses import dataclass

from pydantic import BaseModel

MAX_GAP_S = 1.5
SOFT_MAX_S = 10.0
_KAZAKH_LETTERS = re.compile("[әғқңөұүһі]", re.IGNORECASE)
_CYRILLIC = re.compile("[а-яё]", re.IGNORECASE)
_SENTENCE_END = (".", "?", "!", "…")


@dataclass(frozen=True)
class TimedWord:
    """A word with times in seconds from the start of the fragment and a speaker label."""

    text: str
    start_s: float
    end_s: float
    speaker: str | None


class Segment(BaseModel):
    """One utterance. `start_s`/`end_s` are from the fragment start, `t_v_*` are video seconds."""

    t_v_start: float
    t_v_end: float
    start_s: float
    end_s: float
    speaker: str | None
    language: str
    text: str


def guess_language(text: str) -> str:
    """`kk`, `ru` or `unknown` by the letters of `text` (see the module note)."""
    if _KAZAKH_LETTERS.search(text):
        return "kk"
    return "ru" if _CYRILLIC.search(text) else "unknown"


def _starts_new(prev: TimedWord, word: TimedWord, seg_start_s: float) -> bool:
    if word.speaker != prev.speaker or word.start_s - prev.end_s >= MAX_GAP_S:
        return True
    return prev.text.endswith(_SENTENCE_END) and prev.end_s - seg_start_s >= SOFT_MAX_S


def _segment(words: list[TimedWord], fragment_start_s: float) -> Segment:
    start, end = words[0].start_s, max(w.end_s for w in words)
    text = " ".join(w.text for w in words)
    return Segment(
        t_v_start=fragment_start_s + start,
        t_v_end=fragment_start_s + end,
        start_s=start,
        end_s=end,
        speaker=words[0].speaker,
        language=guess_language(text),
        text=text,
    )


def build_segments(words: list[TimedWord], fragment_start_s: float) -> list[Segment]:
    """Segments of `words`, with times converted to video seconds `t_v`.

    No words give no segments. Words out of order are sorted by start time.
    """
    ordered = sorted(words, key=lambda w: (w.start_s, w.end_s))
    segments: list[Segment] = []
    current: list[TimedWord] = []
    for word in ordered:
        if current and _starts_new(current[-1], word, current[0].start_s):
            segments.append(_segment(current, fragment_start_s))
            current = []
        current.append(word)
    if current:
        segments.append(_segment(current, fragment_start_s))
    return segments


def language_shares(segments: list[Segment]) -> dict[str, float]:
    """Share of segments per language label (empty for no segments)."""
    if not segments:
        return {}
    counts: dict[str, int] = {}
    for seg in segments:
        counts[seg.language] = counts.get(seg.language, 0) + 1
    return {lang: n / len(segments) for lang, n in sorted(counts.items())}
