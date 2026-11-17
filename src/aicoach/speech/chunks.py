"""Cutting the match sound into overlapping chunks and joining the chunk transcripts (pure).

A provider limits a request (OpenRouter: 25 MB of audio and about 60 s of processing per request,
D-070), so the match is sent in chunks. Neighbouring chunks overlap, so a word that falls on a
boundary is heard whole in one of them. To join, the middle of each overlap is the cut: a word
belongs to the chunk whose kept range holds the middle of the word, so a word that two chunks
hear with the same time comes exactly once. The two models time one word a little apart (about
0.1 s), so a word lying right on the cut can in rare cases come twice or be lost; this is the
limit of the scheme. The speaker labels of a chunk are its own ("speaker_0" of two chunks
may be two people); they are mapped to the labels of the previous chunk by who speaks at the same
time in the overlap.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from aicoach.speech.segments import TimedWord


@dataclass(frozen=True)
class Chunk:
    """A stretch of the sound: `[start_s, end_s]` in video seconds."""

    index: int
    start_s: float
    end_s: float

    @property
    def length_s(self) -> float:
        """The length of the chunk."""
        return self.end_s - self.start_s


@dataclass(frozen=True)
class ChunkWords:
    """The words one chunk returned; the times count from the start of the chunk."""

    chunk: Chunk
    words: tuple[TimedWord, ...]


def plan_chunks(duration_s: float, chunk_s: float, overlap_s: float) -> list[Chunk]:
    """Chunks of `chunk_s` seconds that overlap by `overlap_s`, covering `[0, duration_s]`.

    The last chunk ends exactly at `duration_s`; nothing is cut after it. A zero length gives no
    chunks.

    Raises:
        ValueError: A value is not finite, `duration_s` is negative, `chunk_s` is not positive,
            `overlap_s` is negative or not shorter than `chunk_s`.
    """
    if not all(math.isfinite(v) for v in (duration_s, chunk_s, overlap_s)):
        msg = "chunk plan needs finite numbers"
        raise ValueError(msg)
    if duration_s < 0 or chunk_s <= 0 or overlap_s < 0 or overlap_s >= chunk_s:
        msg = (
            f"bad chunk plan: duration {duration_s}, chunk {chunk_s}, overlap {overlap_s} "
            "(need duration >= 0, chunk > 0, 0 <= overlap < chunk)"
        )
        raise ValueError(msg)
    chunks: list[Chunk] = []
    start = 0.0
    while start < duration_s:
        end = min(start + chunk_s, duration_s)
        chunks.append(Chunk(len(chunks), start, end))
        if end >= duration_s:
            break
        start = end - overlap_s
    return chunks


def keep_ranges(chunks: Sequence[Chunk]) -> list[tuple[float, float]]:
    """For each chunk the video seconds `[from, to)` whose words it supplies: the cut between two
    chunks is the middle of their overlap; the first range starts at 0, the last has no end."""
    if not chunks:
        return []
    cuts = [(chunks[i + 1].start_s + chunks[i].end_s) / 2 for i in range(len(chunks) - 1)]
    lows = [0.0, *cuts]
    highs = [*cuts, math.inf]
    return list(zip(lows, highs, strict=True))


def _overlap_s(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _map_speakers(
    current: Sequence[TimedWord], previous: Sequence[TimedWord], known: Sequence[str]
) -> tuple[dict[str, str], int]:
    """Labels of `current` (global times) mapped one to one onto the labels of the earlier chunks.

    A label that speaks together with a word of `previous` (the overlap) takes that word's label;
    the strongest pairs go first and a label is never given twice, because the labels of one chunk
    are different people. A label with no such evidence takes the first label of `known` (the
    labels seen so far, in the order they appeared) that is still free, else a new one; it is
    counted, because that is a guess.
    """
    votes: dict[tuple[str, str], float] = {}
    for word in current:
        if word.speaker is None:
            continue
        for old in previous:
            if old.speaker is None:
                continue
            shared = _overlap_s(word.start_s, word.end_s, old.start_s, old.end_s)
            if shared > 0:
                pair = (word.speaker, old.speaker)
                votes[pair] = votes.get(pair, 0.0) + shared
    mapping: dict[str, str] = {}
    for (label, target), _ in sorted(votes.items(), key=lambda item: -item[1]):
        if label not in mapping and target not in mapping.values():
            mapping[label] = target
    unresolved = 0
    for label in dict.fromkeys(w.speaker for w in current if w.speaker is not None):
        if label in mapping:
            continue
        unresolved += 1
        free = [k for k in known if k not in mapping.values()]
        mapping[label] = free[0] if free else _new_label(known, mapping.values())
    return mapping, unresolved


def _new_label(known: Sequence[str], used: Iterable[str]) -> str:
    """A label that is neither known nor used: the first number not taken."""
    taken = {*known, *used}
    number = 0
    while str(number) in taken:
        number += 1
    return str(number)


def _shifted(words: Sequence[TimedWord], offset_s: float) -> list[TimedWord]:
    return [TimedWord(w.text, w.start_s + offset_s, w.end_s + offset_s, w.speaker) for w in words]


def merge_chunks(results: Sequence[ChunkWords]) -> tuple[list[TimedWord], int]:
    """All words of the chunks in video time, each once, speakers made consistent.

    Args:
        results: The chunks in order with their words (times from the chunk start).

    Returns:
        The words sorted by time with times in video seconds, and how many speaker labels of
        later chunks had no evidence in the overlap (a guess: they took a free known label).
    """
    chunks = [r.chunk for r in results]
    ranges = keep_ranges(chunks)
    merged: list[TimedWord] = []
    previous: list[TimedWord] = []
    unresolved_total = 0
    for result, (low, high) in zip(results, ranges, strict=True):
        words = _shifted(result.words, result.chunk.start_s)
        known = list(dict.fromkeys(w.speaker for w in merged if w.speaker is not None))
        if previous or known:
            window = [w for w in previous if w.end_s > result.chunk.start_s]
            mapping, unresolved = _map_speakers(words, window, known)
            unresolved_total += unresolved
            words = [
                TimedWord(w.text, w.start_s, w.end_s, mapping.get(w.speaker or "", w.speaker))
                for w in words
            ]
        merged.extend(w for w in words if low <= (w.start_s + w.end_s) / 2 < high)
        previous = words
    merged.sort(key=lambda w: (w.start_s, w.end_s))
    return merged, unresolved_total
