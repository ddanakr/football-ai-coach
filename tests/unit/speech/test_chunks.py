"""Cutting the sound into overlapping chunks and joining the answers (pure functions)."""

import math

import pytest

from aicoach.speech.chunks import Chunk, ChunkWords, keep_ranges, merge_chunks, plan_chunks
from aicoach.speech.segments import TimedWord


def w(text: str, start: float, end: float, speaker: str | None = "s0") -> TimedWord:
    return TimedWord(text, start, end, speaker)


@pytest.mark.unit
def test_plan_chunks_covers_the_sound_with_overlap_and_ends_exactly_at_the_end() -> None:
    chunks = plan_chunks(400.0, 180.0, 10.0)

    assert [(c.start_s, c.end_s) for c in chunks] == [(0, 180), (170, 350), (340, 400)]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert chunks[2].length_s == 60.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (0.0, []),
        (50.0, [(0.0, 50.0)]),  # shorter than a chunk
        (180.0, [(0.0, 180.0)]),  # exactly one chunk
        (350.0, [(0.0, 180.0), (170.0, 350.0)]),  # the last chunk ends where the sound ends
    ],
)
def test_plan_chunks_edges(duration: float, expected: list[tuple[float, float]]) -> None:
    assert [(c.start_s, c.end_s) for c in plan_chunks(duration, 180.0, 10.0)] == expected


@pytest.mark.unit
def test_plan_chunks_without_overlap_has_no_shared_seconds() -> None:
    chunks = plan_chunks(30.0, 10.0, 0.0)

    assert [(c.start_s, c.end_s) for c in chunks] == [(0, 10), (10, 20), (20, 30)]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("duration", "chunk", "overlap"),
    [
        (-1.0, 10.0, 1.0),
        (10.0, 0.0, 0.0),
        (10.0, 10.0, 10.0),  # the overlap must be shorter than the chunk
        (10.0, 10.0, -1.0),
        (math.nan, 10.0, 1.0),
        (10.0, math.inf, 1.0),
    ],
)
def test_plan_chunks_bad_values_are_refused(duration: float, chunk: float, overlap: float) -> None:
    with pytest.raises(ValueError, match="chunk plan"):
        plan_chunks(duration, chunk, overlap)


@pytest.mark.unit
def test_plan_chunks_every_second_of_a_long_match_is_in_some_chunk() -> None:
    chunks = plan_chunks(7940.0, 180.0, 10.0)

    assert chunks[0].start_s == 0.0
    assert chunks[-1].end_s == 7940.0
    assert all(b.start_s < a.end_s for a, b in zip(chunks, chunks[1:], strict=False))  # overlap


@pytest.mark.unit
def test_keep_ranges_cut_in_the_middle_of_each_overlap() -> None:
    chunks = plan_chunks(400.0, 180.0, 10.0)

    ranges = keep_ranges(chunks)

    assert ranges == [(0.0, 175.0), (175.0, 345.0), (345.0, math.inf)]


@pytest.mark.unit
def test_keep_ranges_single_chunk_keeps_everything() -> None:
    assert keep_ranges([Chunk(0, 0.0, 50.0)]) == [(0.0, math.inf)]
    assert keep_ranges([]) == []


@pytest.mark.unit
def test_merge_chunks_gives_every_word_of_the_overlap_once() -> None:
    chunks = plan_chunks(30.0, 20.0, 10.0)  # [0, 20] and [10, 30]; the cut is at 15 s
    # The same speech heard by both chunks in the overlap [10, 20] (times from the chunk start).
    first = ChunkWords(
        chunks[0], (w("раз", 2, 3), w("два", 11, 12), w("три", 16, 17), w("че", 18, 19))
    )
    second = ChunkWords(
        chunks[1], (w("два", 1, 2), w("три", 6, 7), w("че", 8, 9), w("пять", 12, 13))
    )

    words, _ = merge_chunks([first, second])

    assert [x.text for x in words] == ["раз", "два", "три", "че", "пять"]
    assert [x.start_s for x in words] == [2, 11, 16, 18, 22]  # video seconds


@pytest.mark.unit
def test_merge_chunks_loses_no_word_that_only_one_chunk_heard() -> None:
    chunks = plan_chunks(30.0, 20.0, 10.0)
    first = ChunkWords(chunks[0], (w("только первый", 12, 13),))  # 12 s: before the cut (15 s)
    second = ChunkWords(chunks[1], (w("только второй", 6, 7),))  # 16 s: after the cut

    words, _ = merge_chunks([first, second])

    assert [x.text for x in words] == ["только первый", "только второй"]


@pytest.mark.unit
def test_merge_chunks_word_on_the_cut_belongs_to_the_later_chunk() -> None:
    chunks = plan_chunks(30.0, 20.0, 10.0)
    first = ChunkWords(chunks[0], (w("граница", 14, 16),))  # middle 15 s
    second = ChunkWords(chunks[1], (w("граница", 4, 6),))  # middle 15 s

    words, _ = merge_chunks([first, second])

    assert len(words) == 1


@pytest.mark.unit
def test_merge_chunks_maps_the_speaker_labels_of_a_later_chunk_by_the_overlap() -> None:
    chunks = plan_chunks(30.0, 20.0, 10.0)
    # Chunk 1 calls the commentators A and B; chunk 2 calls the same people the other way round.
    first = ChunkWords(chunks[0], (w("a", 11, 13, "A"), w("b", 13, 15, "B"), w("c", 17, 19, "B")))
    second = ChunkWords(
        chunks[1], (w("a", 1, 3, "B"), w("b", 3, 5, "A"), w("c", 7, 9, "A"), w("d", 12, 14, "A"))
    )

    words, unresolved = merge_chunks([first, second])

    assert unresolved == 0
    assert [(x.text, x.speaker) for x in words] == [
        ("a", "A"),
        ("b", "B"),
        ("c", "B"),
        ("d", "B"),  # "A" of the second chunk is "B" of the first: mapped
    ]


@pytest.mark.unit
def test_merge_chunks_a_label_without_evidence_takes_a_free_known_label_and_is_counted() -> None:
    chunks = plan_chunks(30.0, 20.0, 10.0)
    first = ChunkWords(chunks[0], (w("a", 2, 3, "A"),))  # silence in the overlap
    second = ChunkWords(chunks[1], (w("d", 12, 14, "X"),))

    words, unresolved = merge_chunks([first, second])

    assert unresolved == 1
    assert words[-1].speaker == "A"  # the only speaker known so far


@pytest.mark.unit
def test_merge_chunks_one_speaker_in_the_overlap_does_not_merge_the_two_speakers() -> None:
    # Only the speaker "A" talks in the overlap (10..20 s); the second chunk calls the same two
    # people "x" and "y". "y" has no evidence and must not take the label that "x" already took.
    chunks = plan_chunks(40.0, 20.0, 10.0)
    first = ChunkWords(chunks[0], (w("a", 1, 2, "A"), w("b", 3, 4, "B"), w("c", 12, 14, "A")))
    second = ChunkWords(chunks[1], (w("c", 2, 4, "x"), w("d", 12, 13, "y"), w("e", 14, 15, "x")))

    words, unresolved = merge_chunks([first, second])

    assert [(x.text, x.speaker) for x in words] == [
        ("a", "A"),
        ("b", "B"),
        ("c", "A"),
        ("d", "B"),  # "y" is the other known speaker, not "A"
        ("e", "A"),
    ]
    assert unresolved == 1


@pytest.mark.unit
def test_merge_chunks_more_speakers_than_known_get_a_new_label() -> None:
    chunks = plan_chunks(30.0, 20.0, 10.0)
    first = ChunkWords(chunks[0], (w("a", 11, 13, "A"),))
    second = ChunkWords(chunks[1], (w("a", 1, 3, "x"), w("z", 12, 14, "y"), w("q", 15, 16, "q")))

    words, unresolved = merge_chunks([first, second])

    labels = [x.speaker for x in words]
    assert labels[0] == "A"
    assert len(set(labels)) == 3  # three different people stay three different labels
    assert unresolved == 2


@pytest.mark.unit
def test_merge_chunks_words_without_speaker_stay_without_speaker() -> None:
    chunks = plan_chunks(30.0, 20.0, 10.0)
    first = ChunkWords(chunks[0], (w("a", 11, 13, None),))
    second = ChunkWords(chunks[1], (w("d", 12, 14, None),))

    words, unresolved = merge_chunks([first, second])

    assert [x.speaker for x in words] == [None, None]
    assert unresolved == 0


@pytest.mark.unit
def test_merge_chunks_empty_answers_give_no_words() -> None:
    chunks = plan_chunks(30.0, 20.0, 10.0)

    assert merge_chunks([ChunkWords(c, ()) for c in chunks]) == ([], 0)
    assert merge_chunks([]) == ([], 0)


@pytest.mark.unit
def test_merge_chunks_timing_spread_on_the_cut_can_double_a_word_documented_limit() -> None:
    # Two models time one word 0.2 s apart; the cut is at 15 s. The middle is 14.9 s in one chunk
    # (kept by the first) and 15.1 s in the other (kept by the second): the word comes twice.
    # This is the limit of the scheme (see the module note), not a bug to hide.
    chunks = plan_chunks(30.0, 20.0, 10.0)
    first = ChunkWords(chunks[0], (w("граница", 14.8, 15.0),))
    second = ChunkWords(chunks[1], (w("граница", 5.0, 5.2),))

    words, _ = merge_chunks([first, second])

    assert [x.text for x in words] == ["граница", "граница"]
