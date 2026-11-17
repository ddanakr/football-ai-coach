"""Segments from timed words, the time conversion to t_v, and the word-coverage metric."""

import pytest

from aicoach.speech.metrics import Gap, coverage, normalize_words, time_gaps
from aicoach.speech.segments import (
    Segment,
    TimedWord,
    build_segments,
    guess_language,
    language_shares,
)


def _w(text: str, start: float, end: float, speaker: str | None = "spk:0") -> TimedWord:
    return TimedWord(text, start, end, speaker)


@pytest.mark.unit
def test_segment_times_become_video_seconds() -> None:
    (seg,) = build_segments([_w("гол", 2.0, 2.5), _w("был", 2.6, 3.0)], 935.0)

    assert (seg.start_s, seg.end_s) == (2.0, 3.0)
    assert (seg.t_v_start, seg.t_v_end) == (937.0, 938.0)
    assert seg.text == "гол был"


@pytest.mark.unit
def test_speaker_change_and_long_pause_split_segments() -> None:
    words = [
        _w("а", 0, 1),
        _w("б", 1.1, 2, "spk:1"),
        _w("в", 2.1, 3, "spk:1"),
        _w("г", 6.0, 7, "spk:1"),
    ]

    segments = build_segments(words, 0.0)

    assert [s.text for s in segments] == ["а", "б в", "г"]
    assert [s.speaker for s in segments] == ["spk:0", "spk:1", "spk:1"]


@pytest.mark.unit
def test_long_run_splits_at_a_sentence_end_only() -> None:
    words = [_w("раз.", 0, 5), _w("два", 5.1, 11), _w("три.", 11.1, 12), _w("четыре", 12.2, 13)]

    segments = build_segments(words, 0.0)

    # "раз." ends at 5 s (< 10 s): no split; "три." ends the run after 10 s: split.
    assert [s.text for s in segments] == ["раз. два три.", "четыре"]


@pytest.mark.unit
def test_no_words_give_no_segments_and_unordered_words_are_sorted() -> None:
    assert build_segments([], 10.0) == []
    (seg,) = build_segments([_w("два", 1.0, 1.5), _w("раз", 0.0, 0.9)], 0.0)
    assert seg.text == "раз два"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Ақтөбе допты алды", "kk"),
        ("мяч в воротах", "ru"),
        ("1:0 !!!", "unknown"),
        ("", "unknown"),
    ],
)
def test_language_guess_by_letters(text: str, expected: str) -> None:
    assert guess_language(text) == expected


@pytest.mark.unit
def test_language_shares_count_segments() -> None:
    segments = build_segments([_w("қарсы", 0, 1), _w("мяч", 5, 6), _w("гол", 9, 10)], 0.0)

    assert language_shares(segments) == {"kk": 1 / 3, "ru": 2 / 3}
    assert language_shares([]) == {}


@pytest.mark.unit
def test_normalize_keeps_kazakh_letters_and_folds_yo_and_case() -> None:
    assert normalize_words("Ақтөбе, ЕЛКА! Ёж 12 мяч-гол") == ["ақтөбе", "елка", "еж", "мяч", "гол"]
    assert normalize_words("123 ...") == []


@pytest.mark.unit
def test_coverage_accepts_case_endings_and_names_missing_words() -> None:
    cov = coverage(["Заруцкий тащил", "за боковую"], "Заруцкого тащили вдоль боковой")

    # "за" is a short word and must match exactly; the others share a stem of 5+ letters.
    assert (cov.found, cov.total, cov.missing) == (3, 4, ("за",))
    assert cov.share == 0.75


@pytest.mark.unit
def test_coverage_distinguishes_kazakh_letters_and_handles_empty_input() -> None:
    assert coverage(["қарсы"], "карсы").found == 0
    assert coverage(["қарсы"], "Қарсыны").found == 1
    empty = coverage([], "что угодно")
    assert (empty.total, empty.share) == (0, None)
    assert coverage(["слово"], "").missing == ("слово",)


def _seg(start: float, end: float) -> Segment:
    return Segment(
        t_v_start=100 + start,
        t_v_end=100 + end,
        start_s=start,
        end_s=end,
        speaker=None,
        language="ru",
        text="x",
    )


@pytest.mark.unit
def test_time_gaps_finds_speech_that_stops_before_the_fragment_ends() -> None:
    # The cut-off of fragment 3 in D-066: speech until 30 s of 65.
    gaps = time_gaps([_seg(0, 12), _seg(13, 30)], 65.0)

    assert gaps == [Gap(30.0, 65.0)]
    assert gaps[0].length_s == 35.0


@pytest.mark.unit
def test_time_gaps_finds_a_gap_in_the_middle_and_one_at_the_start() -> None:
    gaps = time_gaps([_seg(20, 30), _seg(50, 60)], 60.0)

    assert gaps == [Gap(0.0, 20.0), Gap(30.0, 50.0)]


@pytest.mark.unit
def test_time_gaps_ignores_pauses_shorter_than_the_limit_and_overlaps() -> None:
    segments = [_seg(0, 20), _seg(5, 10), _seg(34.9, 60)]  # a 14.9 s pause

    assert time_gaps(segments, 60.0) == []


@pytest.mark.unit
def test_time_gaps_limit_is_inclusive() -> None:
    assert time_gaps([_seg(0, 10), _seg(25, 40)], 40.0) == [Gap(10.0, 25.0)]


@pytest.mark.unit
def test_time_gaps_without_segments_is_one_gap_over_the_fragment() -> None:
    assert time_gaps([], 90.0) == [Gap(0.0, 90.0)]
    assert time_gaps([], 10.0) == []  # a fragment shorter than the limit has no gap
    assert time_gaps([], 0.0) == []


@pytest.mark.unit
def test_time_gaps_sorts_unordered_segments() -> None:
    assert time_gaps([_seg(40, 60), _seg(0, 20)], 60.0) == [Gap(20.0, 40.0)]
