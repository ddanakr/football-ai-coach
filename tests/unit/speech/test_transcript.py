"""Segments, player mentions, halves and gaps from joined words (pure, invented names)."""

import pytest

from aicoach.plate_names.lineup import Player
from aicoach.speech.segments import TimedWord
from aicoach.speech.transcript import (
    TranscribeConfig,
    build_transcript_segments,
    find_gaps,
    half_of,
)

SPANS = {1: (100.0, 500.0), 2: (700.0, 1100.0)}
TESTOV = Player("home", "Тестов Иван", 9, aliases=("Тесто",))
OBRAZTSOV = Player("away", "Образцов Марат", 7)


def say(text: str, start: float, speaker: str = "s0") -> list[TimedWord]:
    """One word per token, each a second long."""
    return [TimedWord(t, start + i, start + i + 0.9, speaker) for i, t in enumerate(text.split())]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("t_v", "half"),
    [(99.9, None), (100.0, 1), (500.0, 1), (500.1, None), (700.0, 2), (1100.0, 2), (2000.0, None)],
)
def test_half_of_includes_the_ends_and_leaves_the_break_without_a_half(
    t_v: float, half: int | None
) -> None:
    assert half_of(t_v, SPANS) == half


@pytest.mark.unit
def test_half_of_without_known_halves_is_none() -> None:
    assert half_of(300.0, {}) is None


@pytest.mark.unit
def test_segments_have_ids_in_the_order_of_time_the_half_and_the_language() -> None:
    words = [*say("Передача на Тестова", 120.0), *say("Қарсы шабуыл", 140.0, "s1")]

    segments = build_transcript_segments(words, SPANS, [TESTOV], 0.88)

    assert [s.id for s in segments] == ["s:0", "s:1"]
    assert [s.half for s in segments] == [1, 1]
    assert [s.language for s in segments] == ["ru", "kk"]
    assert segments[0].t_start == 120.0
    assert segments[0].text == "Передача на Тестова"
    assert [s.speaker for s in segments] == ["s0", "s1"]


@pytest.mark.unit
def test_segment_names_the_player_with_the_lineup_spelling_and_keeps_the_word_said() -> None:
    (segment,) = build_transcript_segments(say("удар Тестова мимо", 120.0), SPANS, [TESTOV], 0.88)

    (mention,) = segment.players
    assert (mention.name, mention.word, mention.side, mention.number) == (
        "Тестов Иван",
        "Тестова",
        "home",
        9,
    )
    assert mention.via == "name"


@pytest.mark.unit
def test_segment_finds_a_nickname_through_the_alias_column() -> None:
    (segment,) = build_transcript_segments(say("и снова Тесто", 120.0), SPANS, [TESTOV], 0.88)

    assert [(m.name, m.via, m.word) for m in segment.players] == [("Тестов Иван", "alias", "Тесто")]


@pytest.mark.unit
def test_without_a_lineup_no_player_is_named() -> None:
    (segment,) = build_transcript_segments(say("удар Тестова", 120.0), SPANS, None, 0.88)

    assert segment.players == []


@pytest.mark.unit
def test_segment_in_the_break_has_no_half_and_no_words_give_no_segments() -> None:
    (segment,) = build_transcript_segments(say("перерыв", 600.0), SPANS, None, 0.88)

    assert segment.half is None
    assert build_transcript_segments([], SPANS, [TESTOV], 0.88) == []


@pytest.mark.unit
def test_find_gaps_reports_a_stretch_without_speech_inside_a_half_only() -> None:
    words = [*say("начало", 100.0), *say("опять", 400.0), *say("второй", 700.0)]
    segments = build_transcript_segments(words, SPANS, None, 0.88)

    gaps, halves = find_gaps(segments, SPANS, 1200.0, 15.0)

    first = [g for g in gaps if g.half == 1]
    assert [(g.t_start, g.t_end) for g in first] == [(100.9, 400.0), (400.9, 500.0)]
    assert halves[0].half == 1
    assert halves[0].n_gaps == 2
    assert halves[0].gap_share == pytest.approx((299.1 + 99.1) / 400.0, abs=1e-3)
    # The break (500..700) is not looked at.
    assert all(g.t_start >= 100.0 for g in gaps)


@pytest.mark.unit
def test_find_gaps_continuous_speech_has_no_gap() -> None:
    words = [w for k in range(0, 400, 10) for w in say("слово слово слово", 100.0 + k)]
    segments = build_transcript_segments(words, {1: (100.0, 500.0)}, None, 0.88)

    gaps, halves = find_gaps(segments, {1: (100.0, 500.0)}, 600.0, 15.0)

    assert gaps == []
    assert halves[0].gap_s == 0
    assert halves[0].gap_share == 0.0


@pytest.mark.unit
def test_find_gaps_no_speech_at_all_is_one_gap_over_each_half() -> None:
    gaps, halves = find_gaps([], SPANS, 1200.0, 15.0)

    assert [(g.half, g.t_start, g.t_end) for g in gaps] == [(1, 100.0, 500.0), (2, 700.0, 1100.0)]
    assert [h.gap_share for h in halves] == [1.0, 1.0]


@pytest.mark.unit
def test_find_gaps_without_halves_counts_the_whole_sound_once() -> None:
    gaps, halves = find_gaps([], {}, 200.0, 15.0)

    assert [(g.half, g.t_start, g.t_end) for g in gaps] == [(None, 0.0, 200.0)]
    assert halves[0].half is None


@pytest.mark.unit
def test_find_gaps_a_half_cut_by_the_end_of_the_sound_is_measured_to_the_sound() -> None:
    gaps, halves = find_gaps([], {1: (100.0, 500.0)}, 300.0, 15.0)

    assert [(g.t_start, g.t_end) for g in gaps] == [(100.0, 300.0)]
    assert halves[0].end == 300.0


@pytest.mark.unit
def test_transcribe_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="chunk_s"):
        TranscribeConfig(chunk_s=0)
    with pytest.raises(ValueError, match="overlap_s"):
        TranscribeConfig(overlap_s=-1)
    assert TranscribeConfig().min_similarity == 0.88
