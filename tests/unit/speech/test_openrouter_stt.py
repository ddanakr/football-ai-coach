"""Speech-to-text through OpenRouter on recorded answers (no network)."""

import base64
from typing import Any  # JSON bodies of the API: their shape is the service's, not ours

import pytest

from aicoach.errors import ExternalServiceError, TranscriptionError
from aicoach.speech.openrouter_stt import (
    OpenRouterTranscriber,
    audio_format,
    parse_transcription,
    request_payload,
)
from aicoach.speech.segments import TimedWord
from tests.fixtures.openrouter_synth import Server, error_response, json_response


def verbose(words: list[dict[str, Any]], text: str = "гол Ақтөбе") -> dict[str, Any]:
    return {
        "text": text,
        "language": "ru",
        "words": words,
        "usage": {"seconds": 99.5, "input_tokens": 0, "output_tokens": 0, "cost": 0.00608},
    }


def word(text: str, start: float, end: float, speaker: str | None = "speaker_0") -> dict[str, Any]:
    return {"word": text, "start": start, "end": end, "speaker": speaker, "type": "word"}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mime", "expected"),
    [("audio/mp3", "mp3"), ("audio/mpeg", "mp3"), ("audio/wav", "wav"), ("audio/x-wav", "wav"),
     ("audio/ogg; codecs=opus", "ogg")],
)  # fmt: skip
def test_audio_format_of_a_mime_type(mime: str, expected: str) -> None:
    assert audio_format(mime) == expected


@pytest.mark.unit
def test_timed_payload_asks_for_word_times_and_speakers() -> None:
    payload = request_payload("elevenlabs/scribe-v2", b"\x00\x01", "audio/mp3", None)

    assert payload["model"] == "elevenlabs/scribe-v2"
    assert payload["input_audio"] == {
        "data": base64.b64encode(b"\x00\x01").decode(),
        "format": "mp3",
    }
    assert payload["response_format"] == "verbose_json"
    assert payload["timestamp_granularities"] == ["word"]
    assert payload["diarize"] is True


@pytest.mark.unit
def test_vocabulary_payload_sends_clean_unique_keyterms_and_no_timing_options() -> None:
    terms = ["Ветров", " Ветров ", "", "x" * 101, "Қайрат"]

    payload = request_payload("m", b"a", "audio/mp3", terms)

    assert payload["keyterms"] == ["Ветров", "Қайрат"]
    assert "diarize" not in payload
    assert "response_format" not in payload


@pytest.mark.unit
def test_words_become_timed_words_with_speakers_and_skip_audio_events() -> None:
    body = verbose(
        [
            word("гол", 1.0, 1.4),
            {"word": "(музыка)", "start": 1.5, "end": 2.0, "type": "audio_event"},
            word("Ақтөбе", 2.1, 2.9, "speaker_1"),
            {"word": " ", "start": 2.9, "end": 3.0, "type": "spacing"},
        ]
    )

    raw = parse_transcription(body, timed=True)

    assert raw.words == (
        TimedWord("гол", 1.0, 1.4, "speaker_0"),
        TimedWord("Ақтөбе", 2.1, 2.9, "speaker_1"),
    )
    assert raw.text == "гол Ақтөбе"
    assert raw.audio_seconds == 99.5
    assert raw.cost_usd == 0.00608


@pytest.mark.unit
def test_segments_stand_in_for_words_when_the_provider_gives_no_word_times() -> None:
    body = {
        "text": "Один. Два.",
        "segments": [
            {"id": 0, "start": 0.0, "end": 4.0, "text": " Один. "},
            {"id": 1, "start": 4.5, "end": 9.0, "text": "Два.", "speaker": "A"},
        ],
    }

    raw = parse_transcription(body, timed=True)

    assert [(w.text, w.start_s, w.end_s, w.speaker) for w in raw.words] == [
        ("Один.", 0.0, 4.0, None),
        ("Два.", 4.5, 9.0, "A"),
    ]
    assert raw.cost_usd is None


@pytest.mark.unit
def test_text_without_any_times_gives_a_transcript_without_words_not_an_error() -> None:
    raw = parse_transcription({"text": "только текст"}, timed=True)

    assert raw.text == "только текст"
    assert raw.words == ()


@pytest.mark.unit
def test_vocabulary_mode_returns_only_the_text() -> None:
    raw = parse_transcription(verbose([word("гол", 1.0, 1.4)]), timed=False)

    assert raw.words == ()
    assert raw.text


@pytest.mark.unit
@pytest.mark.parametrize("bad", [None, "1.0", -1.0, float("inf"), True])
def test_a_bad_word_time_is_a_transcription_error(bad: object) -> None:
    body = verbose([{"word": "гол", "start": bad, "end": 2.0}])

    with pytest.raises(TranscriptionError, match="word start"):
        parse_transcription(body, timed=True)


@pytest.mark.unit
@pytest.mark.parametrize("body", [{}, {"text": None}, {"text": 5}, {"words": []}])
def test_an_answer_without_text_is_a_transcription_error(body: dict[str, Any]) -> None:
    with pytest.raises(TranscriptionError, match="no text"):
        parse_transcription(body, timed=True)


@pytest.mark.unit
def test_silence_is_an_empty_transcript_not_an_error() -> None:
    raw = parse_transcription({"text": "", "words": []}, timed=True)

    assert raw.text == ""
    assert raw.words == ()


def transcriber(server: Server) -> OpenRouterTranscriber:
    return OpenRouterTranscriber(server.http(), "elevenlabs/scribe-v2")


@pytest.mark.unit
def test_transcribe_posts_to_the_transcription_endpoint_and_parses() -> None:
    server = Server([json_response(verbose([word("гол", 1.0, 1.4)]))])

    raw = transcriber(server).transcribe(b"audio", "audio/mp3", duration_s=99.0)

    assert str(server.requests[0].url).endswith("/audio/transcriptions")
    assert server.body()["model"] == "elevenlabs/scribe-v2"
    assert raw.words[0].text == "гол"
    assert transcriber(server).model == "elevenlabs/scribe-v2"


@pytest.mark.unit
def test_a_refused_option_is_dropped_one_at_a_time_and_the_request_repeated() -> None:
    server = Server(
        [
            error_response(400, "diarize is not supported"),
            error_response(400, "timestamp_granularities is not supported"),
            json_response({"text": "всё равно текст"}),
        ]
    )

    raw = transcriber(server).transcribe(b"a", "audio/mp3")

    assert "diarize" in server.body(0)
    assert "diarize" not in server.body(1)
    assert "timestamp_granularities" in server.body(1)
    assert "timestamp_granularities" not in server.body(2)
    assert raw.text == "всё равно текст"


@pytest.mark.unit
def test_a_400_that_no_option_explains_is_raised() -> None:
    server = Server([error_response(400)] * 3)

    with pytest.raises(ExternalServiceError, match="400"):
        transcriber(server).transcribe(b"a", "audio/mp3")

    assert len(server.requests) == 3  # diarize and timestamps dropped, then the error stands


@pytest.mark.unit
def test_rate_limits_are_retried() -> None:
    server = Server([error_response(429), json_response(verbose([word("а", 0, 1)]))])

    raw = transcriber(server).transcribe(b"a", "audio/mp3")

    assert raw.words
    assert len(server.requests) == 2


@pytest.mark.unit
def test_no_credits_is_not_retried() -> None:
    server = Server([error_response(402)])

    with pytest.raises(ExternalServiceError, match="402"):
        transcriber(server).transcribe(b"a", "audio/mp3")

    assert len(server.requests) == 1
