"""Reading the model's answer, the request config, the retries (no network)."""

from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors
from pydantic import SecretStr

from aicoach.errors import ExternalServiceError, TranscriptionError
from aicoach.speech.transcriber import (
    MAX_RETRIES,
    MAX_VOCABULARY,
    GeminiTranscriber,
    parse_offset,
    parse_response,
    request_config,
)


def _word(text: str, start: str, end: str, speaker: str = "spk:0") -> SimpleNamespace:
    return SimpleNamespace(
        type="word_info", text=text, start_offset=start, end_offset=end, speaker=speaker
    )


def _response(text: str, words: list[Any], tokens: tuple[int, int] = (10, 5)) -> SimpleNamespace:
    step = SimpleNamespace(content=[SimpleNamespace(annotations=words)])
    usage = SimpleNamespace(total_input_tokens=tokens[0], total_output_tokens=tokens[1])
    return SimpleNamespace(output_text=text, steps=[step], usage=usage)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "seconds"), [("10.100s", 10.1), ("0s", 0.0), (3, 3.0), (" 2.5 ", 2.5)]
)
def test_parse_offset_reads_seconds(raw: object, seconds: float) -> None:
    assert parse_offset(raw) == seconds


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["", "abc", "-1s", "nan", "inf", None])
def test_parse_offset_rejects_garbage(raw: object) -> None:
    with pytest.raises(TranscriptionError):
        parse_offset(raw)


@pytest.mark.unit
def test_timed_response_gives_words_speakers_and_tokens() -> None:
    words = [_word("Гол.", "1.0s", "1.5s"), _word("Ура", "2s", "2.4s", "spk:1")]

    raw = parse_response(_response("Гол. Ура", words), timed=True)

    assert [(w.text, w.start_s, w.speaker) for w in raw.words] == [
        ("Гол.", 1.0, "spk:0"),
        ("Ура", 2.0, "spk:1"),
    ]
    assert (raw.input_tokens, raw.output_tokens, raw.text) == (10, 5, "Гол. Ура")


@pytest.mark.unit
def test_empty_answer_is_an_empty_transcript_not_an_error() -> None:
    raw = parse_response(SimpleNamespace(output_text="", steps=None, usage=None), timed=True)

    assert (raw.text, raw.words, raw.input_tokens) == ("", (), 0)


@pytest.mark.unit
def test_timed_text_without_word_times_and_bad_times_are_errors() -> None:
    with pytest.raises(TranscriptionError):
        parse_response(_response("слова есть", []), timed=True)
    with pytest.raises(TranscriptionError):
        parse_response(_response("гол", [_word("гол", "oops", "1s")]), timed=True)


@pytest.mark.unit
def test_vocabulary_mode_ignores_annotations_and_blank_words_are_dropped() -> None:
    plain = parse_response(_response("текст", [_word("текст", "0s", "1s")]), timed=False)
    blank = [_word("  ", "0s", "1s"), _word("x", "1s", "2s")]
    blanks = parse_response(_response("x", blank), timed=True)

    assert plain.words == ()
    assert [w.text for w in blanks.words] == ["x"]


@pytest.mark.unit
def test_request_config_modes() -> None:
    timed = request_config(None)["transcription_config"]["mode"]
    assert timed["diarization_mode"] == "speaker"
    assert timed["timestamp_granularities"] == ["word"]

    vocab = request_config(["Заруцкий", " ", "Заруцкий", "Топалов"])["transcription_config"]
    assert vocab == {"custom_vocabulary": ["Заруцкий", "Топалов"]}
    many = request_config([f"t{i}" for i in range(500)])["transcription_config"]
    assert len(many["custom_vocabulary"]) == MAX_VOCABULARY


class _Interactions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client(outcomes: list[object]) -> tuple[GeminiTranscriber, _Interactions, list[float]]:
    pauses: list[float] = []
    fake = _Interactions(outcomes)
    client = GeminiTranscriber(SecretStr("k"), "model-x", sleep=pauses.append, interactions=fake)
    return client, fake, pauses


def _api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"message": "x", "status": "S"}})


@pytest.mark.unit
def test_request_sends_audio_inline_and_the_model_name() -> None:
    client, fake, _ = _client([_response("гол", [_word("гол", "0s", "1s")])])

    raw = client.transcribe(b"\x00\x01", "audio/mp3")

    call = fake.calls[0]
    assert call["model"] == "model-x"
    assert call["input"][0]["type"] == "audio"
    assert call["input"][0]["mime_type"] == "audio/mp3"
    assert raw.text == "гол"
    assert client.model == "model-x"


@pytest.mark.unit
def test_transient_errors_are_retried_with_growing_pauses() -> None:
    ok = _response("a", [_word("a", "0s", "1s")])
    client, fake, pauses = _client([_api_error(429), _api_error(503), ok])

    client.transcribe(b"x", "audio/mp3")

    assert len(fake.calls) == 3
    assert pauses == [4.0, 8.0]


@pytest.mark.unit
@pytest.mark.parametrize(("code", "calls"), [(400, 1), (429, MAX_RETRIES + 1)])
def test_permanent_error_or_used_up_retries_raise_external_service_error(
    code: int, calls: int
) -> None:
    client, fake, _ = _client([_api_error(code) for _ in range(MAX_RETRIES + 1)])

    with pytest.raises(ExternalServiceError):
        client.transcribe(b"x", "audio/mp3")
    assert len(fake.calls) == calls
