"""Speech-to-text through OpenRouter: the second `Transcriber` after Gemini (D-068).

What the OpenRouter docs say (STT API reference, read 2026-11-17): `POST /audio/transcriptions`
takes JSON with `model` and `input_audio` (`data` is raw base64, `format` is `wav`, `mp3`, ...);
`response_format: "verbose_json"` returns `language`, `duration`, `segments` and, with
`timestamp_granularities: ["word"]`, `words` (`word`, `start`, `end`, optional `speaker`, and
`type` of `word` or `audio_event`); `diarize: true` labels the words with speakers (verbose JSON
only) and `keyterms` (up to 1000 terms) bias the recognition; both work with some providers only,
the others answer 400. `usage` carries `seconds`, token counts and `cost` in US dollars.

An option the provider refuses is dropped and the request repeated (diarization first, then the
word timestamps): the transcript is then less detailed, and a warning says so.
"""

import base64
from collections.abc import Sequence
from typing import Any  # JSON of the STT API

import structlog

from aicoach.errors import TranscriptionError
from aicoach.openrouter import OpenRouterError, OpenRouterHttp, count, reported_cost
from aicoach.speech.segments import TimedWord
from aicoach.speech.transcriber import RawTranscript

logger = structlog.get_logger()

TRANSCRIBE_PATH = "/audio/transcriptions"
REQUEST_TIMEOUT_S = 180.0
MAX_RETRIES = 3
MAX_KEYTERMS = 1000  # the documented limit of `keyterms`
MAX_KEYTERM_CHARS = 100
_NOT_SPEECH = frozenset({"audio_event", "spacing"})
_OPTIONS_TO_DROP = ("diarize", "timestamp_granularities", "keyterms")
_FORMATS = {"mpeg": "mp3", "x-wav": "wav", "wave": "wav", "x-m4a": "m4a", "mp4": "m4a"}
_HTTP_BAD_REQUEST = 400


def audio_format(mime_type: str) -> str:
    """The `input_audio.format` of a MIME type (`audio/mp3` gives `mp3`)."""
    subtype = mime_type.split("/")[-1].split(";")[0].strip().lower()
    return _FORMATS.get(subtype, subtype)


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"bad {what} in the answer"
        raise TranscriptionError(msg)
    seconds = float(value)
    if not 0 <= seconds < float("inf"):
        msg = f"bad {what} in the answer"
        raise TranscriptionError(msg)
    return seconds


def _items(value: object) -> list[dict[str, Any]]:
    """The objects of a list in the answer; anything else in it is ignored."""
    return [i for i in value if isinstance(i, dict)] if isinstance(value, list) else []


def _speaker(item: dict[str, Any]) -> str | None:
    label = item.get("speaker", item.get("speaker_label"))
    return None if label is None else str(label)


def _words_of(body: dict[str, Any]) -> list[TimedWord]:
    """The timed words; without word timestamps the segments stand in for words."""
    words: list[TimedWord] = []
    for item in _items(body.get("words")):
        text = str(item.get("word") or "").strip()
        if not text or item.get("type") in _NOT_SPEECH:
            continue
        words.append(
            TimedWord(
                text,
                _number(item.get("start"), "word start"),
                _number(item.get("end"), "word end"),
                _speaker(item),
            )
        )
    if words:
        return words
    for item in _items(body.get("segments")):
        text = str(item.get("text") or "").strip()
        if text:
            words.append(
                TimedWord(
                    text,
                    _number(item.get("start"), "segment start"),
                    _number(item.get("end"), "segment end"),
                    _speaker(item),
                )
            )
    return words


def parse_transcription(body: dict[str, Any], *, timed: bool) -> RawTranscript:
    """The transcript of a response body (pure).

    Raises:
        TranscriptionError: The body has no text, or a time is not a finite non-negative number.
    """
    text = body.get("text")
    if not isinstance(text, str):
        msg = "the answer has no text"
        raise TranscriptionError(msg)
    usage = body.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    seconds = usage.get("seconds")
    words = _words_of(body) if timed else []
    if timed and text.strip() and not words:
        logger.warning("stt_no_timestamps", stage="speech")
    return RawTranscript(
        text.strip(),
        tuple(words),
        count(usage.get("input_tokens")),
        count(usage.get("output_tokens")),
        reported_cost(seconds),
        reported_cost(usage.get("cost")),
    )


def request_payload(
    model: str, audio: bytes, mime_type: str, vocabulary: Sequence[str] | None
) -> dict[str, Any]:
    """The JSON body: timed words with speakers, or (with `vocabulary`) plain text with hints."""
    payload: dict[str, Any] = {
        "model": model,
        "input_audio": {
            "data": base64.b64encode(audio).decode("ascii"),
            "format": audio_format(mime_type),
        },
    }
    if vocabulary is None:
        payload["response_format"] = "verbose_json"
        payload["timestamp_granularities"] = ["word"]
        payload["diarize"] = True
        return payload
    terms = [t.strip() for t in vocabulary if t.strip() and len(t.strip()) <= MAX_KEYTERM_CHARS]
    payload["keyterms"] = list(dict.fromkeys(terms))[:MAX_KEYTERMS]
    return payload


class OpenRouterTranscriber:
    """A speech-to-text model through OpenRouter (`elevenlabs/scribe-v2`, ...)."""

    def __init__(
        self, http: OpenRouterHttp, model: str, *, timeout_s: float = REQUEST_TIMEOUT_S
    ) -> None:
        self._http = http
        self._model = model
        self._timeout_s = timeout_s

    @property
    def model(self) -> str:
        """The model slug."""
        return self._model

    def transcribe(
        self,
        audio: bytes,
        mime_type: str,
        *,
        vocabulary: Sequence[str] | None = None,
        duration_s: float | None = None,  # noqa: ARG002 - part of the `Transcriber` contract
    ) -> RawTranscript:
        """See `Transcriber.transcribe`; retries 429 and 5xx, and drops an option the provider
        refuses with 400.

        Raises:
            ExternalServiceError: The service failed for good.
            TranscriptionError: The answer cannot be read.
        """
        payload = request_payload(self._model, audio, mime_type, vocabulary)
        body = self._post(payload)
        return parse_transcription(body, timed=vocabulary is None)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        dropped: list[str] = []
        while True:
            try:
                return self._http.post(
                    TRANSCRIBE_PATH, payload, timeout_s=self._timeout_s, max_retries=MAX_RETRIES
                )
            except OpenRouterError as err:
                option = next((o for o in _OPTIONS_TO_DROP if o in payload), None)
                if err.status != _HTTP_BAD_REQUEST or option is None:
                    raise
                payload = {k: v for k, v in payload.items() if k != option}
                dropped.append(option)
                logger.warning(
                    "stt_option_refused", stage="speech", model=self._model, dropped=dropped
                )
