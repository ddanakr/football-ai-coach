"""The transcription model: a `Protocol` and the Gemini implementation.

Two implementations justify the `Protocol` (CLAUDE.md section 6): the real client and the test
double. Gemini specifics (docs checked 2026-11-16, `gemini-3.5-transcribe`): it is called through
the Interactions API (`client.interactions.create`, `v1beta`) with a `transcription_config`.
Word timestamps and speaker labels cannot be combined with a custom vocabulary, so a request is
either `timed` (words with times and speakers) or `vocabulary` (plain text, terms as hints).
The model documents no JSON-schema output: the structure comes from the word annotations.
"""

import base64
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx2
import structlog
from google import genai
from google.genai import errors
from pydantic import SecretStr

from aicoach.errors import ExternalServiceError, TranscriptionError
from aicoach.speech.segments import TimedWord

logger = structlog.get_logger()

REQUEST_TIMEOUT_S = 180.0
MAX_RETRIES = 3
BACKOFF_BASE_S = 4.0
TRANSIENT_CODES = frozenset({429, 500, 502, 503, 504})
MAX_VOCABULARY = 100  # "best results at about 100 terms" (docs)


@dataclass(frozen=True)
class RawTranscript:
    """What the model returned: the text, timed words (empty in vocabulary mode), token counts.

    `audio_seconds` and `cost_usd` are what the service itself reported (OpenRouter does, the
    direct Gemini API does not): `None` when unknown.
    """

    text: str
    words: tuple[TimedWord, ...]
    input_tokens: int
    output_tokens: int
    audio_seconds: float | None = None
    cost_usd: float | None = None


class Transcriber(Protocol):
    """A speech-to-text model."""

    @property
    def model(self) -> str:
        """The model name, for the result files."""
        ...

    def transcribe(
        self,
        audio: bytes,
        mime_type: str,
        *,
        vocabulary: Sequence[str] | None = None,
        duration_s: float | None = None,
    ) -> RawTranscript:
        """Transcribe `audio`. Without `vocabulary`: timed words with speakers. With it: text
        only, the terms are hints. `duration_s` is the length of the sound when the caller knows
        it (the spending estimate uses it; the model does not).

        Raises:
            ExternalServiceError: The service failed for good.
            TranscriptionError: The answer cannot be read.
        """
        ...


def parse_offset(value: object) -> float:
    """Seconds from an offset such as `10.100s`.

    Raises:
        TranscriptionError: Not a finite non-negative number with an optional `s`.
    """
    shown = str(value)[:20]
    try:
        seconds = float(str(value).strip().removesuffix("s"))
    except ValueError as err:
        msg = f"bad time offset in the answer: {shown!r}"
        raise TranscriptionError(msg) from err
    if not 0 <= seconds < float("inf"):
        msg = f"bad time offset in the answer: {shown!r}"
        raise TranscriptionError(msg)
    return seconds


def _annotations(response: Any) -> list[Any]:  # noqa: ANN401 - SDK objects are duck-typed
    found: list[Any] = []
    for step in getattr(response, "steps", None) or []:
        for content in getattr(step, "content", None) or []:
            found += [
                a
                for a in getattr(content, "annotations", None) or []
                if getattr(a, "type", None) == "word_info"
            ]
    return found


def parse_response(response: Any, *, timed: bool) -> RawTranscript:  # noqa: ANN401 - see above
    """The transcript of an Interactions response.

    Raises:
        TranscriptionError: A timed answer has text but no usable word annotations, or a word
            has a bad time.
    """
    text = (getattr(response, "output_text", None) or "").strip()
    usage = getattr(response, "usage", None)
    tokens_in = int(getattr(usage, "total_input_tokens", 0) or 0)
    tokens_out = int(getattr(usage, "total_output_tokens", 0) or 0)
    words: list[TimedWord] = []
    if timed:
        for ann in _annotations(response):
            word = str(getattr(ann, "text", "") or "").strip()
            if word:
                start = parse_offset(getattr(ann, "start_offset", None))
                end = parse_offset(getattr(ann, "end_offset", None))
                words.append(TimedWord(word, start, end, getattr(ann, "speaker", None)))
        if text and not words:
            msg = "the answer has text but no word times"
            raise TranscriptionError(msg)
    return RawTranscript(text, tuple(words), tokens_in, tokens_out)


def request_config(vocabulary: Sequence[str] | None) -> dict[str, Any]:
    """The `generation_config` of a timed (no vocabulary) or a vocabulary request (pure)."""
    if vocabulary is None:
        mode = {
            "type": "verbatim",
            "diarization_mode": "speaker",
            "timestamp_granularities": ["word"],
        }
        return {"transcription_config": {"mode": mode}}
    terms = list(dict.fromkeys(t.strip() for t in vocabulary if t.strip()))
    if len(terms) > MAX_VOCABULARY:
        logger.warning("vocabulary_cut", stage="speech", terms=len(terms), kept=MAX_VOCABULARY)
        terms = terms[:MAX_VOCABULARY]
    return {"transcription_config": {"custom_vocabulary": terms}}


class GeminiTranscriber:
    """`gemini-3.5-transcribe` (or the model of the settings) through `google-genai`."""

    def __init__(
        self,
        api_key: SecretStr,
        model: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
        interactions: Any = None,  # noqa: ANN401 - the SDK resource, replaceable in tests
    ) -> None:
        self._model = model
        self._sleep = sleep
        self._client: genai.Client | None = None  # keep referenced: it closes on collection
        if interactions is None:
            self._client = genai.Client(
                api_key=api_key.get_secret_value(), http_options={"api_version": "v1beta"}
            )
            interactions = self._client.interactions
        self._interactions = interactions

    @property
    def model(self) -> str:
        """The model name."""
        return self._model

    def transcribe(
        self,
        audio: bytes,
        mime_type: str,
        *,
        vocabulary: Sequence[str] | None = None,
        duration_s: float | None = None,  # noqa: ARG002 - part of the `Transcriber` contract
    ) -> RawTranscript:
        """See `Transcriber.transcribe`; retries 429 and 5xx with growing pauses."""
        data = base64.b64encode(audio).decode("ascii")
        payload = [{"type": "audio", "data": data, "mime_type": mime_type}]
        config = request_config(vocabulary)
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._interactions.create(
                    model=self._model,
                    input=payload,
                    generation_config=config,
                    timeout=REQUEST_TIMEOUT_S,
                )
            except errors.APIError as err:
                self._pause_or_raise(attempt, err, err.code in TRANSIENT_CODES, str(err.code))
            except httpx2.HTTPError as err:
                self._pause_or_raise(attempt, err, True, type(err).__name__)
            else:
                return parse_response(response, timed=vocabulary is None)
        raise AssertionError  # pragma: no cover - the loop always returns or raises

    def _pause_or_raise(self, attempt: int, err: Exception, transient: bool, what: str) -> None:
        if not transient or attempt == MAX_RETRIES:
            msg = f"Gemini transcription failed ({what})"
            raise ExternalServiceError(msg) from err
        pause = BACKOFF_BASE_S * 2**attempt
        logger.warning("llm_retry", stage="speech", error=what, attempt=attempt + 1, pause_s=pause)
        self._sleep(pause)
