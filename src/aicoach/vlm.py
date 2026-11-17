"""A model that watches a video clip, through OpenRouter (FR-11, D-073).

What the OpenRouter docs say (guides/overview/multimodal/videos, read 2026-11-18): the Chat
Completions API takes a `video_url` content part whose `url` is a public address or a base64 data
URL (`data:video/mp4;base64,...`); formats `video/mp4`, `video/mpeg`, `video/mov`, `video/webm`;
only models whose `architecture.input_modalities` has `video` accept it. A clip of a local match is
not public, so it is always sent as a data URL. The answer is an ordinary chat answer, and the
cost the service charged is in `usage.cost`, as for the agent.
"""

import base64
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar  # Any: JSON of the chat API

from aicoach.agent.messages import Usage
from aicoach.agent.openrouter_llm import CHAT_PATH, from_response
from aicoach.openrouter import OpenRouterHttp

# A clip upload is megabytes and a video answer takes longer than a text one.
REQUEST_TIMEOUT_S = 180.0
MAX_RETRIES = 3
# A cap on the answer, thinking tokens included: a description of 2-4 sentences is far shorter,
# the cap only stops a model that runs on.
MAX_OUTPUT_TOKENS = 1500
VIDEO_MIME = "video/mp4"


@dataclass(frozen=True)
class VideoReply:
    """What a video model said; `truncated` is set when it stopped at the length cap."""

    text: str
    usage: Usage = field(default_factory=Usage)
    truncated: bool = False


class VideoModel(Protocol):
    """A model that answers a question about a video clip (OpenRouter, and a fake in tests)."""

    @property
    def model(self) -> str:
        """The model name."""
        ...

    def describe(
        self, system: str, prompt: str, video: bytes, *, video_seconds: float | None = None
    ) -> VideoReply:
        """Answer `prompt` about `video` (MP4 bytes).

        `video_seconds` is the length of the clip, used only to estimate the cost beforehand.
        """
        ...


def build_payload(model: str, system: str, prompt: str, video: bytes) -> dict[str, Any]:
    """The chat request with the clip inline (pure)."""
    data_url = f"data:{VIDEO_MIME};base64,{base64.b64encode(video).decode('ascii')}"
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": data_url}},
                ],
            },
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
    }


class OpenRouterVideoModel:
    """A video-capable chat model through OpenRouter; `model` is the OpenRouter slug."""

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

    def describe(
        self,
        system: str,
        prompt: str,
        video: bytes,
        *,
        video_seconds: float | None = None,  # noqa: ARG002 - for the budget estimate only
    ) -> VideoReply:
        """One request with retries on 429, 5xx and network errors.

        Raises:
            ExternalServiceError: A permanent error, retries used up, or an unusable answer.
        """
        body = self._http.post(
            CHAT_PATH,
            build_payload(self._model, system, prompt, video),
            timeout_s=self._timeout_s,
            max_retries=MAX_RETRIES,
        )
        turn = from_response(body)
        return VideoReply(turn.text, turn.usage, turn.truncated)


_FENCE = re.compile(r"^```[A-Za-z]*\s*|\s*```$")
_T = TypeVar("_T")


def extract_json_object(text: str) -> dict[str, Any] | None:  # Any: JSON from a model
    """The JSON object a model answered with, or `None` when there is no usable one.

    Accepts the bare object and the one wrapped in a Markdown code fence or in a sentence of
    prose (the first `{` to the last `}`). A list, a scalar or a broken text give `None`.
    """
    body = _FENCE.sub("", text.strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(body[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class JsonAnswer(Generic[_T]):
    """The outcome of `ask_json`: the parsed value (`None` when every attempt was invalid), the
    number of requests made and the tokens and cost of all of them."""

    value: _T | None
    attempts: int
    usage: Usage = field(default_factory=Usage)


def ask_json(
    model: VideoModel,
    system: str,
    prompt: str,
    video: bytes,
    parse: Callable[[str], _T | None],
    *,
    video_seconds: float | None = None,
    max_attempts: int = 2,
) -> JsonAnswer[_T]:
    """Ask a video model for a strict answer: `parse` turns the text into a value or `None`.

    An answer that `parse` rejects (not JSON, wrong values, cut off) is asked for once more
    (`max_attempts` requests in all); after that the value is `None` and the caller decides what
    "no answer" means. A service failure is not retried here (`OpenRouterVideoModel` already
    retries 429 and 5xx) and propagates.

    Raises:
        ExternalServiceError: The service failed for good.
        BudgetExceededError: The budget does not allow a request.
    """
    usage = Usage()
    for attempt in range(1, max_attempts + 1):
        reply = model.describe(system, prompt, video, video_seconds=video_seconds)
        usage = usage + reply.usage
        value = None if reply.truncated else parse(reply.text)
        if value is not None:
            return JsonAnswer(value, attempt, usage)
    return JsonAnswer(None, max_attempts, usage)
