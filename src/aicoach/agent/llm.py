"""The language-model client: a `Protocol` and the Gemini implementation (google-genai SDK).

The loop talks to `LLMClient` only: the real client and the test double are the two
implementations that justify the `Protocol` (CLAUDE.md section 6).

Gemini specifics kept here: function declarations from the JSON schema of `ToolSpec`, the manual
tool loop (the SDK's automatic function calling is switched off so that the loop counts and
limits the calls), the model's own `Content` is passed back unchanged (thought signatures), the
thinking parts are dropped from the text, and the retry on 429 and 5xx with growing pauses.
"""

import time
from collections.abc import Callable, Sequence
from typing import Protocol

import httpx2
import structlog
from google import genai
from google.genai import errors, types
from pydantic import SecretStr

from aicoach.agent.messages import (
    HistoryItem,
    ModelTurn,
    ToolCall,
    ToolResults,
    ToolSpec,
    Usage,
    UserMessage,
)
from aicoach.errors import ExternalServiceError

logger = structlog.get_logger()

REQUEST_TIMEOUT_S = 60.0
EMPTY_TURN_TEXT = "(пустой ответ)"
MAX_RETRIES = 4
BACKOFF_BASE_S = 4.0  # pauses 4, 8, 16, 32 s: about a minute, enough for a per-minute quota
TRANSIENT_CODES = frozenset({429, 500, 502, 503, 504})
_MAX_ERROR_TEXT = 200


class LLMClient(Protocol):
    """A language model with function calling."""

    @property
    def model(self) -> str:
        """The model name, for the journal and the price."""
        ...

    def generate(
        self,
        system: str,
        history: Sequence[HistoryItem],
        tools: Sequence[ToolSpec],
        *,
        allow_tools: bool = True,
    ) -> ModelTurn:
        """One request: the next turn of the model.

        Raises:
            ExternalServiceError: The service failed for good (after the retries).
        """
        ...


class _Models(Protocol):
    def generate_content(
        self, *, model: str, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse: ...


def to_contents(history: Sequence[HistoryItem]) -> list[types.Content]:
    """The history as Gemini `Content` objects (pure)."""
    contents: list[types.Content] = []
    for item in history:
        if isinstance(item, UserMessage):
            contents.append(types.Content(role="user", parts=[types.Part(text=item.text)]))
        elif isinstance(item, ModelTurn):
            contents.append(_model_content(item))
        else:
            contents.append(_results_content(item))
    return contents


def _model_content(turn: ModelTurn) -> types.Content:
    if isinstance(turn.raw, types.Content) and turn.raw.parts:
        return turn.raw  # unchanged: it carries the thought signatures
    parts = [types.Part(text=turn.text)] if turn.text else []
    parts += [
        types.Part(function_call=types.FunctionCall(name=c.name, args=c.args, id=c.call_id))
        for c in turn.calls
    ]
    # A turn with nothing in it (an empty answer) cannot be sent back as it is: the API refuses
    # a content without parts.
    return types.Content(role="model", parts=parts or [types.Part(text=EMPTY_TURN_TEXT)])


def _results_content(results: ToolResults) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name=r.name, id=r.call_id, response={"result": r.result}
                )
            )
            for r in results.results
        ],
    )


def from_response(response: types.GenerateContentResponse) -> ModelTurn:
    """The turn from a Gemini response (pure).

    Raises:
        ExternalServiceError: The response has no candidate (for example it was blocked).
    """
    usage_meta = response.usage_metadata
    usage = Usage(
        (usage_meta.prompt_token_count or 0) if usage_meta else 0,
        # Thinking tokens are billed as output.
        ((usage_meta.candidates_token_count or 0) + (usage_meta.thoughts_token_count or 0))
        if usage_meta
        else 0,
    )
    if not response.candidates:
        reason = getattr(response.prompt_feedback, "block_reason", None)
        msg = f"Gemini returned no answer (block reason: {reason})"
        raise ExternalServiceError(msg)
    content = response.candidates[0].content
    texts: list[str] = []
    calls: list[ToolCall] = []
    for part in (content.parts or []) if content else []:
        if part.function_call is not None and part.function_call.name:
            call = part.function_call
            calls.append(ToolCall(call.name or "", dict(call.args or {}), call.id))
        elif part.text and not part.thought:
            texts.append(part.text)
    cut_off = response.candidates[0].finish_reason == types.FinishReason.MAX_TOKENS
    return ModelTurn("".join(texts), tuple(calls), usage, content, truncated=cut_off)


def _declarations(tools: Sequence[ToolSpec]) -> types.ToolListUnion:
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t.name, description=t.description, parameters_json_schema=t.parameters
                )
                for t in tools
            ]
        )
    ]


class GeminiClient:
    """Gemini through `google-genai`. The key is passed in as `SecretStr` and used only to open
    the connection; it is never logged."""

    def __init__(
        self,
        api_key: SecretStr,
        model: str,
        *,
        timeout_s: float = REQUEST_TIMEOUT_S,
        sleep: Callable[[float], None] = time.sleep,
        models: _Models | None = None,
    ) -> None:
        self._model = model
        self._timeout_ms = int(timeout_s * 1000)
        self._sleep = sleep
        # The client object must stay referenced: it closes the connection when it is collected.
        self._client: genai.Client | None = None
        if models is None:
            self._client = genai.Client(api_key=api_key.get_secret_value())
            models = self._client.models
        self._models = models

    @property
    def model(self) -> str:
        """The model name."""
        return self._model

    def generate(
        self,
        system: str,
        history: Sequence[HistoryItem],
        tools: Sequence[ToolSpec],
        *,
        allow_tools: bool = True,
    ) -> ModelTurn:
        """One request with retries on 429 and 5xx.

        Raises:
            ExternalServiceError: A permanent error, retries used up, or an empty answer.
        """
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=_declarations(tools),
            # The loop makes the calls itself: it counts them and enforces the limit.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tool_config=None
            if allow_tools
            else types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.NONE
                )
            ),
            http_options=types.HttpOptions(timeout=self._timeout_ms),
        )
        response = self._request(to_contents(history), config)
        return from_response(response)

    def _request(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self._models.generate_content(
                    model=self._model, contents=contents, config=config
                )
            except errors.APIError as err:
                self._pause_or_raise(
                    attempt,
                    err,
                    err.code in TRANSIENT_CODES,
                    f"{err.code} {err.status}",
                    err.message,
                )
            except httpx2.HTTPError as err:  # timeout, connection refused, dropped
                self._pause_or_raise(attempt, err, True, type(err).__name__, None)
        raise AssertionError  # pragma: no cover - the loop always returns or raises

    def _pause_or_raise(
        self, attempt: int, err: Exception, transient: bool, what: str, text: str | None
    ) -> None:
        """Wait before the next try, or give up with `ExternalServiceError` (cause kept)."""
        if not transient or attempt == MAX_RETRIES:
            detail = f": {text[:_MAX_ERROR_TEXT]}" if text else ""
            msg = f"Gemini request failed ({what}){detail}"
            raise ExternalServiceError(msg) from err
        pause = BACKOFF_BASE_S * 2**attempt
        logger.warning("llm_retry", stage="agent", error=what, attempt=attempt + 1, pause_s=pause)
        self._sleep(pause)
