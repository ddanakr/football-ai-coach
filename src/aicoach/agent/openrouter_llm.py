"""The OpenRouter chat client for the agent: the second `LLMClient` after Gemini (D-068).

OpenRouter speaks the OpenAI chat format (docs read 2026-11-17): the tools go in `tools`
(`{"type": "function", "function": {name, description, parameters}}`), a call comes back in
`message.tool_calls` with a JSON string in `function.arguments`, and the result goes back as a
`role: "tool"` message with the same `tool_call_id`. `finish_reason` is `length` when the answer
was cut. The assistant message of the response is kept as `ModelTurn.raw` and sent back unchanged,
so the reasoning details some models need with the next request are not lost.
"""

import json
from collections.abc import Sequence
from typing import Any  # JSON of the chat API

import structlog

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
from aicoach.openrouter import OpenRouterHttp, count, reported_cost

logger = structlog.get_logger()

REQUEST_TIMEOUT_S = 60.0
MAX_RETRIES = 4
EMPTY_TURN_TEXT = "(пустой ответ)"
CHAT_PATH = "/chat/completions"


def to_messages(system: str, history: Sequence[HistoryItem]) -> list[dict[str, Any]]:
    """The history as OpenAI chat messages, the system prompt first (pure)."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for item in history:
        if isinstance(item, UserMessage):
            messages.append({"role": "user", "content": item.text})
        elif isinstance(item, ModelTurn):
            messages.append(_assistant_message(item))
        else:
            messages.extend(_tool_messages(item))
    return messages


def _assistant_message(turn: ModelTurn) -> dict[str, Any]:
    if isinstance(turn.raw, dict) and (turn.raw.get("content") or turn.raw.get("tool_calls")):
        # Unchanged but for the role: it carries the reasoning details of the model.
        return {"role": "assistant", **turn.raw}
    message: dict[str, Any] = {"role": "assistant", "content": turn.text or EMPTY_TURN_TEXT}
    if turn.calls:
        message["content"] = turn.text or None
        message["tool_calls"] = [
            {
                "id": call.call_id or f"call_{n}",
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.args)},
            }
            for n, call in enumerate(turn.calls)
        ]
    return message


def _tool_messages(results: ToolResults) -> list[dict[str, Any]]:
    return [
        {
            "role": "tool",
            "tool_call_id": r.call_id or f"call_{n}",
            "content": json.dumps(r.result, ensure_ascii=False),
        }
        for n, r in enumerate(results.results)
    ]


def to_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """The tools in the OpenAI format (pure)."""
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def _calls_of(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    raw_calls = message.get("tool_calls")
    for raw in raw_calls if isinstance(raw_calls, list) else []:
        function = raw.get("function") if isinstance(raw, dict) else None
        function = function if isinstance(function, dict) else {}
        name = str(function.get("name") or "")
        if not name:
            continue
        arguments = function.get("arguments") or "{}"
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError as err:
            msg = f"OpenRouter: the model sent unreadable arguments for the tool {name}"
            raise ExternalServiceError(msg) from err
        if not isinstance(args, dict):
            msg = f"OpenRouter: the arguments of the tool {name} are not an object"
            raise ExternalServiceError(msg)
        call_id = raw.get("id")
        calls.append(ToolCall(name, args, call_id if isinstance(call_id, str) else None))
    return tuple(calls)


def _usage_of(body: dict[str, Any]) -> Usage:
    usage = body.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return Usage(
        count(usage.get("prompt_tokens")),
        # The completion count includes the reasoning tokens, which are billed as output.
        count(usage.get("completion_tokens")),
        reported_cost(usage.get("cost")),
    )


def from_response(body: dict[str, Any]) -> ModelTurn:
    """The turn from a chat response (pure).

    Raises:
        ExternalServiceError: The response has no choice, or a call has unreadable arguments.
    """
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        msg = "OpenRouter returned no answer (no choices)"
        raise ExternalServiceError(msg)
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        msg = "OpenRouter returned no answer (no message)"
        raise ExternalServiceError(msg)
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    return ModelTurn(
        text,
        _calls_of(message),
        _usage_of(body),
        message,
        truncated=choice.get("finish_reason") == "length",
    )


class OpenRouterClient:
    """A chat model through OpenRouter. `model` is the OpenRouter slug (`google/...`)."""

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

    def generate(
        self,
        system: str,
        history: Sequence[HistoryItem],
        tools: Sequence[ToolSpec],
        *,
        allow_tools: bool = True,
    ) -> ModelTurn:
        """One request with retries on 429, 5xx and network errors.

        Raises:
            ExternalServiceError: A permanent error, retries used up, or an unusable answer.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": to_messages(system, history),
            "tools": to_tools(tools),
            # The loop counts the calls: when its limit is reached the tools are switched off,
            # but they stay in the request, because the history still holds tool messages.
            "tool_choice": "auto" if allow_tools else "none",
        }
        body = self._http.post(
            CHAT_PATH, payload, timeout_s=self._timeout_s, max_retries=MAX_RETRIES
        )
        return from_response(body)
