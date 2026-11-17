"""A recorded OpenRouter: an `httpx2` client over a mock transport (no network)."""

import json
from collections.abc import Callable
from typing import Any  # JSON bodies of the API: their shape is the service's, not ours

import httpx2
from pydantic import SecretStr

from aicoach.openrouter import OpenRouterHttp

FAKE_KEY = "sk-or-v1-fake-key-for-tests-0123456789"  # pragma: allowlist secret

Handler = Callable[[httpx2.Request], httpx2.Response]


def json_response(body: dict[str, Any], status: int = 200) -> httpx2.Response:
    """A response with a JSON body."""
    return httpx2.Response(status, json=body)


def error_response(status: int, message: str = "boom", code: int | None = None) -> httpx2.Response:
    """The error body OpenRouter sends: `{"error": {"code", "message"}}`."""
    return httpx2.Response(status, json={"error": {"code": code or status, "message": message}})


class Server:
    """Plays a script of responses (or exceptions) and records the requests it got."""

    def __init__(self, script: list[httpx2.Response | Exception]) -> None:
        self.script = list(script)
        self.requests: list[httpx2.Request] = []
        self.pauses: list[float] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        """The mock transport handler."""
        self.requests.append(request)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def body(self, index: int = 0) -> dict[str, Any]:
        """The JSON body of the `index`-th request."""
        body: dict[str, Any] = json.loads(self.requests[index].content)
        return body

    def http(self) -> OpenRouterHttp:
        """The OpenRouter access that talks to this server."""
        client = httpx2.Client(transport=httpx2.MockTransport(self.handle))
        return OpenRouterHttp(
            SecretStr(FAKE_KEY),
            base_url="https://or.test/api/v1",
            client=client,
            sleep=self.pauses.append,
        )


def chat_body(
    text: str | None = "ответ",
    calls: list[tuple[str, str, str]] | None = None,
    *,
    finish: str = "stop",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A chat completion body; `calls` are (id, name, JSON arguments)."""
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {"id": i, "type": "function", "function": {"name": n, "arguments": a}}
            for i, n, a in calls
        ]
    return {
        "id": "gen-1",
        "choices": [{"finish_reason": finish, "message": message}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0001},
    }
