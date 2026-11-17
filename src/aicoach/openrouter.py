"""HTTP access to OpenRouter, shared by the chat client and the speech-to-text client (D-068).

What the OpenRouter docs say (read 2026-11-17): the base URL is `https://openrouter.ai/api/v1`, the
key goes in `Authorization: Bearer <key>`, chat is `POST /chat/completions` (OpenAI format, tools as
`tools` and `tool_calls`), speech-to-text is `POST /audio/transcriptions`; 402 means no credits,
429 is a rate limit, 5xx and 524 are provider or timeout errors; the response `usage.cost` is what
was charged. The key is held as `SecretStr`, put into the request header only, and never logged.
"""

import time
from collections.abc import Callable
from typing import Any  # JSON bodies: their shape is OpenRouter's, not ours

import httpx2
import structlog
from pydantic import SecretStr

from aicoach.errors import ExternalServiceError

logger = structlog.get_logger()

BASE_URL = "https://openrouter.ai/api/v1"
BACKOFF_BASE_S = 4.0  # pauses 4, 8, 16, 32 s, as for the Gemini client
TRANSIENT_CODES = frozenset({408, 429, 500, 502, 503, 504, 524, 529})
_MAX_ERROR_TEXT = 200


def reported_cost(value: object) -> float | None:
    """A cost from a response: a finite non-negative number, else `None` (a NaN or a negative
    figure would break the monthly sum of the budget)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    cost = float(value)
    return cost if 0 <= cost < float("inf") else None


def count(value: object) -> int:
    """A token count from a response: a non-negative integer, else 0 (garbage is not a count)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value) if 0 <= value < float("inf") else 0


class OpenRouterError(ExternalServiceError):
    """OpenRouter refused or failed for good; `status` is the HTTP (or body) code, if any."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _error_of(body: object) -> tuple[int | None, str]:
    """The code and text of the `error` object of a response body (empty when there is none)."""
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return None, ""
    code = error.get("code")
    message = str(error.get("message", ""))[:_MAX_ERROR_TEXT]
    return (code if isinstance(code, int) else None), message


class OpenRouterHttp:
    """POST JSON to OpenRouter with retries on transient errors.

    Args:
        api_key: The OpenRouter key.
        base_url: The API root (a test server in tests).
        client: The `httpx2` client; built here when absent (tests give one with a mock transport).
        sleep: The pause function (a fake in tests).
    """

    def __init__(
        self,
        api_key: SecretStr,
        *,
        base_url: str = BASE_URL,
        client: httpx2.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key.get_secret_value()}"}
        self._client = client or httpx2.Client()
        self._sleep = sleep

    def post(
        self, path: str, payload: dict[str, Any], *, timeout_s: float, max_retries: int
    ) -> dict[str, Any]:
        """Send `payload` and return the JSON answer.

        Raises:
            OpenRouterError: A permanent error (bad request, no credits, bad key), a transient
                error that outlasted `max_retries` retries. A 200 answer that is not a JSON
                object comes back as `{}`; the caller reports what it misses.
        """
        for attempt in range(max_retries + 1):
            try:
                response = self._client.post(
                    f"{self._base_url}{path}",
                    json=payload,
                    headers=self._headers,
                    timeout=timeout_s,
                )
            except httpx2.HTTPError as err:  # timeout, connection refused, dropped
                self._pause_or_raise(attempt, max_retries, err, True, type(err).__name__, "", None)
                continue
            body = self._json(response)
            status = response.status_code
            body_code, text = _error_of(body)
            if status < httpx2.codes.BAD_REQUEST and body_code is None:
                return body if isinstance(body, dict) else {}
            code = body_code or status  # a provider error can come with HTTP 200
            self._pause_or_raise(
                attempt, max_retries, None, code in TRANSIENT_CODES, f"HTTP {code}", text, code
            )
        raise AssertionError  # pragma: no cover - the loop always returns or raises

    @staticmethod
    def _json(response: httpx2.Response) -> object:
        try:
            return response.json()
        except ValueError:  # not JSON: an error page of a proxy, for example
            return {}

    def _pause_or_raise(
        self,
        attempt: int,
        max_retries: int,
        err: Exception | None,
        transient: bool,
        what: str,
        text: str,
        status: int | None,
    ) -> None:
        """Wait before the next try, or give up with `OpenRouterError` (cause kept)."""
        if not transient or attempt == max_retries:
            detail = f": {text}" if text else ""
            msg = f"OpenRouter request failed ({what}){detail}"
            raise OpenRouterError(msg, status) from err
        pause = BACKOFF_BASE_S * 2**attempt
        logger.warning(
            "llm_retry", stage="openrouter", error=what, attempt=attempt + 1, pause_s=pause
        )
        self._sleep(pause)
