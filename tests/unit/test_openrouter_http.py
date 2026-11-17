"""The OpenRouter HTTP access: the key, retries, errors (a recorded server, no network)."""

import httpx2
import pytest

from aicoach.errors import ExternalServiceError
from aicoach.openrouter import BACKOFF_BASE_S, OpenRouterError
from tests.fixtures.openrouter_synth import (
    FAKE_KEY,
    Server,
    error_response,
    json_response,
)


@pytest.mark.unit
def test_post_sends_the_key_in_the_header_and_returns_the_json() -> None:
    server = Server([json_response({"ok": True})])

    body = server.http().post("/x", {"a": 1}, timeout_s=5, max_retries=2)

    request = server.requests[0]
    assert body == {"ok": True}
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert str(request.url) == "https://or.test/api/v1/x"
    assert server.body() == {"a": 1}


@pytest.mark.unit
def test_transient_errors_are_retried_with_growing_pauses() -> None:
    server = Server([error_response(429), error_response(503), json_response({"ok": 1})])

    body = server.http().post("/x", {}, timeout_s=5, max_retries=4)

    assert body == {"ok": 1}
    assert len(server.requests) == 3
    assert server.pauses == [BACKOFF_BASE_S, BACKOFF_BASE_S * 2]


@pytest.mark.unit
def test_used_up_retries_raise_with_the_status_and_no_key() -> None:
    server = Server([error_response(429, "slow down")] * 3)

    with pytest.raises(OpenRouterError, match="HTTP 429") as caught:
        server.http().post("/x", {}, timeout_s=5, max_retries=2)

    assert caught.value.status == 429
    assert len(server.requests) == 3
    assert FAKE_KEY not in str(caught.value)
    assert isinstance(caught.value, ExternalServiceError)


@pytest.mark.unit
@pytest.mark.parametrize("status", [400, 401, 402, 404])
def test_permanent_errors_are_not_retried(status: int) -> None:
    server = Server([error_response(status, "no credits")])

    with pytest.raises(OpenRouterError, match=f"HTTP {status}.*no credits"):
        server.http().post("/x", {}, timeout_s=5, max_retries=4)

    assert len(server.requests) == 1
    assert server.pauses == []


@pytest.mark.unit
def test_provider_error_inside_an_http_200_is_an_error_too() -> None:
    server = Server([json_response({"error": {"code": 502, "message": "upstream"}})] * 2)

    with pytest.raises(OpenRouterError, match="HTTP 502"):
        server.http().post("/x", {}, timeout_s=5, max_retries=1)

    assert len(server.requests) == 2  # 502 is transient: one retry


@pytest.mark.unit
def test_network_errors_are_retried_then_raised() -> None:
    server = Server([httpx2.ReadTimeout("t"), httpx2.ReadTimeout("t")])

    with pytest.raises(OpenRouterError, match="ReadTimeout"):
        server.http().post("/x", {}, timeout_s=5, max_retries=1)

    assert len(server.requests) == 2


@pytest.mark.unit
def test_an_error_page_that_is_not_json_is_a_failure_not_a_crash() -> None:
    server = Server([httpx2.Response(502, text="<html>bad gateway</html>")])

    with pytest.raises(OpenRouterError, match="HTTP 502"):
        server.http().post("/x", {}, timeout_s=5, max_retries=0)
