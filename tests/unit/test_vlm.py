"""The video model through OpenRouter: the request with the clip inline, the answer, the errors."""

import base64

import pytest

from aicoach.errors import ExternalServiceError
from aicoach.openrouter import OpenRouterError
from aicoach.vlm import MAX_OUTPUT_TOKENS, OpenRouterVideoModel, build_payload
from tests.fixtures.openrouter_synth import Server, chat_body, error_response, json_response

CLIP = b"\x00\x00\x00\x18ftypmp42 not really a video"


@pytest.mark.unit
def test_payload_has_the_prompt_and_the_clip_as_a_base64_data_url() -> None:
    payload = build_payload("google/x", "система", "опиши", CLIP)

    system, user = payload["messages"]
    assert system == {"role": "system", "content": "система"}
    text_part, video_part = user["content"]
    assert text_part == {"type": "text", "text": "опиши"}
    url = video_part["video_url"]["url"]
    assert video_part["type"] == "video_url"
    assert url == "data:video/mp4;base64," + base64.b64encode(CLIP).decode("ascii")
    assert payload["model"] == "google/x"
    assert payload["max_tokens"] == MAX_OUTPUT_TOKENS


@pytest.mark.unit
def test_describe_returns_the_text_the_usage_and_the_reported_cost() -> None:
    server = Server([json_response(chat_body("Игрок бьёт по воротам."))])
    model = OpenRouterVideoModel(server.http(), "google/x")

    result = model.describe("s", "p", CLIP, video_seconds=20.0)

    assert result.text == "Игрок бьёт по воротам."
    assert (result.usage.input_tokens, result.usage.output_tokens) == (100, 20)
    assert result.usage.cost_usd == 0.0001
    assert not result.truncated
    assert model.model == "google/x"
    assert server.requests[0].url.path.endswith("/chat/completions")


@pytest.mark.unit
def test_a_cut_off_answer_is_marked_truncated() -> None:
    server = Server([json_response(chat_body("Игрок бьёт", finish="length"))])

    result = OpenRouterVideoModel(server.http(), "m").describe("s", "p", CLIP)

    assert result.truncated


@pytest.mark.unit
def test_transient_errors_are_retried_with_pauses_then_the_answer_is_used() -> None:
    server = Server([error_response(429), error_response(503), json_response(chat_body("ок"))])

    result = OpenRouterVideoModel(server.http(), "m").describe("s", "p", CLIP)

    assert result.text == "ок"
    assert len(server.requests) == 3
    assert server.pauses == [4.0, 8.0]


@pytest.mark.unit
def test_a_permanent_error_is_raised_without_a_retry() -> None:
    server = Server([error_response(402, "no credits")])

    with pytest.raises(OpenRouterError, match="402"):
        OpenRouterVideoModel(server.http(), "m").describe("s", "p", CLIP)

    assert len(server.requests) == 1


@pytest.mark.unit
@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"message": None}]}])
def test_an_answer_without_a_message_is_an_external_service_error(body: dict[str, object]) -> None:
    server = Server([json_response(body)])

    with pytest.raises(ExternalServiceError):
        OpenRouterVideoModel(server.http(), "m").describe("s", "p", CLIP)
