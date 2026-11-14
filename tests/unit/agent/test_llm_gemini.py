"""The Gemini client: conversion of messages and responses, retries (no network)."""

import httpx2
import pytest
from google.genai import errors, types
from pydantic import SecretStr

from aicoach.agent.llm import (
    BACKOFF_BASE_S,
    EMPTY_TURN_TEXT,
    MAX_RETRIES,
    GeminiClient,
    from_response,
    to_contents,
)
from aicoach.agent.messages import (
    HistoryItem,
    ModelTurn,
    ToolCall,
    ToolResult,
    ToolResults,
    ToolSpec,
    Usage,
    UserMessage,
)
from aicoach.errors import ExternalServiceError

SPEC = ToolSpec("match_info", "обзор", {"type": "object", "properties": {}})


def response(
    *parts: types.Part, prompt: int = 11, output: int = 7, thoughts: int | None = None
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt,
            candidates_token_count=output,
            thoughts_token_count=thoughts,
        ),
    )


class FakeModels:
    """Plays a script of responses or exceptions and records the requests."""

    def __init__(self, script: list[types.GenerateContentResponse | Exception]) -> None:
        self.script = script
        self.requests: list[dict[str, object]] = []

    def generate_content(
        self, *, model: str, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        self.requests.append({"model": model, "contents": contents, "config": config})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def api_error(code: int, status: str = "RESOURCE_EXHAUSTED") -> errors.APIError:
    return errors.APIError(code, {"error": {"status": status, "message": "quota"}})


def client(models: FakeModels, pauses: list[float] | None = None) -> GeminiClient:
    sleeps = pauses if pauses is not None else []
    return GeminiClient(SecretStr("k"), "gemini-test", sleep=sleeps.append, models=models)


@pytest.mark.unit
def test_response_text_calls_and_tokens_are_read_and_thinking_is_dropped() -> None:
    turn = from_response(
        response(
            types.Part(text="размышление", thought=True),
            types.Part(text="Ответ"),
            types.Part(
                function_call=types.FunctionCall(name="query_events", args={"type": "goal"}, id="x")
            ),
            thoughts=3,
        )
    )

    assert turn.text == "Ответ"
    assert turn.calls == (ToolCall("query_events", {"type": "goal"}, "x"),)
    assert turn.usage == Usage(11, 10)  # thinking tokens are output
    assert isinstance(turn.raw, types.Content)


@pytest.mark.unit
def test_response_without_usage_or_parts_is_an_empty_turn() -> None:
    empty = types.GenerateContentResponse(candidates=[types.Candidate()])

    turn = from_response(empty)

    assert (turn.text, turn.calls, turn.usage) == ("", (), Usage())


@pytest.mark.unit
def test_response_without_candidates_is_a_service_error() -> None:
    with pytest.raises(ExternalServiceError, match="no answer"):
        from_response(types.GenerateContentResponse(candidates=[]))


@pytest.mark.unit
def test_history_becomes_contents_and_the_models_own_content_is_kept_unchanged() -> None:
    raw = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(name="match_info", args={}),
                thought_signature=b"sig",
            )
        ],
    )
    history: list[HistoryItem] = [
        UserMessage("Привет"),
        ModelTurn("", (ToolCall("match_info", {}, "a"),), Usage(), raw),
        ToolResults((ToolResult("match_info", {"events_total": 7}, "a"),)),
        ModelTurn("Итог", (), Usage()),
    ]

    contents = to_contents(history)

    assert [c.role for c in contents] == ["user", "model", "user", "model"]
    assert contents[1] is raw
    assert contents[2].parts and contents[2].parts[0].function_response
    assert contents[2].parts[0].function_response.response == {"result": {"events_total": 7}}
    assert contents[3].parts and contents[3].parts[0].text == "Итог"


@pytest.mark.unit
def test_request_disables_automatic_calling_and_declares_the_tools() -> None:
    models = FakeModels([response(types.Part(text="ok"))])

    client(models).generate("система", [UserMessage("?")], [SPEC])

    config = models.requests[0]["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.system_instruction == "система"
    assert config.automatic_function_calling
    assert config.automatic_function_calling.disable is True
    assert config.tool_config is None
    assert config.tools
    tool = config.tools[0]
    assert isinstance(tool, types.Tool)
    assert tool.function_declarations
    declaration = tool.function_declarations[0]
    assert declaration.name == "match_info"
    assert models.requests[0]["model"] == "gemini-test"


@pytest.mark.unit
def test_tools_off_sets_the_none_calling_mode() -> None:
    models = FakeModels([response(types.Part(text="ok"))])

    client(models).generate("s", [UserMessage("?")], [SPEC], allow_tools=False)

    config = models.requests[0]["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.tool_config
    assert config.tool_config.function_calling_config
    assert config.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.NONE


@pytest.mark.unit
@pytest.mark.parametrize("code", [429, 500, 503])
def test_transient_errors_are_retried_with_growing_pauses(code: int) -> None:
    pauses: list[float] = []
    models = FakeModels([api_error(code), api_error(code), response(types.Part(text="ok"))])

    turn = client(models, pauses).generate("s", [UserMessage("?")], [SPEC])

    assert turn.text == "ok"
    assert pauses == [BACKOFF_BASE_S, BACKOFF_BASE_S * 2]


@pytest.mark.unit
def test_network_errors_are_retried_too() -> None:
    pauses: list[float] = []
    models = FakeModels([httpx2.ConnectTimeout("timed out"), response(types.Part(text="ok"))])

    assert client(models, pauses).generate("s", [UserMessage("?")], [SPEC]).text == "ok"
    assert len(pauses) == 1


@pytest.mark.unit
def test_retries_are_limited_and_then_a_typed_error_with_the_cause() -> None:
    pauses: list[float] = []
    models = FakeModels([api_error(429) for _ in range(MAX_RETRIES + 1)])

    with pytest.raises(ExternalServiceError, match="429 RESOURCE_EXHAUSTED") as info:
        client(models, pauses).generate("s", [UserMessage("?")], [SPEC])

    assert len(pauses) == MAX_RETRIES
    assert len(models.requests) == MAX_RETRIES + 1
    assert isinstance(info.value.__cause__, errors.APIError)


@pytest.mark.unit
@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_permanent_errors_are_not_retried(code: int) -> None:
    pauses: list[float] = []
    models = FakeModels([api_error(code, "INVALID_ARGUMENT")])

    with pytest.raises(ExternalServiceError, match=str(code)):
        client(models, pauses).generate("s", [UserMessage("?")], [SPEC])

    assert pauses == []


@pytest.mark.unit
def test_an_empty_turn_goes_back_to_the_api_with_a_placeholder_part() -> None:
    empty = from_response(types.GenerateContentResponse(candidates=[types.Candidate()]))

    (content,) = to_contents([empty])

    assert content.role == "model"
    assert content.parts
    assert content.parts[0].text == EMPTY_TURN_TEXT


@pytest.mark.unit
def test_a_response_stopped_at_the_length_limit_is_marked_truncated() -> None:
    cut = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text="Система")]),
                finish_reason=types.FinishReason.MAX_TOKENS,
            )
        ]
    )

    assert from_response(cut).truncated is True
    assert from_response(response(types.Part(text="ok"))).truncated is False


@pytest.mark.unit
def test_the_real_client_path_keeps_the_key_out_of_errors_logs_and_repr(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    key = "AIzaFakeKeyForTests0123456789abcdefXYZ"  # pragma: allowlist secret
    gemini = GeminiClient(SecretStr(key), "gemini-test", sleep=lambda _s: None)
    # The client is real (it only opens a connection when a request is sent); its requests
    # go to a fake that fails twice with a rate limit and then for good.
    gemini._models = FakeModels(
        [api_error(429), api_error(429), api_error(403, "PERMISSION_DENIED")]
    )

    with pytest.raises(ExternalServiceError) as info:
        gemini.generate("s", [UserMessage("?")], [SPEC])

    captured = capsys.readouterr()
    assert key not in str(info.value)
    assert key not in repr(info.value)
    assert key not in caplog.text
    assert key not in captured.out
    assert key not in captured.err
