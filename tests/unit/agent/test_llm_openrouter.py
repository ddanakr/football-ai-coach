"""The OpenRouter chat client on recorded answers: tools, usage and cost, errors (no network)."""

import json

import pytest

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
from aicoach.agent.openrouter_llm import (
    MAX_RETRIES,
    OpenRouterClient,
    from_response,
    to_messages,
    to_tools,
)
from aicoach.errors import ExternalServiceError
from tests.fixtures.openrouter_synth import Server, chat_body, error_response, json_response

SPEC = ToolSpec(
    "match_info",
    "сводка по матчу",
    {"type": "object", "properties": {"half": {"type": "integer"}}},
)


def client(server: Server) -> OpenRouterClient:
    return OpenRouterClient(server.http(), "google/gemini-test")


@pytest.mark.unit
def test_tools_go_in_the_openai_format() -> None:
    assert to_tools([SPEC]) == [
        {
            "type": "function",
            "function": {
                "name": "match_info",
                "description": "сводка по матчу",
                "parameters": SPEC.parameters,
            },
        }
    ]
    assert to_tools([]) == []


@pytest.mark.unit
def test_history_becomes_system_user_assistant_and_tool_messages() -> None:
    call = ToolCall("match_info", {"half": 1}, "call_a")
    history: list[HistoryItem] = [
        UserMessage("Сколько голов?"),
        ModelTurn("", (call,), Usage()),
        ToolResults((ToolResult("match_info", {"goals": 4}, "call_a"),)),
        ModelTurn("Четыре гола.", (), Usage()),
    ]

    messages = to_messages("система", history)

    assert messages[0] == {"role": "system", "content": "система"}
    assert messages[1] == {"role": "user", "content": "Сколько голов?"}
    assistant = messages[2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["id"] == "call_a"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"half": 1}
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "call_a",
        "content": '{"goals": 4}',
    }
    assert messages[4] == {"role": "assistant", "content": "Четыре гола."}


@pytest.mark.unit
def test_the_raw_assistant_message_is_sent_back_with_its_reasoning_details() -> None:
    raw = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "m", "arguments": "{}"}}
        ],
        "reasoning_details": [{"type": "reasoning.encrypted", "data": "xyz"}],
    }
    turn = ModelTurn("", (ToolCall("m", {}, "c1"),), Usage(), raw)

    assert to_messages("s", [turn])[1] == raw


@pytest.mark.unit
def test_an_empty_turn_is_sent_back_as_a_placeholder_not_as_nothing() -> None:
    message = to_messages("s", [ModelTurn("", (), Usage())])[1]

    assert message["role"] == "assistant"
    assert message["content"]


@pytest.mark.unit
def test_calls_without_ids_get_matching_ids_in_the_call_and_its_result() -> None:
    history: list[HistoryItem] = [
        ModelTurn("", (ToolCall("a", {}), ToolCall("b", {})), Usage()),
        ToolResults((ToolResult("a", {}), ToolResult("b", {}))),
    ]

    messages = to_messages("s", history)

    call_ids = [c["id"] for c in messages[1]["tool_calls"]]
    assert call_ids == [m["tool_call_id"] for m in messages[2:]]
    assert len(set(call_ids)) == 2


@pytest.mark.unit
def test_a_tool_call_is_parsed_with_its_arguments_usage_and_reported_cost() -> None:
    body = chat_body(
        None,
        [("call_1", "match_info", '{"half": 2}')],
        usage={"prompt_tokens": 120, "completion_tokens": 30, "cost": 0.00042},
    )

    turn = from_response(body)

    assert turn.calls == (ToolCall("match_info", {"half": 2}, "call_1"),)
    assert turn.text == ""
    assert turn.usage == Usage(120, 30, 0.00042)
    assert not turn.truncated


@pytest.mark.unit
def test_a_text_answer_cut_by_the_length_limit_is_marked_truncated() -> None:
    assert from_response(chat_body("часть отв", finish="length")).truncated


@pytest.mark.unit
def test_missing_usage_and_cost_are_zero_tokens_and_no_cost() -> None:
    body = chat_body("ок")
    del body["usage"]

    assert from_response(body).usage == Usage(0, 0, None)


@pytest.mark.unit
@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": ["x"]}])
def test_an_answer_without_a_choice_is_an_external_service_error(body: dict[str, object]) -> None:
    with pytest.raises(ExternalServiceError, match="no choices"):
        from_response(body)


@pytest.mark.unit
@pytest.mark.parametrize("arguments", ["{not json", "[1, 2]", '"text"'])
def test_unreadable_tool_arguments_are_an_external_service_error(arguments: str) -> None:
    body = chat_body(None, [("c", "match_info", arguments)])

    with pytest.raises(ExternalServiceError, match="match_info"):
        from_response(body)


@pytest.mark.unit
def test_empty_arguments_mean_no_arguments() -> None:
    turn = from_response(chat_body(None, [("c", "match_info", "")]))

    assert turn.calls[0].args == {}


@pytest.mark.unit
def test_generate_sends_the_model_messages_and_tools_and_returns_the_turn() -> None:
    server = Server([json_response(chat_body("Ответ"))])

    turn = client(server).generate("сист", [UserMessage("в?")], [SPEC])

    body = server.body()
    assert body["model"] == "google/gemini-test"
    assert body["tool_choice"] == "auto"
    assert body["tools"][0]["function"]["name"] == "match_info"
    assert body["messages"][0]["content"] == "сист"
    assert turn.text == "Ответ"


@pytest.mark.unit
def test_generate_without_tools_allowed_sets_tool_choice_none_but_keeps_the_tools() -> None:
    server = Server([json_response(chat_body("Ответ"))])

    client(server).generate("s", [UserMessage("в?")], [SPEC], allow_tools=False)

    assert server.body()["tool_choice"] == "none"
    assert server.body()["tools"]


@pytest.mark.unit
def test_generate_retries_a_rate_limit_and_then_succeeds() -> None:
    server = Server([error_response(429), json_response(chat_body("Ответ"))])

    turn = client(server).generate("s", [UserMessage("в?")], [])

    assert turn.text == "Ответ"
    assert len(server.requests) == 2
    assert server.pauses == [4.0]


@pytest.mark.unit
def test_generate_gives_up_after_the_retries_with_an_external_service_error() -> None:
    server = Server([error_response(429)] * (MAX_RETRIES + 1))

    with pytest.raises(ExternalServiceError, match="429"):
        client(server).generate("s", [UserMessage("в?")], [])

    assert len(server.requests) == MAX_RETRIES + 1


@pytest.mark.unit
def test_generate_does_not_retry_no_credits() -> None:
    server = Server([error_response(402, "insufficient credits")])

    with pytest.raises(ExternalServiceError, match="402"):
        client(server).generate("s", [UserMessage("в?")], [])

    assert len(server.requests) == 1
