"""Garbage in OpenRouter answers and in the ledger: no crash, no silent wrong number."""

import json
from pathlib import Path
from typing import Any  # JSON bodies of the API: their shape is the service's, not ours

import pytest

from aicoach.agent.messages import Usage
from aicoach.agent.openrouter_llm import from_response
from aicoach.errors import ExternalServiceError, StorageError
from aicoach.openrouter import count, reported_cost
from aicoach.speech.openrouter_stt import OpenRouterTranscriber, parse_transcription
from aicoach.spend import LEDGER_FILE, SpendLedger
from tests.fixtures.openrouter_synth import Server, chat_body, error_response, json_response


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.5, 0.5), (0, 0.0), (3, 3.0), (-0.1, None), (float("nan"), None),
     (float("inf"), None), ("0.5", None), (True, None), (None, None)],
)  # fmt: skip
def test_a_reported_cost_must_be_finite_and_not_negative(
    value: object, expected: float | None
) -> None:
    assert reported_cost(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [(5, 5), (5.9, 5), (-1, 0), (float("nan"), 0), (float("inf"), 0), ("7", 0), (None, 0)],
)
def test_a_token_count_is_a_non_negative_integer_or_zero(value: object, expected: int) -> None:
    assert count(value) == expected


@pytest.mark.unit
def test_chat_usage_with_a_nan_cost_has_no_cost_so_that_the_table_price_applies() -> None:
    body = chat_body(
        "ок", usage={"prompt_tokens": 10, "completion_tokens": 2, "cost": float("nan")}
    )

    assert from_response(body).usage == Usage(10, 2, None)


@pytest.mark.unit
@pytest.mark.parametrize("usage", ["text", 5, ["x"], {"prompt_tokens": "many"}])
def test_chat_usage_of_the_wrong_shape_is_zero_tokens(usage: Any) -> None:  # noqa: ANN401
    body = chat_body("ок")
    body["usage"] = usage

    assert from_response(body).usage == Usage(0, 0, None)


@pytest.mark.unit
@pytest.mark.parametrize("message", [None, "text", ["x"]])
def test_a_choice_without_a_message_object_is_an_external_service_error(message: Any) -> None:  # noqa: ANN401
    body = {"choices": [{"finish_reason": "stop", "message": message}]}

    with pytest.raises(ExternalServiceError, match="no message"):
        from_response(body)


@pytest.mark.unit
def test_tool_calls_of_the_wrong_shape_are_skipped_or_get_no_id() -> None:
    body = chat_body("ок")
    body["choices"][0]["message"]["tool_calls"] = [
        "junk",
        {"function": "junk"},
        {"id": 7, "function": {"name": "match_info", "arguments": "{}"}},
    ]

    turn = from_response(body)

    assert [(c.name, c.call_id) for c in turn.calls] == [("match_info", None)]


@pytest.mark.unit
def test_transcription_usage_with_a_bad_cost_or_seconds_is_unknown_not_a_crash() -> None:
    body = {"text": "т", "usage": {"seconds": -3, "cost": float("inf"), "input_tokens": "x"}}

    raw = parse_transcription(body, timed=True)

    assert (raw.audio_seconds, raw.cost_usd, raw.input_tokens) == (None, None, 0)


@pytest.mark.unit
@pytest.mark.parametrize("usage", ["text", 5, None])
def test_transcription_usage_of_the_wrong_shape_is_ignored(usage: Any) -> None:  # noqa: ANN401
    raw = parse_transcription({"text": "т", "usage": usage}, timed=True)

    assert raw.cost_usd is None


@pytest.mark.unit
def test_words_and_segments_that_are_not_objects_are_ignored() -> None:
    body = {
        "text": "один",
        "words": ["junk", 5, {"word": "один", "start": 0.0, "end": 1.0}],
        "segments": "junk",
    }

    raw = parse_transcription(body, timed=True)

    assert [w.text for w in raw.words] == ["один"]
    assert parse_transcription({"text": "т", "words": "junk"}, timed=True).words == ()


@pytest.mark.unit
def test_keyterms_are_dropped_when_the_provider_refuses_them() -> None:
    server = Server(
        [error_response(400, "keyterms are not supported"), json_response({"text": "т"})]
    )

    raw = OpenRouterTranscriber(server.http(), "m").transcribe(
        b"a", "audio/mp3", vocabulary=["Ветров"]
    )

    assert "keyterms" in server.body(0)
    assert "keyterms" not in server.body(1)
    assert raw.text == "т"


def entry(**changes: object) -> str:
    row: dict[str, object] = {
        "ts": "2026-11-02T00:00:00+00:00",
        "provider": "openrouter",
        "model": "m",
        "kind": "agent",
        "cost_usd": 1.0,
        "basis": "actual",
    }
    return json.dumps({**row, **changes})


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad",
    [
        {"cost_usd": float("nan")},
        {"cost_usd": float("inf")},
        {"cost_usd": -1.0},
        {"ts": "2026-11-02T00:00:00"},  # no time zone
        {"ts": "yesterday"},
        {"basis": "guess"},
    ],
)
def test_a_ledger_line_with_a_bad_cost_time_or_basis_is_a_storage_error(
    tmp_path: Path, bad: dict[str, object]
) -> None:
    (tmp_path / LEDGER_FILE).write_text(entry(**bad) + "\n", encoding="utf-8")
    ledger = SpendLedger(tmp_path / LEDGER_FILE, 5.0)

    with pytest.raises(StorageError, match="line 1"):
        ledger.month_total()


@pytest.mark.unit
@pytest.mark.parametrize("choices", [5, "x", {"a": 1}, [None], [[]]])
def test_choices_of_the_wrong_type_are_an_external_service_error(choices: Any) -> None:  # noqa: ANN401
    with pytest.raises(ExternalServiceError, match="no choices"):
        from_response({"choices": choices})


@pytest.mark.unit
@pytest.mark.parametrize("calls", [5, "x", {"a": 1}])
def test_tool_calls_that_are_not_a_list_are_ignored(calls: Any) -> None:  # noqa: ANN401
    body = chat_body("ответ")
    body["choices"][0]["message"]["tool_calls"] = calls

    turn = from_response(body)

    assert turn.calls == ()
    assert turn.text == "ответ"
