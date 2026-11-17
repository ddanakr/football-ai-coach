"""The `describe_episode` tool: offered only with a describer, marked "not verified", and its
text never makes a name or a number "seen" (D-073)."""

from pathlib import Path

import pytest

from aicoach.agent.loop import Agent
from aicoach.agent.tools import EPISODE_NOTE, ToolBox
from aicoach.agent.validator import find_violations
from aicoach.episode import EpisodeDescription
from aicoach.errors import BudgetExceededError, ExportError, ExternalServiceError
from tests.fixtures.agent_synth import FakeLLM, make_context, turn

# What a video model can write although it was told not to: a surname, a shirt number, a score,
# a minute. None of it comes from the tools of this match.
RISKY = "Игрок Выдумов с номером 77 бьёт на 88-й минуте, счёт 5:4. Вратарь отбивает мяч."


def description(text: str = RISKY) -> EpisodeDescription:
    return EpisodeDescription.model_validate(
        {
            "event_id": "p:2",
            "type": "goal",
            "t_v": 1500.0,
            "model": "test/vlm-1",
            "before_s": 8.0,
            "after_s": 12.0,
            "text": text,
            "created_at": "2026-11-18T10:00:00+00:00",
        }
    )


class Describer:
    """A scripted `EpisodeDescriber`: records the event ids, returns or raises."""

    def __init__(self, result: EpisodeDescription | Exception | None = None) -> None:
        self.result = result or description()
        self.calls: list[str] = []

    def __call__(self, event_id: str) -> EpisodeDescription:
        self.calls.append(event_id)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def box(describer: Describer | None) -> ToolBox:
    return ToolBox(make_context(), episode_describer=describer)


@pytest.mark.unit
def test_the_tool_is_offered_only_when_there_is_a_describer() -> None:
    with_it = [s.name for s in box(Describer()).specs()]
    without = [s.name for s in box(None).specs()]

    assert "describe_episode" in with_it
    assert "describe_episode" not in without
    assert "describe_episode" not in [
        s.name for s in ToolBox(make_context(), lambda *_: Path()).specs()
    ]


@pytest.mark.unit
def test_the_result_is_marked_as_unverified_data_and_carries_the_text() -> None:
    describer = Describer()

    result = box(describer).call("describe_episode", {"event_id": "p:2"})

    assert describer.calls == ["p:2"]
    assert result == {
        "event_id": "p:2",
        "status": "described_from_clip_unverified",
        "description": RISKY,
        "note": EPISODE_NOTE,
    }
    assert "не проверено" in EPISODE_NOTE
    assert "model" not in str(result)  # the model name has digits that must not become facts


@pytest.mark.unit
def test_numbers_and_names_of_the_description_are_not_confirmed_data() -> None:
    tools = box(Describer())
    tools.call("describe_episode", {"event_id": "p:2"})

    assert not {77.0, 88.0, 5.0, 4.0} & tools.seen_numbers
    assert not any("Выдумов" in name for name in tools.allowed_names)
    assert tools.seen_event_ids == set()


@pytest.mark.unit
@pytest.mark.parametrize(
    "answer",
    [
        "По клипу, не проверено: Выдумов бьёт по воротам [E:p:2].",  # a surname of the description
        "По клипу, не проверено: бьёт игрок №77 [E:p:2].",  # a number of the description
        "По клипу, не проверено: гол на 88-й минуте [E:p:2].",
    ],
)
def test_an_answer_that_repeats_a_name_or_a_number_of_the_description_is_rejected(
    answer: str,
) -> None:
    tools = box(Describer())
    tools.call("query_events", {"type": "goal"})  # the goal p:2 and its real facts
    tools.call("describe_episode", {"event_id": "p:2"})

    violations = find_violations(
        answer,
        seen_ids=tools.seen_event_ids,
        existing_ids=tools.ctx.ref_ids(),
        allowed_names=tools.allowed_names,
        allowed_numbers=tools.seen_numbers,
    )

    assert violations


@pytest.mark.unit
def test_a_name_and_a_minute_from_the_real_tools_are_still_allowed_next_to_the_description() -> (
    None
):
    tools = box(Describer())
    tools.call("query_events", {"type": "goal"})
    tools.call("describe_episode", {"event_id": "p:2"})

    violations = find_violations(
        "Гол Тестов на 23-й минуте [E:p:2]. По клипу, не проверено: вратарь отбивает мяч.",
        seen_ids=tools.seen_event_ids,
        existing_ids=tools.ctx.ref_ids(),
        allowed_names=tools.allowed_names,
        allowed_numbers=tools.seen_numbers,
    )

    assert violations == []


@pytest.mark.unit
def test_an_unknown_event_gives_an_error_and_no_request() -> None:
    describer = Describer()

    result = box(describer).call("describe_episode", {"event_id": "p:999"})

    assert "нет в данных матча" in result["error"]
    assert describer.calls == []


@pytest.mark.unit
@pytest.mark.parametrize("failure", [ExportError("ffmpeg gone"), ExternalServiceError("HTTP 500")])
def test_a_clip_or_service_failure_is_an_error_for_the_model_without_details(
    failure: Exception,
) -> None:
    result = box(Describer(failure)).call("describe_episode", {"event_id": "p:2"})

    assert result == {"error": "описание эпизода не получено"}


@pytest.mark.unit
def test_a_budget_refusal_is_not_swallowed() -> None:
    with pytest.raises(BudgetExceededError):
        box(Describer(BudgetExceededError("budget"))).call("describe_episode", {"event_id": "p:2"})


@pytest.mark.unit
@pytest.mark.parametrize("args", [{}, {"event_id": 5}, {"event_id": "p:2", "x": 1}])
def test_bad_arguments_are_an_error_result(args: dict[str, object]) -> None:
    describer = Describer()

    result = box(describer).call("describe_episode", args)

    assert result["error"].startswith("неверные аргументы")
    assert describer.calls == []


@pytest.mark.unit
def test_a_call_to_a_tool_that_is_not_offered_is_an_unknown_tool() -> None:
    assert (
        "неизвестный инструмент" in box(None).call("describe_episode", {"event_id": "p:2"})["error"]
    )


@pytest.mark.unit
def test_the_agent_relays_the_description_with_the_mark_and_real_facts() -> None:
    llm = FakeLLM(
        [
            turn("", ("query_events", {"type": "goal"})),
            turn("", ("describe_episode", {"event_id": "p:2"})),
            turn("Гол Тестова [E:p:2]. По клипу, не проверено: вратарь отбивает мяч."),
        ]
    )

    answer = Agent(llm, "system").ask("Опиши первый гол", box(Describer()))

    assert answer.outcome == "first_try"
    assert [c["name"] for c in answer.tool_calls] == ["query_events", "describe_episode"]


@pytest.mark.unit
def test_the_agent_refuses_an_answer_with_the_surname_of_the_description_after_one_retry() -> None:
    bad = "Гол [E:p:2]. По клипу, не проверено: Выдумов бьёт по воротам."
    llm = FakeLLM(
        [
            turn("", ("query_events", {"type": "goal"})),
            turn("", ("describe_episode", {"event_id": "p:2"})),
            turn(bad),
            turn(bad),
        ]
    )

    answer = Agent(llm, "system").ask("Опиши первый гол", box(Describer()))

    assert answer.outcome == "refused"
    assert "unknown_name" in answer.violations


MARK = "Описание эпизода: по клипу, не проверено."


def ask_with_description(final: str, describer: Describer | None = None) -> tuple[str, bool]:
    llm = FakeLLM(
        [
            turn("", ("query_events", {"type": "goal"})),
            turn("", ("describe_episode", {"event_id": "p:2"})),
            turn(final),
        ]
    )
    answer = Agent(llm, "system").ask("Опиши первый гол", box(describer or Describer()))
    return answer.text, answer.outcome == "first_try"


@pytest.mark.unit
def test_the_code_adds_the_mark_when_the_model_did_not_write_it() -> None:
    text, passed = ask_with_description("Гол Тестова [E:p:2]. Вратарь отбивает мяч.")

    assert passed
    assert text.endswith("\n\n" + MARK)


@pytest.mark.unit
@pytest.mark.parametrize(
    "written",
    [
        "По клипу, не проверено: вратарь отбивает мяч.",
        "Вратарь отбивает мяч (по клипу, не проверено).",
    ],
)
def test_the_mark_is_not_repeated_when_the_model_wrote_it(written: str) -> None:
    text, passed = ask_with_description(f"Гол Тестова [E:p:2]. {written}")

    assert passed
    assert MARK not in text


@pytest.mark.unit
def test_no_mark_without_a_description() -> None:
    llm = FakeLLM([turn("", ("query_events", {"type": "goal"})), turn("Два гола [E:p:2].")])

    answer = Agent(llm, "system").ask("Сколько голов?", box(Describer()))

    assert "по клипу" not in answer.text


@pytest.mark.unit
def test_no_mark_when_the_description_was_not_received() -> None:
    describer = Describer(ExternalServiceError("HTTP 500"))
    tools = box(describer)

    tools.call("describe_episode", {"event_id": "p:2"})

    assert not tools.episode_described
    text, _ = ask_with_description("Описание не получено [E:p:2].", describer)
    assert "по клипу" not in text


@pytest.mark.unit
def test_a_refused_answer_gets_no_mark() -> None:
    bad = "Гол [E:p:2], бьёт Выдумов по воротам."
    llm = FakeLLM(
        [
            turn("", ("query_events", {"type": "goal"})),
            turn("", ("describe_episode", {"event_id": "p:2"})),
            turn(bad),
            turn(bad),
        ]
    )

    answer = Agent(llm, "system").ask("Опиши первый гол", box(Describer()))

    assert answer.outcome == "refused"
    assert "по клипу" not in answer.text


@pytest.mark.unit
def test_the_mark_is_added_after_a_retry_too() -> None:
    llm = FakeLLM(
        [
            turn("", ("query_events", {"type": "goal"})),
            turn("", ("describe_episode", {"event_id": "p:2"})),
            turn("Гол [E:p:2], бьёт Выдумов по воротам."),
            turn("Гол Тестова [E:p:2]. Вратарь отбивает мяч."),
        ]
    )

    answer = Agent(llm, "system").ask("Опиши первый гол", box(Describer()))

    assert answer.outcome == "retry"
    assert answer.text.endswith(MARK)


@pytest.mark.unit
def test_the_two_phrases_in_different_places_are_not_the_mark() -> None:
    text, passed = ask_with_description(
        "Гол Тестова [E:p:2]. Это видно по клипу. Счёт пока не проверено мной, но вратарь отбил."
    )

    assert passed
    assert text.endswith(MARK)
