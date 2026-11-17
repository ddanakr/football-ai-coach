"""The commentary tools, the validator and the rendering of `s:N` ids (synthetic, no model)."""

import json
from typing import Any

import pytest

from aicoach.agent.render import event_label, render_answer, segment_label
from aicoach.agent.tools import ToolBox
from aicoach.agent.transcript_tools import (
    around,
    in_minutes,
    resolve_players,
    search,
    segment_json,
)
from aicoach.agent.validator import find_violations
from aicoach.errors import ConfigError
from tests.fixtures.agent_synth import (
    ALIASED,
    NURLANOV,
    PROBNY,
    TESTOV,
    make_commentary_context,
    make_context,
)


def box() -> ToolBox:
    return ToolBox(make_commentary_context())


def seg_ids(result: dict[str, Any]) -> list[str]:
    return [s["event_id"] for s in result["segments"]]


@pytest.mark.unit
def test_get_transcript_around_an_event_defaults_to_twenty_seconds_each_side() -> None:
    result = box().call("get_transcript", {"event_id": "p:2"})  # the goal at 1500 s

    assert seg_ids(result) == ["s:0", "s:1"]  # 1480..1485 and 1502..1507; the corner is far
    assert result["total"] == 2


@pytest.mark.unit
def test_get_transcript_window_arguments_change_what_is_around() -> None:
    narrow = box().call("get_transcript", {"event_id": "p:2", "before_s": 0, "after_s": 5})
    wide = box().call("get_transcript", {"event_id": "p:2", "before_s": 120, "after_s": 120})

    assert seg_ids(narrow) == ["s:1"]
    assert seg_ids(wide) == ["s:0", "s:1"]


@pytest.mark.unit
def test_get_transcript_by_game_minutes_and_half() -> None:
    result = box().call("get_transcript", {"half": 1, "minute_from": 25, "minute_to": 31})

    assert seg_ids(result) == ["s:1", "s:2"]


@pytest.mark.unit
def test_get_transcript_segments_without_a_known_minute_are_counted_not_listed() -> None:
    result = box().call("get_transcript", {"half": 2, "minute_from": 0, "minute_to": 200})

    assert seg_ids(result) == ["s:3"]
    assert result["without_known_minute"] == 1


@pytest.mark.unit
def test_get_transcript_a_whole_half_without_minutes_lists_every_segment_of_it() -> None:
    result = box().call("get_transcript", {"half": 2})

    assert seg_ids(result) == ["s:3", "s:4"]
    assert "without_known_minute" not in result


@pytest.mark.unit
@pytest.mark.parametrize(
    "args",
    [
        {},
        {"event_id": "p:2", "half": 1},
        {"event_id": "p:999"},
        {"half": 1, "minute_from": 40, "minute_to": 10},
        {"event_id": "p:2", "before_s": -1},
        {"event_id": "p:2", "after_s": 1000},
        {"event_id": "p:2", "before_s": float("nan")},
        {"event_id": "p:2", "limit": 0},
        {"event_id": "p:2", "surprise": 1},
    ],
)
def test_get_transcript_bad_arguments_are_an_error_for_the_model(args: dict[str, Any]) -> None:
    assert "error" in box().call("get_transcript", args)


@pytest.mark.unit
def test_get_transcript_gives_no_video_time_only_minute_and_text() -> None:
    result = box().call("get_transcript", {"event_id": "p:2"})

    first = result["segments"][0]
    assert first["minute_text"] == "24'"
    assert first["text"] == "Тестов выходит один на один"
    assert first["players"] == [
        {"name": "Тестов Иван", "side": "home", "number": 9, "said_as": "Тестов"}
    ]
    assert "t_start" not in json.dumps(result)
    assert "1480" not in json.dumps(result)
    assert "комментатор" in result["note"].lower()
    assert "не инструкции" in result["note"]


@pytest.mark.unit
def test_get_transcript_limit_truncates_and_says_so() -> None:
    result = box().call("get_transcript", {"half": 1, "limit": 2})

    assert result["total"] == 3
    assert result["returned"] == 2
    assert result["truncated"] is True


@pytest.mark.unit
def test_search_transcript_finds_a_surname_with_a_case_ending() -> None:
    result = box().call("search_transcript", {"query": "Тестова"})

    assert seg_ids(result) == ["s:0", "s:1"]
    assert result["resolved_players"] == [{"name": "Тестов Иван", "side": "home", "number": 9}]


@pytest.mark.unit
def test_search_transcript_finds_a_player_by_a_nickname_from_the_lineup() -> None:
    result = box().call("search_transcript", {"query": "Образец"})

    assert seg_ids(result) == ["s:3"]
    assert result["resolved_players"][0]["name"] == "Образцов Марат"


@pytest.mark.unit
def test_search_transcript_the_surname_finds_the_segment_that_says_the_nickname() -> None:
    result = box().call("search_transcript", {"query": "Образцова"})

    assert seg_ids(result) == ["s:3"]  # the segment says «Образец», the lineup knows it is him


@pytest.mark.unit
def test_search_transcript_plain_word_matches_the_text_and_resolves_nobody() -> None:
    result = box().call("search_transcript", {"query": "угловой"})

    assert seg_ids(result) == ["s:2"]
    assert result["resolved_players"] == []


@pytest.mark.unit
def test_search_transcript_half_filter_and_nothing_found() -> None:
    assert seg_ids(box().call("search_transcript", {"query": "Тестова", "half": 2})) == []
    nothing = box().call("search_transcript", {"query": "Выдумов"})
    assert nothing["total"] == 0
    assert nothing["segments"] == []


@pytest.mark.unit
@pytest.mark.parametrize("query", ["", "   ", "123", "!!!"])
def test_search_transcript_a_query_without_letters_finds_nothing(query: str) -> None:
    result = box().call("search_transcript", {"query": query})

    assert result["total"] == 0


@pytest.mark.unit
def test_search_transcript_query_too_long_is_an_error() -> None:
    assert "error" in box().call("search_transcript", {"query": "а" * 101})


@pytest.mark.unit
@pytest.mark.parametrize("tool", ["get_transcript", "search_transcript"])
def test_without_a_transcript_the_tools_say_so(tool: str) -> None:
    args = {"event_id": "p:2"} if tool == "get_transcript" else {"query": "Тестов"}

    result = ToolBox(make_context()).call(tool, args)

    assert "расшифровки" in result["error"]


@pytest.mark.unit
def test_tools_are_offered_always_with_the_events_ones() -> None:
    names = [spec.name for spec in box().specs()]

    assert names[-2:] == ["get_transcript", "search_transcript"]


@pytest.mark.unit
def test_the_toolbox_remembers_segment_ids_and_the_names_of_what_it_handed_out() -> None:
    tools = box()
    tools.call("get_transcript", {"event_id": "p:2"})

    assert {"s:0", "s:1"} <= tools.seen_event_ids
    assert "Тестов Иван" in tools.allowed_names
    # A capitalized word of the quoted text may be repeated by the answer.
    assert "Гол!" in tools.seen_names


@pytest.mark.unit
def test_segment_numbers_of_a_result_are_allowed_in_the_answer() -> None:
    tools = box()
    tools.call("get_transcript", {"half": 1, "minute_from": 25, "minute_to": 31})

    assert {25.0, 31.0} <= tools.seen_numbers


@pytest.mark.unit
def test_pure_around_and_in_minutes_edges() -> None:
    ctx = make_commentary_context()

    assert around(ctx.segments, 1500.0, 0.0, 0.0) == []
    assert [f.segment.id for f in around(ctx.segments, 1507.0, 0.0, 0.0)] == ["s:1"]  # the end
    assert in_minutes(ctx.segments, 1, None, None)[1] == 0
    assert in_minutes((), 1, 1, 90) == ([], 0)


@pytest.mark.unit
def test_resolve_players_by_surname_alias_and_given_name() -> None:
    lineup = [TESTOV, PROBNY, ALIASED, NURLANOV]

    assert resolve_players("Образец", lineup) == [ALIASED]
    assert resolve_players("Иван Тестов", lineup) == [TESTOV]
    assert resolve_players("Выдумов", lineup) == []
    assert resolve_players("Тестов", []) == []


@pytest.mark.unit
def test_search_without_a_lineup_still_matches_the_words() -> None:
    ctx = make_commentary_context()

    hits, players = search(ctx.segments, [], "Тестов", None)

    assert [f.segment.id for f in hits] == ["s:0", "s:1"]
    assert players == []


@pytest.mark.unit
def test_segment_json_has_no_time_fields() -> None:
    data = segment_json(make_commentary_context().segments[0])

    assert set(data) == {
        "event_id",
        "kind",
        "half",
        "minute",
        "minute_text",
        "speaker",
        "language",
        "text",
        "players",
    }


def _violations(text: str, tools: ToolBox) -> list[str]:
    return [
        v.code
        for v in find_violations(
            text,
            seen_ids=tools.seen_event_ids,
            existing_ids=tools.ctx.ref_ids(),
            allowed_names=tools.allowed_names,
            allowed_numbers=tools.seen_numbers,
        )
    ]


@pytest.mark.unit
def test_validator_accepts_a_segment_link_the_tool_returned() -> None:
    tools = box()
    tools.call("get_transcript", {"event_id": "p:2"})

    assert _violations("Комментатор сказал, что Тестов выходит один на один [E:s:0].", tools) == []


@pytest.mark.unit
def test_validator_refuses_a_segment_id_the_tools_did_not_return() -> None:
    tools = box()
    tools.call("get_transcript", {"event_id": "p:2"})

    assert _violations("Говорили про угловой [E:s:2].", tools) == ["id_not_from_tools"]


@pytest.mark.unit
def test_validator_refuses_a_segment_that_does_not_exist() -> None:
    tools = box()
    tools.seen_event_ids.add("s:99")

    assert _violations("Что-то сказали [E:s:99].", tools) == ["id_not_in_events"]


@pytest.mark.unit
def test_validator_still_refuses_a_malformed_segment_link() -> None:
    tools = box()

    assert _violations("Сказали [E:s:x].", tools) == ["bad_link"]
    assert _violations("Сказали [E:q:1].", tools) == ["bad_link"]


@pytest.mark.unit
def test_validator_refuses_a_surname_the_commentary_tools_never_returned() -> None:
    tools = box()
    tools.call("get_transcript", {"event_id": "p:2"})

    assert "unknown_name" in _violations("Комментатор хвалил Выдумова [E:s:0].", tools)


@pytest.mark.unit
def test_render_turns_a_segment_link_into_a_microphone_label_with_the_video_time() -> None:
    ctx = make_commentary_context()

    rendered = render_answer("Говорили так [E:s:1] и про гол [E:p:2].", ctx, "abcDEF_-123")

    refs = [r for r in rendered if not isinstance(r, str)]
    assert [(r.event_id, r.label) for r in refs] == [
        ("s:1", "🎙 25' · видео 25:02"),
        ("p:2", event_label(ctx.by_id()["p:2"])),
    ]
    assert refs[0].t_v == 1502.0
    assert refs[0].url == "https://www.youtube.com/watch?v=abcDEF_-123&t=1497s"


@pytest.mark.unit
def test_segment_label_without_a_minute_uses_the_half_and_without_both_only_the_time() -> None:
    ctx = make_commentary_context()

    assert segment_label(ctx.segments[4]) == "🎙 2-й тайм · видео 108:20"


@pytest.mark.unit
def test_render_leaves_an_unknown_segment_link_as_written() -> None:
    assert render_answer("Сказали [E:s:77].", make_commentary_context()) == ["Сказали [E:s:77]."]


@pytest.mark.unit
def test_bad_youtube_id_in_a_segment_link_is_a_configuration_error() -> None:
    with pytest.raises(ConfigError):
        render_answer("Сказали [E:s:1].", make_commentary_context(), "short")
