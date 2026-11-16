"""The agent tools over a synthetic match (no model, no files)."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from aicoach.agent.context import MatchContext
from aicoach.agent.tools import ToolBox
from aicoach.domain.events import EventType
from aicoach.errors import ExportError
from aicoach.plate_names.lineup import Player
from tests.fixtures.agent_synth import AWAY, HOME, make_context, make_event, read


def box(*, lineup: bool = True, clip_maker: Any = None) -> ToolBox:
    return ToolBox(make_context(lineup=lineup), clip_maker)


def ids(result: dict[str, Any]) -> list[str]:
    return [e["event_id"] for e in result["events"]]


@pytest.mark.unit
def test_match_info_gives_teams_counts_and_half_minutes_but_no_video_time() -> None:
    result = box().call("match_info", {})

    assert result["teams"] == {"home": HOME, "away": AWAY}
    assert result["events_total"] == 7
    assert result["events_by_type"] == {"corner": 1, "foul": 1, "goal": 2, "shot": 2, "subst": 1}
    assert result["events_by_half"] == {"1": 4, "2": 3}
    assert result["halves"]["1"] == {"first_minute": 1, "last_minute": 47}
    assert "t_v" not in str(result)


@pytest.mark.unit
def test_match_info_without_lineup_uses_home_and_away_words() -> None:
    result = box(lineup=False).call("match_info", {})

    assert result["teams"] == {"home": "хозяева", "away": "гости"}
    assert result["lineup_available"] is False


@pytest.mark.unit
def test_match_info_with_arguments_is_an_error_for_the_model() -> None:
    assert "error" in box().call("match_info", {"half": 1})


@pytest.mark.unit
def test_query_events_filters_by_type_and_half() -> None:
    result = box().call("query_events", {"type": "goal", "half": 2})

    assert ids(result) == ["p:6"]
    assert result["total"] == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"minute_from": 20, "minute_to": 30}, ["p:2", "p:3"]),
        ({"minute_from": 46}, ["p:4", "p:5", "p:6"]),
        ({"minute_to": 5}, ["p:1"]),
        ({"minute_from": 23, "minute_to": 23}, ["p:2"]),
        ({"minute_from": 90}, []),
    ],
)
def test_query_events_filters_by_game_minute_in_code(
    args: dict[str, int], expected: list[str]
) -> None:
    result = box().call("query_events", args)

    assert ids(result) == expected
    assert result["total"] == len(expected)


@pytest.mark.unit
def test_query_events_minute_filter_reports_events_without_a_known_minute() -> None:
    result = box().call("query_events", {"type": "shot", "minute_from": 0})

    assert ids(result) == ["p:1"]
    assert result["without_known_minute"] == 1


@pytest.mark.unit
def test_query_events_minute_range_reversed_is_an_error() -> None:
    assert "error" in box().call("query_events", {"minute_from": 50, "minute_to": 10})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("side", "expected"),
    [("home", ["p:1", "p:2"]), ("away", ["p:3", "p:5", "p:6"]), ("unknown", ["p:4", "p:7"])],
)
def test_query_events_filters_by_side_and_unknown_team_is_findable(
    side: str, expected: list[str]
) -> None:
    assert ids(box().call("query_events", {"side": side})) == expected


@pytest.mark.unit
def test_query_events_filters_by_player_with_a_declined_surname() -> None:
    assert ids(box().call("query_events", {"player": "Тестова"})) == ["p:2"]


@pytest.mark.unit
def test_query_events_limit_truncates_and_says_so() -> None:
    result = box().call("query_events", {"limit": 3})

    assert result["total"] == 7
    assert result["returned"] == 3
    assert result["truncated"] is True


@pytest.mark.unit
def test_query_events_nothing_found_is_an_empty_result_not_an_error() -> None:
    result = box().call("query_events", {"type": "goal", "half": 1, "side": "away"})

    assert result["total"] == 0
    assert result["events"] == []
    assert "error" not in result


@pytest.mark.unit
@pytest.mark.parametrize(
    "args",
    [
        {"type": "penalty"},
        {"half": 3},
        {"limit": 0},
        {"limit": 1000},
        {"minute_from": -1},
        {"x": 1},
    ],
)
def test_query_events_wrong_arguments_return_an_error_for_the_model(args: dict[str, Any]) -> None:
    result = box().call("query_events", args)

    assert result["error"].startswith("неверные аргументы")


@pytest.mark.unit
def test_event_json_has_no_video_time_and_says_when_the_team_is_unknown() -> None:
    result = box().call("get_event", {"event_id": "p:4"})["event"]

    assert result["team"] is None
    assert result["team_name"] is None
    assert result["minute_text"] == "45+1'"
    assert not {"t_v", "video_time", "time"} & result.keys()
    assert result["players"] == [
        {"role": "player", "name": "Образцов Марат", "checked_against_lineup": True}
    ]


@pytest.mark.unit
def test_event_json_counts_unrecognized_names_and_marks_unchecked_ones() -> None:
    sub = box().call("get_event", {"event_id": "p:5"})["event"]
    goal = box().call("get_event", {"event_id": "p:6"})["event"]

    assert sub["unrecognized_names"] == 1
    assert [p["name"] for p in sub["players"]] == ["Нұрланов Ұлан"]
    assert goal["players"][0]["checked_against_lineup"] is False


@pytest.mark.unit
def test_get_event_unknown_id_is_an_error_and_is_not_remembered() -> None:
    tools = box()

    result = tools.call("get_event", {"event_id": "p:99"})

    assert "error" in result
    assert tools.seen_event_ids == set()


@pytest.mark.unit
def test_toolbox_remembers_event_ids_and_names_it_handed_out() -> None:
    tools = box()
    tools.call("query_events", {"type": "goal"})

    assert tools.seen_event_ids == {"p:2", "p:6"}
    assert {"Тестов Иван", "Ұлан Нұрланов"} <= tools.seen_names
    assert {HOME, AWAY} <= tools.allowed_names
    assert "Образцов Марат" not in tools.allowed_names


@pytest.mark.unit
def test_find_player_by_surname_returns_the_player_and_events_of_his_side() -> None:
    result = box().call("find_player", {"query": "нурланов"})

    (player,) = result["players"]
    assert player["name"] == "Нұрланов Ұлан"
    assert player["number"] == 11
    assert player["team_name"] == AWAY
    assert [e["event_id"] for e in player["events"]] == ["p:5", "p:6"]


@pytest.mark.unit
def test_find_player_by_number_and_side() -> None:
    result = box().call("find_player", {"number": 9, "side": "home"})

    assert [p["name"] for p in result["players"]] == ["Тестов Иван"]
    assert result["players"][0]["events"][0]["event_id"] == "p:2"


@pytest.mark.unit
def test_find_player_does_not_take_events_of_the_other_side() -> None:
    result = box().call("find_player", {"query": "Тестов", "side": "away"})

    assert result["players"] == []
    assert result["note"] == "игрок не найден"


@pytest.mark.unit
def test_find_player_unknown_name_is_not_found_and_nothing_is_remembered() -> None:
    tools = box()

    result = tools.call("find_player", {"query": "Несуществующий"})

    assert result["players"] == []
    assert tools.seen_names == set()


@pytest.mark.unit
def test_find_player_needs_a_query_or_a_number() -> None:
    assert "error" in box().call("find_player", {})


@pytest.mark.unit
def test_find_player_without_lineup_searches_the_plate_names() -> None:
    result = box(lineup=False).call("find_player", {"query": "Тестов"})

    (player,) = result["players"]
    assert player["name"] == "Тестов Иван"
    assert player["from_plates_only"] is True
    assert [e["event_id"] for e in player["events"]] == ["p:2"]


@pytest.mark.unit
def test_find_player_without_lineup_cannot_search_by_number() -> None:
    assert "состава" in box(lineup=False).call("find_player", {"number": 9})["error"]


@pytest.mark.unit
def test_unknown_tool_is_an_error_for_the_model() -> None:
    assert "error" in box().call("delete_everything", {})


@pytest.mark.unit
def test_make_clip_is_offered_only_with_a_clip_maker() -> None:
    assert "make_clip" not in [s.name for s in box().specs()]
    assert "make_clip" in [s.name for s in box(clip_maker=lambda *_: Path("c.mp4")).specs()]
    assert "error" in box().call("make_clip", {"event_id": "p:2"})


@pytest.mark.unit
def test_make_clip_calls_the_maker_with_default_window_and_records_the_clip(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, float, float]] = []

    def maker(event_id: str, before: float, after: float) -> Path:
        calls.append((event_id, before, after))
        return tmp_path / "clip.mp4"

    tools = box(clip_maker=maker)

    result = tools.call("make_clip", {"event_id": "p:2"})

    assert calls == [("p:2", 8.0, 12.0)]
    assert result["status"] == "clip_ready"
    assert [(c.event_id, c.path) for c in tools.clips] == [("p:2", tmp_path / "clip.mp4")]


@pytest.mark.unit
def test_make_clip_export_failure_is_an_error_for_the_model_not_a_crash() -> None:
    def maker(_event_id: str, _before: float, _after: float) -> Path:
        msg = "ffmpeg is missing"
        raise ExportError(msg)

    result = box(clip_maker=maker).call("make_clip", {"event_id": "p:2"})

    # The details (paths, FFmpeg output) go to the log, not to the model.
    assert result == {"error": "клип не сделан"}


@pytest.mark.unit
@pytest.mark.parametrize("args", [{"event_id": "p:99"}, {"event_id": "p:2", "before_s": 600}])
def test_make_clip_unknown_event_or_huge_window_is_refused(args: dict[str, Any]) -> None:
    called: list[str] = []

    def maker(event_id: str, _before: float, _after: float) -> Path:
        called.append(event_id)
        return Path("c.mp4")

    tools = box(clip_maker=maker)

    assert "error" in tools.call("make_clip", args)
    assert called == []


@pytest.mark.unit
def test_find_player_remembers_only_the_players_it_returned() -> None:
    firsts = ["Аскар", "Берик", "Ерлан", "Марат", "Нурлан", "Талгат", "Данияр"]
    crowd = tuple(Player("home", f"Общий {first}", i) for i, first in enumerate(firsts))
    events = tuple(
        make_event(i, EventType.GOAL, 100.0 * i, 1, i, "home", [read(p.name)])
        for i, p in enumerate(crowd, start=1)
    )
    tools = ToolBox(MatchContext(match_id="m", events=events, has_lineup=True, lineup=crowd))

    result = tools.call("find_player", {"query": "Общий"})

    assert len(result["players"]) == 5
    assert tools.seen_event_ids == {f"p:{i}" for i in range(1, 6)}
    assert tools.seen_names == {p.name for p in crowd[:5]}


@pytest.mark.unit
def test_find_player_says_when_more_players_fit_than_are_shown() -> None:
    firsts = ["Аскар", "Берик", "Ерлан", "Марат", "Нурлан", "Талгат", "Данияр"]
    crowd = tuple(Player("home", f"Общий {first}", i) for i, first in enumerate(firsts))
    tools = ToolBox(MatchContext(match_id="m", events=(), has_lineup=True, lineup=crowd))

    result = tools.call("find_player", {"query": "Общий"})

    assert len(result["players"]) == 5
    assert "подошло игроков: 7" in result["note"]


@pytest.mark.unit
def test_find_player_without_lineup_says_that_the_number_was_not_used() -> None:
    result = box(lineup=False).call("find_player", {"query": "Тестов", "number": 9})

    assert [p["name"] for p in result["players"]] == ["Тестов Иван"]
    assert "номер не учтён" in result["note"]


@pytest.mark.unit
def test_toolbox_remembers_numbers_of_results_and_arguments_but_not_digits_of_ids() -> None:
    tools = box()

    tools.call("query_events", {"type": "goal", "half": 2})

    assert {1.0, 2.0, 81.0} <= tools.seen_numbers  # total 1, the half, the minute
    assert 6.0 not in tools.seen_numbers  # the digit of the id p:6


@pytest.mark.unit
def test_numbers_of_an_error_result_are_not_remembered() -> None:
    tools = box()

    tools.call("query_events", {"limit": 999})

    assert tools.seen_numbers == set()


@pytest.mark.unit
def test_digits_of_an_event_id_or_a_limit_in_the_arguments_are_not_seen_numbers() -> None:
    tools = box()

    tools.call("get_event", {"event_id": "p:5"})
    tools.call("query_events", {"type": "shot", "half": 1, "limit": 17})

    assert 17.0 not in tools.seen_numbers
    tools2 = box()
    tools2.call("get_event", {"event_id": "p:99"})
    assert 99.0 not in tools2.seen_numbers


@pytest.mark.unit
def test_a_foul_with_a_card_plate_shows_the_card_and_the_name_from_the_plate() -> None:
    event = box().call("get_event", {"event_id": "p:4"})["event"]

    assert event["card"] == "yellow"
    assert event["players"][0]["name"] == "Образцов Марат"
    assert "card" not in box().call("get_event", {"event_id": "p:1"})["event"]


@pytest.mark.unit
def test_query_events_card_filter_keeps_only_fouls_with_a_card() -> None:
    assert ids(box().call("query_events", {"card": True})) == ["p:4"]
    assert "p:4" not in ids(box().call("query_events", {"card": False}))


@pytest.mark.unit
def test_match_info_counts_the_cards_by_colour() -> None:
    assert box().call("match_info", {})["cards"] == {"yellow": 1, "red": 0}


@pytest.mark.unit
def test_find_player_shows_the_cards_of_the_player() -> None:
    (player,) = box().call("find_player", {"query": "Образцов"})["players"]

    assert player["yellow_cards"] == 1
    assert [e["card"] for e in player["events"] if "card" in e] == ["yellow"]


@pytest.mark.unit
def test_a_substitution_names_who_came_on_and_who_went_off_or_none_when_unread() -> None:
    event = box().call("get_event", {"event_id": "p:5"})["event"]

    assert event["substitution"] == {"came_on": "Нұрланов Ұлан", "went_off": None}
    assert "substitution" not in box().call("get_event", {"event_id": "p:1"})["event"]


@pytest.mark.unit
def test_side_filter_reports_the_events_whose_team_is_unknown_instead_of_hiding_them() -> None:
    result = box().call("query_events", {"type": "shot", "side": "home"})

    assert ids(result) == ["p:1"]
    assert result["team_unknown"] == 1  # the shot p:7 has no team
    assert "команда не определена" in result["team_unknown_note"]


@pytest.mark.unit
def test_side_filter_with_nothing_but_unknown_team_events_says_so() -> None:
    result = box().call("query_events", {"type": "foul", "side": "away"})

    assert (result["total"], result["team_unknown"]) == (0, 1)


@pytest.mark.unit
def test_no_team_unknown_flag_without_a_side_filter_or_with_side_unknown() -> None:
    assert "team_unknown" not in box().call("query_events", {"type": "shot"})
    assert "team_unknown" not in box().call("query_events", {"type": "shot", "side": "unknown"})


@pytest.mark.unit
def test_without_known_minute_is_counted_for_the_asked_side_only() -> None:
    result = box().call("query_events", {"type": "shot", "side": "home", "minute_from": 0})

    assert ids(result) == ["p:1"]
    assert "without_known_minute" not in result  # p:7 has no minute but no team either


@pytest.mark.unit
def test_team_unknown_is_counted_over_all_fitting_events_not_only_the_shown_ones() -> None:
    result = box().call("query_events", {"side": "home", "limit": 1})

    assert result["returned"] == 1
    assert result["team_unknown"] == 2  # p:4 and p:7


def box_with_red_card() -> ToolBox:
    ctx = make_context()
    events = tuple(replace(f, card="red") if f.event.event_id == "p:4" else f for f in ctx.events)
    return ToolBox(replace(ctx, events=events))


@pytest.mark.unit
def test_red_card_is_shown_counted_and_found_by_the_card_filter() -> None:
    tools = box_with_red_card()

    assert tools.call("get_event", {"event_id": "p:4"})["event"]["card"] == "red"
    assert tools.call("match_info", {})["cards"] == {"yellow": 0, "red": 1}
    assert ids(tools.call("query_events", {"card": True})) == ["p:4"]
    (player,) = tools.call("find_player", {"query": "Образцов"})["players"]
    assert (player["yellow_cards"], player["red_cards"]) == (0, 1)
