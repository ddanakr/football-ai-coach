"""The corner tags in the agent: filters of `query_events`, the tags of an event, the mark added
by code, and the context that carries them (D-078). The tags are a model's reading of a clip."""

from dataclasses import replace
from pathlib import Path

import pytest

from aicoach.agent.context import MatchContext, load_context
from aicoach.agent.loop import Agent
from aicoach.agent.render import plain_text
from aicoach.agent.tools import CORNER_TAGS_NOTE, QUERY_EVENTS, ToolBox
from aicoach.corner_tags.schemas import CornerFields, CornerTag
from aicoach.corner_tags.view import CornerView
from aicoach.domain.events import EventType
from aicoach.i18n import ru
from tests.fixtures.agent_synth import FakeLLM, make_context, make_event, turn
from tests.fixtures.corner_synth import MATCH_ID as SYNTH_MATCH
from tests.fixtures.corner_synth import write_db, write_lineup, write_tag_files

FIELDS = {
    "is_corner": "yes",
    "attacking_kit": "white",
    "delivery": "cross",
    "zone": "far_post",
    "first_contact": "defence",
    "outcome": "cleared",
}


def view(side: str | None = "home", **changes: str) -> CornerView:
    tag = CornerTag(
        event_id="p:3",
        t_v=1900.0,
        half=1,
        fields=CornerFields.model_validate({**FIELDS, **changes}),
        status="ok",
        model="test/vlm",
        prompt_version=1,
        before_s=4.0,
        after_s=12.0,
    )
    return CornerView(tag, side)  # type: ignore[arg-type]


def context_with_tags() -> MatchContext:
    """The synthetic match where the corner p:3 has tags and a second corner p:8 has others."""
    ctx = make_context()
    events = list(ctx.events)
    events[2] = replace(events[2], corner=view("home"))
    second = make_event(8, EventType.CORNER, 5200.0, 2, 55, None)
    events.append(
        replace(
            second, corner=view(None, attacking_kit="unclear", zone="near_post", outcome="shot")
        )
    )
    events.append(make_event(9, EventType.CORNER, 5400.0, 2, 57, None))  # a corner without tags
    return replace(ctx, events=tuple(events), team_kits={"home": "white", "away": None})


def ids(result: dict) -> list[str]:  # type: ignore[type-arg]
    return [e["event_id"] for e in result["events"]]


@pytest.mark.unit
def test_the_tool_offers_the_corner_filters_with_only_allowed_values() -> None:
    props = QUERY_EVENTS.parameters["properties"]

    assert props["corner_team"]["enum"] == ["home", "away", "unknown"]
    assert "cross" in props["corner_delivery"]["enum"]
    assert "unclear" in props["corner_outcome"]["enum"]
    assert "light_blue" in props["corner_kit"]["enum"]
    assert all("не проверено" in props[k]["description"] for k in props if k.startswith("corner_"))


@pytest.mark.unit
def test_an_event_with_tags_carries_them_with_the_side_the_name_and_the_source() -> None:
    box = ToolBox(context_with_tags())

    result = box.call("get_event", {"event_id": "p:3"})

    tags = result["event"]["corner_tags"]
    assert tags["delivery"] == "cross"
    assert tags["zone"] == "far_post"
    assert tags["attacking_team"] == "home"
    assert tags["attacking_team_name"] == "Альфа"
    assert tags["source"] == "clip_model_unverified"
    assert box.corner_tags_shown


@pytest.mark.unit
def test_an_unknown_attacking_team_is_null_not_a_guess() -> None:
    box = ToolBox(context_with_tags())

    tags = box.call("get_event", {"event_id": "p:8"})["event"]["corner_tags"]

    assert tags["attacking_team"] is None
    assert tags["attacking_team_name"] is None
    assert tags["attacking_kit"] == "unclear"


@pytest.mark.unit
def test_an_event_without_tags_has_no_tags_key_and_does_not_set_the_flag() -> None:
    box = ToolBox(context_with_tags())

    result = box.call("get_event", {"event_id": "p:9"})

    assert "corner_tags" not in result["event"]
    assert not box.corner_tags_shown


@pytest.mark.unit
def test_filter_by_the_attacking_team() -> None:
    box = ToolBox(context_with_tags())

    home = box.call("query_events", {"corner_team": "home"})
    unknown = box.call("query_events", {"corner_team": "unknown"})
    away = box.call("query_events", {"corner_team": "away"})

    assert ids(home) == ["p:3"]
    assert ids(unknown) == ["p:8"]  # a corner without tags is not "unknown", it has no tags
    assert away["total"] == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"corner_zone": "far_post"}, ["p:3"]),
        ({"corner_zone": "дальняя штанга"}, ["p:3"]),  # a Russian name is accepted
        ({"corner_outcome": "shot"}, ["p:8"]),
        ({"corner_kit": "unclear"}, ["p:8"]),
        ({"corner_delivery": "cross", "corner_first_contact": "defence"}, ["p:3", "p:8"]),
        ({"corner_delivery": "cross", "corner_zone": "near_post"}, ["p:8"]),
        ({"corner_zone": "center"}, []),
    ],
)
def test_filter_by_the_fields(args: dict[str, str], expected: list[str]) -> None:
    result = ToolBox(context_with_tags()).call("query_events", args)

    assert ids(result) == expected
    assert result["total"] == len(expected)


@pytest.mark.unit
def test_a_corner_filter_never_returns_events_that_are_not_corners_or_have_no_tags() -> None:
    result = ToolBox(context_with_tags()).call("query_events", {"corner_delivery": "cross"})

    assert set(ids(result)) <= {"p:3", "p:8"}
    assert "p:9" not in ids(result)


@pytest.mark.unit
def test_the_filters_combine_with_the_other_filters() -> None:
    box = ToolBox(context_with_tags())

    assert ids(box.call("query_events", {"corner_delivery": "cross", "half": 2})) == ["p:8"]
    assert ids(box.call("query_events", {"type": "corner", "corner_team": "home"})) == ["p:3"]
    assert box.call("query_events", {"type": "goal", "corner_team": "home"})["total"] == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "args",
    [
        {"corner_zone": "penalty spot"},
        {"corner_delivery": "zone"},
        {"corner_team": "both"},
        {"corner_kit": "pink"},
    ],
)
def test_a_wrong_filter_value_is_an_error_for_the_model_naming_the_argument(
    args: dict[str, str],
) -> None:
    result = ToolBox(context_with_tags()).call("query_events", args)

    assert list(result) == ["error"]
    assert next(iter(args)) in result["error"]


@pytest.mark.unit
def test_the_result_carries_the_not_verified_note_when_tags_are_in_it() -> None:
    box = ToolBox(context_with_tags())

    assert box.call("query_events", {"corner_team": "home"})["corner_tags_note"] == CORNER_TAGS_NOTE
    assert "corner_tags_note" in box.call("query_events", {"type": "corner"})
    assert "corner_tags_note" not in box.call("query_events", {"type": "goal"})
    assert "по клипу" in CORNER_TAGS_NOTE
    assert "не проверено" in CORNER_TAGS_NOTE


@pytest.mark.unit
def test_the_answer_gets_the_mark_from_the_code_when_the_model_forgets_it() -> None:
    llm = FakeLLM(
        [
            turn("", ("query_events", {"corner_team": "home"})),
            turn("Хозяева подали один угловой с навесом [E:p:3]."),
        ]
    )

    answer = Agent(llm, "system").ask("Угловые хозяев?", ToolBox(context_with_tags()))

    assert answer.outcome == "first_try"
    text = plain_text(answer.segments)
    assert ru.CORNER_TAGS_ANSWER_MARK in text
    assert text.endswith(ru.CORNER_ZONE_ANSWER_MARK)  # the zone of p:3 was shown too


@pytest.mark.unit
def test_the_mark_is_not_added_twice_when_the_model_wrote_it() -> None:
    llm = FakeLLM(
        [
            turn("", ("query_events", {"corner_team": "home"})),
            turn("Один угловой с навесом, по клипу, не проверено [E:p:3]."),
        ]
    )

    answer = Agent(llm, "system").ask("Угловые хозяев?", ToolBox(context_with_tags()))

    assert plain_text(answer.segments).count("не проверено") == 1


@pytest.mark.unit
def test_no_mark_when_no_tags_reached_the_model() -> None:
    llm = FakeLLM([turn("", ("query_events", {"type": "goal"})), turn("Гол [E:p:2].")])

    answer = Agent(llm, "system").ask("Кто забил?", ToolBox(make_context()))

    assert ru.CORNER_TAGS_ANSWER_MARK not in plain_text(answer.segments)


@pytest.mark.unit
def test_the_tag_words_in_the_answer_do_not_break_the_validator() -> None:
    llm = FakeLLM(
        [
            turn("", ("query_events", {"corner_zone": "far_post"})),
            turn("Мяч полетел на дальнюю штангу, защита выбила [E:p:3]."),
        ]
    )

    answer = Agent(llm, "system").ask("Куда?", ToolBox(context_with_tags()))

    assert answer.outcome == "first_try"
    assert answer.violations == []


@pytest.mark.unit
def test_load_context_reads_the_tags_the_team_names_and_the_kits_of_the_match(
    tmp_path: Path,
) -> None:
    db = write_db(tmp_path)
    write_tag_files(tmp_path)

    ctx = load_context(db, SYNTH_MATCH, tmp_path)

    corners = [f for f in ctx.events if f.event.type == EventType.CORNER]
    assert [f.corner is not None for f in corners] == [True, True, True]
    assert [f.corner.side for f in corners if f.corner] == ["home", "away", None]
    assert ctx.team_names == {"home": "Актобе", "away": "Кайрат"}
    assert ctx.team_kits == {"home": "white", "away": "red"}


@pytest.mark.unit
def test_load_context_without_tag_files_works_as_before(tmp_path: Path) -> None:
    db = write_db(tmp_path)
    write_lineup(tmp_path, "Тобол", "Астана")

    ctx = load_context(db, SYNTH_MATCH, tmp_path)

    assert all(f.corner is None for f in ctx.events)
    assert ctx.team_names == {"home": "Тобол", "away": "Астана"}
    assert ctx.team_kits == {"home": None, "away": None}


@pytest.mark.unit
def test_a_manual_override_is_what_the_agent_sees(tmp_path: Path) -> None:
    from aicoach.teams.store import SideOverride, TeamsOverride, save_override

    db = write_db(tmp_path)
    write_tag_files(tmp_path)
    save_override(tmp_path, TeamsOverride(home=SideOverride(name="FC Aktobe", kit="blue")))

    ctx = load_context(db, SYNTH_MATCH, tmp_path)

    assert ctx.team_names["home"] == "FC Aktobe"
    assert ctx.team_kits["home"] == "blue"
    first = next(f for f in ctx.events if f.event.event_id == "p:3")
    assert first.corner is not None
    assert first.corner.side is None  # white is no longer the home colour


@pytest.mark.unit
def test_an_event_with_tags_marks_the_zone_as_low_reliability_for_the_model() -> None:
    box = ToolBox(context_with_tags())

    tags = box.call("get_event", {"event_id": "p:3"})["event"]["corner_tags"]

    assert tags["zone_reliability"] == "low"
    assert "zone" in CORNER_TAGS_NOTE
    assert "низкой надёжности" in CORNER_TAGS_NOTE


@pytest.mark.unit
def test_the_zone_flag_is_set_by_a_shown_zone_or_a_zone_filter_not_by_an_unclear_zone() -> None:
    ctx = context_with_tags()
    events = list(ctx.events)
    events[2] = replace(events[2], corner=view("home", zone="unclear"))
    unclear = ToolBox(replace(ctx, events=tuple(events)))

    unclear.call("get_event", {"event_id": "p:3"})
    assert unclear.corner_tags_shown
    assert not unclear.zone_shown

    shown = ToolBox(ctx)
    shown.call("get_event", {"event_id": "p:3"})
    assert shown.zone_shown

    filtered = ToolBox(ctx)
    filtered.call("query_events", {"corner_zone": "center"})  # nothing found, still asked
    assert filtered.zone_shown


@pytest.mark.unit
def test_the_answer_gets_the_zone_caveat_from_the_code_when_the_model_forgets_it() -> None:
    llm = FakeLLM(
        [
            turn("", ("query_events", {"corner_zone": "far_post"})),
            turn("Мяч полетел на дальнюю штангу, по клипу, не проверено [E:p:3]."),
        ]
    )

    answer = Agent(llm, "system").ask("Куда?", ToolBox(context_with_tags()))

    assert plain_text(answer.segments).endswith(ru.CORNER_ZONE_ANSWER_MARK)
    assert "4 из 8" in ru.CORNER_ZONE_ANSWER_MARK


@pytest.mark.unit
def test_the_zone_caveat_is_not_added_when_the_model_said_it() -> None:
    llm = FakeLLM(
        [
            turn("", ("query_events", {"corner_zone": "far_post"})),
            turn("Зона дальняя штанга, низкая надёжность, по клипу, не проверено [E:p:3]."),
        ]
    )

    answer = Agent(llm, "system").ask("Куда?", ToolBox(context_with_tags()))

    assert ru.CORNER_ZONE_ANSWER_MARK not in plain_text(answer.segments)


@pytest.mark.unit
def test_no_zone_caveat_when_the_zone_did_not_reach_the_model() -> None:
    llm = FakeLLM(
        [
            turn("", ("get_event", {"event_id": "p:9"})),  # a corner without tags
            turn("Тегов у этого углового нет [E:p:9]."),
        ]
    )

    answer = Agent(llm, "system").ask("Что за угловой?", ToolBox(context_with_tags()))

    assert ru.CORNER_ZONE_ANSWER_MARK not in plain_text(answer.segments)
