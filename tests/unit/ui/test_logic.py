"""Pure logic of the viewer: time text, links, filters, rows."""

import math

import pytest
from tests.fixtures.ui_synth import make_event

from aicoach.domain.events import EventType
from aicoach.errors import ConfigError
from aicoach.storage import CurrentEvent
from aicoach.ui.logic import (
    PLAYER_PRE_ROLL_S,
    EventFilter,
    EventRow,
    available_types,
    filter_events,
    format_mmss,
    player_start,
    selected_row,
    table_records,
    to_row,
    to_rows,
    type_label,
    validate_video_id,
    watch_url,
    youtube_link,
)

VID = "abcDEF_-123"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("t_v", "expected"),
    [
        (0.0, "00:00"),
        (59.9, "00:59"),
        (60.0, "01:00"),
        (754.3, "12:34"),
        (3600.0, "60:00"),
        (7501.0, "125:01"),
    ],
)
def test_format_mmss_cuts_seconds_and_lets_minutes_pass_an_hour(t_v: float, expected: str) -> None:
    assert format_mmss(t_v) == expected


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, -math.inf])
def test_format_mmss_bad_time_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="t_v"):
        format_mmss(bad)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("t_v", "pre_roll", "expected"),
    [
        (100.0, 5.0, 95),
        (100.9, 5.0, 95),
        (5.0, 5.0, 0),
        (3.0, 5.0, 0),
        (0.0, 5.0, 0),
        (50.0, 0.0, 50),
    ],
)
def test_player_start_subtracts_pre_roll_and_stops_at_zero(
    t_v: float, pre_roll: float, expected: int
) -> None:
    assert player_start(t_v, pre_roll) == expected


@pytest.mark.unit
def test_player_start_default_pre_roll_is_five_seconds() -> None:
    assert PLAYER_PRE_ROLL_S == 5.0
    assert player_start(65.0) == 60


@pytest.mark.unit
def test_player_start_bad_pre_roll_raises() -> None:
    with pytest.raises(ValueError, match="t_v"):
        player_start(10.0, -1.0)


@pytest.mark.unit
def test_youtube_link_points_to_the_moment_with_pre_roll() -> None:
    assert youtube_link(VID, 125.4) == f"https://www.youtube.com/watch?v={VID}&t=120s"


@pytest.mark.unit
def test_youtube_link_near_the_start_gives_zero() -> None:
    assert youtube_link(VID, 2.0) == f"https://www.youtube.com/watch?v={VID}&t=0s"


@pytest.mark.unit
def test_youtube_link_custom_pre_roll() -> None:
    assert youtube_link(VID, 100.0, pre_roll_s=10.0).endswith("&t=90s")


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad", ["", "short", "abcDEF_-1234", 'abc"><script>', "abc DEF_-12", "абвгдеёжзик"]
)
def test_video_id_with_wrong_form_is_refused(bad: str) -> None:
    with pytest.raises(ConfigError, match="11 characters"):
        validate_video_id(bad)
    with pytest.raises(ConfigError):
        youtube_link(bad, 1.0)


@pytest.mark.unit
def test_video_id_starting_with_a_dash_is_valid() -> None:
    assert validate_video_id("-abcDEF_123") == "-abcDEF_123"


@pytest.mark.unit
def test_watch_url_has_no_time() -> None:
    assert watch_url(VID) == f"https://www.youtube.com/watch?v={VID}"


@pytest.mark.unit
def test_type_label_is_russian_for_every_category() -> None:
    assert [type_label(t) for t in EventType] == ["Гол", "Удар", "Угловой", "Фол", "Замена"]


@pytest.fixture
def events() -> list[CurrentEvent]:
    return [
        make_event("p:1", EventType.GOAL, 10.0, 1, 1.0),
        make_event("p:2", EventType.SHOT, 20.0, 1, 0.3),
        make_event("p:3", EventType.FOUL, 30.0, 2, 0.5),
        make_event("p:4", EventType.SUBST, 40.0, 2, None, source="plates"),
        make_event(
            "e:5", EventType.CORNER, 50.0, None, None, status="manual", source=None, team=None
        ),
    ]


def ids(found: list[CurrentEvent]) -> list[str]:
    return [e.event_id for e in found]


@pytest.mark.unit
def test_filter_all_types_keeps_everything_in_order(events: list[CurrentEvent]) -> None:
    flt = EventFilter(types=frozenset(EventType))

    assert ids(filter_events(events, flt)) == ["p:1", "p:2", "p:3", "p:4", "e:5"]


@pytest.mark.unit
def test_filter_by_types(events: list[CurrentEvent]) -> None:
    flt = EventFilter(types=frozenset({EventType.GOAL, EventType.FOUL}))

    assert ids(filter_events(events, flt)) == ["p:1", "p:3"]


@pytest.mark.unit
def test_filter_no_types_shows_nothing(events: list[CurrentEvent]) -> None:
    assert filter_events(events, EventFilter(types=frozenset())) == []


@pytest.mark.unit
def test_filter_by_half_hides_unknown_half(events: list[CurrentEvent]) -> None:
    flt = EventFilter(types=frozenset(EventType), half=2)

    assert ids(filter_events(events, flt)) == ["p:3", "p:4"]


@pytest.mark.unit
def test_filter_min_score_keeps_events_without_score_and_the_boundary(
    events: list[CurrentEvent],
) -> None:
    flt = EventFilter(types=frozenset(EventType), min_score=0.5)

    assert ids(filter_events(events, flt)) == ["p:1", "p:3", "p:4", "e:5"]


@pytest.mark.unit
def test_filter_empty_input_gives_empty_list() -> None:
    assert filter_events([], EventFilter(types=frozenset(EventType))) == []


@pytest.mark.unit
def test_available_types_follow_the_taxonomy_order(events: list[CurrentEvent]) -> None:
    assert available_types(reversed(events)) == list(EventType)
    assert available_types([]) == []


@pytest.mark.unit
def test_to_row_formats_a_model_event() -> None:
    row = to_row(make_event("p:7", EventType.SHOT, 754.3, 2, 0.8123, source="merge"))

    assert (row.event_id, row.t_v, row.time_text) == ("p:7", 754.3, "12:34")
    assert (row.type_label, row.half_text, row.score_text) == ("Удар", "2", "0.81")
    assert (row.source_text, row.team_text, row.status_text) == ("модель и плашка", "хозяева", "")
    assert row.type_text == "Удар"


@pytest.mark.unit
def test_to_row_user_event_shows_dashes_and_status() -> None:
    row = to_row(make_event("e:1", half=None, score=None, status="manual", source=None, team=None))

    assert (row.half_text, row.score_text, row.source_text, row.team_text) == ("—",) * 4
    assert row.type_text == "Гол (добавлено вручную)"


@pytest.mark.unit
def test_to_row_unknown_status_is_shown_as_is_and_away_team_is_guests() -> None:
    row = to_row(make_event(status="odd", team="away"))

    assert row.type_text == "Гол (odd)"
    assert row.team_text == "гости"


@pytest.mark.unit
def test_table_records_have_russian_columns_in_order_and_empty_unknowns() -> None:
    rows = to_rows(
        [
            make_event("p:1", EventType.SHOT, 754.3, 2, 0.8123, source="merge", team="away"),
            make_event("e:2", half=None, score=None, status="manual", source=None, team=None),
        ]
    )

    records = table_records(rows)

    assert list(records[0]) == [
        "Время",
        "Тип",
        "Тайм",
        "Уверенность",
        "Источник",
        "Команда",
        "Правка",
    ]
    assert list(records[0].values()) == [
        "12:34",
        "Удар",
        "2",
        0.8123,
        "модель и плашка",
        "гости",
        "",
    ]
    assert list(records[1].values()) == [
        "01:40",
        "Гол (добавлено вручную)",
        "",
        None,
        "",
        "",
        "добавлено",
    ]


@pytest.mark.unit
def test_table_records_empty_input_gives_no_records() -> None:
    assert table_records([]) == []


@pytest.fixture
def rows() -> list[EventRow]:
    return to_rows([make_event(f"p:{i}", t_v=10.0 * i) for i in range(3)])


@pytest.mark.unit
@pytest.mark.parametrize(("selected", "expected"), [([0], "p:0"), ([2], "p:2"), ([1, 0], "p:1")])
def test_selected_row_returns_the_event_of_the_first_picked_index(
    rows: list[EventRow], selected: list[int], expected: str
) -> None:
    chosen = selected_row(rows, selected)

    assert chosen is not None
    assert chosen.event_id == expected


@pytest.mark.unit
@pytest.mark.parametrize("selected", [[], [3], [-1], [99]])
def test_selected_row_nothing_or_out_of_range_gives_none(
    rows: list[EventRow], selected: list[int]
) -> None:
    assert selected_row(rows, selected) is None


@pytest.mark.unit
def test_selected_row_with_no_rows_gives_none() -> None:
    assert selected_row([], [0]) is None


@pytest.mark.unit
def test_to_rows_keeps_order(events: list[CurrentEvent]) -> None:
    assert [r.event_id for r in to_rows(events)] == ids(events)
