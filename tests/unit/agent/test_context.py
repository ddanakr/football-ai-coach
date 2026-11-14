"""The match snapshot: game minutes, plate names, lineup, loading from a match directory."""

import os
from pathlib import Path

import pytest

from aicoach.agent.context import (
    build_facts,
    game_minute,
    load_context,
    minute_text,
)
from aicoach.domain.events import EventType
from aicoach.errors import StorageError
from aicoach.plate_names.lineup import parse_team_names
from aicoach.scoreboard.schemas import OffsetSegment
from tests.fixtures.agent_synth import read
from tests.fixtures.plate_names_synth import LINEUP_CSV
from tests.fixtures.ui_synth import make_event, write_match_dir
from tests.unit.ui.test_players import names, write_names
from tests.unit.ui.test_players import read as pread


@pytest.mark.unit
@pytest.mark.parametrize(
    ("clock_s", "expected"),
    [
        (0.0, 1),
        (59.9, 1),
        (60.0, 2),
        (22 * 60 + 40.0, 23),
        (45 * 60 + 5.0, 46),
        (None, None),
        (-1.0, None),
        (float("nan"), None),
        (float("inf"), None),
    ],
)
def test_game_minute_is_the_clock_minute_plus_one(
    clock_s: float | None, expected: int | None
) -> None:
    assert game_minute(clock_s) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("minute", "half", "expected"),
    [
        (23, 1, "23'"),
        (45, 1, "45'"),
        (47, 1, "45+2'"),
        (46, 2, "46'"),
        (90, 2, "90'"),
        (93, 2, "90+3'"),
        (47, None, "47'"),
        (None, 1, None),
    ],
)
def test_minute_text_writes_added_time_like_the_broadcast(
    minute: int | None, half: int | None, expected: str | None
) -> None:
    assert minute_text(minute, half) == expected


SEGMENTS = (
    OffsetSegment(
        start=100.0, end=1000.0, clock_start_s=0, clock_end_s=900, offset_s=100.0, n_readings=900
    ),
)


@pytest.mark.unit
def test_build_facts_adds_the_minute_and_the_plate_names_by_type_and_time() -> None:
    goal = make_event("p:1", EventType.GOAL, 220.0)
    shot = make_event("p:2", EventType.SHOT, 5000.0)  # no clock segment near it
    names = {(EventType.GOAL, 220.0): (read("Тестов Иван"),)}

    facts = build_facts([goal, shot], SEGMENTS, names, 30.0)

    assert [(f.minute, f.minute_text) for f in facts] == [(3, "3'"), (None, None)]
    assert [p.name for p in facts[0].players] == ["Тестов Иван"]
    assert facts[1].players == ()


@pytest.mark.unit
def test_build_facts_gives_no_name_to_a_moved_event_or_to_one_added_by_the_user() -> None:
    names = {(EventType.GOAL, 220.0): (read("Тестов Иван"),)}
    moved = make_event("p:1", EventType.GOAL, 221.0)
    added = make_event("e:1", EventType.GOAL, 220.0, source=None)

    facts = build_facts([moved, added], SEGMENTS, names, 30.0)

    assert [f.players for f in facts] == [(), ()]


@pytest.mark.unit
def test_build_facts_of_nothing_is_empty() -> None:
    assert build_facts([], SEGMENTS, {}, 30.0) == ()


@pytest.mark.unit
def test_team_names_come_from_the_team_column_of_the_lineup() -> None:
    assert parse_team_names(LINEUP_CSV.encode()) == {"home": "Север", "away": "Юг"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "data", [b"", "side,name\nhome,Иванов\n".encode(), b"\xff\xfe", b"team,side\n"]
)
def test_team_names_of_a_lineup_without_team_column_or_with_garbage_are_empty(data: bytes) -> None:
    assert parse_team_names(data) == {}


@pytest.mark.unit
def test_load_context_reads_events_minutes_and_half_bounds(tmp_path: Path) -> None:
    db = write_match_dir(tmp_path, "m1")

    ctx = load_context(db, "m1", tmp_path)

    assert ctx.match_id == "m1"
    assert [str(f.event.type) for f in ctx.events] == ["goal", "subst", "foul"]
    assert all(f.minute is not None and f.minute_text for f in ctx.events)
    assert ctx.has_lineup is False
    assert ctx.team_names == {"home": "хозяева", "away": "гости"}
    assert set(ctx.half_minutes) == {1, 2}


@pytest.mark.unit
def test_load_context_with_a_lineup_has_team_names_and_players(tmp_path: Path) -> None:
    db = write_match_dir(tmp_path, "m1")
    (tmp_path / "lineups.csv").write_text(LINEUP_CSV, encoding="utf-8")

    ctx = load_context(db, "m1", tmp_path)

    assert ctx.has_lineup is True
    assert ctx.team_names == {"home": "Север", "away": "Юг"}
    assert len(ctx.lineup) == 6


@pytest.mark.unit
def test_load_context_survives_missing_and_broken_optional_files(tmp_path: Path) -> None:
    db = write_match_dir(tmp_path, "m1")
    (tmp_path / "scoreboard.json").write_text("{ not json", encoding="utf-8")
    (tmp_path / "lineups.csv").write_text("garbage without columns", encoding="utf-8")
    (tmp_path / "plate_names.json").unlink(missing_ok=True)

    ctx = load_context(db, "m1", tmp_path)

    assert len(ctx.events) == 3
    assert [f.minute for f in ctx.events] == [None, None, None]
    assert ctx.has_lineup is False


@pytest.mark.unit
def test_load_context_without_database_is_a_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="does not exist"):
        load_context(tmp_path / "none.db", "m1", tmp_path)


@pytest.mark.unit
def test_load_context_of_another_match_id_has_no_events(tmp_path: Path) -> None:
    db = write_match_dir(tmp_path, "m1")

    assert load_context(db, "other", tmp_path).events == ()


@pytest.mark.unit
def test_load_context_joins_plate_names_to_the_goal_and_follows_a_changed_file(
    tmp_path: Path,
) -> None:
    db = write_match_dir(tmp_path, "m1")
    path = write_names(tmp_path, names("p1", "goal", pread("player", "Лебедев Роман"), t_v=230.0))

    goal = load_context(db, "m1", tmp_path).events[0]

    assert [p.name for p in goal.players] == ["Лебедев Роман"]

    write_names(tmp_path, names("p1", "goal", pread("player", "Нурланов Айдар"), t_v=230.0))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))  # a new stamp
    assert [p.name for p in load_context(db, "m1", tmp_path).events[0].players] == [
        "Нурланов Айдар"
    ]


@pytest.mark.unit
def test_load_context_ignores_names_of_a_plate_with_another_time(tmp_path: Path) -> None:
    db = write_match_dir(tmp_path, "m1")
    write_names(tmp_path, names("p1", "goal", pread("player", "Лебедев Роман"), t_v=999.0))

    assert load_context(db, "m1", tmp_path).events[0].players == ()


@pytest.mark.unit
def test_load_context_survives_a_scoreboard_file_with_bad_bytes(tmp_path: Path) -> None:
    db = write_match_dir(tmp_path, "m1")
    (tmp_path / "scoreboard.json").write_bytes(b"\xff\xfe\x00broken")
    (tmp_path / "candidates.json").write_bytes(b"\xff\xfe\x00broken")

    ctx = load_context(db, "m1", tmp_path)

    assert len(ctx.events) == 3
    assert ctx.half_minutes == {}


@pytest.mark.unit
def test_load_context_derives_the_half_of_an_event_added_by_the_user(tmp_path: Path) -> None:
    from aicoach.storage import add_edit, connect

    db = write_match_dir(tmp_path, "m1")
    with connect(db) as conn:
        add_edit(conn, "m1", "add", author="test", type=EventType.SHOT, t_v=300.0)

    added = [f for f in load_context(db, "m1", tmp_path).events if f.event.origin == "user"]

    assert [f.event.half for f in added] == [1]
    assert added[0].players == ()
