"""The "player" column of the events table: texts from plate names, the join with the events,
the loader of the two files, and the table in the app (`AppTest`)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.fixtures.ui_synth import make_event, write_match_dir

from aicoach.domain.events import EventType
from aicoach.fusion.schemas import Candidate, Evidence
from aicoach.i18n import ru
from aicoach.plate_names.schemas import (
    PlateNames,
    PlateNamesConfig,
    PlateNamesResult,
    PlateNamesStats,
    PlayerRead,
    Role,
    Status,
)
from aicoach.ui.data import load_player_texts
from aicoach.ui.launch import AppArgs, app_script, build_streamlit_command
from aicoach.ui.logic import player_text, player_texts, table_records, to_row, to_rows

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1"])
VID = "abcDEF_-123"
PLATE_T = 10.0  # the first-seen time of the plates in `names` and in the evidence


def run_app(monkeypatch: pytest.MonkeyPatch, db: Path, match_id: str) -> AppTest:
    """Run `app.py` with the arguments `cli ui` would give it (the match directory is the
    database's directory)."""
    command = build_streamlit_command(AppArgs(db=db, match_id=match_id, youtube_id=VID))
    monkeypatch.setattr("sys.argv", [str(app_script()), *command[6:]])
    return AppTest.from_file(str(app_script()), default_timeout=20).run()


def read(role: Role, name: str | None, status: Status = "matched") -> PlayerRead:
    return PlayerRead(
        role=role,
        raw_text="raw",
        name=name,
        source="ocr+lineup" if status == "matched" else "ocr",
        status=status,
    )


def names(plate_id: str, kind: str, *players: PlayerRead, t_v: float = 10.0) -> PlateNames:
    return PlateNames.model_validate(
        {
            "plate_id": plate_id,
            "type": kind,
            "t_v": t_v,
            "half": 1,
            "side": "home",
            "players": [p.model_dump() for p in players],
        }
    )


def candidate(kind: EventType, t_v: float, *refs: tuple[str, str]) -> Candidate:
    return Candidate(
        type=kind,
        t_v=t_v,
        score=1.0,
        source="plates",
        rule="r",
        evidence=[Evidence(kind=k, ref=ref, t_v=PLATE_T) for k, ref in refs],  # type: ignore[arg-type]
    )


@pytest.mark.unit
def test_player_text_goal_gives_the_lineup_name() -> None:
    assert player_text(names("a", "goal", read("player", "Лебедев Роман"))) == "Лебедев Роман"


@pytest.mark.unit
def test_player_text_unchecked_name_carries_the_note() -> None:
    text = player_text(names("a", "card", read("player", "Роман Лебедев", "unchecked")))

    assert text == "Роман Лебедев" + ru.PLAYER_UNCHECKED_MARK


@pytest.mark.unit
def test_player_text_unrecognized_name_says_so() -> None:
    assert player_text(names("a", "goal", read("player", None, "unrecognized"))) == (
        ru.PLAYER_UNRECOGNIZED
    )


@pytest.mark.unit
def test_player_text_substitution_names_who_came_on_and_who_left() -> None:
    text = player_text(
        names("a", "subst", read("in", "Мирзоев Тимур"), read("out", None, "unrecognized"))
    )

    assert (
        text == f"{ru.PLAYER_WENT_ON} Мирзоев Тимур; {ru.PLAYER_WENT_OFF} {ru.PLAYER_UNRECOGNIZED}"
    )


@pytest.mark.unit
def test_player_text_substitution_with_unchecked_names_has_one_note_at_the_end() -> None:
    text = player_text(
        names(
            "a", "subst", read("in", "Тимур Мирзоев", "unchecked"), read("out", "Иван", "unchecked")
        )
    )

    assert text.count(ru.PLAYER_UNCHECKED_MARK) == 1
    assert text.endswith(ru.PLAYER_UNCHECKED_MARK)


@pytest.mark.unit
def test_player_texts_join_events_to_plates_by_the_plate_id_and_kind() -> None:
    goal = candidate(EventType.GOAL, 200.0, ("score_change", "score_changes[0]"), ("plate", "g1"))
    foul = candidate(EventType.FOUL, 700.0, ("plate", "c1"))
    subst = candidate(EventType.SUBST, 600.0, ("plate", "s1"))
    plates = [
        names("g1", "goal", read("player", "Лебедев Роман")),
        names("c1", "card", read("player", "Нурланов Айдар")),
        names("s1", "subst", read("in", "А Б"), read("out", "В Г")),
    ]

    texts = player_texts([goal, foul, subst], plates)

    assert texts[(EventType.GOAL, 200.0)] == "Лебедев Роман"
    assert texts[(EventType.FOUL, 700.0)] == "Нурланов Айдар"
    assert texts[(EventType.SUBST, 600.0)].startswith(ru.PLAYER_WENT_ON)


@pytest.mark.unit
def test_player_texts_names_of_an_older_plates_file_do_not_land_on_another_plate() -> None:
    # The same id and type, but the plate was seen at another time: plates were recomputed.
    stale = names("g1", "goal", read("player", "Лебедев Роман")).model_copy(update={"t_v": 99.0})
    goal = candidate(EventType.GOAL, 200.0, ("plate", "g1"))  # the evidence says 10.0

    assert player_texts([goal], [stale]) == {}


@pytest.mark.unit
def test_player_texts_skip_events_without_a_matching_plate() -> None:
    plates = [names("g1", "goal", read("player", "Лебедев Роман"))]
    candidates = [
        candidate(EventType.SHOT, 10.0, ("model", "model:Shots on target")),  # no plate evidence
        candidate(
            EventType.SUBST, 20.0, ("plate", "g1")
        ),  # a goal plate cannot make a substitution
        candidate(EventType.GOAL, 30.0, ("plate", "nope")),  # plate without names
        candidate(EventType.SHOT, 40.0, ("plate", "g1")),  # a shot has no plate kind
        candidate(EventType.GOAL, 50.0),
    ]

    assert player_texts(candidates, plates) == {}
    assert player_texts([], []) == {}


@pytest.mark.unit
def test_to_row_shows_the_player_of_an_untouched_prediction_only() -> None:
    event = make_event("p:1", EventType.GOAL, 200.0)
    players = {(EventType.GOAL, 200.0): "Лебедев Роман"}

    assert to_row(event, players).player_text == "Лебедев Роман"
    assert to_row(make_event("p:1", EventType.GOAL, 201.0), players).player_text == ""  # moved
    assert to_row(make_event("p:1", EventType.SHOT, 200.0), players).player_text == ""  # retyped
    assert to_row(make_event("e:1", EventType.GOAL, 200.0, source=None), players).player_text == ""
    assert to_row(event).player_text == ""
    assert to_row(event, {}).player_text == ""


@pytest.mark.unit
def test_table_records_put_the_player_after_the_team() -> None:
    rows = to_rows(
        [make_event("p:1", EventType.GOAL, 200.0)], {(EventType.GOAL, 200.0): "Лебедев Роман"}
    )

    record = table_records(rows)[0]

    keys = list(record)
    assert keys[keys.index(ru.COLUMN_TEAM) + 1] == ru.COLUMN_PLAYER
    assert record[ru.COLUMN_PLAYER] == "Лебедев Роман"


def write_names(match_dir: Path, *plates: PlateNames) -> Path:
    result = PlateNamesResult(
        cache_key="k",
        plates_version="v",
        lineup=None,
        lineup_version=None,
        config=PlateNamesConfig(),
        names=list(plates),
        stats=PlateNamesStats(
            n_plates=len(plates), n_names=0, n_matched=0, n_unchecked=0, n_unrecognized=0
        ),
    )
    path = match_dir / "plate_names.json"
    path.write_text(result.model_dump_json(), encoding="utf-8")
    return path


@pytest.mark.unit
def test_load_player_texts_without_the_files_is_empty(tmp_path: Path) -> None:
    assert load_player_texts(tmp_path) == {}
    write_match_dir(tmp_path)  # candidates.json is there, plate_names.json is not
    assert load_player_texts(tmp_path) == {}


@pytest.mark.unit
def test_load_player_texts_joins_the_two_files_and_follows_changes(tmp_path: Path) -> None:
    write_match_dir(tmp_path)
    path = write_names(tmp_path, names("p1", "goal", read("player", "Лебедев Роман"), t_v=230.0))

    first = load_player_texts(tmp_path)
    assert list(first.values()) == ["Лебедев Роман"]

    write_names(tmp_path, names("p1", "goal", read("player", "Нурланов Айдар"), t_v=230.0))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))  # a new stamp
    assert list(load_player_texts(tmp_path).values()) == ["Нурланов Айдар"]


@pytest.mark.unit
def test_load_player_texts_invalid_names_file_is_empty_not_an_error(tmp_path: Path) -> None:
    write_match_dir(tmp_path)
    (tmp_path / "plate_names.json").write_text("{not json", encoding="utf-8")

    assert load_player_texts(tmp_path) == {}


@pytest.mark.unit
def test_app_table_shows_the_players_of_goal_substitution_and_card_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = write_match_dir(tmp_path)
    write_names(
        tmp_path,
        names("p1", "goal", read("player", "Лебедев Роман"), t_v=230.0),
        names(
            "p2", "subst", read("in", "Мирзоев Тимур"), read("out", "Стрельцов Антон"), t_v=600.0
        ),
        names("p3", "card", read("player", "Роман Лебедев", "unchecked"), t_v=700.0),
    )

    app = run_app(monkeypatch, db, "m1")

    assert not app.exception
    table = app.dataframe[0].value
    by_type = {row[ru.COLUMN_TYPE]: row[ru.COLUMN_PLAYER] for _, row in table.iterrows()}
    assert by_type["Гол"] == "Лебедев Роман"
    assert by_type["Замена"] == (
        f"{ru.PLAYER_WENT_ON} Мирзоев Тимур; {ru.PLAYER_WENT_OFF} Стрельцов Антон"
    )
    assert by_type["Фол"] == "Роман Лебедев" + ru.PLAYER_UNCHECKED_MARK


@pytest.mark.unit
def test_app_table_without_names_has_an_empty_player_column(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = write_match_dir(tmp_path)

    app: AppTest = run_app(monkeypatch, db, "m1")

    assert not app.exception
    assert set(app.dataframe[0].value[ru.COLUMN_PLAYER]) == {""}
