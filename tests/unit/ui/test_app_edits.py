"""Manual edits through the Streamlit script (`AppTest`): every action, errors, no double writes.

The match directory is a real result of the fusion stage on synthetic inputs, so the half
bounds and the video length that the edits are checked against are the real files.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.fixtures.ui_synth import write_events_db, write_match_dir

from aicoach.domain.events import EventType
from aicoach.fusion.stage import run_fuse
from aicoach.i18n import ru
from aicoach.storage import connect, init_db
from aicoach.ui.data import count_edits, load_events
from aicoach.ui.launch import AppArgs, app_script, build_streamlit_command

VID = "abcDEF_-123"
MATCH = "m1"

# Same loopback exception as `test_app.py`: AppTest needs an asyncio socket pair on Windows.
pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1"])


def start(monkeypatch: pytest.MonkeyPatch, db: Path, match_dir: Path) -> AppTest:
    """Run `app.py` with the arguments `cli ui` would give it."""
    args = AppArgs(db=db, match_id=MATCH, youtube_id=VID, match_dir=match_dir)
    monkeypatch.setattr("sys.argv", [str(app_script()), *build_streamlit_command(args)[6:]])
    return AppTest.from_file(str(app_script()), default_timeout=20).run()


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    """The page over a synthetic match: goal 03:20, substitution 10:00, foul 11:18."""
    return start(monkeypatch, write_match_dir(tmp_path, MATCH), tmp_path)


def pick(app: AppTest, index: int) -> AppTest:
    """Pick a table row the way a click does: through the table's selection state.

    The selection is cleared first, as the browser table does when its rows change; without it
    the test would "click" the index that the previous pick left in the state.
    """
    app.session_state["events_table"] = {"selection": {"rows": [], "columns": []}}
    app.run()
    app.session_state["events_table"] = {"selection": {"rows": [index], "columns": []}}
    return app.run()


def table(app: AppTest) -> list[list[str]]:
    """The events table as rows of text: time, type, ..., edit mark."""
    frame = app.dataframe[0].value
    return [[str(v) for v in row] for row in frame.itertuples(index=False)]


def edit_marks(app: AppTest) -> list[str]:
    return [row[-1] for row in table(app)]


def types(app: AppTest) -> list[str]:
    return [row[1] for row in table(app)]


def journal(tmp_path: Path) -> list[tuple[str, str | None]]:
    """`(action, target)` of every row of the edits journal, oldest first."""
    with connect(tmp_path / "aicoach.db") as conn:
        rows = conn.execute("SELECT action, target FROM edits ORDER BY id").fetchall()
    return [(r["action"], r["target"]) for r in rows]


def messages(app: AppTest) -> list[str]:
    return [str(m.value) for m in (*app.success, *app.error)]


@pytest.mark.unit
def test_no_edit_controls_for_a_selection_that_does_not_exist_yet(app: AppTest) -> None:
    assert not app.exception
    assert [c.value for c in app.caption if c.value == ru.EDIT_PICK_FIRST] == [ru.EDIT_PICK_FIRST]
    assert not [b for b in app.button if b.key == "btn_move"]
    assert app.button(key="btn_undo").disabled  # no edits yet
    assert ru.EDIT_NO_EDITS in [c.value for c in app.caption]


@pytest.mark.unit
def test_delete_hides_the_event_and_logs_one_edit(app: AppTest, tmp_path: Path) -> None:
    ids = [e.event_id for e in load_events(tmp_path / "aicoach.db", MATCH)]
    pick(app, 0)

    app.button(key="btn_delete").click().run()

    assert not app.exception
    assert len(table(app)) == 2
    assert [e.event_id for e in load_events(tmp_path / "aicoach.db", MATCH)] == ids[1:]
    assert journal(tmp_path) == [("delete", ids[0])]
    assert "удалено" in messages(app)[0]
    assert not app.button(key="btn_undo").disabled


@pytest.mark.unit
def test_rerunning_the_page_never_writes_an_edit_twice(app: AppTest, tmp_path: Path) -> None:
    pick(app, 0)
    app.button(key="btn_delete").click().run()

    for _ in range(3):
        app.run()
    app.sidebar.slider[0].set_value(0.1).run()
    app.sidebar.radio[0].set_value(1).run()

    assert count_edits(tmp_path / "aicoach.db", MATCH) == 1
    assert len(journal(tmp_path)) == 1
    assert messages(app) == []  # the message is shown once


@pytest.mark.unit
def test_message_is_shown_once_after_the_click_and_then_gone(app: AppTest) -> None:
    pick(app, 0)

    app.button(key="btn_delete").click().run()
    assert len(app.success) == 1

    assert len(app.run().success) == 0


@pytest.mark.unit
def test_retype_changes_the_type_and_marks_the_row(app: AppTest, tmp_path: Path) -> None:
    pick(app, 1)  # the substitution
    assert types(app)[1] == "Замена"

    app.selectbox[0].set_value(EventType.CORNER)
    app.button(key="btn_retype").click().run()

    assert not app.exception
    assert types(app)[1] == "Угловой"
    assert edit_marks(app) == ["", ru.MARK_EDITED, ""]
    assert [a for a, _ in journal(tmp_path)] == ["retype"]
    assert messages(app) == [ru.EDIT_RETYPED.format(old="Замена", new="Угловой")]


@pytest.mark.unit
def test_retype_to_the_same_type_shows_an_error_and_writes_nothing(
    app: AppTest, tmp_path: Path
) -> None:
    pick(app, 1)

    app.button(key="btn_retype").click().run()

    assert [str(e.value) for e in app.error] == [ru.ERR_TYPE_UNCHANGED]
    assert journal(tmp_path) == []


@pytest.mark.unit
def test_move_changes_the_time_and_keeps_the_event_chosen(app: AppTest, tmp_path: Path) -> None:
    pick(app, 0)  # the goal at 03:20
    time_field = next(t for t in app.text_input if t.key and t.key.startswith("move_time_"))
    assert time_field.value == "03:20"  # the default is the time of the event

    time_field.set_value("05:00")
    app.button(key="btn_move").click().run()

    assert not app.exception
    assert [row[0] for row in table(app)] == ["05:00", "10:00", "11:18"]
    assert edit_marks(app) == [ru.MARK_EDITED, "", ""]
    assert messages(app) == [ru.EDIT_MOVED.format(type="Гол", time="05:00", half=1)]
    selected = [
        m for m in app.markdown if ru.PLAYING.format(type="Гол", time="05:00") in str(m.value)
    ]
    assert selected  # the player and the panel follow the event to its new time
    assert journal(tmp_path)[0][0] == "move"
    new_field = next(t for t in app.text_input if t.key and t.key.startswith("move_time_"))
    assert new_field.value == "05:00"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("abc", "мм:сс"),
        ("-1:00", "мм:сс"),
        ("20:00", "вне таймов"),  # the break between the halves
        ("00:10", "вне таймов"),  # before the first half
        ("99:00", "дальше конца видео"),
    ],
)
def test_move_to_a_bad_time_shows_an_error_and_writes_nothing(
    app: AppTest, tmp_path: Path, text: str, error: str
) -> None:
    pick(app, 0)
    next(t for t in app.text_input if t.key and t.key.startswith("move_time_")).set_value(text)

    app.button(key="btn_move").click().run()

    assert not app.exception
    assert len(app.error) == 1
    assert error in str(app.error[0].value)
    assert journal(tmp_path) == []
    assert table(app)[0][0] == "03:20"


@pytest.mark.unit
def test_move_into_the_other_half_shows_the_new_half_in_the_table(app: AppTest) -> None:
    pick(app, 0)
    next(t for t in app.text_input if t.key and t.key.startswith("move_time_")).set_value("30:00")

    app.button(key="btn_move").click().run()

    row = next(r for r in table(app) if r[0] == "30:00")
    assert row[2] == "2"  # the half column follows the time, not the stored half of the prediction


@pytest.mark.unit
def test_add_creates_a_manual_event_and_selects_it(app: AppTest, tmp_path: Path) -> None:
    pick(app, 2)  # the form starts with the time of the chosen event
    time_field = next(t for t in app.text_input if t.key and t.key.startswith("add_time_"))
    assert time_field.value == "11:18"
    time_field.set_value("04:00")
    next(s for s in app.selectbox if s.key and s.key.startswith("add_type_")).set_value(
        EventType.SHOT
    )

    app.button(key="btn_add").click().run()

    assert not app.exception
    assert [row[0] for row in table(app)] == ["03:20", "04:00", "10:00", "11:18"]
    assert table(app)[1][1] == f"Удар ({ru.STATUS_LABELS['manual']})"
    assert edit_marks(app)[1] == ru.MARK_ADDED
    assert table(app)[1][2] == "1"  # the half is derived from the time
    assert messages(app) == [ru.EDIT_ADDED.format(type="Удар", time="04:00", half=1)]
    assert [a for a, _ in journal(tmp_path)] == ["add"]
    assert any(ru.PLAYING.format(type="Удар", time="04:00") in str(m.value) for m in app.markdown)


@pytest.mark.unit
def test_add_without_a_chosen_event_needs_a_typed_time(app: AppTest, tmp_path: Path) -> None:
    app.button(key="btn_add").click().run()

    assert [str(e.value) for e in app.error] == [ru.ERR_TIME_FORMAT]
    assert journal(tmp_path) == []

    next(t for t in app.text_input if t.key and t.key.startswith("add_time_")).set_value("25:00")
    app.button(key="btn_add").click().run()

    assert len(table(app)) == 4
    assert [a for a, _ in journal(tmp_path)] == ["add"]


@pytest.mark.unit
@pytest.mark.parametrize("text", ["20:00", "00:10", "70:00", "x"])
def test_add_at_a_bad_time_is_refused(app: AppTest, tmp_path: Path, text: str) -> None:
    next(t for t in app.text_input if t.key and t.key.startswith("add_time_")).set_value(text)

    app.button(key="btn_add").click().run()

    assert len(app.error) == 1
    assert journal(tmp_path) == []


@pytest.mark.unit
def test_undo_takes_back_the_last_edit_and_then_the_one_before(
    app: AppTest, tmp_path: Path
) -> None:
    pick(app, 0)
    app.button(key="btn_delete").click().run()  # the goal is gone, the substitution is row 0
    pick(app, 0)
    app.selectbox[0].set_value(EventType.CORNER)
    app.button(key="btn_retype").click().run()
    assert types(app) == ["Угловой", "Фол"]

    app.button(key="btn_undo").click().run()
    assert types(app) == ["Замена", "Фол"]
    assert messages(app) == [ru.EDIT_UNDONE]
    assert not app.button(key="btn_undo").disabled

    app.button(key="btn_undo").click().run()
    assert types(app) == ["Гол", "Замена", "Фол"]
    assert app.button(key="btn_undo").disabled
    # The journal only grows: two edits and two undo records.
    assert [a for a, _ in journal(tmp_path)] == ["delete", "retype", "undo", "undo"]


@pytest.mark.unit
def test_undo_of_an_add_removes_the_event_and_clears_the_choice(
    app: AppTest, tmp_path: Path
) -> None:
    next(t for t in app.text_input if t.key and t.key.startswith("add_time_")).set_value("04:00")
    app.button(key="btn_add").click().run()
    assert len(table(app)) == 4

    app.button(key="btn_undo").click().run()

    assert not app.exception
    assert len(table(app)) == 3
    assert count_edits(tmp_path / "aicoach.db", MATCH) == 0
    assert ru.EDIT_PICK_FIRST in [c.value for c in app.caption]


@pytest.mark.unit
def test_choice_survives_an_edit_that_reorders_the_table(app: AppTest) -> None:
    pick(app, 0)  # the goal at 03:20
    next(t for t in app.text_input if t.key and t.key.startswith("move_time_")).set_value("11:30")
    app.button(key="btn_move").click().run()

    # The goal now stands last; the table index the page reported before the move is stale,
    # but the panel still shows the goal, not the event that took row 0.
    assert [row[0] for row in table(app)][-1] == "11:30"
    assert any(
        ru.EDIT_SELECTED.format(type="Гол", time="11:30") in str(m.value) for m in app.markdown
    )


@pytest.mark.unit
def test_deleting_every_event_keeps_undo_available(app: AppTest) -> None:
    for _ in range(3):
        pick(app, 0)
        app.button(key="btn_delete").click().run()

    assert not app.exception
    assert [w.value for w in app.warning] == [ru.NO_EVENTS.format(match_id=MATCH)]
    assert not app.button(key="btn_undo").disabled

    app.button(key="btn_undo").click().run()

    assert len(table(app)) == 1


@pytest.mark.unit
def test_edit_without_the_stage_files_is_refused_with_a_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    write_events_db(db, MATCH)  # no candidates.json and scoreboard.json next to it
    app = start(monkeypatch, db, tmp_path)
    pick(app, 0)

    next(t for t in app.text_input if t.key and t.key.startswith("move_time_")).set_value("11:00")
    app.button(key="btn_move").click().run()

    assert [str(e.value) for e in app.error] == [ru.ERR_NO_BOUNDS]
    assert count_edits(db, MATCH) == 0
    # Deleting needs no bounds: it still works.
    app.button(key="btn_delete").click().run()
    assert count_edits(db, MATCH) == 1


@pytest.mark.unit
def test_empty_database_has_no_edit_panel_and_stays_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    init_db(db)

    app = start(monkeypatch, db, tmp_path)

    assert not app.exception
    assert len(app.warning) == 1
    assert not [b for b in app.button if b.key == "btn_undo"]
    assert count_edits(db, MATCH) == 0


@pytest.mark.unit
def test_a_new_fusion_run_replacing_the_events_gives_a_message_not_a_crash(
    app: AppTest, tmp_path: Path
) -> None:
    pick(app, 0)
    # The viewer still shows the old ids; a forced run with the edit guard off replaces them.
    run_fuse(tmp_path, MATCH, force=True, discard_edits=True, run_id="run2")
    app.button(key="btn_delete").click().run()

    assert not app.exception
    assert [str(e.value) for e in app.error] == [ru.EDIT_EVENT_GONE]
    assert app.session_state["selected_event_id"] is None  # the page dropped the stale choice
