"""Smoke tests of the Streamlit script through `AppTest` (no browser, no network)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.fixtures.ui_synth import write_events_db

from aicoach.i18n import ru
from aicoach.storage import init_db
from aicoach.ui.launch import AppArgs, app_script, build_streamlit_command

VID = "abcDEF_-123"

# AppTest runs an asyncio loop, and on Windows the loop needs a loopback socket pair.
# Only loopback is allowed; the network stays closed.
pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1"])


def run_app(monkeypatch: pytest.MonkeyPatch, db: Path, match_id: str = "m1") -> AppTest:
    """Run `app.py` with the arguments `cli ui` would give it."""
    command = build_streamlit_command(AppArgs(db=db, match_id=match_id, youtube_id=VID))
    monkeypatch.setattr("sys.argv", [str(app_script()), *command[6:]])
    app = AppTest.from_file(str(app_script()), default_timeout=20)
    return app.run()


def markdown_text(app: AppTest) -> str:
    return "\n".join(str(m.value) for m in app.markdown)


def table_rows(app: AppTest) -> int:
    """Number of rows in the events table."""
    return len(app.dataframe[0].value)


def pick_row(app: AppTest, index: int) -> AppTest:
    """Select a table row the way a click on it does: through the table's selection state."""
    app.session_state["events_table"] = {"selection": {"rows": [index], "columns": []}}
    return app.run()


@pytest.mark.unit
def test_app_without_database_tells_to_run_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = run_app(monkeypatch, tmp_path / "none.db")

    assert not app.exception
    assert len(app.error) == 1
    assert "cli run" in app.error[0].value
    assert not (tmp_path / "none.db").exists()


@pytest.mark.unit
def test_app_with_empty_database_says_there_are_no_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    init_db(db)

    app = run_app(monkeypatch, db)

    assert not app.exception
    assert len(app.warning) == 1
    assert "m1" in app.warning[0].value


@pytest.mark.unit
def test_app_with_unreadable_database_shows_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "bad.db"
    db.write_text("not a database, only text" * 20, encoding="utf-8")

    app = run_app(monkeypatch, db)

    assert not app.exception
    assert len(app.error) == 1
    assert str(db) in app.error[0].value


@pytest.mark.unit
def test_app_with_bad_arguments_shows_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["app.py"])

    app = AppTest.from_file(str(app_script()), default_timeout=20).run()

    assert not app.exception
    assert len(app.error) == 1
    assert "cli ui" in app.error[0].value


@pytest.mark.unit
def test_app_lists_all_events_in_one_table_and_shows_the_start_of_the_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)

    app = run_app(monkeypatch, db)

    assert not app.exception
    assert len(app.dataframe) == 1
    assert table_rows(app) == 5
    assert list(app.dataframe[0].value.columns) == [
        ru.COLUMN_TIME,
        ru.COLUMN_TYPE,
        ru.COLUMN_HALF,
        ru.COLUMN_SCORE,
        ru.COLUMN_SOURCE,
        ru.COLUMN_TEAM,
    ]
    text = markdown_text(app)
    assert f"https://www.youtube.com/watch?v={VID}" in text
    assert "&t=" not in text
    assert ru.SHOWN.format(shown=5, total=5) in [c.value for c in app.caption]


@pytest.mark.unit
def test_table_shows_mmss_russian_type_and_empty_unknowns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)

    table = run_app(monkeypatch, db).dataframe[0].value

    first = table.iloc[0]
    assert (first[ru.COLUMN_TIME], first[ru.COLUMN_TYPE]) == ("10:00", "Гол")
    assert (first[ru.COLUMN_HALF], first[ru.COLUMN_SOURCE]) == ("1", "табло и плашки")
    assert first[ru.COLUMN_TEAM] == "хозяева"
    assert table.iloc[1][ru.COLUMN_TEAM] == ""  # a model event has no team


@pytest.mark.unit
def test_picking_a_row_links_to_five_seconds_before_the_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)
    app = run_app(monkeypatch, db)

    pick_row(app, 0)

    assert not app.exception
    text = markdown_text(app)
    assert f"https://www.youtube.com/watch?v={VID}&t=595s" in text
    assert ru.PLAYING.format(type="Гол", time="10:00") in text


@pytest.mark.unit
def test_picking_another_row_moves_the_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)
    app = run_app(monkeypatch, db)

    pick_row(app, 0)
    pick_row(app, 3)

    text = markdown_text(app)
    assert "&t=3995s" in text
    assert "&t=595s" not in text


@pytest.mark.unit
def test_player_variant_st_video_starts_at_the_picked_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)
    app = run_app(monkeypatch, db)

    app.sidebar.radio[0].set_value(ru.PLAYER_ST_VIDEO).run()
    pick_row(app, 2)

    assert not app.exception
    assert "&t=1195s" in markdown_text(app)


@pytest.mark.unit
def test_changing_the_variant_keeps_the_picked_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)
    app = run_app(monkeypatch, db)
    pick_row(app, 1)

    app.sidebar.radio[0].set_value(ru.PLAYER_ST_VIDEO).run()

    assert "&t=895s" in markdown_text(app)


@pytest.mark.unit
def test_filter_by_type_and_half_and_score(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)
    app = run_app(monkeypatch, db)

    app.sidebar.radio[1].set_value(2).run()
    assert table_rows(app) == 2  # the corner and the substitution of the second half

    app.sidebar.slider[0].set_value(0.7).run()
    assert table_rows(app) == 1  # the corner (0.60) is below, the substitution has no score

    app.sidebar.radio[1].set_value(None).run()
    app.sidebar.multiselect[0].set_value(["goal", "shot"]).run()
    assert table_rows(app) == 1


@pytest.mark.unit
def test_pick_after_filtering_maps_to_the_filtered_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)
    app = run_app(monkeypatch, db)
    app.sidebar.radio[1].set_value(2).run()

    pick_row(app, 1)  # second row of the filtered table: the substitution at 4300 s

    assert "&t=4295s" in markdown_text(app)


@pytest.mark.unit
def test_filters_that_leave_nothing_show_a_message_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)
    app = run_app(monkeypatch, db)

    app.sidebar.multiselect[0].set_value([]).run()

    assert not app.exception
    assert len(app.dataframe) == 0
    assert [i.value for i in app.info] == [ru.NOTHING_MATCHES]
