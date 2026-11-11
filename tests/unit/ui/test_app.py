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
def test_app_lists_all_events_and_shows_the_start_of_the_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)

    app = run_app(monkeypatch, db)

    assert not app.exception
    assert len(app.button) == 5
    text = markdown_text(app)
    assert f"https://www.youtube.com/watch?v={VID}" in text
    assert "&t=" not in text
    assert ru.SHOWN.format(shown=5, total=5) in [c.value for c in app.caption]


@pytest.mark.unit
def test_click_on_play_selects_the_event_and_links_to_five_seconds_before(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)
    app = run_app(monkeypatch, db)

    app.button[0].click().run()

    assert not app.exception
    text = markdown_text(app)
    assert f"https://www.youtube.com/watch?v={VID}&t=595s" in text
    assert ru.PLAYING.format(type="Гол", time="10:00") in text


@pytest.mark.unit
def test_click_on_another_event_moves_the_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)
    app = run_app(monkeypatch, db)

    app.button[0].click().run()
    app.button[3].click().run()

    text = markdown_text(app)
    assert "&t=3995s" in text
    assert "&t=595s" not in text


@pytest.mark.unit
def test_player_variant_st_video_starts_at_the_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)
    app = run_app(monkeypatch, db)

    app.sidebar.radio[0].set_value(ru.PLAYER_ST_VIDEO).run()
    app.button[2].click().run()

    assert not app.exception
    assert "&t=1195s" in markdown_text(app)


@pytest.mark.unit
def test_filter_by_type_and_half_and_score(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)
    app = run_app(monkeypatch, db)

    app.sidebar.radio[1].set_value(2).run()
    assert len(app.button) == 2  # the corner and the substitution of the second half

    app.sidebar.slider[0].set_value(0.7).run()
    assert len(app.button) == 1  # the corner (0.60) is below, the substitution has no score

    app.sidebar.radio[1].set_value(None).run()
    app.sidebar.multiselect[0].set_value(["goal", "shot"]).run()
    assert len(app.button) == 1


@pytest.mark.unit
def test_filters_that_leave_nothing_show_a_message_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db = tmp_path / "e.db"
    write_events_db(db)
    app = run_app(monkeypatch, db)

    app.sidebar.multiselect[0].set_value([]).run()

    assert not app.exception
    assert len(app.button) == 0
    assert [i.value for i in app.info] == [ru.NOTHING_MATCHES]
