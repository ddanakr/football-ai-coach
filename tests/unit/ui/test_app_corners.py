"""The corner tags in the viewer (`AppTest`): the column, the filters and the team block (D-078)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.fixtures.corner_synth import write_db, write_tag_files
from tests.unit.ui.test_app_edits import VID, table

from aicoach.domain.events import EventType
from aicoach.i18n import ru
from aicoach.teams.store import OVERRIDE_FILE, read_override
from aicoach.ui.launch import AppArgs, app_script, build_streamlit_command

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1"])
MATCH = "m1"


def start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, tags: bool = True) -> AppTest:
    db = write_db(tmp_path)
    if tags:
        write_tag_files(tmp_path)
    args = AppArgs(db=db, match_id=MATCH, youtube_id=VID, match_dir=tmp_path)
    monkeypatch.setattr("sys.argv", [str(app_script()), *build_streamlit_command(args)[6:]])
    return AppTest.from_file(str(app_script()), default_timeout=20).run()


def corner_cells(app: AppTest) -> list[str]:
    frame = app.dataframe[0].value
    return [str(v) for v in frame[ru.COLUMN_CORNER]]


@pytest.mark.unit
def test_the_table_has_the_corner_tags_column_with_the_not_verified_mark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = start(monkeypatch, tmp_path)

    assert not app.exception
    cells = corner_cells(app)
    corners = [c for c in cells if c]
    assert len(corners) == 3
    assert all(c.endswith("(по клипу, не проверено)") for c in corners)
    assert any("подаёт: Актобе" in c for c in corners)
    assert any("подаёт: Кайрат" in c for c in corners)
    assert any("команда не определена" in c for c in corners)


@pytest.mark.unit
def test_without_tags_there_is_no_column_and_the_filters_say_how_to_make_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = start(monkeypatch, tmp_path, tags=False)

    assert not app.exception
    assert ru.COLUMN_CORNER not in list(app.dataframe[0].value.columns)
    assert ru.CORNER_FILTERS_NO_TAGS in [str(c.value) for c in app.caption]


@pytest.mark.unit
def test_the_team_filter_narrows_the_corners_and_keeps_other_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = start(monkeypatch, tmp_path)
    before = len(table(app))

    app.selectbox(key="corner_filter_team").select("home").run()

    rows = table(app)
    assert not app.exception
    assert len(rows) == before - 2  # two corners of three left the table
    assert sum(r[1] == ru.TYPE_LABELS[EventType.GOAL] for r in rows) == 4  # goals stay
    assert sum("подаёт: Актобе" in r[-1] or "подаёт: Актобе" in " ".join(r) for r in rows) == 1


@pytest.mark.unit
def test_a_field_filter_and_the_unknown_team_filter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = start(monkeypatch, tmp_path)
    app.selectbox(key="corner_filter_outcome").select("shot").run()

    assert not app.exception
    assert sum("исход: удар" in " ".join(r) for r in table(app)) == 1

    app.selectbox(key="corner_filter_outcome").select(None).run()
    app.selectbox(key="corner_filter_team").select("unknown").run()

    assert sum("команда не определена" in " ".join(r) for r in table(app)) == 1


@pytest.mark.unit
def test_the_team_block_shows_the_names_and_kits_and_saves_a_manual_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = start(monkeypatch, tmp_path)
    shown = [str(t.value) for t in app.text]
    assert any("Актобе" in line and "белый" in line for line in shown)

    app.text_input(key="teams_name_home").set_value("FC Aktobe")
    app.selectbox(key="teams_kit_away").select("blue")
    app.button(key="btn_teams_save").click().run()

    assert not app.exception
    saved = read_override(tmp_path)
    assert (saved.home.name, saved.home.kit) == ("FC Aktobe", None)
    assert saved.away.kit == "blue"
    assert ru.TEAMS_SAVED in [str(s.value) for s in app.success]
    assert any("FC Aktobe" in str(t.value) and "вручную" in str(t.value) for t in app.text)


@pytest.mark.unit
def test_reset_removes_the_override_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = start(monkeypatch, tmp_path)
    app.text_input(key="teams_name_home").set_value("FC Aktobe")
    app.button(key="btn_teams_save").click().run()
    assert (tmp_path / OVERRIDE_FILE).is_file()

    app.button(key="btn_teams_reset").click().run()

    assert not (tmp_path / OVERRIDE_FILE).exists()
    assert app.text_input(key="teams_name_home").value == ""


@pytest.mark.unit
def test_a_manual_kit_changes_who_is_the_attacking_team_in_the_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = start(monkeypatch, tmp_path)
    app.selectbox(key="teams_kit_home").select("red")
    app.button(key="btn_teams_save").click().run()

    cells = corner_cells(app)

    # Both sides now wear red: no corner can be given to a side.
    assert sum("команда не определена" in c for c in cells) == 3
