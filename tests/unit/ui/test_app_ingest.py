"""The "new match" block of the viewer (`AppTest`): start, refused second start, step list.

`subprocess.Popen` is replaced by a fake: no ingest process is started here.
"""

import os
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest
from tests.fixtures.ui_synth import write_events_db

from aicoach.i18n import ru
from aicoach.ingest.lock import LOCK_FILE
from aicoach.status import write_status
from aicoach.ui.ingest_jobs import LOG_FILE
from aicoach.ui.launch import StartArgs, app_script, build_streamlit_command

VID = "abcDEF_-123"  # invented id
URL = f"https://www.youtube.com/watch?v={VID}&t=10s"

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1"])


class FakePopen:
    """Records the started commands; `exit_code` None means the process still runs."""

    started: list["FakePopen"] = []
    exit_code: int | None = None

    # Any: Popen keyword arguments are mixed.
    def __init__(self, args: list[str], **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.pid = 4242
        FakePopen.started.append(self)

    def poll(self) -> int | None:
        return FakePopen.exit_code


@pytest.fixture
def popen(monkeypatch: pytest.MonkeyPatch) -> type[FakePopen]:
    FakePopen.started = []
    FakePopen.exit_code = None
    monkeypatch.setattr("aicoach.ui.ingest_jobs.subprocess.Popen", FakePopen)
    return FakePopen


def open_app(monkeypatch: pytest.MonkeyPatch, data_root: Path | None) -> AppTest:
    command = build_streamlit_command(StartArgs(data_root=data_root, templates=Path("tpl")))
    monkeypatch.setattr("sys.argv", [str(app_script()), *command[6:]])
    return AppTest.from_file(str(app_script()), default_timeout=20).run()


def submit(app: AppTest, url: str = URL) -> AppTest:
    app.text_input(key="ingest_url").input(url)
    return app.button(key="btn_ingest").click().run()


def sidebar_text(app: AppTest) -> str:
    parts = [str(t.value) for t in app.sidebar.text]
    for kind in (app.sidebar.success, app.sidebar.error, app.sidebar.info, app.sidebar.warning):
        parts += [str(m.value) for m in kind]
    return "\n".join(parts)


@pytest.mark.unit
def test_ingest_start_runs_cli_ingest_in_the_background_and_shows_the_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, popen: type[FakePopen]
) -> None:
    app = submit(open_app(monkeypatch, tmp_path))

    assert not app.exception
    assert len(popen.started) == 1
    args = popen.started[0].args
    assert args[1:5] == ["-m", "aicoach.cli", "ingest", f"https://www.youtube.com/watch?v={VID}"]
    assert args[args.index("--data-root") + 1] == str(tmp_path)
    assert args[args.index("--templates") + 1] == "tpl"
    assert (tmp_path / VID / LOG_FILE).is_file()
    text = sidebar_text(app)
    assert ru.INGEST_STARTED.format(log=tmp_path / VID / LOG_FILE) in text
    pending = ru.INGEST_STATE_LABELS["pending"]
    assert f"{ru.INGEST_STEP_LABELS['download']}: {pending}" in text
    assert [i.value for i in app.info] == [ru.NO_MATCH_OPEN]  # the main page is not blocked


@pytest.mark.unit
def test_ingest_second_start_while_running_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, popen: type[FakePopen]
) -> None:
    app = submit(open_app(monkeypatch, tmp_path))

    app = app.button(key="btn_ingest").click().run()

    assert len(popen.started) == 1
    assert ru.INGEST_BUSY.format(pid=4242) in sidebar_text(app)


@pytest.mark.unit
def test_ingest_start_refused_while_another_process_holds_the_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, popen: type[FakePopen]
) -> None:
    (tmp_path / VID).mkdir()
    (tmp_path / VID / LOCK_FILE).write_text(str(os.getpid()))  # this process is alive

    app = submit(open_app(monkeypatch, tmp_path))

    assert popen.started == []
    assert ru.INGEST_BUSY.format(pid=os.getpid()) in sidebar_text(app)


@pytest.mark.unit
def test_ingest_bad_link_is_refused_before_starting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, popen: type[FakePopen]
) -> None:
    app = submit(open_app(monkeypatch, tmp_path / "data"), "https://example.com/v")

    assert popen.started == []
    assert "Ссылка не принята" in sidebar_text(app)
    assert not (tmp_path / "data").exists()


@pytest.mark.unit
@pytest.mark.usefixtures("popen")
def test_ingest_steps_show_the_states_and_progress_from_status_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    match_dir = tmp_path / VID
    match_dir.mkdir()
    write_status(match_dir, "download", "done", 1.0)
    write_status(match_dir, "scoreboard", "running", 0.5)

    app = submit(open_app(monkeypatch, tmp_path))

    text = sidebar_text(app)
    labels, states = ru.INGEST_STEP_LABELS, ru.INGEST_STATE_LABELS
    assert f"{labels['download']}: {states['done']}" in text
    assert f"{labels['scoreboard']}: {states['running']} (50%)" in text
    assert f"{labels['fuse']}: {states['pending']}" in text


@pytest.mark.unit
def test_ingest_waiting_for_tdeed_shows_what_to_bring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, popen: type[FakePopen]
) -> None:
    app = submit(open_app(monkeypatch, tmp_path))
    match_dir = tmp_path / VID
    write_status(match_dir, "tdeed", "waiting", 0.0)
    write_status(match_dir, "ingest", "waiting", 4 / 6)
    popen.exit_code = 3

    app = app.run()

    text = sidebar_text(app)
    assert f"{ru.INGEST_STEP_LABELS['tdeed']}: {ru.INGEST_STATE_LABELS['waiting']}" in text
    assert "tdeed_<имя>.json" in text
    assert str(match_dir) in text
    assert ru.TDEED_RESUME_UI in text


@pytest.mark.unit
def test_ingest_failed_and_vanished_processes_are_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, popen: type[FakePopen]
) -> None:
    app = submit(open_app(monkeypatch, tmp_path))
    match_dir = tmp_path / VID
    popen.exit_code = 2
    write_status(match_dir, "ingest", "failed", 0.0, error="stage download failed: 403")

    app = app.run()
    assert ru.INGEST_FAILED.format(error="stage download failed: 403") in sidebar_text(app)

    write_status(match_dir, "ingest", "running", 0.1)  # left behind by a killed process
    app = app.run()
    assert ru.INGEST_PROCESS_GONE.format(log=match_dir / LOG_FILE) in sidebar_text(app)


@pytest.mark.unit
def test_ingest_done_opens_the_new_match_in_the_viewer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, popen: type[FakePopen]
) -> None:
    app = submit(open_app(monkeypatch, tmp_path))
    match_dir = tmp_path / VID
    write_events_db(match_dir / "aicoach.db", match_id=VID)
    write_status(match_dir, "ingest", "done", 1.0)
    popen.exit_code = 0
    app = app.run()
    assert ru.INGEST_DONE in sidebar_text(app)

    app = app.button(key="btn_open_match").click().run()

    assert not app.exception
    assert not app.info  # "no match open" is gone
    assert len(app.dataframe[0].value) == 5
    assert any(VID in str(m.value) for m in app.markdown)  # the YouTube link of the new match


@pytest.mark.unit
def test_ingest_block_without_data_root_explains_how_to_enable_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = open_app(monkeypatch, None)

    assert not app.exception
    assert any(ru.INGEST_NO_ROOT in c.value for c in app.sidebar.caption)
    assert not app.button  # no start button without a data root
