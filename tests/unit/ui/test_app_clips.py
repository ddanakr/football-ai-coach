"""Clip export from the viewer (`AppTest`): button state, result kept across reruns, errors.

`aicoach.clips.export_clip` is replaced by a fake: FFmpeg is not run here (the real run is in
`tests/integration/test_ffmpeg_clip_export.py`).
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.fixtures.ui_synth import write_match_dir
from tests.unit.ui.test_app_edits import MATCH, VID, pick

from aicoach.errors import ExportError
from aicoach.i18n import ru
from aicoach.storage import CurrentEvent
from aicoach.ui.data import load_events
from aicoach.ui.launch import AppArgs, app_script, build_streamlit_command

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1"])


class FakeExport:
    """Stands in for `export_clip`: records the call and writes a small file, or fails."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.raises = raises
        self.calls: list[tuple[Path, CurrentEvent, float, float]] = []

    def __call__(self, video: Path, event: CurrentEvent, before: float, after: float) -> Path:
        self.calls.append((video, event, before, after))
        if self.raises is not None:
            raise self.raises
        path = video.parent / "clips" / f"{event.event_id.replace(':', '')}.mp4"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fake mp4")
        return path


def start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, video: Path | None) -> AppTest:
    args = AppArgs(
        db=write_match_dir(tmp_path, MATCH),
        match_id=MATCH,
        youtube_id=VID,
        match_dir=tmp_path,
        video=video,
    )
    monkeypatch.setattr("sys.argv", [str(app_script()), *build_streamlit_command(args)[6:]])
    return AppTest.from_file(str(app_script()), default_timeout=20).run()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeExport:
    export = FakeExport()
    monkeypatch.setattr("aicoach.clips.export_clip", export)
    return export


def clip_downloads(app: AppTest) -> list[object]:
    """The download buttons of the clip (the one of the labels is always on the page)."""
    return [b for b in app.get("download_button") if b.proto.id and "clip" in str(b.key)]


def captions(app: AppTest) -> list[str]:
    return [str(c.value) for c in app.caption]


@pytest.mark.unit
def test_clip_button_disabled_until_an_event_is_picked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake: FakeExport
) -> None:
    app = start(monkeypatch, tmp_path, tmp_path / "src.mp4")
    assert not app.exception
    assert app.button(key="btn_clip").disabled
    assert ru.CLIP_PICK_FIRST in captions(app)
    assert [app.number_input(key=k).value for k in ("clip_before", "clip_after")] == [8.0, 12.0]
    assert fake.calls == []


@pytest.mark.unit
@pytest.mark.usefixtures("fake")
def test_clip_button_disabled_without_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = pick(start(monkeypatch, tmp_path, None), 0)
    assert not app.exception
    assert app.button(key="btn_clip").disabled
    assert ru.CLIP_NO_VIDEO in captions(app)


@pytest.mark.unit
def test_clip_export_shows_clip_and_keeps_it_and_the_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake: FakeExport
) -> None:
    video = tmp_path / "src.mp4"
    app = pick(start(monkeypatch, tmp_path, video), 1)
    app.number_input(key="clip_before").set_value(20.0)
    app.number_input(key="clip_after").set_value(10.0).run()

    app.button(key="btn_clip").click().run()

    assert not app.exception
    ((called_video, event, before, after),) = fake.calls
    assert (called_video, before, after) == (video, 20.0, 10.0)
    events = load_events(tmp_path / "aicoach.db", MATCH)
    assert event.event_id == events[1].event_id
    assert len(app.get("video")) == 2  # the player and the clip
    assert len(clip_downloads(app)) == 1
    assert any(c.startswith("Клип: ") and "20 с до и 10 с после" in c for c in captions(app))

    app.run()  # a rerun (any click elsewhere) keeps the clip and the chosen event
    assert len(clip_downloads(app)) == 1
    assert app.session_state["selected_event_id"] == events[1].event_id
    assert len(fake.calls) == 1  # nothing is exported again by a rerun


@pytest.mark.unit
def test_clip_export_error_is_shown_without_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "aicoach.clips.export_clip", FakeExport(ExportError("FFmpeg failed (exit code 1): boom"))
    )
    app = pick(start(monkeypatch, tmp_path, tmp_path / "src.mp4"), 0)

    app.button(key="btn_clip").click().run()

    assert not app.exception
    assert [e.value for e in app.error] == [
        ru.CLIP_FAILED.format(error="FFmpeg failed (exit code 1): boom")
    ]
    assert clip_downloads(app) == []


@pytest.mark.unit
@pytest.mark.usefixtures("fake")
def test_clip_file_removed_after_export_shows_error_and_forgets_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = pick(start(monkeypatch, tmp_path, tmp_path / "src.mp4"), 0)
    app.button(key="btn_clip").click().run()
    for clip in (tmp_path / "clips").iterdir():
        clip.unlink()

    app.run()

    assert [e.value for e in app.error] == [ru.CLIP_FILE_GONE.format(name="p1.mp4")]
    app.run()
    assert app.error == []
