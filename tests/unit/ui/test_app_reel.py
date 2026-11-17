"""The "Reel by category" and "Download the labels" blocks of the viewer (`AppTest`): state of the
button, the result kept across reruns, errors, the labels download. `build_reel` is replaced by a
fake: FFmpeg is not run here (the real run is in `tests/integration/test_reel_ffmpeg.py`)."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.unit.ui.test_app_clips import captions, start

from aicoach.agent.context import EventFacts
from aicoach.errors import ExportError
from aicoach.i18n import ru
from aicoach.reel import ReelFilter, ReelOptions, ReelResult, select_events
from aicoach.ui.reel_panel import BUTTON_KEY

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1"])


class FakeBuild:
    """Stands in for `build_reel`: records the call, writes a small file, or fails."""

    def __init__(
        self, raises: Exception | None = None, *, captioned: bool = True, cached: bool = False
    ) -> None:
        self.raises = raises
        self.captioned = captioned
        self.cached = cached
        self.calls: list[tuple[Path, Path, int, ReelFilter, ReelOptions]] = []

    def __call__(
        self,
        video: Path,
        match_dir: Path,
        facts: Sequence[EventFacts],
        flt: ReelFilter,
        options: ReelOptions,
        *,
        progress: object = None,
    ) -> ReelResult:
        self.calls.append((video, match_dir, len(select_events(facts, flt).events), flt, options))
        if callable(progress):
            progress(0, 2)
            progress(2, 2)
        if self.raises is not None:
            raise self.raises
        path = match_dir / "clips" / "reels" / "reel_fake.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake reel")
        return ReelResult(path, 1, self.cached, self.captioned, False, 25.0)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeBuild:
    build = FakeBuild()
    monkeypatch.setattr("aicoach.ui.reel_panel.build_reel", build)
    return build


@pytest.mark.unit
def test_reel_block_shows_the_count_and_enables_the_button(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake: FakeBuild
) -> None:
    app = start(monkeypatch, tmp_path, tmp_path / "src.mp4")

    assert not app.exception
    assert not app.button(key=BUTTON_KEY).disabled
    assert ru.REEL_COUNT.format(count=1) in captions(app)  # one goal in the synthetic match
    assert [app.number_input(key=k).value for k in ("reel_before", "reel_after")] == [8.0, 12.0]
    assert fake.calls == []


@pytest.mark.unit
@pytest.mark.usefixtures("fake")
def test_reel_button_is_disabled_without_a_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = start(monkeypatch, tmp_path, None)

    assert not app.exception
    assert app.button(key=BUTTON_KEY).disabled
    assert ru.REEL_NO_VIDEO in captions(app)


@pytest.mark.unit
@pytest.mark.usefixtures("fake")
def test_reel_button_is_disabled_when_no_event_fits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = start(monkeypatch, tmp_path, tmp_path / "src.mp4")

    app.selectbox(key="reel_kind").select("corner").run()  # the synthetic match has no corner

    assert app.button(key=BUTTON_KEY).disabled
    assert ru.REEL_EMPTY in captions(app)


@pytest.mark.unit
@pytest.mark.usefixtures("fake")
def test_reel_button_is_disabled_above_the_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("aicoach.ui.reel_panel.DEFAULT_MAX_EVENTS", 0)

    app = start(monkeypatch, tmp_path, tmp_path / "src.mp4")

    assert app.button(key=BUTTON_KEY).disabled
    assert ru.REEL_TOO_MANY.format(count=1, limit=0) in captions(app)


@pytest.mark.unit
def test_reel_side_filter_reports_events_with_an_unknown_team(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake: FakeBuild
) -> None:
    app = start(monkeypatch, tmp_path, tmp_path / "src.mp4")

    app.selectbox(key="reel_side").select("away").run()  # the one goal has the side home

    assert ru.REEL_EMPTY in captions(app)
    assert fake.calls == []


@pytest.mark.unit
def test_reel_build_shows_the_video_and_download_and_keeps_them_across_reruns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake: FakeBuild
) -> None:
    video = tmp_path / "src.mp4"
    app = start(monkeypatch, tmp_path, video)
    app.selectbox(key="reel_kind").select("card").run()
    app.number_input(key="reel_before").set_value(5.0).run()

    app.button(key=BUTTON_KEY).click().run()

    assert not app.exception
    ((called_video, match_dir, n, flt, options),) = fake.calls
    assert (called_video, match_dir, n) == (video, tmp_path, 1)
    assert flt == ReelFilter("card", None, None)
    assert (options.before_s, options.after_s) == (5.0, 12.0)
    assert ru.REEL_READY.format(count=1, duration="00:25") in captions(app)
    assert any(b.proto.id and "reel" in str(b.key) for b in app.get("download_button"))

    app.run()  # a rerun keeps the reel and does not cut it again
    assert any("reel" in str(b.key) for b in app.get("download_button"))
    assert len(fake.calls) == 1


@pytest.mark.unit
def test_reel_from_the_cache_and_without_captions_is_said_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("aicoach.ui.reel_panel.build_reel", FakeBuild(captioned=False, cached=True))
    app = start(monkeypatch, tmp_path, tmp_path / "src.mp4")

    app.button(key=BUTTON_KEY).click().run()

    texts = captions(app)
    assert ru.REEL_NO_CAPTIONS in texts
    assert any(ru.REEL_FROM_CACHE in t for t in texts)


@pytest.mark.unit
def test_reel_error_is_shown_without_crash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "aicoach.ui.reel_panel.build_reel", FakeBuild(ExportError("FFmpeg failed (exit code 1)"))
    )
    app = start(monkeypatch, tmp_path, tmp_path / "src.mp4")

    app.button(key=BUTTON_KEY).click().run()

    assert not app.exception
    assert [e.value for e in app.error] == [
        ru.REEL_FAILED.format(error="FFmpeg failed (exit code 1)")
    ]


@pytest.mark.unit
@pytest.mark.usefixtures("fake")
def test_reel_file_removed_after_the_build_shows_an_error_and_forgets_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = start(monkeypatch, tmp_path, tmp_path / "src.mp4")
    app.button(key=BUTTON_KEY).click().run()
    for reel in (tmp_path / "clips" / "reels").iterdir():
        reel.unlink()

    app.run()

    assert [e.value for e in app.error] == [ru.REEL_FILE_GONE.format(name="reel_fake.mp4")]
    app.run()
    assert not app.error  # forgotten: the message is shown once


# --- labels --------------------------------------------------------------------------------


def labels_button(app: AppTest) -> object:
    (button,) = [b for b in app.get("download_button") if "labels" in str(b.key)]
    return button


@pytest.mark.unit
def test_labels_block_offers_the_download_with_the_event_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = start(monkeypatch, tmp_path, None)

    assert not app.exception
    assert labels_button(app) is not None
    assert ru.LABELS_SUMMARY.format(count=3) in captions(app)  # goal, substitution, foul


@pytest.mark.unit
def test_labels_reviewed_only_checkbox_narrows_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = start(monkeypatch, tmp_path, None)

    app.checkbox(key="labels_reviewed_only").check().run()

    assert not app.exception
    assert ru.LABELS_SUMMARY.format(count=0) in captions(app)  # nobody reviewed anything yet
    assert labels_button(app) is not None
