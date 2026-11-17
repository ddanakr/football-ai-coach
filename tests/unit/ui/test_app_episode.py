"""The "Describe the episode" button of the viewer (`AppTest`): state, result kept across reruns,
the mark "by the clip, not verified", errors. The clip cutter and the video model are fakes."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.fixtures.episode_synth import TEXT, FakeClipper, FakeVideoModel
from tests.unit.ui.test_app_clips import captions, start
from tests.unit.ui.test_app_edits import pick

from aicoach.errors import BudgetExceededError, ExternalServiceError
from aicoach.i18n import ru
from aicoach.ui.data import load_events
from aicoach.ui.episode_panel import BUTTON_KEY

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1"])


class Fakes:
    """The video model and the clip cutter the page talks to."""

    def __init__(self, vlm: FakeVideoModel | None = None) -> None:
        self.vlm = vlm or FakeVideoModel()
        self.clipper = FakeClipper()
        self.models: list[str] = []

    def make_video_model(self, _settings: object, _dir: Path) -> FakeVideoModel:
        self.models.append(self.vlm.model)
        return self.vlm


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    made = Fakes()
    monkeypatch.setattr("aicoach.ui.episode_panel.make_video_model", made.make_video_model)
    monkeypatch.setattr("aicoach.episode.export_clip", made.clipper)
    return made


def infos(app: AppTest) -> list[str]:
    return [str(i.value) for i in app.info]


@pytest.mark.unit
@pytest.mark.usefixtures("fakes")
def test_button_is_disabled_until_an_event_is_picked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = start(monkeypatch, tmp_path, tmp_path / "src.mp4")

    assert not app.exception
    assert app.button(key=BUTTON_KEY).disabled
    assert ru.EPISODE_PICK_FIRST in captions(app)
    assert app.button(key=BUTTON_KEY).label == "Описать эпизод"

    pick(app, 0)
    assert not app.button(key=BUTTON_KEY).disabled


@pytest.mark.unit
@pytest.mark.usefixtures("fakes")
def test_button_is_disabled_without_the_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = pick(start(monkeypatch, tmp_path, None), 0)

    assert not app.exception
    assert app.button(key=BUTTON_KEY).disabled
    assert ru.EPISODE_NO_VIDEO in captions(app)


@pytest.mark.unit
def test_click_shows_the_text_marked_not_verified_with_model_tokens_and_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fakes: Fakes
) -> None:
    app = pick(start(monkeypatch, tmp_path, tmp_path / "src.mp4"), 1)

    app.button(key=BUTTON_KEY).click().run()

    assert not app.exception
    assert infos(app) == [TEXT]
    marked = [c for c in captions(app) if c.startswith(ru.EPISODE_UNVERIFIED)]
    assert len(marked) == 1
    assert "модель test/vlm-1" in marked[0]
    assert "токены 2100/480" in marked[0]
    assert "0.0031 USD" in marked[0]
    assert len(fakes.vlm.requests) == 1
    events = load_events(tmp_path / "aicoach.db", "m1")
    assert fakes.clipper.calls[0][1].event_id == events[1].event_id
    assert any(c.startswith("Эпизод: «") for c in captions(app))


@pytest.mark.unit
def test_the_clip_window_fields_are_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fakes: Fakes
) -> None:
    app = pick(start(monkeypatch, tmp_path, tmp_path / "src.mp4"), 0)
    app.number_input(key="clip_before").set_value(20.0)
    app.number_input(key="clip_after").set_value(10.0).run()

    app.button(key=BUTTON_KEY).click().run()

    assert fakes.clipper.calls[0][2:] == (20.0, 10.0)
    assert fakes.vlm.requests[0][3] == 30.0


@pytest.mark.unit
def test_a_rerun_keeps_the_result_and_asks_nothing_and_a_second_click_comes_from_the_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fakes: Fakes
) -> None:
    app = pick(start(monkeypatch, tmp_path, tmp_path / "src.mp4"), 0)
    app.button(key=BUTTON_KEY).click().run()

    app.run()
    app.run()

    assert infos(app) == [TEXT]
    assert len(fakes.vlm.requests) == 1

    app.button(key=BUTTON_KEY).click().run()

    assert len(fakes.vlm.requests) == 1  # the cache answered
    assert any(ru.EPISODE_META_CACHED in c for c in captions(app))
    assert infos(app) == [TEXT]


@pytest.mark.unit
@pytest.mark.parametrize(
    "failure",
    [
        ExternalServiceError("OpenRouter request failed (HTTP 500)"),
        BudgetExceededError("Бюджет месяца исчерпан"),
    ],
)
def test_a_failure_is_shown_as_an_error_and_the_page_survives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: Exception
) -> None:
    made = Fakes(FakeVideoModel([failure]))
    monkeypatch.setattr("aicoach.ui.episode_panel.make_video_model", made.make_video_model)
    monkeypatch.setattr("aicoach.episode.export_clip", made.clipper)
    app = pick(start(monkeypatch, tmp_path, tmp_path / "src.mp4"), 0)

    app.button(key=BUTTON_KEY).click().run()

    assert not app.exception
    assert [e.value for e in app.error] == [ru.EPISODE_FAILED.format(error=failure)]
    assert infos(app) == []


@pytest.mark.unit
def test_a_failure_is_replaced_by_the_next_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    made = Fakes(FakeVideoModel([ExternalServiceError("HTTP 503")]))
    monkeypatch.setattr("aicoach.ui.episode_panel.make_video_model", made.make_video_model)
    monkeypatch.setattr("aicoach.episode.export_clip", made.clipper)
    app = pick(start(monkeypatch, tmp_path, tmp_path / "src.mp4"), 0)
    app.button(key=BUTTON_KEY).click().run()
    assert len(app.error) == 1

    app.button(key=BUTTON_KEY).click().run()

    assert app.error == []
    assert infos(app) == [TEXT]


@pytest.mark.unit
def test_without_the_openrouter_key_the_error_names_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = pick(start(monkeypatch, tmp_path, tmp_path / "src.mp4"), 0)

    app.button(key=BUTTON_KEY).click().run()

    assert not app.exception
    assert any("OPENROUTER_API_KEY" in str(e.value) for e in app.error)
