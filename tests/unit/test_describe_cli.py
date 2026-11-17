"""The `describe` command: arguments, defaults, output and errors (no model, no FFmpeg)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from aicoach.cli import main
from aicoach.errors import ExternalServiceError
from aicoach.ui.data import load_events
from tests.fixtures.episode_synth import TEXT, FakeClipper, FakeVideoModel
from tests.fixtures.ui_synth import write_match_dir

KEY = "key-for-the-fake-client"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    """`main` points structlog at the captured stderr, which pytest closes after the test."""
    yield
    structlog.reset_defaults()


class Made:
    """What the command built: the video model it was given, and the arguments of the call."""

    def __init__(self) -> None:
        self.vlm = FakeVideoModel()
        self.models: list[str | None] = []

    def make(self, _settings: object, _dir: Path, model: str | None = None) -> FakeVideoModel:
        self.models.append(model)
        return self.vlm


def event_id_of(match_dir: Path) -> str:
    return load_events(match_dir / "aicoach.db", "m1")[0].event_id


@pytest.mark.unit
def test_describe_prints_the_text_with_the_mark_the_model_and_the_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_match_dir(tmp_path, "m1")
    made = Made()
    monkeypatch.setattr("aicoach.cli.make_video_model", made.make)
    monkeypatch.setattr("aicoach.episode.export_clip", FakeClipper())
    event_id = event_id_of(tmp_path)

    code = main(
        ["describe", "--video", str(tmp_path / "src.mp4"), "--event-id", event_id]
        + ["--match-id", "m1"]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert f"event: {event_id}" in out
    assert "model: test/vlm-1" in out
    assert "cached: False" in out
    assert "cost: 0.0031 USD" in out
    assert f"по клипу, не проверено: {TEXT}" in out
    assert made.models == [None]  # no flag: the setting decides


@pytest.mark.unit
def test_describe_passes_the_model_flag_and_the_window_and_reads_the_cache_the_second_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_match_dir(tmp_path, "m1")
    made = Made()
    clipper = FakeClipper()
    monkeypatch.setattr("aicoach.cli.make_video_model", made.make)
    monkeypatch.setattr("aicoach.episode.export_clip", clipper)
    argv = ["describe", "--video", str(tmp_path / "src.mp4"), "--event-id", event_id_of(tmp_path)]
    argv += ["--match-id", "m1", "--model", "x/y", "--before", "5", "--after", "7"]

    assert main(argv) == 0
    assert main(argv) == 0

    out = capsys.readouterr().out
    assert made.models == ["x/y", "x/y"]
    assert [c[2:] for c in clipper.calls] == [(5.0, 7.0)]  # the second run took the cache
    assert out.count("cached: True") == 1
    assert len(made.vlm.requests) == 1


@pytest.mark.unit
def test_describe_reports_a_service_failure_with_exit_code_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_match_dir(tmp_path, "m1")
    made = Made()
    made.vlm = FakeVideoModel([ExternalServiceError("OpenRouter request failed (HTTP 500)")])
    monkeypatch.setattr("aicoach.cli.make_video_model", made.make)
    monkeypatch.setattr("aicoach.episode.export_clip", FakeClipper())
    argv = ["describe", "--video", str(tmp_path / "src.mp4"), "--event-id", event_id_of(tmp_path)]

    code = main([*argv, "--match-id", "m1"])

    assert code == 2
    assert "HTTP 500" in capsys.readouterr().err


@pytest.mark.unit
def test_describe_without_the_openrouter_key_says_which_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_match_dir(tmp_path, "m1")

    code = main(
        ["describe", "--video", str(tmp_path / "src.mp4"), "--event-id", "p:1"]
        + ["--match-id", "m1"]
    )

    assert code == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


@pytest.mark.unit
def test_describe_for_an_unknown_event_is_an_error_before_any_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_match_dir(tmp_path, "m1")
    made = Made()
    monkeypatch.setattr("aicoach.cli.make_video_model", made.make)

    code = main(
        ["describe", "--video", str(tmp_path / "src.mp4"), "--event-id", "p:999"]
        + ["--match-id", "m1"]
    )

    assert code == 2
    assert "p:999" in capsys.readouterr().err
    assert made.vlm.requests == []
