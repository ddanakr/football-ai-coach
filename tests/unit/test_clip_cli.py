"""The `clip` command: arguments, defaults and errors (FFmpeg is not run)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from aicoach.cli import main
from aicoach.errors import ExportError


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    """`main` points structlog at the captured stderr, which pytest closes after the test."""
    yield
    structlog.reset_defaults()


class FakeExport:
    """Stands in for `export_event_clip`: records the call, returns a path or fails."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.raises = raises
        self.calls: list[tuple[Path, str, Path, str, float, float]] = []

    def __call__(
        self, db: Path, match_id: str, video: Path, event_id: str, before: float, after: float
    ) -> Path:
        self.calls.append((db, match_id, video, event_id, before, after))
        if self.raises is not None:
            raise self.raises
        return video.parent / "clips" / "p3_goal_t1000ms_b8_a12.mp4"


@pytest.mark.unit
def test_clip_defaults_come_from_the_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeExport()
    monkeypatch.setattr("aicoach.cli.export_event_clip", fake)
    video = tmp_path / "match01" / "src.mp4"

    code = main(["clip", "--video", str(video), "--event-id", "p:3"])

    assert code == 0
    assert fake.calls == [(video.parent / "aicoach.db", "match01", video, "p:3", 8.0, 12.0)]
    assert "p3_goal_t1000ms_b8_a12.mp4" in capsys.readouterr().out


@pytest.mark.unit
def test_clip_explicit_options_are_passed_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeExport()
    monkeypatch.setattr("aicoach.cli.export_event_clip", fake)
    video, db = tmp_path / "src.mp4", tmp_path / "other.db"
    argv = ["clip", "--video", str(video), "--event-id", "e:2", "--db", str(db)]
    argv += ["--match-id", "m9", "--before", "20", "--after", "10"]

    assert main(argv) == 0
    assert fake.calls == [(db, "m9", video, "e:2", 20.0, 10.0)]


@pytest.mark.unit
def test_clip_export_error_exits_with_code_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("aicoach.cli.export_event_clip", FakeExport(ExportError("FFmpeg failed")))

    code = main(["clip", "--video", str(tmp_path / "s.mp4"), "--event-id", "p:1"])

    assert code == 2
    assert "error: FFmpeg failed" in capsys.readouterr().err
