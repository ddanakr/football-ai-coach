"""Clip export (`aicoach.clips`): window, file name, FFmpeg command, runner errors, cache.

FFmpeg itself is not run here: a fake runner records the command and writes the output file.
The real FFmpeg run is in `tests/integration/test_ffmpeg_clip_export.py`.
"""

import math
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from aicoach.clips import (
    CLIPS_DIR,
    ClipCaption,
    build_ffmpeg_command,
    clip_filename,
    clip_window,
    drawtext_filter,
    export_clip,
    find_current_event,
    run_ffmpeg,
    stderr_tail,
)
from aicoach.domain.events import EventType
from aicoach.errors import ExportError, ScoreboardParseError
from aicoach.scoreboard.video import VideoInfo
from aicoach.storage import add_edit, connect, get_current_events
from tests.fixtures.ui_synth import make_event, write_events_db


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    """`main` points structlog at the captured stderr, which pytest closes after the test."""
    yield
    structlog.reset_defaults()


INFO = VideoInfo(fps=25.0, frame_count=25 * 300, width=64, height=36)  # 300 s


class FakeRunner:
    """Stands in for `subprocess.run`: records commands, writes the output or fails."""

    def __init__(
        self, returncode: int = 0, stderr: bytes = b"", raises: Exception | None = None
    ) -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.raises = raises
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], _timeout_s: float) -> subprocess.CompletedProcess[bytes]:
        self.calls.append(command)
        Path(command[-1]).write_bytes(b"partial clip")  # FFmpeg creates the file before failing
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(command, self.returncode, b"", self.stderr)


def fake_probe(_video: Path) -> VideoInfo:
    return INFO


def files_in(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


# --- clip_window --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("t_v", "before", "after", "expected"),
    [
        (100.0, 8.0, 12.0, (92.0, 112.0)),  # inside the video
        (3.0, 8.0, 12.0, (0.0, 15.0)),  # trimmed at the start
        (0.0, 8.0, 12.0, (0.0, 12.0)),  # event in the first frame
        (295.0, 8.0, 12.0, (287.0, 300.0)),  # trimmed at the end
        (300.0, 8.0, 12.0, (292.0, 300.0)),  # event in the last frame
        (100.0, 0.0, 12.0, (100.0, 112.0)),  # no time before
        (100.0, 8.0, 0.0, (92.0, 100.0)),  # no time after
        (100.0, 20.0, 10.0, (80.0, 110.0)),  # the Gemini window, same function
        (5.0, 400.0, 400.0, (0.0, 300.0)),  # wider than the video
    ],
)
def test_clip_window_valid_input_returns_trimmed_window(
    t_v: float, before: float, after: float, expected: tuple[float, float]
) -> None:
    assert clip_window(t_v, before, after, 300.0) == expected


@pytest.mark.unit
def test_clip_window_zero_margins_returns_empty_window_without_min_length() -> None:
    assert clip_window(100.0, 0.0, 0.0, 300.0) == (100.0, 100.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("t_v", "expected"),
    [(100.0, (100.0, 100.04)), (300.0, (299.96, 300.0)), (0.0, (0.0, 0.04))],
)
def test_clip_window_zero_margins_with_min_length_widens_inside_video(
    t_v: float, expected: tuple[float, float]
) -> None:
    start, end = clip_window(t_v, 0.0, 0.0, 300.0, min_length_s=0.04)
    assert (start, end) == pytest.approx(expected)


@pytest.mark.unit
def test_clip_window_video_shorter_than_min_length_returns_whole_video() -> None:
    assert clip_window(0.01, 0.0, 0.0, 0.02, min_length_s=0.04) == (0.0, 0.02)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("t_v", "before", "after", "duration", "min_length", "fragment"),
    [
        (100.0, -1.0, 12.0, 300.0, 0.0, "must not be negative"),
        (100.0, 8.0, -0.5, 300.0, 0.0, "must not be negative"),
        (300.5, 8.0, 12.0, 300.0, 0.0, "outside the video"),
        (-0.1, 8.0, 12.0, 300.0, 0.0, "outside the video"),
        (1.0, 8.0, 12.0, 0.0, 0.0, "duration must be positive"),
        (1.0, 8.0, 12.0, -5.0, 0.0, "duration must be positive"),
        (math.nan, 8.0, 12.0, 300.0, 0.0, "t_v must be a finite number"),
        (100.0, math.inf, 12.0, 300.0, 0.0, "before_s must be a finite number"),
        (100.0, 8.0, 12.0, math.nan, 0.0, "duration_s must be a finite number"),
        (100.0, 8.0, 12.0, 300.0, -1.0, "min_length_s must not be negative"),
    ],
)
def test_clip_window_invalid_input_raises_value_error(
    t_v: float, before: float, after: float, duration: float, min_length: float, fragment: str
) -> None:
    with pytest.raises(ValueError, match=fragment):
        clip_window(t_v, before, after, duration, min_length_s=min_length)


# --- name and command ----------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("event_id", "event_type", "t_v", "before", "after", "expected"),
    [
        ("p:12", EventType.GOAL, 200.0, 8.0, 12.0, "p12_goal_t200000ms_b8_a12.mp4"),
        ("e:3", EventType.SUBST, 61.2345, 20.0, 10.0, "e3_subst_t61234ms_b20_a10.mp4"),
        ("p:1", EventType.FOUL, 0.0, 0.0, 7.5, "p1_foul_t0ms_b0_a7.5.mp4"),
    ],
)
def test_clip_filename_event_returns_name_without_colon(
    event_id: str, event_type: EventType, t_v: float, before: float, after: float, expected: str
) -> None:
    event = make_event(event_id=event_id, type=event_type, t_v=t_v)
    assert clip_filename(event, before, after) == expected


@pytest.mark.unit
def test_build_ffmpeg_command_returns_reencode_with_seek_before_input() -> None:
    command = build_ffmpeg_command(Path("v/src.mp4"), Path("v/clips/c.mp4"), 92.0, 20.0)
    assert command[0] == "ffmpeg"
    assert command.index("-ss") < command.index("-i")
    assert command[command.index("-ss") + 1] == "92.000"
    assert command[command.index("-i") + 1] == str(Path("v/src.mp4"))
    assert command[command.index("-t") + 1] == "20.000"
    pairs = list(zip(command, command[1:], strict=False))
    for pair in [
        ("-map", "0:v:0"),
        ("-map", "0:a:0?"),
        ("-c:v", "libx264"),
        ("-preset", "veryfast"),
        ("-crf", "23"),
        ("-pix_fmt", "yuv420p"),
        ("-c:a", "aac"),
        ("-b:a", "128k"),
        ("-movflags", "+faststart"),
        ("-f", "mp4"),
    ]:
        assert pair in pairs
    assert "copy" not in command
    assert command[-1] == str(Path("v/clips/c.mp4"))


@pytest.mark.unit
def test_build_ffmpeg_command_custom_binary_is_first_argument() -> None:
    assert build_ffmpeg_command(Path("a"), Path("b"), 0, 1, ffmpeg="C:/ff/ffmpeg.exe")[0] == (
        "C:/ff/ffmpeg.exe"
    )


# --- run_ffmpeg -----------------------------------------------------------------------------


@pytest.mark.unit
def test_run_ffmpeg_success_returns_none(tmp_path: Path) -> None:
    runner = FakeRunner()
    run_ffmpeg(["ffmpeg", str(tmp_path / "x")], 5.0, runner)
    assert len(runner.calls) == 1


@pytest.mark.unit
def test_run_ffmpeg_nonzero_exit_raises_with_stderr_tail(tmp_path: Path) -> None:
    runner = FakeRunner(returncode=1, stderr=b"line 1\nInvalid data found when processing input")
    with pytest.raises(ExportError, match="(?s)exit code 1.*Invalid data found"):
        run_ffmpeg(["ffmpeg", str(tmp_path / "x")], 5.0, runner)


@pytest.mark.unit
def test_run_ffmpeg_timeout_raises_export_error(tmp_path: Path) -> None:
    runner = FakeRunner(raises=subprocess.TimeoutExpired("ffmpeg", 5.0))
    with pytest.raises(ExportError, match="did not finish in 5 s"):
        run_ffmpeg(["ffmpeg", str(tmp_path / "x")], 5.0, runner)


@pytest.mark.unit
def test_run_ffmpeg_missing_binary_raises_install_hint(tmp_path: Path) -> None:
    runner = FakeRunner(raises=FileNotFoundError("ffmpeg"))
    with pytest.raises(ExportError, match="install FFmpeg"):
        run_ffmpeg(["ffmpeg", str(tmp_path / "x")], 5.0, runner)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        (None, "(no output)"),
        (b"", "(no output)"),
        (b"\xff\xfebad bytes", "\ufffd\ufffdbad bytes"),
        (
            b"\n".join(f"l{i}".encode() for i in range(50)),
            "\n".join(f"l{i}" for i in range(30, 50)),
        ),
    ],
)
def test_stderr_tail_any_bytes_returns_last_lines(stderr: bytes | None, expected: str) -> None:
    assert stderr_tail(stderr) == expected


@pytest.mark.unit
def test_stderr_tail_long_line_is_shortened() -> None:
    assert len(stderr_tail(b"x" * 10_000)) == 2000


# --- export_clip ------------------------------------------------------------------------------


@pytest.mark.unit
def test_export_clip_new_event_writes_clip_in_clips_dir(tmp_path: Path) -> None:
    video = tmp_path / "src.mp4"
    runner = FakeRunner()
    event = make_event(event_id="p:7", type=EventType.CORNER, t_v=3.0)
    path = export_clip(video, event, 8, 12, runner=runner, probe=fake_probe)
    assert path == tmp_path / CLIPS_DIR / "p7_corner_t3000ms_b8_a12.mp4"
    assert path.read_bytes() == b"partial clip"
    assert files_in(path.parent) == [path.name]  # the temporary file was renamed
    command = runner.calls[0]
    assert command[command.index("-ss") + 1] == "0.000"  # trimmed at the start
    assert command[command.index("-t") + 1] == "15.000"


@pytest.mark.unit
def test_export_clip_existing_clip_does_not_run_ffmpeg(tmp_path: Path) -> None:
    video, event = tmp_path / "src.mp4", make_event(t_v=100.0)
    first = FakeRunner()
    path = export_clip(video, event, runner=first, probe=fake_probe)
    again = FakeRunner()

    def no_probe(_video: Path) -> VideoInfo:
        msg = "the video must not be read for a cached clip"
        raise AssertionError(msg)

    assert export_clip(video, event, runner=again, probe=no_probe) == path
    assert (len(first.calls), len(again.calls)) == (1, 0)


@pytest.mark.unit
def test_export_clip_moved_event_gets_new_clip(tmp_path: Path) -> None:
    video, runner = tmp_path / "src.mp4", FakeRunner()
    a = export_clip(video, make_event(t_v=100.0), runner=runner, probe=fake_probe)
    b = export_clip(video, make_event(t_v=130.0), runner=runner, probe=fake_probe)
    assert a != b
    assert len(runner.calls) == 2


@pytest.mark.unit
def test_export_clip_event_in_last_frame_gives_at_least_one_frame(tmp_path: Path) -> None:
    runner = FakeRunner()
    export_clip(tmp_path / "v.mp4", make_event(t_v=300.0), 0, 0, runner=runner, probe=fake_probe)
    command = runner.calls[0]
    assert command[command.index("-ss") + 1] == "299.960"
    assert command[command.index("-t") + 1] == "0.040"


@pytest.mark.unit
@pytest.mark.parametrize(
    "runner",
    [
        FakeRunner(returncode=1, stderr=b"Conversion failed!"),
        FakeRunner(raises=subprocess.TimeoutExpired("ffmpeg", 1.0)),
    ],
)
def test_export_clip_ffmpeg_failure_leaves_no_file(tmp_path: Path, runner: FakeRunner) -> None:
    with pytest.raises(ExportError):
        export_clip(tmp_path / "src.mp4", make_event(), runner=runner, probe=fake_probe)
    assert files_in(tmp_path / CLIPS_DIR) == []


@pytest.mark.unit
def test_export_clip_rename_failure_raises_export_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_replace(_self: Path, _target: Path) -> Path:
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "replace", broken_replace)
    with pytest.raises(ExportError, match="cannot write the clip.*locked"):
        export_clip(tmp_path / "src.mp4", make_event(), runner=FakeRunner(), probe=fake_probe)
    assert files_in(tmp_path / CLIPS_DIR) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("t_v", "before", "after", "fragment"),
    [(100.0, -1.0, 12.0, "must not be negative"), (301.0, 8.0, 12.0, "outside the video")],
)
def test_export_clip_bad_window_raises_without_ffmpeg(
    tmp_path: Path, t_v: float, before: float, after: float, fragment: str
) -> None:
    runner = FakeRunner()
    with pytest.raises(ExportError, match=fragment):
        export_clip(
            tmp_path / "s.mp4", make_event(t_v=t_v), before, after, runner=runner, probe=fake_probe
        )
    assert runner.calls == []


@pytest.mark.unit
def test_export_clip_unreadable_video_raises_export_error(tmp_path: Path) -> None:
    def broken(_video: Path) -> VideoInfo:
        msg = "video file not found: s.mp4"
        raise ScoreboardParseError(msg)

    with pytest.raises(ExportError, match="cannot read the video.*not found"):
        export_clip(tmp_path / "s.mp4", make_event(), runner=FakeRunner(), probe=broken)


# --- find_current_event -------------------------------------------------------------------


@pytest.mark.unit
def test_find_current_event_moved_event_returns_edited_time(tmp_path: Path) -> None:
    db = tmp_path / "aicoach.db"
    write_events_db(db, "m1")
    with connect(db) as conn:
        target = get_current_events(conn, "m1")[0].event_id
        add_edit(conn, "m1", "move", author="test", target=target, t_v=640.0)
    event = find_current_event(db, "m1", target)
    assert event.t_v == 640.0


@pytest.mark.unit
def test_find_current_event_deleted_event_raises(tmp_path: Path) -> None:
    db = tmp_path / "aicoach.db"
    write_events_db(db, "m1")
    with connect(db) as conn:
        target = get_current_events(conn, "m1")[0].event_id
        add_edit(conn, "m1", "delete", author="test", target=target)
    with pytest.raises(ExportError, match="not among the current events"):
        find_current_event(db, "m1", target)


@pytest.mark.unit
@pytest.mark.parametrize(("match_id", "event_id"), [("m1", "p:999"), ("other", "p:1")])
def test_find_current_event_unknown_event_raises(
    tmp_path: Path, match_id: str, event_id: str
) -> None:
    db = tmp_path / "aicoach.db"
    write_events_db(db, "m1")
    with pytest.raises(ExportError, match="not among the current events"):
        find_current_event(db, match_id, event_id)


@pytest.mark.unit
def test_find_current_event_missing_db_raises_and_creates_nothing(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="does not exist"):
        find_current_event(tmp_path / "none.db", "m1", "p:1")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_export_clip_clips_dir_not_creatable_raises_export_error(tmp_path: Path) -> None:
    (tmp_path / CLIPS_DIR).write_text("a file where the directory should be")
    runner = FakeRunner()
    with pytest.raises(ExportError, match="cannot write the clip"):
        export_clip(tmp_path / "src.mp4", make_event(), runner=runner, probe=fake_probe)
    assert runner.calls == []


@pytest.mark.unit
def test_run_ffmpeg_cannot_start_raises_export_error(tmp_path: Path) -> None:
    runner = FakeRunner(raises=PermissionError("access denied"))
    with pytest.raises(ExportError, match="cannot start ffmpeg: access denied"):
        run_ffmpeg(["ffmpeg", str(tmp_path / "x")], 5.0, runner)


# --- captions (a reel piece) ---------------------------------------------------------------


@pytest.mark.unit
def test_clip_filename_caption_adds_a_hash_that_follows_the_text() -> None:
    event = make_event(event_id="p:12", type=EventType.GOAL, t_v=200.0)
    plain = clip_filename(event, 8.0, 12.0)

    one = clip_filename(event, 8.0, 12.0, "Гол · 5'")

    assert one != plain and one.startswith("p12_goal_t200000ms_b8_a12_c") and one.endswith(".mp4")
    assert one == clip_filename(event, 8.0, 12.0, "Гол · 5'")
    assert one != clip_filename(event, 8.0, 12.0, "Гол · 6'")


@pytest.mark.unit
def test_drawtext_filter_reads_the_text_from_a_file_and_escapes_the_windows_drive_colon() -> None:
    caption = ClipCaption("Гол · 5'", Path("C:/Windows/Fonts/arial.ttf"), seconds=2.0)

    chain = drawtext_filter(Path("C:/work/it's/c.mp4.part.txt"), caption)

    assert chain.startswith(r"drawtext=fontfile='C\:/Windows/Fonts/arial.ttf':textfile='C\:/work/")
    assert r"it'\''s" in chain  # a quote in the path is closed, escaped and reopened
    assert "Гол" not in chain  # the text itself never goes onto the command line
    assert chain.endswith(r"enable=lt(t\,2)")


@pytest.mark.unit
def test_build_ffmpeg_command_video_filter_goes_before_the_encoder() -> None:
    command = build_ffmpeg_command(
        Path("src.mp4"), Path("o.mp4"), 1.0, 2.0, video_filter="drawtext=x=1"
    )

    assert command[command.index("-vf") + 1] == "drawtext=x=1"
    assert command.index("-vf") < command.index("-c:v")
    assert "-vf" not in build_ffmpeg_command(Path("src.mp4"), Path("o.mp4"), 1.0, 2.0)


class CaptionRunner(FakeRunner):
    """Records the caption file's content while FFmpeg "runs"."""

    def __init__(self) -> None:
        super().__init__()
        self.caption_seen: list[str] = []

    def __call__(self, command: list[str], timeout_s: float) -> subprocess.CompletedProcess[bytes]:
        chain = command[command.index("-vf") + 1]
        text_path = chain.split("textfile='")[1].split("':")[0].replace(r"\:", ":")
        self.caption_seen.append(Path(text_path).read_text(encoding="utf-8"))
        return super().__call__(command, timeout_s)


@pytest.mark.unit
def test_export_clip_with_caption_writes_the_text_file_for_ffmpeg_and_removes_it(
    tmp_path: Path,
) -> None:
    runner = CaptionRunner()
    caption = ClipCaption("Гол · 5'", tmp_path / "font.ttf")

    path = export_clip(
        tmp_path / "src.mp4", make_event(), runner=runner, probe=fake_probe, caption=caption
    )

    assert runner.caption_seen == ["Гол · 5'"]
    assert files_in(tmp_path / CLIPS_DIR) == [path.name]  # no .part and no caption file left
    assert "_c" in path.name


@pytest.mark.unit
def test_export_clip_with_caption_ffmpeg_failure_removes_the_caption_file(tmp_path: Path) -> None:
    runner = CaptionRunner()
    runner.returncode = 1
    caption = ClipCaption("Гол · 5'", tmp_path / "font.ttf")

    with pytest.raises(ExportError):
        export_clip(
            tmp_path / "src.mp4", make_event(), runner=runner, probe=fake_probe, caption=caption
        )

    assert files_in(tmp_path / CLIPS_DIR) == []
