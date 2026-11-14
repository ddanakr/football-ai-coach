"""Clip export with the real FFmpeg on a tiny synthetic video (lavfi `testsrc` and `sine`).

Skipped when FFmpeg or FFprobe is not in PATH. No network and no match data.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from aicoach.clips import CLIPS_DIR, export_event_clip
from aicoach.domain.events import EventType
from aicoach.storage import NewPrediction, add_predictions, connect, get_current_events, init_db

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="FFmpeg and FFprobe are needed in PATH",
    ),
]

SECONDS = 6
FPS = 25


def run_tool(*args: str) -> str:
    # Fixed tool names and our own arguments; no shell.
    done = subprocess.run(args, capture_output=True, check=True, timeout=60)  # noqa: S603
    return done.stdout.decode("utf-8")


def make_video(path: Path) -> None:
    """A 6 s video, 25 fps, with a 440 Hz tone."""
    run_tool(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={SECONDS}:size=160x90:rate={FPS}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={SECONDS}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    )  # fmt: skip


def probe(path: Path) -> dict[str, object]:
    out = run_tool(
        "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,codec_type",
        "-of", "json", str(path),
    )  # fmt: skip
    data: dict[str, object] = json.loads(out)
    return data


def make_db(db: Path, t_v: float) -> str:
    init_db(db)
    with connect(db) as conn:
        add_predictions(
            conn, "m1", "run1", [NewPrediction(type=EventType.GOAL, t_v=t_v, source="plates")]
        )
        return get_current_events(conn, "m1")[0].event_id


def test_export_event_clip_real_ffmpeg_writes_trimmed_h264_aac_clip(tmp_path: Path) -> None:
    video, db = tmp_path / "src.mp4", tmp_path / "aicoach.db"
    make_video(video)
    event_id = make_db(db, 2.0)

    clip = export_event_clip(db, "m1", video, event_id, before_s=8, after_s=1)

    assert clip.parent == tmp_path / CLIPS_DIR
    info = probe(clip)
    streams = info["streams"]
    assert isinstance(streams, list)
    assert sorted((s["codec_type"], s["codec_name"]) for s in streams) == [
        ("audio", "aac"),
        ("video", "h264"),
    ]
    fmt = info["format"]
    assert isinstance(fmt, dict)
    # Window 0..3 s (trimmed at the start); AAC frames may add a few hundredths.
    assert float(fmt["duration"]) == pytest.approx(3.0, abs=0.1)
    assert [p.name for p in clip.parent.iterdir()] == [clip.name]


def test_export_event_clip_second_call_returns_cached_file(tmp_path: Path) -> None:
    video, db = tmp_path / "src.mp4", tmp_path / "aicoach.db"
    make_video(video)
    event_id = make_db(db, 5.5)
    first = export_event_clip(db, "m1", video, event_id, before_s=1, after_s=12)
    stamp = first.stat().st_mtime_ns
    video.unlink()  # the cached clip must not need the video any more

    assert export_event_clip(db, "m1", video, event_id, before_s=1, after_s=12) == first
    assert first.stat().st_mtime_ns == stamp
