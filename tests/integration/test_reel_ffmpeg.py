"""Reel export with the real FFmpeg on a tiny synthetic video (lavfi `testsrc` and `sine`).

Skipped when FFmpeg or FFprobe is not in PATH. The captions need a font: without one on the
machine the reel is built without captions (the same code path as in production), and the test
still checks the join. No network and no match data.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from aicoach.clips import CLIPS_DIR
from aicoach.domain.events import EventType
from aicoach.reel import REELS_DIR, ReelFilter, ReelOptions, build_reel, find_font
from tests.fixtures.reel_synth import make_facts

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="FFmpeg and FFprobe are needed in PATH",
    ),
]

SECONDS = 12
FPS = 25


def run_tool(*args: str) -> str:
    # Fixed tool names and our own arguments; no shell.
    done = subprocess.run(args, capture_output=True, check=True, timeout=120)  # noqa: S603
    return done.stdout.decode("utf-8")


def make_video(path: Path) -> None:
    """A 12 s video, 25 fps, with a 440 Hz tone."""
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


def test_build_reel_real_ffmpeg_joins_two_captioned_pieces_without_encoding_again(
    tmp_path: Path,
) -> None:
    video = tmp_path / "src.mp4"
    make_video(video)
    facts = [
        make_facts("p:2", EventType.GOAL, 9.0, minute=9),
        make_facts("p:1", EventType.GOAL, 3.0, minute=3),
    ]
    font = find_font(None)

    result = build_reel(
        video,
        tmp_path,
        facts,
        ReelFilter("goal"),
        ReelOptions(before_s=2.0, after_s=2.0, font_file=font),
    )

    assert result.path.parent == tmp_path / CLIPS_DIR / REELS_DIR
    assert result.n_events == 2 and not result.cached
    assert result.captioned == (font is not None)
    assert not result.reencoded  # the two pieces came out of one encoder: a stream copy joins them
    info = probe(result.path)
    streams = info["streams"]
    assert isinstance(streams, list)
    assert sorted((s["codec_type"], s["codec_name"]) for s in streams) == [
        ("audio", "aac"),
        ("video", "h264"),
    ]
    fmt = info["format"]
    assert isinstance(fmt, dict)
    # Two windows of 4 s each; AAC frames may add a few hundredths at every join.
    assert float(fmt["duration"]) == pytest.approx(8.0, abs=0.3)
    assert result.duration_s == pytest.approx(8.0, abs=0.3)
    assert sorted(p.name for p in result.path.parent.iterdir()) == [result.path.name]

    again = build_reel(
        video,
        tmp_path,
        facts,
        ReelFilter("goal"),
        ReelOptions(before_s=2.0, after_s=2.0, font_file=font),
    )
    assert again.cached and again.path == result.path
