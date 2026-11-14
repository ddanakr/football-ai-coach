"""Frame size check and audio extraction (FFmpeg is a fake runner here)."""

import subprocess
from pathlib import Path

import pytest

from aicoach.errors import IngestError
from aicoach.ingest.media import (
    AUDIO_RATE_HZ,
    build_audio_command,
    check_frame_size,
    extract_audio,
)
from aicoach.scoreboard.video import VideoInfo


def info(width: int, height: int) -> VideoInfo:
    return VideoInfo(fps=60.0, frame_count=600, width=width, height=height)


@pytest.mark.unit
def test_check_frame_size_1280x720_passes() -> None:
    check_frame_size(info(1280, 720))


@pytest.mark.unit
@pytest.mark.parametrize(("width", "height"), [(1920, 1080), (854, 480), (720, 1280), (1280, 719)])
def test_check_frame_size_other_sizes_raise_with_the_reason(width: int, height: int) -> None:
    with pytest.raises(IngestError, match=rf"{width}x{height}.*calibrated for 1280x720"):
        check_frame_size(info(width, height))


@pytest.mark.unit
def test_build_audio_command_asks_for_16khz_mono_pcm_wav(tmp_path: Path) -> None:
    command = build_audio_command(tmp_path / "v.mp4", tmp_path / "a.wav.part")

    assert command[0] == "ffmpeg"
    joined = " ".join(command)
    assert f"-ar {AUDIO_RATE_HZ}" in joined
    assert "-ac 1" in joined
    assert "-c:a pcm_s16le" in joined
    assert "-f wav" in joined
    assert command[-1] == str(tmp_path / "a.wav.part")


def writing_runner(command: list[str], timeout_s: float) -> "subprocess.CompletedProcess[bytes]":
    """Fake FFmpeg: writes the output file it was asked for."""
    assert timeout_s > 0
    Path(command[-1]).write_bytes(b"RIFF")
    return subprocess.CompletedProcess(command, 0, b"", b"")


@pytest.mark.unit
def test_extract_audio_success_renames_the_partial_file(tmp_path: Path) -> None:
    out = extract_audio(tmp_path / "v.mp4", tmp_path / "audio.wav", writing_runner)

    assert out.read_bytes() == b"RIFF"
    assert not (tmp_path / "audio.wav.part").exists()


@pytest.mark.unit
def test_extract_audio_ffmpeg_failure_raises_and_leaves_no_file(tmp_path: Path) -> None:
    def failing(command: list[str], _timeout_s: float) -> "subprocess.CompletedProcess[bytes]":
        Path(command[-1]).write_bytes(b"half")
        stderr = b"Stream map '0:a:0' matches no streams"
        return subprocess.CompletedProcess(command, 1, b"", stderr)

    with pytest.raises(IngestError, match="audio extraction failed.*matches no streams"):
        extract_audio(tmp_path / "v.mp4", tmp_path / "audio.wav", failing)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_extract_audio_missing_ffmpeg_raises(tmp_path: Path) -> None:
    def missing(command: list[str], _timeout_s: float) -> "subprocess.CompletedProcess[bytes]":
        raise FileNotFoundError(command[0])

    with pytest.raises(IngestError, match="install FFmpeg"):
        extract_audio(tmp_path / "v.mp4", tmp_path / "audio.wav", missing)


@pytest.mark.unit
def test_extract_audio_rename_failure_raises(tmp_path: Path) -> None:
    (tmp_path / "audio.wav").mkdir()
    (tmp_path / "audio.wav" / "keep").write_text("x")

    with pytest.raises(IngestError, match="cannot write"):
        extract_audio(tmp_path / "v.mp4", tmp_path / "audio.wav", writing_runner)

    assert not (tmp_path / "audio.wav.part").exists()
