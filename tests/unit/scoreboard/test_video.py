"""Video reading on a tiny synthetic clip: gray level of frame i equals i."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.video import (
    DECODE_CPU,
    DECODE_GPU,
    SampledReader,
    frame_step,
    probe_video,
    sample_count,
)

FPS = 30
N_FRAMES = 150  # 5 s


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 5 s, 30 fps clip; the gray level of frame i is i."""
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (64, 48))
    for i in range(N_FRAMES):
        writer.write(np.full((48, 64, 3), i, dtype=np.uint8))
    writer.release()
    return path


@pytest.mark.unit
def test_probe_video_reads_metadata(clip: Path) -> None:
    info = probe_video(clip)
    assert info.fps == pytest.approx(FPS)
    assert info.frame_count == N_FRAMES
    assert (info.width, info.height) == (64, 48)
    assert info.duration_s == pytest.approx(5.0)


@pytest.mark.unit
def test_probe_video_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="not found"):
        probe_video(tmp_path / "nope.mp4")


@pytest.mark.unit
def test_probe_video_garbage_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video" * 100)
    with pytest.raises(ScoreboardParseError):
        probe_video(bad)


@pytest.mark.unit
@pytest.mark.parametrize("hw_accel", [True, False])
def test_sampled_reader_returns_one_frame_per_step(clip: Path, hw_accel: bool) -> None:
    got = list(SampledReader(clip, FPS, hw_accel))
    assert [k for k, _ in got] == [0, 1, 2, 3, 4]
    for k, frame in got:
        assert float(frame.mean()) == pytest.approx(k * FPS, abs=6)


@pytest.mark.unit
def test_sampled_reader_counts_every_decoded_frame(clip: Path) -> None:
    reader = SampledReader(clip, FPS)
    list(reader)
    assert reader.frames_decoded == N_FRAMES


@pytest.mark.unit
def test_sampled_reader_step_larger_than_the_file_gives_the_first_frame_only(clip: Path) -> None:
    got = list(SampledReader(clip, 1000))
    assert [k for k, _ in got] == [0]


@pytest.mark.unit
def test_sampled_reader_reports_the_decode_mode(clip: Path) -> None:
    cpu = SampledReader(clip, FPS, hw_accel=False)
    list(cpu)
    assert cpu.decode_mode == DECODE_CPU
    gpu = SampledReader(clip, FPS, hw_accel=True)
    list(gpu)
    assert gpu.decode_mode in (DECODE_GPU, DECODE_CPU)  # the machine may have no GPU decoder


@pytest.mark.unit
def test_sampled_reader_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="cannot open"):
        list(SampledReader(tmp_path / "nope.mp4", 30))


@pytest.mark.unit
def test_sampled_reader_cut_off_file_without_index_raises(clip: Path, tmp_path: Path) -> None:
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(clip.read_bytes()[: clip.stat().st_size // 2])  # the mp4 index is at the end
    with pytest.raises(ScoreboardParseError, match="cannot open"):
        list(SampledReader(cut, FPS, hw_accel=False))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fps", "sample_fps", "expected"),
    [(60.0, 1.0, 60), (30.0, 1.0, 30), (59.94, 1.0, 60), (25.0, 2.0, 12), (10.0, 30.0, 1)],
)
def test_frame_step_rounds_and_never_drops_below_one(
    fps: float, sample_fps: float, expected: int
) -> None:
    assert frame_step(fps, sample_fps) == expected


@pytest.mark.unit
@pytest.mark.parametrize(("fps", "sample_fps"), [(0.0, 1.0), (30.0, 0.0), (-1.0, 1.0)])
def test_frame_step_rejects_non_positive_rates(fps: float, sample_fps: float) -> None:
    with pytest.raises(ScoreboardParseError):
        frame_step(fps, sample_fps)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("frames", "step", "expected"),
    [(150, 30, 5), (151, 30, 6), (1, 30, 1), (0, 30, 0), (10, 0, 0)],
)
def test_sample_count(frames: int, step: int, expected: int) -> None:
    assert sample_count(frames, step) == expected


@pytest.mark.unit
@pytest.mark.parametrize("step", [0, -1])
def test_sampled_reader_rejects_a_step_below_one(clip: Path, step: int) -> None:
    with pytest.raises(ScoreboardParseError, match="at least 1"):
        SampledReader(clip, step)


@pytest.mark.unit
def test_sampled_reader_second_pass_counts_frames_from_zero(clip: Path) -> None:
    reader = SampledReader(clip, FPS)
    first = [k for k, _ in reader]
    second = [k for k, _ in reader]
    assert first == second == [0, 1, 2, 3, 4]
    assert reader.frames_decoded == N_FRAMES
