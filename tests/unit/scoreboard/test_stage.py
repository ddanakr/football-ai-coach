"""The stage end to end on a tiny clip with a fake frame source (no real footage)."""

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.config import ScoreboardConfig
from aicoach.scoreboard.schemas import DecodeMode, ScoreboardResult
from aicoach.scoreboard.stage import (
    RESULT_FILE,
    TEMPLATES_DIR,
    SourceFactory,
    duration_warning,
    load_context,
    load_labels,
    run_calibration,
    run_labelling,
    run_scoreboard,
)
from aicoach.scoreboard.video import Frame, VideoInfo, probe_video
from aicoach.status import read_status
from tests.fixtures.scoreboard_synth import grass
from tests.unit.scoreboard.conftest import PARAMS, match_frames, write_calibration


class FakeSource:
    """Stands in for SampledReader: yields prepared frames and counts how often it was used."""

    used = 0

    def __init__(
        self, frames: list[Frame], mode: DecodeMode = "opencv_gpu", decoded: int | None = None
    ) -> None:
        self._frames = frames
        self.decode_mode: DecodeMode = mode
        self.frames_decoded = 60 if decoded is None else decoded

    def __iter__(self) -> Iterator[tuple[int, Frame]]:
        FakeSource.used += 1
        yield from enumerate(self._frames)


def _factory(
    frames: list[Frame], mode: DecodeMode = "opencv_gpu", decoded: int | None = None
) -> SourceFactory:
    return lambda _path, _step, _hw: FakeSource(frames, mode, decoded)


def _broken_factory() -> SourceFactory:
    return lambda _path, _step, _hw: BrokenSource(match_frames())


class BrokenSource(FakeSource):
    """A source that dies in the middle of the pass."""

    def __iter__(self) -> Iterator[tuple[int, Frame]]:
        yield from enumerate(self._frames[:3])
        msg = "decoder crashed"
        raise RuntimeError(msg)


@pytest.fixture(autouse=True)
def _reset_counter() -> None:
    FakeSource.used = 0


@pytest.fixture
def calibrated(match_dir: Path) -> Path:
    write_calibration(match_dir)
    return match_dir


@pytest.mark.unit
def test_run_scoreboard_writes_the_result_beside_the_video(calibrated: Path) -> None:
    video = calibrated / "match.mp4"
    result = run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    stored = ScoreboardResult.model_validate_json((calibrated / RESULT_FILE).read_text("utf-8"))
    assert stored == result
    assert result.video == "match.mp4"
    assert result.decode_mode == "opencv_gpu"
    assert result.sample_fps == pytest.approx(1.0)
    assert result.cache_key != ""
    assert result.stats.n_samples == 12
    assert result.stats.n_visible == 12
    assert result.stats.n_unreadable == 0


@pytest.mark.unit
def test_run_scoreboard_reads_the_clock_score_and_time(calibrated: Path) -> None:
    result = run_scoreboard(
        calibrated / "match.mp4", "match01", source_factory=_factory(match_frames())
    )
    assert result.samples[3].clock_s == 603
    assert result.samples[3].score == (0, 0)
    assert result.samples[9].score == (1, 0)
    [change] = result.score_changes
    assert (change.t_old_last_seen, change.t_new_first_seen) == (5.0, 6.0)
    assert (change.old, change.new, change.direction) == ((0, 0), (1, 0), "up")
    assert [s.t_v for s in result.samples] == [float(k) for k in range(12)]


@pytest.mark.unit
def test_run_scoreboard_writes_status_done(calibrated: Path) -> None:
    run_scoreboard(calibrated / "match.mp4", "match01", source_factory=_factory(match_frames()))
    status = read_status(calibrated).stages["scoreboard"]
    assert (status.state, status.progress, status.error) == ("done", 1.0, None)


@pytest.mark.unit
def test_run_scoreboard_skips_the_video_when_the_result_is_up_to_date(calibrated: Path) -> None:
    video = calibrated / "match.mp4"
    first = run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    assert FakeSource.used == 1
    with capture_logs() as logs:
        second = run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    assert FakeSource.used == 1
    assert second == first
    assert any(entry["event"] == "stage_skipped" for entry in logs)


@pytest.mark.unit
def test_run_scoreboard_force_reads_again(calibrated: Path) -> None:
    video = calibrated / "match.mp4"
    run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    run_scoreboard(video, "match01", force=True, source_factory=_factory(match_frames()))
    assert FakeSource.used == 2


@pytest.mark.unit
def test_run_scoreboard_same_input_gives_the_same_result(calibrated: Path) -> None:
    video = calibrated / "match.mp4"
    first = run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    second = run_scoreboard(video, "match01", force=True, source_factory=_factory(match_frames()))
    assert second == first


@pytest.mark.unit
def test_run_scoreboard_changed_templates_invalidate_the_cache(calibrated: Path) -> None:
    video = calibrated / "match.mp4"
    run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    extra = calibrated / TEMPLATES_DIR / "score" / "0__999.png"
    extra.write_bytes((calibrated / TEMPLATES_DIR / "score" / "0__0.png").read_bytes())
    run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    assert FakeSource.used == 2


@pytest.mark.unit
def test_run_scoreboard_changed_parameters_invalidate_the_cache(calibrated: Path) -> None:
    video = calibrated / "match.mp4"
    run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    other = ScoreboardConfig(duration_tol_s=5.0)
    run_scoreboard(video, "match01", other, source_factory=_factory(match_frames()))
    assert FakeSource.used == 2


@pytest.mark.unit
def test_run_scoreboard_broken_cache_is_recomputed_with_a_warning(calibrated: Path) -> None:
    video = calibrated / "match.mp4"
    run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    (calibrated / RESULT_FILE).write_text("{broken", encoding="utf-8")
    with capture_logs() as logs:
        run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    assert FakeSource.used == 2
    assert any(entry["event"] == "cache_unreadable" for entry in logs)


@pytest.mark.unit
def test_run_scoreboard_without_templates_fails_with_a_hint_and_status(match_dir: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="scoreboard-calibrate"):
        run_scoreboard(match_dir / "match.mp4", "match01", source_factory=_factory([]))
    status = read_status(match_dir).stages["scoreboard"]
    assert status.state == "failed"
    assert "scoreboard-calibrate" in (status.error or "")
    assert not (match_dir / RESULT_FILE).exists()


@pytest.mark.unit
def test_run_scoreboard_other_decode_mode_than_calibration_raises(calibrated: Path) -> None:
    factory = _factory(match_frames(), mode="opencv_cpu")
    with pytest.raises(ScoreboardParseError, match="differs from the calibration"):
        run_scoreboard(calibrated / "match.mp4", "match01", source_factory=factory)
    assert read_status(calibrated).stages["scoreboard"].state == "failed"
    assert not (calibrated / RESULT_FILE).exists()


@pytest.mark.unit
def test_run_scoreboard_missing_video_fails(calibrated: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="not found"):
        run_scoreboard(calibrated / "nope.mp4", "match01", source_factory=_factory([]))


@pytest.mark.unit
def test_run_scoreboard_warns_when_the_file_is_shorter_than_its_metadata(calibrated: Path) -> None:
    factory = _factory(match_frames(), decoded=40)
    with capture_logs() as logs:
        result = run_scoreboard(calibrated / "match.mp4", "match01", source_factory=factory)
    assert result.stats.duration_warning is not None
    assert any(entry["event"] == "duration_mismatch" for entry in logs)


@pytest.mark.unit
def test_run_scoreboard_video_without_a_board_gives_an_empty_but_valid_result(
    calibrated: Path,
) -> None:
    frames = [grass(seed=k) for k in range(12)]
    result = run_scoreboard(calibrated / "match.mp4", "match01", source_factory=_factory(frames))
    assert result.stats.n_visible == 0
    assert result.visible_segments == []
    assert result.score_changes == []


@pytest.mark.unit
def test_run_scoreboard_no_frames_gives_an_empty_result(calibrated: Path) -> None:
    result = run_scoreboard(calibrated / "match.mp4", "match01", source_factory=_factory([]))
    assert result.samples == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("decoded", "tol", "expect_warning"),
    [
        (300, 1.0, False),
        (331, 1.0, True),
        (269, 1.0, True),
        (330, 1.0, False),
        (310, 1.0, False),
        (311, 0.1, True),
    ],
)
def test_duration_warning(decoded: int, tol: float, expect_warning: bool) -> None:
    info = VideoInfo(fps=30.0, frame_count=300, width=64, height=48)
    assert (duration_warning(info, decoded, tol) is not None) is expect_warning


@pytest.mark.unit
def test_load_context_reads_reference_and_templates(calibrated: Path) -> None:
    ctx, mode = load_context(calibrated / TEMPLATES_DIR, PARAMS)
    assert mode == "opencv_gpu"
    assert len(ctx.score_templates) > 0
    assert ctx.added_templates is None


@pytest.mark.unit
def test_load_context_missing_template_directory_names_it(calibrated: Path) -> None:
    shutil.rmtree(calibrated / TEMPLATES_DIR / "clock")
    with pytest.raises(ScoreboardParseError, match="template directory not found"):
        load_context(calibrated / TEMPLATES_DIR, PARAMS)


@pytest.mark.unit
def test_run_calibration_and_labelling_end_to_end(match_dir: Path, tmp_path: Path) -> None:
    frames = [match_frames()[k % 12] for k in range(24)]
    summary = run_calibration(
        match_dir / "match.mp4", tmp_path / "work", "match01", source_factory=_factory(frames)
    )
    assert summary.n_frames == 24
    assert summary.n_board_frames == 24
    assert summary.decode_mode == "opencv_gpu"
    assert (tmp_path / "work" / "sheet_clock.png").is_file()
    assert (match_dir / TEMPLATES_DIR / "board_reference.npz").is_file()

    clusters = json.loads((tmp_path / "work" / "clusters.json").read_text("utf-8"))["clusters"]
    labels = {  # label the first cluster of each kind; enough to prove the flow
        "score": {"0": "0"},
        "clock": {"0": "colon"},
    }
    assert len(clusters["clock"]) > 0
    labels_file = tmp_path / "labels.json"
    labels_file.write_text(json.dumps(labels), encoding="utf-8")
    report = run_labelling(match_dir / "match.mp4", tmp_path / "work", labels_file)
    assert report.templates["clock"] >= 1
    assert (match_dir / TEMPLATES_DIR / "calibration.json").is_file()


@pytest.mark.unit
def test_run_calibration_without_a_board_raises(match_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="no frames"):
        run_calibration(
            match_dir / "match.mp4",
            tmp_path / "work",
            "match01",
            source_factory=_factory([grass(seed=1)]),
        )


@pytest.mark.unit
@pytest.mark.parametrize("content", ["{broken", "[]", '{"clock": {"0": 5}}', ""])
def test_load_labels_bad_files_raise(tmp_path: Path, content: str) -> None:
    path = tmp_path / "labels.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ScoreboardParseError, match="cannot read labels file"):
        load_labels(path)


@pytest.mark.unit
def test_load_labels_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="cannot read labels file"):
        load_labels(tmp_path / "nope.json")


@pytest.mark.unit
def test_probe_of_the_tiny_clip_matches_the_fixture(match_dir: Path) -> None:
    info = probe_video(match_dir / "match.mp4")
    assert info.frame_count == 60
    assert info.fps == pytest.approx(5.0)


@pytest.mark.unit
def test_run_scoreboard_failure_keeps_the_previous_result_untouched(calibrated: Path) -> None:
    video = calibrated / "match.mp4"
    run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    before = (calibrated / RESULT_FILE).read_bytes()
    with pytest.raises(ScoreboardParseError, match="differs from the calibration"):
        run_scoreboard(
            video, "match01", force=True, source_factory=_factory(match_frames(), "opencv_cpu")
        )
    assert (calibrated / RESULT_FILE).read_bytes() == before
    assert not list(calibrated.glob("*.tmp"))
    assert read_status(calibrated).stages["scoreboard"].state == "failed"


@pytest.mark.unit
def test_run_scoreboard_unexpected_error_marks_failed_and_is_raised_again(
    calibrated: Path,
) -> None:
    video = calibrated / "match.mp4"
    run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    before = (calibrated / RESULT_FILE).read_bytes()
    broken = _broken_factory()
    with capture_logs() as logs, pytest.raises(RuntimeError, match="decoder crashed"):
        run_scoreboard(video, "match01", force=True, source_factory=broken)
    status = read_status(calibrated).stages["scoreboard"]
    assert status.state == "failed"
    assert "decoder crashed" in (status.error or "")
    assert (calibrated / RESULT_FILE).read_bytes() == before
    failed = [entry for entry in logs if entry["event"] == "stage_failed"]
    assert len(failed) == 1
    assert isinstance(failed[0]["duration_ms"], int)


@pytest.mark.unit
def test_run_scoreboard_failed_status_write_does_not_hide_the_real_error(
    calibrated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aicoach.errors import StorageError
    from aicoach.status import State, write_status

    def write_unless_failed(
        match_dir: Path, name: str, state: State, progress: float, error: str | None = None
    ) -> None:
        if state == "failed":
            msg = "disk full"
            raise StorageError(msg)
        write_status(match_dir, name, state, progress, error)

    monkeypatch.setattr("aicoach.stage_common.write_status", write_unless_failed)
    with capture_logs() as logs, pytest.raises(RuntimeError, match="decoder crashed"):
        run_scoreboard(calibrated / "match.mp4", "match01", source_factory=_broken_factory())
    assert any(entry["event"] == "status_write_failed" for entry in logs)


@pytest.mark.unit
def test_run_scoreboard_unwritable_result_is_a_storage_error_and_leaves_no_tmp(
    calibrated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aicoach.errors import StorageError

    original = Path.replace

    def failing_replace(self: Path, target: Path) -> Path:
        if self.name == "scoreboard.json.tmp":
            msg = "locked"
            raise PermissionError(msg)
        return original(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(StorageError, match="cannot write"):
        run_scoreboard(calibrated / "match.mp4", "match01", source_factory=_factory(match_frames()))
    assert not (calibrated / "scoreboard.json.tmp").exists()
    assert not (calibrated / RESULT_FILE).exists()


@pytest.mark.unit
def test_run_scoreboard_writes_one_event_with_duration_for_a_run_and_one_for_a_skip(
    calibrated: Path,
) -> None:
    video = calibrated / "match.mp4"
    with capture_logs() as first:
        run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    with capture_logs() as second:
        run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    assert [e["event"] for e in first if e["event"].startswith("stage_")] == ["stage_done"]
    assert [e["event"] for e in second if e["event"].startswith("stage_")] == ["stage_skipped"]
    assert isinstance(first[-1]["duration_ms"], int)
    assert isinstance(second[-1]["duration_ms"], int)


@pytest.mark.unit
def test_run_scoreboard_cache_follows_the_video_file_and_the_decode_mode(
    calibrated: Path,
) -> None:
    video = calibrated / "match.mp4"
    run_scoreboard(video, "match01", source_factory=_factory(match_frames()))
    record = calibrated / TEMPLATES_DIR / "calibration.json"
    record.write_text(record.read_text("utf-8").replace("opencv_gpu", "opencv_cpu"), "utf-8")
    run_scoreboard(video, "match01", source_factory=_factory(match_frames(), "opencv_cpu"))
    assert FakeSource.used == 2  # the calibration mode changed, so the key changed
    stat = video.stat()
    os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    run_scoreboard(video, "match01", source_factory=_factory(match_frames(), "opencv_cpu"))
    assert FakeSource.used == 3  # the video file changed


@pytest.mark.unit
def test_run_scoreboard_status_goes_through_running_to_failed_with_progress(
    calibrated: Path,
) -> None:
    broken = _broken_factory()
    with pytest.raises(RuntimeError):
        run_scoreboard(calibrated / "match.mp4", "match01", source_factory=broken)
    status = read_status(calibrated).stages["scoreboard"]
    assert status.state == "failed"
    assert 0.0 < status.progress < 1.0  # 3 of 12 samples were read


@pytest.mark.unit
def test_load_context_reads_added_templates_when_present(calibrated: Path) -> None:
    score_dir = calibrated / TEMPLATES_DIR / "score"
    first = sorted(score_dir.glob("1__*.png"))[0]
    added_dir = calibrated / TEMPLATES_DIR / "added"
    added_dir.mkdir()
    (added_dir / "1__0.png").write_bytes(first.read_bytes())
    ctx, _mode = load_context(calibrated / TEMPLATES_DIR, PARAMS)
    assert ctx.added_templates is not None


@pytest.mark.unit
def test_run_calibration_rejects_a_non_positive_step(match_dir: Path, tmp_path: Path) -> None:
    from aicoach.errors import ConfigError

    with pytest.raises(ConfigError, match="positive"):
        run_calibration(
            match_dir / "match.mp4", tmp_path, "m", step_s=0.0, source_factory=_factory([])
        )


@pytest.mark.unit
def test_run_scoreboard_unopenable_video_fails_before_any_pass(calibrated: Path) -> None:
    (calibrated / "bad.mp4").write_bytes(b"not a video" * 50)
    with pytest.raises(ScoreboardParseError):
        run_scoreboard(calibrated / "bad.mp4", "match01", source_factory=_factory([]))
    assert FakeSource.used == 0
    assert read_status(calibrated).stages["scoreboard"].state == "failed"


@pytest.mark.unit
def test_run_scoreboard_unwritable_status_does_not_stop_the_stage(
    calibrated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aicoach.errors import StorageError

    def always_failing(*_args: object, **_kwargs: object) -> None:
        msg = "file is open in another program"
        raise StorageError(msg)

    monkeypatch.setattr("aicoach.stage_common.write_status", always_failing)
    with capture_logs() as logs:
        result = run_scoreboard(
            calibrated / "match.mp4", "match01", source_factory=_factory(match_frames())
        )
    assert result.stats.n_samples == 12
    assert (calibrated / RESULT_FILE).is_file()
    assert any(entry["event"] == "status_write_failed" for entry in logs)
