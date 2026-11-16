"""The `run` pipeline: order, arguments, failure handling and status (stages are fakes)."""

from collections.abc import Callable
from pathlib import Path

import pytest

from aicoach.errors import AicoachError, FusionError, SpottingError, StorageError
from aicoach.fusion.config import FuseConfig
from aicoach.logging_setup import stage_logger
from aicoach.pipeline import PipelineResult, run_pipeline, run_step
from aicoach.status import read_status, write_status
from tests.fixtures.fusion_synth import write_tdeed
from tests.fixtures.pipeline_fakes import Recorder


@pytest.mark.unit
def test_run_pipeline_runs_the_three_stages_in_order_and_marks_the_run_done(
    tmp_path: Path,
) -> None:
    rec = Recorder()
    video = tmp_path / "match.mp4"

    result = run_pipeline(video, "m", stages=rec.stages())

    assert isinstance(result, PipelineResult)
    assert [c[0] for c in rec.calls] == ["scoreboard", "plates", "fusion"]
    assert read_status(tmp_path).stages["run"].state == "done"


@pytest.mark.unit
def test_run_pipeline_passes_the_options_to_every_stage(tmp_path: Path) -> None:
    rec = Recorder()
    video = tmp_path / "match.mp4"
    predictions = tmp_path / "tdeed_x.json"
    write_tdeed(predictions, [])
    config = FuseConfig(goal_plate_window_s=100.0)

    run_pipeline(
        video,
        "m",
        predictions_path=predictions,
        fuse_config=config,
        force=True,
        hw_accel=False,
        discard_edits=True,
        stages=rec.stages(),
    )

    (_, board_args, board_kwargs), (_, plate_args, plate_kwargs), (_, fuse_args, fuse_kwargs) = (
        rec.calls
    )
    assert board_args[:2] == (video, "m")
    assert board_args[2].hw_accel is False
    assert board_kwargs == {"force": True}
    assert plate_args[2].hw_accel is False
    assert plate_kwargs == {"force": True}
    assert fuse_args == (tmp_path, "m", predictions, config)
    assert fuse_kwargs == {"force": True, "discard_edits": True}


@pytest.mark.unit
@pytest.mark.parametrize(("fail_at", "ran"), [("scoreboard", 1), ("plates", 2), ("fuse", 3)])
def test_run_pipeline_stops_at_the_failing_stage_and_names_it(
    tmp_path: Path, fail_at: str, ran: int
) -> None:
    rec = Recorder(fail_at="fusion" if fail_at == "fuse" else fail_at, error=FusionError("boom"))

    with pytest.raises(FusionError, match=f"stage {fail_at} failed: boom"):
        run_pipeline(tmp_path / "match.mp4", "m", stages=rec.stages())

    assert len(rec.calls) == ran
    status = read_status(tmp_path).stages["run"]
    assert status.state == "failed"
    assert status.error == f"stage {fail_at} failed: boom"


@pytest.mark.unit
def test_run_pipeline_foreign_error_becomes_a_handled_error_that_names_the_stage(
    tmp_path: Path,
) -> None:
    rec = Recorder(fail_at="plates", error=ValueError("odd"))

    with pytest.raises(AicoachError, match="stage plates failed: odd") as info:
        run_pipeline(tmp_path / "match.mp4", "m", stages=rec.stages())

    assert isinstance(info.value.__cause__, ValueError)
    assert read_status(tmp_path).stages["run"].state == "failed"


@pytest.mark.unit
def test_run_pipeline_keeps_the_type_and_object_of_our_own_error(tmp_path: Path) -> None:
    error = FusionError("boom")
    rec = Recorder(fail_at="scoreboard", error=error)

    with pytest.raises(FusionError) as info:
        run_pipeline(tmp_path / "match.mp4", "m", stages=rec.stages())

    assert info.value is error


@pytest.mark.unit
@pytest.mark.parametrize("content", [None, "not json"])
def test_run_pipeline_bad_predictions_file_fails_before_any_stage(
    tmp_path: Path, content: str | None
) -> None:
    rec = Recorder()
    predictions = tmp_path / "tdeed_x.json"
    if content is not None:
        predictions.write_text(content, encoding="utf-8")

    with pytest.raises(SpottingError):
        run_pipeline(tmp_path / "match.mp4", "m", predictions_path=predictions, stages=rec.stages())

    assert rec.calls == []


@pytest.mark.unit
def test_run_pipeline_after_a_failure_a_new_run_marks_the_run_done(tmp_path: Path) -> None:
    video = tmp_path / "match.mp4"
    with pytest.raises(FusionError):
        run_pipeline(video, "m", stages=Recorder(fail_at="plates", error=FusionError("x")).stages())
    assert read_status(tmp_path).stages["run"].state == "failed"

    run_pipeline(video, "m", stages=Recorder().stages())

    assert read_status(tmp_path).stages["run"].state == "done"


@pytest.mark.unit
def test_run_pipeline_interrupt_does_not_leave_running_behind(tmp_path: Path) -> None:
    rec = Recorder(fail_at="scoreboard", error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(tmp_path / "match.mp4", "m", stages=rec.stages())

    assert read_status(tmp_path).stages["run"].state == "failed"


@pytest.mark.unit
def test_run_step_reports_under_the_given_entry_and_names_the_stage(tmp_path: Path) -> None:
    log = stage_logger("test", "m")

    def boom() -> int:
        msg = "bad input"
        raise FusionError(msg)

    assert run_step(tmp_path, "ingest", "audio", 0.25, lambda: 7, log) == 7
    with pytest.raises(FusionError, match="stage fuse failed: bad input"):
        run_step(tmp_path, "ingest", "fuse", 0.5, boom, log)

    entry = read_status(tmp_path).stages["ingest"]
    assert (entry.state, entry.progress) == ("failed", 0.5)
    assert entry.error == "stage fuse failed: bad input"
    assert "run" not in read_status(tmp_path).stages


def works(match_dir: Path, name: str, final: str = "done") -> Callable[[], None]:
    """A stage that does real work: `running` first, then its final state (inside the call)."""

    def call() -> None:
        write_status(match_dir, name, "running", 0.0)
        write_status(match_dir, name, final, 1.0)  # type: ignore[arg-type]  # state literal

    return call


@pytest.mark.unit
def test_run_step_writes_the_times_of_a_stage_that_wrote_its_entry(tmp_path: Path) -> None:
    log = stage_logger("test", "m")
    run_step(tmp_path, "ingest", "audio", 0.0, works(tmp_path, "audio"), log)

    entry = read_status(tmp_path).stages["audio"]
    assert entry.state == "done"
    assert entry.started_at is not None
    assert entry.finished_at is not None
    assert entry.started_at.endswith("+00:00")
    assert entry.finished_at >= entry.started_at
    assert entry.duration_s is not None
    assert entry.duration_s >= 0


@pytest.mark.unit
def test_run_step_gives_no_times_to_a_skipped_or_entryless_step(tmp_path: Path) -> None:
    log = stage_logger("test", "m")
    write_status(tmp_path, "plate_names", "skipped", 1.0, error="no tesseract")

    run_step(tmp_path, "ingest", "plate_names", 0.0, lambda: None, log)
    run_step(tmp_path, "ingest", "templates", 0.0, lambda: None, log)

    stages = read_status(tmp_path).stages
    assert stages["plate_names"].duration_s is None
    assert "templates" not in stages


@pytest.mark.unit
def test_run_step_writes_the_times_of_a_failed_stage_too(tmp_path: Path) -> None:
    log = stage_logger("test", "m")

    def boom() -> int:
        write_status(tmp_path, "fuse", "running", 0.0)
        write_status(tmp_path, "fuse", "failed", 0.0, error="bad input")
        msg = "bad input"
        raise FusionError(msg)

    with pytest.raises(FusionError):
        run_step(tmp_path, "ingest", "fuse", 0.5, boom, log)

    entry = read_status(tmp_path).stages["fuse"]
    assert entry.state == "failed"
    assert entry.duration_s is not None


@pytest.mark.unit
def test_a_second_run_step_over_a_cached_stage_keeps_the_real_times(tmp_path: Path) -> None:
    log = stage_logger("test", "m")
    # First run: the stage writes "running" (clearing), then "done".
    run_step(tmp_path, "run", "scoreboard", 0.0, works(tmp_path, "scoreboard"), log)
    first = read_status(tmp_path).stages["scoreboard"]

    # Second run: the stage finds its cache and writes only "done".
    def cached() -> None:
        write_status(tmp_path, "scoreboard", "done", 1.0)

    run_step(tmp_path, "run", "scoreboard", 0.0, cached, log)

    assert read_status(tmp_path).stages["scoreboard"] == first
    assert first.duration_s is not None


@pytest.mark.unit
def test_a_forced_rerun_replaces_the_times(tmp_path: Path) -> None:
    log = stage_logger("test", "m")
    run_step(tmp_path, "run", "scoreboard", 0.0, works(tmp_path, "scoreboard"), log)
    first = read_status(tmp_path).stages["scoreboard"]

    run_step(tmp_path, "run", "scoreboard", 0.0, works(tmp_path, "scoreboard"), log)

    second = read_status(tmp_path).stages["scoreboard"]
    assert second.started_at is not None
    assert first.started_at is not None
    assert second.started_at >= first.started_at
    assert second.duration_s is not None


@pytest.mark.unit
def test_a_failure_before_the_stage_wrote_its_state_gets_no_times(tmp_path: Path) -> None:
    log = stage_logger("test", "m")
    write_status(tmp_path, "scoreboard", "done", 1.0)  # from an earlier run

    def boom() -> None:
        msg = "bad input"
        raise FusionError(msg)

    with pytest.raises(FusionError):
        run_step(tmp_path, "run", "scoreboard", 0.0, boom, log)

    assert read_status(tmp_path).stages["scoreboard"].duration_s is None


@pytest.mark.unit
def test_a_timing_write_failure_does_not_hide_the_error_of_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = stage_logger("test", "m")

    def broken(*_args: object) -> bool:
        msg = "disk full"
        raise StorageError(msg)

    monkeypatch.setattr("aicoach.stage_common.write_timing", broken)

    def boom() -> None:
        write_status(tmp_path, "fuse", "running", 0.0)
        write_status(tmp_path, "fuse", "failed", 0.0, error="x")
        msg = "bad input"
        raise FusionError(msg)

    with pytest.raises(FusionError, match="bad input"):
        run_step(tmp_path, "run", "fuse", 0.0, boom, log)
    assert run_step(tmp_path, "run", "fuse", 0.0, lambda: 5, log) == 5


@pytest.mark.unit
def test_a_cache_hit_over_an_entry_without_times_stays_without_times(tmp_path: Path) -> None:
    log = stage_logger("test", "m")
    write_status(tmp_path, "plates", "done", 1.0)  # a status file from before the times

    def cached() -> None:
        write_status(tmp_path, "plates", "done", 1.0)

    run_step(tmp_path, "run", "plates", 0.0, cached, log)

    entry = read_status(tmp_path).stages["plates"]
    assert (entry.started_at, entry.duration_s) == (None, None)


@pytest.mark.unit
def test_a_timing_value_error_does_not_hide_the_error_of_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = stage_logger("test", "m")

    def broken(*_args: object) -> bool:
        msg = "bad timing"
        raise ValueError(msg)

    monkeypatch.setattr("aicoach.stage_common.write_timing", broken)

    def boom() -> None:
        write_status(tmp_path, "fuse", "running", 0.0)
        write_status(tmp_path, "fuse", "failed", 0.0, error="x")
        msg = "bad input"
        raise FusionError(msg)

    with pytest.raises(FusionError, match="bad input"):
        run_step(tmp_path, "run", "fuse", 0.0, boom, log)


@pytest.mark.unit
def test_a_step_that_is_not_timed_gets_no_duration(tmp_path: Path) -> None:
    log = stage_logger("test", "m")
    write_status(tmp_path, "tdeed", "running", 0.0)
    write_status(tmp_path, "tdeed", "done", 1.0)

    run_step(tmp_path, "ingest", "tdeed", 0.0, lambda: None, log, timed=False)

    assert read_status(tmp_path).stages["tdeed"].duration_s is None


@pytest.mark.unit
def test_a_start_left_by_a_killed_run_does_not_get_a_duration(tmp_path: Path) -> None:
    log = stage_logger("test", "m")
    write_status(tmp_path, "plates", "running", 0.3)  # earlier run, killed: never finished

    def cached() -> None:
        write_status(tmp_path, "plates", "done", 1.0)

    run_step(tmp_path, "run", "plates", 0.0, cached, log)

    entry = read_status(tmp_path).stages["plates"]
    assert entry.state == "done"
    assert entry.duration_s is None
