"""The `run` pipeline: order, arguments, failure handling and status (stages are fakes)."""

from pathlib import Path

import pytest

from aicoach.errors import AicoachError, FusionError, SpottingError
from aicoach.fusion.config import FuseConfig
from aicoach.logging_setup import stage_logger
from aicoach.pipeline import PipelineResult, run_pipeline, run_step
from aicoach.status import read_status
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
