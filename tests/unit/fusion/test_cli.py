"""Command line of `fuse` and `run`: wiring, messages and exit codes."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog

from aicoach import cli
from aicoach.cli import main
from aicoach.pipeline import run_pipeline
from aicoach.storage import add_edit, connect, get_current_events
from tests.fixtures.fusion_synth import plate, write_inputs, write_tdeed
from tests.fixtures.pipeline_fakes import Recorder
from tests.fixtures.plates_synth import change


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


@pytest.fixture
def match_dir(tmp_path: Path) -> Path:
    """A match directory with one goal, a goal plate and a card plate."""
    write_inputs(
        tmp_path,
        [change((0, 0), (1, 0), 199.0, 200.0)],
        [plate("p1", "goal", 230.0), plate("p2", "card", 700.0)],
    )
    return tmp_path


@pytest.mark.unit
def test_main_fuse_runs_and_reports_counts_mode_and_warning(
    match_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["fuse", "--video", str(match_dir / "match.mp4"), "--match-id", "m1"])

    assert code == 0
    out = capsys.readouterr().out
    assert "goal 1 (1st half 1, 2nd half 0)" in out
    assert "foul 1" in out
    assert "mode no_model" in out
    assert "warning: no model predictions file" in out
    assert (match_dir / "aicoach.db").is_file()
    assert (match_dir / "candidates.json").is_file()


@pytest.mark.unit
def test_main_fuse_missing_scoreboard_exits_with_2_and_says_what_to_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["fuse", "--video", str(tmp_path / "match.mp4")]) == 2
    assert "aicoach.cli scoreboard" in capsys.readouterr().err


@pytest.mark.unit
def test_main_fuse_uses_the_config_file_and_reports_a_bad_one(
    match_dir: Path, tmp_path_factory: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path_factory.mktemp("cfg") / "fuse.json"
    cfg.write_text(json.dumps({"goal_plate_window_s": 1.0}), encoding="utf-8")
    video = str(match_dir / "match.mp4")

    assert main(["fuse", "--video", video, "--config", str(cfg)]) == 0
    cfg.write_text(json.dumps({"goal_plate_window_s": -5}), encoding="utf-8")
    assert main(["fuse", "--video", video, "--config", str(cfg), "--force"]) == 2

    assert "goal_plate_window_s" in capsys.readouterr().err


@pytest.mark.unit
def test_main_fuse_refuses_with_edits_until_discard_edits_is_given(
    match_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video = str(match_dir / "match.mp4")
    assert main(["fuse", "--video", video, "--match-id", "m1"]) == 0
    with connect(match_dir / "aicoach.db") as conn:
        target = get_current_events(conn, "m1")[0].event_id
        add_edit(conn, "m1", "verify", author="a", target=target)
    capsys.readouterr()

    assert main(["fuse", "--video", video, "--match-id", "m1", "--force"]) == 2
    assert "--discard-edits" in capsys.readouterr().err
    assert main(["fuse", "--video", video, "--match-id", "m1", "--force", "--discard-edits"]) == 0


@pytest.mark.unit
def test_main_run_passes_the_arguments_to_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = Recorder()
    seen: dict[str, Any] = {}

    def fake(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return run_pipeline(*args, **kwargs, stages=rec.stages())

    monkeypatch.setattr(cli, "run_pipeline", fake)
    predictions = tmp_path / "tdeed_x.json"
    write_tdeed(predictions, [])

    code = main(
        ["run", "--video", str(tmp_path / "m.mp4"), "--predictions", str(predictions), "--cpu"]
        + ["--force", "--discard-edits"]
    )

    assert code == 0
    assert seen["predictions_path"] == predictions
    assert (seen["force"], seen["hw_accel"], seen["discard_edits"]) == (True, False, True)
    assert [c[0] for c in rec.calls] == ["scoreboard", "plates", "fusion"]
    assert "candidates.json" in capsys.readouterr().out


@pytest.mark.unit
def test_main_run_failure_exits_with_2_and_names_the_stage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # No video and no scoreboard templates: the first stage fails, the run stops there.
    assert main(["run", "--video", str(tmp_path / "match.mp4")]) == 2
    assert "stage scoreboard failed" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.parametrize("command", ["fuse", "run"])
@pytest.mark.parametrize("content", [None, "not json"])
def test_main_missing_or_broken_predictions_exits_with_2_and_names_the_file(
    match_dir: Path, command: str, content: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    predictions = match_dir / "tdeed_x.json"
    if content is not None:
        predictions.write_text(content, encoding="utf-8")

    code = main(
        [command, "--video", str(match_dir / "match.mp4"), "--predictions", str(predictions)]
    )

    assert code == 2
    assert "tdeed_x.json" in capsys.readouterr().err


@pytest.mark.unit
def test_main_run_bad_config_exits_with_2_before_any_stage(
    match_dir: Path, tmp_path_factory: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path_factory.mktemp("cfg") / "fuse.json"
    cfg.write_text(json.dumps({"nope": 1}), encoding="utf-8")

    code = main(["run", "--video", str(match_dir / "match.mp4"), "--config", str(cfg)])

    assert code == 2
    assert "invalid fusion config" in capsys.readouterr().err


@pytest.mark.unit
def test_main_fuse_db_option_chooses_the_database_and_needs_no_video_file(
    match_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    db = tmp_path_factory.mktemp("db") / "events.db"
    video = match_dir / "match.mp4"
    assert not video.exists()

    assert main(["fuse", "--video", str(video), "--db", str(db)]) == 0

    assert db.is_file()
    assert not (match_dir / "aicoach.db").exists()
