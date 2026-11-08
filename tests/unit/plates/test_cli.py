"""Command line of the plates stage: argument errors give exit code 2, a run gives 0."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from aicoach.cli import main
from aicoach.scoreboard.stage import RESULT_FILE as SCOREBOARD_FILE
from tests.fixtures.plates_synth import scoreboard_result


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0", "-1", "nan"])
def test_main_plates_invalid_sample_fps_exits_with_2_and_names_the_option(
    value: str, match_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["plates", "--video", str(match_dir / "match.mp4"), "--sample-fps", value])

    assert code == 2
    assert "sample_fps" in capsys.readouterr().err


@pytest.mark.unit
def test_main_plates_without_scoreboard_json_exits_with_2_and_says_what_to_run(
    match_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["plates", "--video", str(match_dir / "match.mp4"), "--cpu"])

    assert code == 2
    assert "aicoach.cli scoreboard" in capsys.readouterr().err


@pytest.mark.unit
def test_main_plates_missing_video_exits_with_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["plates", "--video", str(tmp_path / "nope.mp4")]) == 2
    assert "not found" in capsys.readouterr().err


@pytest.mark.unit
def test_main_plates_runs_the_stage_and_reports(
    match_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    board = scoreboard_result(decode_mode="opencv_cpu")
    (match_dir / SCOREBOARD_FILE).write_text(board.model_dump_json(), encoding="utf-8")

    code = main(["plates", "--video", str(match_dir / "match.mp4"), "--cpu"])

    assert code == 0
    out = capsys.readouterr().out
    assert "plates.json: 0 plates" in out
    assert "decode mode opencv_cpu" in out
    assert (match_dir / "plates.json").is_file()
