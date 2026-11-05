"""Command line: argument errors give exit code 2 and a message, a run gives 0."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from aicoach.cli import _match_id, build_parser, main
from tests.unit.scoreboard.conftest import write_calibration


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0", "-1", "nan"])
def test_main_invalid_sample_fps_exits_with_2_and_names_the_option(
    value: str, match_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["scoreboard", "--video", str(match_dir / "match.mp4"), "--sample-fps", value])
    assert code == 2
    assert "sample_fps" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0", "-5"])
def test_main_invalid_calibration_step_exits_with_2(
    value: str, match_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["scoreboard-calibrate", "--video", str(match_dir / "match.mp4"), "--step-s", value]
    )
    assert code == 2
    assert "positive" in capsys.readouterr().err


@pytest.mark.unit
def test_main_without_templates_exits_with_2_and_says_how_to_calibrate(
    match_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["scoreboard", "--video", str(match_dir / "match.mp4"), "--cpu"])
    assert code == 2
    assert "scoreboard-calibrate" in capsys.readouterr().err


@pytest.mark.unit
def test_main_missing_video_exits_with_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["scoreboard", "--video", str(tmp_path / "nope.mp4")]) == 2
    assert "not found" in capsys.readouterr().err


@pytest.mark.unit
def test_main_runs_the_stage_and_reports(
    match_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_calibration(match_dir, mode="opencv_cpu")
    code = main(["scoreboard", "--video", str(match_dir / "match.mp4"), "--cpu"])
    out = capsys.readouterr().out
    assert code == 0
    assert "scoreboard.json" in out
    assert (match_dir / "scoreboard.json").is_file()


@pytest.mark.unit
def test_main_labels_file_with_a_wrong_shape_exits_with_2(
    match_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = tmp_path / "labels.json"
    labels.write_text("[]", encoding="utf-8")
    code = main(
        [
            "scoreboard-calibrate",
            "--video",
            str(match_dir / "match.mp4"),
            "--workdir",
            str(tmp_path / "work"),
            "--labels",
            str(labels),
        ]
    )
    assert code == 2
    assert "labels file" in capsys.readouterr().err


@pytest.mark.unit
def test_parser_rejects_an_unknown_command_and_a_missing_video() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["unknown"])
    with pytest.raises(SystemExit):
        parser.parse_args(["scoreboard"])


@pytest.mark.unit
def test_parser_match_id_defaults_to_none_so_the_directory_name_is_used() -> None:
    args = build_parser().parse_args(["scoreboard", "--video", "x/match01/src.mp4"])
    assert args.match_id is None
    assert args.sample_fps is None


@pytest.mark.unit
def test_main_unknown_log_level_exits_with_2(
    match_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--log-level", "FOO", "scoreboard", "--video", str(match_dir / "match.mp4")])
    assert code == 2
    assert "unknown log level" in capsys.readouterr().err


@pytest.mark.unit
def test_match_id_of_a_bare_file_name_is_the_name_of_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "match07"
    folder.mkdir()
    monkeypatch.chdir(folder)
    args = build_parser().parse_args(["scoreboard", "--video", "src.mp4"])
    assert _match_id(args) == "match07"
    named = build_parser().parse_args(["scoreboard", "--video", "src.mp4", "--match-id", "x1"])
    assert _match_id(named) == "x1"
