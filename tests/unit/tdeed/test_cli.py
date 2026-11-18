"""The commands `tdeed` and `tdeed-compare` through `main`, and the report of a run."""

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pytest
import structlog

from aicoach import tdeed_cli
from aicoach.cli import main
from aicoach.scoreboard.video import VideoInfo
from aicoach.tdeed.stage import TdeedResult, TdeedTools, Timings, run_tdeed
from aicoach.tdeed.timecode import Segment
from tests.fixtures.fusion_synth import write_tdeed


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    """`cli.main` binds the logger to the captured stderr; later tests must not inherit it."""
    yield
    structlog.reset_defaults()


@pytest.mark.unit
def test_tdeed_compare_same_files_exit_0_and_say_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    items = [(2500, "Foul", 0.5), (2800, "Corner", 0.3)]
    write_tdeed(tmp_path / "a.json", items)
    write_tdeed(tmp_path / "b.json", items)

    code = main(
        [
            "tdeed-compare",
            str(tmp_path / "a.json"),
            str(tmp_path / "b.json"),
            "--window",
            "00:00-10:00",
            "--edge-margin",
            "0",
        ]
    )

    assert code == 0
    assert capsys.readouterr().out.strip().endswith("совпадает")


@pytest.mark.unit
def test_tdeed_compare_different_files_exit_1_and_list_the_differences(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_tdeed(tmp_path / "a.json", [(2500, "Foul", 0.5), (2800, "Foul", 0.3)])
    write_tdeed(tmp_path / "b.json", [(2500, "Foul", 0.5)])

    code = main(
        [
            "tdeed-compare",
            str(tmp_path / "a.json"),
            str(tmp_path / "b.json"),
            "--window",
            "00:00-10:00",
            "--edge-margin",
            "0",
        ]
    )

    assert code == tdeed_cli.EXIT_DIFFERENT == 1
    assert "различия: Foul: 2 против 1" in capsys.readouterr().out


@pytest.mark.unit
def test_tdeed_compare_uses_the_default_margin_of_10_s(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The only difference is a prediction 5 s from the window start: inside the default margin.
    write_tdeed(tmp_path / "a.json", [(125, "Foul", 0.5), (2500, "Foul", 0.5)])
    write_tdeed(tmp_path / "b.json", [(2500, "Foul", 0.5)])

    code = main(
        [
            "tdeed-compare",
            str(tmp_path / "a.json"),
            str(tmp_path / "b.json"),
            "--window",
            "00:00-10:00",
        ]
    )

    assert code == 0
    assert "окно сравнения 00:10-09:50" in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.parametrize("args", [["--window", "bad"], ["--fps", "0"], ["--edge-margin", "-1"]])
def test_tdeed_compare_bad_options_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    write_tdeed(tmp_path / "a.json", [(2500, "Foul", 0.5)])

    code = main(["tdeed-compare", str(tmp_path / "a.json"), str(tmp_path / "a.json"), *args])

    assert code == 2
    assert capsys.readouterr().err.startswith("error:")


@pytest.mark.unit
def test_tdeed_compare_broken_file_exit_2(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{not json")
    write_tdeed(tmp_path / "b.json", [])

    assert main(["tdeed-compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 2


@pytest.mark.unit
def test_tdeed_without_cuda_exit_2_with_the_notebook_link(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # no torch: no video card
    tools = TdeedTools(probe=lambda _path: VideoInfo(25.0, 250_000, 1280, 720))
    monkeypatch.setattr(tdeed_cli, "run_tdeed", partial(run_tdeed, tools=tools))
    (tmp_path / "src.mp4").write_bytes(b"v")

    code = main(["tdeed", "--match-dir", str(tmp_path), "--cache-dir", str(tmp_path / "c")])

    err = capsys.readouterr().err
    assert code == 2
    assert "colab.research.google.com" in err


@pytest.mark.unit
@pytest.mark.parametrize("args", [["--from", "bad"], ["--to", "1:99:00"]])
def test_tdeed_bad_time_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    code = main(["tdeed", "--match-dir", str(tmp_path), *args])

    assert code == 2
    assert "bad time" in capsys.readouterr().err


@pytest.mark.unit
def test_tdeed_command_passes_the_options_to_the_stage_and_prints_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    segment = Segment(800.0, 792.0, 1392.0, 1452.0)
    timings = Timings(10.0, 60.0, 120.0)

    def fake_run(match_dir: Path, match_id: str, **kwargs: object) -> TdeedResult:
        seen.update(kwargs, match_dir=match_dir, match_id=match_id)
        return TdeedResult(tmp_path / "o.json", "cpu", False, 7, 7800.0, segment, timings, ["w"])

    monkeypatch.setattr(tdeed_cli, "run_tdeed", fake_run)

    code = main(
        [
            "tdeed",
            "--match-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--from",
            "13:20",
            "--to",
            "23:12",
            "--force",
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert seen["device"] == "cpu"
    assert (seen["t_from"], seen["t_to"], seen["force"]) == (800.0, 1392.0, True)
    assert seen["cache_dir"] == tmp_path / "cache"
    assert seen["match_id"] == tmp_path.name
    assert "процессоре" in out  # the warning of the cpu mode
    assert "warning: w" in out
    assert "o.json" in out


@pytest.mark.unit
def test_format_result_segment_gives_per_minute_time_and_a_labelled_estimate() -> None:
    segment = Segment(800.0, 792.0, 1392.0, 1452.0)  # cut length 660 s = 11 min
    result = TdeedResult(
        Path("x.json"), "cuda", False, 3, 7920.0, segment, Timings(5.0, 66.0, 132.0)
    )

    text = tdeed_cli.format_result(result)

    assert "13:12" in text
    assert "(было запрошено 13:20)" in text
    assert "FFmpeg 6.0 с, инференс 12.0 с" in text  # per minute: 66/11 and 132/11
    assert "Оценка на весь матч (132.0 мин видео): FFmpeg 792 с, инференс 1584 с" in text
    assert "Это оценка по отрезку, не измерение" in text


@pytest.mark.unit
def test_format_result_whole_match_has_no_estimate_and_cached_run_says_so() -> None:
    whole = TdeedResult(Path("x.json"), "cuda", False, 3, 600.0, None, Timings(1.0, 2.0, 3.0))
    cached = TdeedResult(Path("x.json"), "cuda", True, 3, 600.0)

    assert "Оценка" not in tdeed_cli.format_result(whole)
    assert "Уже есть: x.json (3 срабатываний)" in tdeed_cli.format_result(cached)
