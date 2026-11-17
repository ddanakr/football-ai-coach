"""Command line of `review-candidates`, `review-stats` and `review-eval`: wiring and exit codes."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import structlog

from aicoach.cli import main
from aicoach.domain.events import EventType
from aicoach.eval.schemas import GroundTruth, GroundTruthEvent, HalfBounds
from aicoach.review.log import append_entry, make_entry
from aicoach.review.stage import RESULT_FILE
from tests.fixtures.review_synth import TDEED_NAME, write_review_match

MATCH = "m1"


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


@pytest.fixture
def match_dir(tmp_path: Path) -> Path:
    path = tmp_path / MATCH
    path.mkdir()
    write_review_match(path, MATCH)
    return path


@pytest.mark.unit
def test_review_candidates_finds_the_tdeed_file_and_reports_the_counts(
    match_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["review-candidates", "--match-dir", str(match_dir)]) == 0

    out = capsys.readouterr().out
    assert "shot 4 (1st half 3, 2nd half 1)" in out
    assert "goal" not in out and "foul" not in out
    assert "8 shot detections, 2 near an event, 1 merged, 1 outside the halves" in out
    assert (match_dir / RESULT_FILE).is_file()


@pytest.mark.unit
def test_review_candidates_with_an_explicit_file_and_force(match_dir: Path) -> None:
    args = ["review-candidates", "--match-dir", str(match_dir), "--predictions"]
    args += [str(match_dir / TDEED_NAME), "--force", "--match-id", "other"]

    assert main(args) == 0

    assert '"match_id": "other"' in (match_dir / RESULT_FILE).read_text(encoding="utf-8")


@pytest.mark.unit
def test_review_candidates_without_a_tdeed_file_exits_with_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["review-candidates", "--match-dir", str(tmp_path)]) == 2
    assert "no T-DEED file" in capsys.readouterr().err


@pytest.mark.unit
def test_review_candidates_without_fuse_output_says_what_to_run(
    match_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (match_dir / "candidates.json").unlink()

    assert main(["review-candidates", "--match-dir", str(match_dir)]) == 2
    assert "aicoach.cli fuse" in capsys.readouterr().err


@pytest.mark.unit
def test_review_stats_prints_the_window(
    match_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["review-candidates", "--match-dir", str(match_dir)]) == 0
    capsys.readouterr()
    now = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
    append_entry(match_dir, make_entry("r:1", "reject", 210.0, EventType.SHOT, now=now))

    code = main(["review-stats", "--match-dir", str(match_dir), "--from", "03:00", "--to", "04:00"])

    assert code == 0
    out = capsys.readouterr().out
    assert "window: 180-240 s of video" in out
    assert "rejected (candidates): 1" in out
    assert "rows in the window: 2" in out  # the goal event at 200 s and its candidate at 210 s


@pytest.mark.unit
@pytest.mark.parametrize(
    "window", [["--from", "xx", "--to", "04:00"], ["--from", "05:00", "--to", "04:00"]]
)
def test_review_stats_with_a_bad_window_exits_with_2(
    match_dir: Path, window: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["review-stats", "--match-dir", str(match_dir), *window]) == 2
    assert capsys.readouterr().err.startswith("error:")


@pytest.mark.unit
def test_review_stats_without_the_database_exits_with_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["review-stats", "--match-dir", str(tmp_path), "--from", "0:00", "--to", "1:00"])

    assert code == 2
    assert "does not exist" in capsys.readouterr().err


@pytest.mark.unit
def test_review_eval_writes_the_section(
    match_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (match_dir / "aicoach_model.db").write_bytes((match_dir / "aicoach.db").read_bytes())
    truth = tmp_path / "truth.json"
    truth.write_text(
        GroundTruth(
            video="v.mp4",
            halves={
                "1": HalfBounds(start=0.0, end=1000.0),
                "2": HalfBounds(start=1500.0, end=2500.0),
            },
            events=(GroundTruthEvent(type=EventType.GOAL, t_v=200.0, half=1),),
        ).model_dump_json(),
        encoding="utf-8",
    )
    output = tmp_path / "metrics.md"

    code = main(
        [
            "review-eval",
            "--match-dir",
            str(match_dir),
            "--tdeed",
            TDEED_NAME,
            "--ground-truth",
            str(truth),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    assert f"result: {output}" in capsys.readouterr().out
    assert "Порог 0.05 задан заранее" in output.read_text(encoding="utf-8")


@pytest.mark.unit
def test_review_eval_with_a_missing_database_exits_with_2(
    match_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    truth = tmp_path / "truth.json"
    truth.write_text('{"video": "v", "events": []}', encoding="utf-8")

    code = main(
        ["review-eval", "--match-dir", str(match_dir), "--tdeed", TDEED_NAME]
        + ["--ground-truth", str(truth), "--output", str(tmp_path / "o.md")]
    )

    assert code == 2
    assert "database not found" in capsys.readouterr().err
