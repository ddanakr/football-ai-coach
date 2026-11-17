"""The `reel` and `export-labels` commands: arguments, defaults, the printed summary, errors."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from aicoach.cli import main
from aicoach.errors import ExportError
from aicoach.reel import ReelFilter, ReelOptions, ReelResult
from tests.fixtures.ui_synth import write_match_dir


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    """`main` points structlog at the captured stderr, which pytest closes after the test."""
    yield
    structlog.reset_defaults()


class FakeReel:
    """Stands in for `export_event_reel`: records the call and returns a reel file or fails."""

    def __init__(self, tmp_path: Path, raises: Exception | None = None) -> None:
        self.tmp_path = tmp_path
        self.raises = raises
        self.calls: list[tuple[Path, str, Path, Path, ReelFilter, ReelOptions]] = []

    def __call__(
        self,
        db: Path,
        match_id: str,
        match_dir: Path,
        video: Path,
        flt: ReelFilter,
        options: ReelOptions,
    ) -> ReelResult:
        self.calls.append((db, match_id, match_dir, video, flt, options))
        if self.raises is not None:
            raise self.raises
        path = self.tmp_path / "reel_goal_x.mp4"
        path.write_bytes(b"x" * 2_000_000)
        return ReelResult(path, 4, False, True, False, 80.0, ("a warning",), team_unknown=1)


@pytest.mark.unit
def test_reel_defaults_come_from_the_match_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeReel(tmp_path)
    monkeypatch.setattr("aicoach.cli.export_event_reel", fake)
    match = tmp_path / "match01"

    code = main(["reel", "--match-dir", str(match), "--type", "goal"])

    assert code == 0
    ((db, match_id, match_dir, video, flt, options),) = fake.calls
    assert (db, match_id, match_dir) == (match / "aicoach.db", "match01", match)
    assert video == match / "src.mp4"
    assert flt == ReelFilter("goal", None, None)
    assert (options.before_s, options.after_s, options.max_events) == (8.0, 12.0, 30)
    out = capsys.readouterr().out
    assert "events: 4" in out and "length: 80.0 s" in out and "size: 2.0 MB" in out
    assert (
        "build time:" in out and "skipped, team unknown: 1" in out and "warning: a warning" in out
    )


@pytest.mark.unit
def test_reel_explicit_options_are_passed_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeReel(tmp_path)
    monkeypatch.setattr("aicoach.cli.export_event_reel", fake)
    argv = ["reel", "--match-dir", str(tmp_path), "--type", "card", "--half", "2"]
    argv += ["--side", "away", "--before", "5", "--after", "7", "--max", "10"]
    argv += ["--db", str(tmp_path / "other.db"), "--video", str(tmp_path / "v.mp4")]
    argv += ["--match-id", "m9"]

    assert main(argv) == 0

    ((db, match_id, _, video, flt, options),) = fake.calls
    assert (db, match_id, video) == (tmp_path / "other.db", "m9", tmp_path / "v.mp4")
    assert flt == ReelFilter("card", 2, "away")
    assert (options.before_s, options.after_s, options.max_events) == (5.0, 7.0, 10)


@pytest.mark.unit
def test_reel_export_error_gives_exit_code_2_and_a_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    error = ExportError("31 events of kind 'shot' match, more than --max 30")
    monkeypatch.setattr("aicoach.cli.export_event_reel", FakeReel(tmp_path, error))

    code = main(["reel", "--match-dir", str(tmp_path), "--type", "shot"])

    assert code == 2
    assert "31 events" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.parametrize(
    "argv",
    [
        ["--type", "penalty"],
        ["--type", "goal", "--half", "3"],
        ["--type", "goal", "--side", "both"],
        [],  # --type is required
    ],
)
def test_reel_bad_arguments_are_refused_by_the_parser(
    tmp_path: Path, argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["reel", "--match-dir", str(tmp_path), *argv])

    assert raised.value.code == 2
    assert capsys.readouterr().err


@pytest.mark.unit
def test_export_labels_writes_the_file_and_prints_the_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = write_match_dir(tmp_path)
    out = tmp_path / "out" / "labels.json"

    code = main(["export-labels", "--db", str(db), "--out", str(out), "--match-id", "m1"])

    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["video"] == "src.mp4" and len(data["events"]) == 3
    printed = capsys.readouterr().out
    assert "events: 3 (foul 1, goal 1, subst 1)" in printed
    assert "not reviewed" not in printed


@pytest.mark.unit
def test_export_labels_reviewed_only_prints_what_was_left_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = write_match_dir(tmp_path)
    out = tmp_path / "labels.json"

    code = main(
        ["export-labels", "--db", str(db), "--out", str(out), "--match-id", "m1", "--reviewed-only"]
    )

    assert code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["events"] == []
    assert "not reviewed, left out: 3" in capsys.readouterr().out


@pytest.mark.unit
def test_export_labels_warns_when_the_half_bounds_are_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = write_match_dir(tmp_path)
    (tmp_path / "candidates.json").unlink()

    code = main(["export-labels", "--db", str(db), "--out", str(tmp_path / "l.json")])

    assert code == 0
    assert "no half bounds" in capsys.readouterr().out


@pytest.mark.unit
def test_export_labels_missing_database_gives_exit_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["export-labels", "--db", str(tmp_path / "none.db"), "--out", str(tmp_path / "l.json")]
    )

    assert code == 2
    assert "does not exist" in capsys.readouterr().err
    assert not (tmp_path / "l.json").exists()
