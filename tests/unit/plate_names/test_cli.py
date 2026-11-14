"""Tests of the `plate-names` command (the stage runs for real, the reader is a fake)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from aicoach.cli import main
from aicoach.plate_names.reader import OcrUnavailableError, TesseractReader
from aicoach.plate_names.stage import RESULT_FILE
from tests.fixtures.plate_names_synth import LINEUP_CSV, FakeReader, make_plate, write_match

GOAL = make_plate("plate_001", "goal", 100.0)


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()  # main() points the logger at the captured stderr


def use_reader(monkeypatch: pytest.MonkeyPatch, reader: FakeReader) -> None:
    monkeypatch.setattr(TesseractReader, "create", lambda *_a, **_k: reader)


@pytest.mark.unit
def test_plate_names_command_prints_the_counts_and_writes_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_match(tmp_path, [GOAL], LINEUP_CSV)
    use_reader(monkeypatch, FakeReader("9 РОМАН ЛЕБЕДЕВ"))

    code = main(["plate-names", "--video", str(tmp_path / "src.mp4")])

    assert code == 0
    out = capsys.readouterr().out
    assert "1 plates, 1 names: 1 matched, 0 unchecked, 0 unrecognized; lineup lineups.csv" in out
    assert (tmp_path / RESULT_FILE).is_file()


@pytest.mark.unit
def test_plate_names_command_without_a_lineup_says_names_are_not_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_match(tmp_path, [GOAL])
    use_reader(monkeypatch, FakeReader("9 РОМАН ЛЕБЕДЕВ"))

    code = main(["plate-names", "--video", str(tmp_path / "src.mp4")])

    assert code == 0
    assert "no lineup (names are not checked)" in capsys.readouterr().out


@pytest.mark.unit
def test_plate_names_command_lineups_option_and_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_match(tmp_path, [make_plate("plate_003", "card", 5.0, team=None)])
    squad = tmp_path / "squad.csv"
    squad.write_text(LINEUP_CSV, encoding="utf-8")
    use_reader(monkeypatch, FakeReader("9 РОМАН ЛЕБЕДЕВ"))

    code = main(
        ["plate-names", "--video", str(tmp_path / "src.mp4"), "--lineups", str(squad), "--force"]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "lineup squad.csv" in out
    assert "warning: plate_003: side unknown" in out


@pytest.mark.unit
def test_plate_names_command_without_tesseract_says_so_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_match(tmp_path, [GOAL])

    def no_tesseract(*_a: object, **_k: object) -> TesseractReader:
        msg = "tesseract not found"
        raise OcrUnavailableError(msg)

    monkeypatch.setattr(TesseractReader, "create", no_tesseract)

    code = main(["plate-names", "--video", str(tmp_path / "src.mp4")])

    assert code == 0
    out = capsys.readouterr().out
    assert out.startswith("plate_names skipped: ")
    assert "Tesseract" in out


@pytest.mark.unit
def test_plate_names_command_without_plates_json_is_a_handled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_reader(monkeypatch, FakeReader())

    code = main(["plate-names", "--video", str(tmp_path / "src.mp4")])

    assert code == 2
    assert "run first: python -m aicoach.cli plates" in capsys.readouterr().err
