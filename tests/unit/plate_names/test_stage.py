"""Tests of the plate-names stage on a synthetic match directory with a fake reader."""

import json
from pathlib import Path

import numpy as np
import pytest

from aicoach.errors import PlatesError
from aicoach.i18n import ru
from aicoach.plate_names.layout import NameLayout
from aicoach.plate_names.reader import OcrUnavailableError, TesseractReader
from aicoach.plate_names.schemas import PlateNamesConfig, PlateNamesResult
from aicoach.plate_names.stage import RESULT_FILE, STAGE, run_plate_names
from aicoach.status import read_status
from tests.fixtures.plate_names_synth import (
    CROP_SHAPE,
    LINEUP_CSV,
    FakeReader,
    make_plate,
    write_crop,
    write_match,
)

GOAL = make_plate("plate_001", "goal", 100.0, team="home")
SUBST = make_plate("plate_002", "subst", 900.0, team="away", half=2)


def state(match_dir: Path) -> str:
    return read_status(match_dir).stages[STAGE].state


@pytest.mark.unit
def test_stage_reads_every_plate_and_checks_the_names_against_the_lineup(tmp_path: Path) -> None:
    write_match(tmp_path, [GOAL, SUBST], LINEUP_CSV)
    reader = FakeReader("9 РОМАН ЛЕБЕДЕВ", "21 ЕРЛАН ҚОСАНОВ", "10 АЙДАР НУРЛАНОВ")

    result = run_plate_names(tmp_path, "m", reader=reader)

    assert result is not None
    assert [(n.plate_id, n.type, n.half, n.side) for n in result.names] == [
        ("plate_001", "goal", 1, "home"),
        ("plate_002", "subst", 2, "away"),
    ]
    assert [(p.role, p.name, p.source) for p in result.names[0].players] == [
        ("player", "Лебедев Роман", "ocr+lineup")
    ]
    assert [(p.role, p.name) for p in result.names[1].players] == [
        ("in", "Қосанов Ерлан"),
        ("out", "Нурланов Айдар"),
    ]
    assert result.lineup == "lineups.csv"
    assert (result.stats.n_plates, result.stats.n_names, result.stats.n_matched) == (2, 3, 3)
    assert len(reader.images) == 3  # one line for the goal, two for the substitution
    assert state(tmp_path) == "done"
    stored = PlateNamesResult.model_validate_json((tmp_path / RESULT_FILE).read_text("utf-8"))
    assert stored == result


@pytest.mark.unit
def test_stage_without_a_lineup_gives_the_cleaned_text_marked_unchecked(tmp_path: Path) -> None:
    write_match(tmp_path, [GOAL])

    result = run_plate_names(tmp_path, "m", reader=FakeReader("9 РОМАН ЛЕБЕДЕВ"))

    assert result is not None
    (player,) = result.names[0].players
    assert (player.name, player.status, player.source) == ("Роман Лебедев", "unchecked", "ocr")
    assert player.similarity is None
    assert result.lineup is None
    assert result.lineup_version is None
    assert result.warnings == []


@pytest.mark.unit
def test_stage_plate_without_a_side_is_unchecked_and_warned_about(tmp_path: Path) -> None:
    write_match(tmp_path, [make_plate("plate_003", "card", 50.0, team=None)], LINEUP_CSV)

    result = run_plate_names(tmp_path, "m", reader=FakeReader("9 РОМАН ЛЕБЕДЕВ"))

    assert result is not None
    assert result.names[0].players[0].status == "unchecked"
    assert result.warnings == ["plate_003: side unknown, names not checked against the lineup"]


@pytest.mark.unit
def test_stage_a_name_not_in_the_lineup_is_unrecognized(tmp_path: Path) -> None:
    write_match(tmp_path, [GOAL], LINEUP_CSV)

    result = run_plate_names(tmp_path, "m", reader=FakeReader("7 ИГОРЬ КОВАЛЬЧУК"))

    assert result is not None
    (player,) = result.names[0].players
    assert (player.status, player.name) == ("unrecognized", None)
    assert player.raw_text == "7 ИГОРЬ КОВАЛЬЧУК"
    assert result.stats.n_unrecognized == 1


@pytest.mark.unit
def test_stage_reads_the_upper_line_for_the_player_who_came_on(tmp_path: Path) -> None:
    write_match(tmp_path, [SUBST], crops=False)
    crop = np.full(CROP_SHAPE, 230, dtype=np.uint8)
    layout = NameLayout()
    height, width = CROP_SHAPE
    down = layout.went_off
    crop[
        int(down.y0 * height) : int(down.y1 * height), int(down.x0 * width) : int(down.x1 * width)
    ] = 20
    write_crop(tmp_path, SUBST, crop)
    reader = FakeReader("a", "b")

    run_plate_names(tmp_path, "m", reader=reader)

    upper, lower = reader.images
    assert not (upper == 0).any()  # text only in the lower line
    assert (lower == 0).any()


@pytest.mark.unit
def test_stage_up_to_date_result_is_returned_without_reading_or_tesseract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_match(tmp_path, [GOAL], LINEUP_CSV)
    first = run_plate_names(tmp_path, "m", reader=FakeReader("9 РОМАН ЛЕБЕДЕВ"))
    monkeypatch.setattr(TesseractReader, "create", _no_tesseract)

    again = run_plate_names(tmp_path, "m")

    assert again == first
    assert state(tmp_path) == "done"


@pytest.mark.unit
def test_stage_force_reads_again(tmp_path: Path) -> None:
    write_match(tmp_path, [GOAL])
    run_plate_names(tmp_path, "m", reader=FakeReader("9 РОМАН ЛЕБЕДЕВ"))
    reader = FakeReader("9 РОМАН ЛЕБЕДЕВ")

    run_plate_names(tmp_path, "m", reader=reader, force=True)

    assert len(reader.images) == 1


@pytest.mark.unit
def test_stage_a_changed_lineup_file_invalidates_the_result(tmp_path: Path) -> None:
    write_match(tmp_path, [GOAL], LINEUP_CSV)
    first = run_plate_names(tmp_path, "m", reader=FakeReader("9 РОМАН ЛЕБЕДЕВ"))
    (tmp_path / "lineups.csv").write_text(LINEUP_CSV + "Север,home,3,Орлов Пётр,DF,1,,\n", "utf-8")
    reader = FakeReader("9 РОМАН ЛЕБЕДЕВ")

    second = run_plate_names(tmp_path, "m", reader=reader)

    assert len(reader.images) == 1
    assert first is not None
    assert second is not None
    assert second.cache_key != first.cache_key


@pytest.mark.unit
def test_stage_changed_parameters_invalidate_the_result(tmp_path: Path) -> None:
    write_match(tmp_path, [GOAL])
    run_plate_names(tmp_path, "m", reader=FakeReader("9 РОМАН ЛЕБЕДЕВ"))
    reader = FakeReader("9 РОМАН ЛЕБЕДЕВ")

    run_plate_names(tmp_path, "m", config=PlateNamesConfig(scale=4), reader=reader)

    assert len(reader.images) == 1
    assert reader.images[0].shape[0] > 0


@pytest.mark.unit
def test_stage_corrupt_cache_file_is_recomputed(tmp_path: Path) -> None:
    write_match(tmp_path, [GOAL])
    run_plate_names(tmp_path, "m", reader=FakeReader("9 РОМАН ЛЕБЕДЕВ"))
    (tmp_path / RESULT_FILE).write_text("{not json", encoding="utf-8")
    reader = FakeReader("9 РОМАН ЛЕБЕДЕВ")

    result = run_plate_names(tmp_path, "m", reader=reader)

    assert result is not None
    assert len(reader.images) == 1


def _no_tesseract(*_args: object, **_kwargs: object) -> TesseractReader:
    msg = "tesseract not found (test)"
    raise OcrUnavailableError(msg)


@pytest.mark.unit
def test_stage_without_tesseract_is_skipped_with_a_reason_and_writes_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_match(tmp_path, [GOAL])
    monkeypatch.setattr(TesseractReader, "create", _no_tesseract)

    result = run_plate_names(tmp_path, "m")

    assert result is None
    status = read_status(tmp_path).stages[STAGE]
    assert status.state == "skipped"
    assert status.error == ru.PLATE_NAMES_SKIPPED.format(reason="tesseract not found (test)")
    assert not (tmp_path / RESULT_FILE).exists()


@pytest.mark.unit
def test_stage_skipped_keeps_an_earlier_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_match(tmp_path, [GOAL])
    run_plate_names(tmp_path, "m", reader=FakeReader("9 РОМАН ЛЕБЕДЕВ"))
    before = (tmp_path / RESULT_FILE).read_text("utf-8")
    monkeypatch.setattr(TesseractReader, "create", _no_tesseract)

    assert run_plate_names(tmp_path, "m", force=True) is None

    assert (tmp_path / RESULT_FILE).read_text("utf-8") == before


@pytest.mark.unit
def test_stage_no_plates_gives_an_empty_result(tmp_path: Path) -> None:
    write_match(tmp_path, [])

    result = run_plate_names(tmp_path, "m", reader=FakeReader())

    assert result is not None
    assert result.names == []
    assert result.stats.n_names == 0


@pytest.mark.unit
def test_stage_missing_plates_json_fails_with_a_hint(tmp_path: Path) -> None:
    with pytest.raises(PlatesError, match="run first: python -m aicoach.cli plates"):
        run_plate_names(tmp_path, "m", reader=FakeReader())

    assert state(tmp_path) == "failed"


@pytest.mark.unit
def test_stage_unreadable_plates_json_fails_with_the_error_kind(tmp_path: Path) -> None:
    (tmp_path / "plates.json").mkdir()  # reading a directory fails with an OS error

    with pytest.raises(PlatesError, match="cannot read"):
        run_plate_names(tmp_path, "m", reader=FakeReader())


@pytest.mark.unit
def test_stage_invalid_plates_json_fails(tmp_path: Path) -> None:
    (tmp_path / "plates.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PlatesError, match="not a valid plates result"):
        run_plate_names(tmp_path, "m", reader=FakeReader())


@pytest.mark.unit
def test_stage_plates_made_with_another_plate_region_fail(tmp_path: Path) -> None:
    write_match(tmp_path, [GOAL])
    data = json.loads((tmp_path / "plates.json").read_text("utf-8"))
    data["roi"]["x0"] += 0.01
    (tmp_path / "plates.json").write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(PlatesError, match="another plate region"):
        run_plate_names(tmp_path, "m", reader=FakeReader("x"))


@pytest.mark.unit
def test_stage_missing_crop_names_the_plate(tmp_path: Path) -> None:
    write_match(tmp_path, [GOAL], crops=False)

    with pytest.raises(PlatesError, match=r"plate_001.*run the plates stage again"):
        run_plate_names(tmp_path, "m", reader=FakeReader("x"))

    assert state(tmp_path) == "failed"


@pytest.mark.unit
def test_stage_crop_that_is_not_an_image_fails(tmp_path: Path) -> None:
    write_match(tmp_path, [GOAL], crops=False)
    (tmp_path / GOAL.crop).parent.mkdir()
    (tmp_path / GOAL.crop).write_bytes(b"not a png")

    with pytest.raises(PlatesError, match="is not an image"):
        run_plate_names(tmp_path, "m", reader=FakeReader("x"))


@pytest.mark.unit
def test_stage_crop_path_outside_the_match_directory_is_refused(tmp_path: Path) -> None:
    match_dir = tmp_path / "match"
    write_match(match_dir, [GOAL.model_copy(update={"crop": "../outside.png"})], crops=False)
    (tmp_path / "outside.png").write_bytes(b"x")

    with pytest.raises(PlatesError, match="outside the match directory"):
        run_plate_names(match_dir, "m", reader=FakeReader("x"))


@pytest.mark.unit
def test_stage_broken_lineup_file_fails_the_stage(tmp_path: Path) -> None:
    write_match(tmp_path, [GOAL], "side,name\nleft,Лебедев Роман\n")

    with pytest.raises(PlatesError, match="side must be home or away"):
        run_plate_names(tmp_path, "m", reader=FakeReader("x"))

    assert state(tmp_path) == "failed"


@pytest.mark.unit
def test_stage_explicit_lineup_file_is_used_and_a_missing_one_fails(tmp_path: Path) -> None:
    write_match(tmp_path, [GOAL])
    other = tmp_path / "squad.csv"
    other.write_text(LINEUP_CSV, encoding="utf-8")

    result = run_plate_names(tmp_path, "m", lineup_path=other, reader=FakeReader("9 РОМАН ЛЕБЕДЕВ"))

    assert result is not None
    assert (result.lineup, result.names[0].players[0].status) == ("squad.csv", "matched")
    with pytest.raises(PlatesError, match="cannot read"):
        run_plate_names(tmp_path, "m", lineup_path=tmp_path / "none.csv", reader=FakeReader("x"))


@pytest.mark.unit
def test_stage_reader_failure_fails_the_stage_and_keeps_the_old_result(tmp_path: Path) -> None:
    write_match(tmp_path, [GOAL])
    run_plate_names(tmp_path, "m", reader=FakeReader("9 РОМАН ЛЕБЕДЕВ"))
    before = (tmp_path / RESULT_FILE).read_text("utf-8")

    class Failing:
        def read(self, _image: object) -> str:
            msg = "tesseract failed on a line"
            raise PlatesError(msg)

    with pytest.raises(PlatesError, match="tesseract failed"):
        run_plate_names(tmp_path, "m", reader=Failing(), force=True)

    assert state(tmp_path) == "failed"
    assert (tmp_path / RESULT_FILE).read_text("utf-8") == before
