"""The work directory of `inference.py`, and the conversion of its output."""

import json
from pathlib import Path

import pytest

from aicoach.errors import SpottingError
from aicoach.spotting.schemas import TDeedFile, TDeedLabel, TDeedPrediction, load_tdeed_file
from aicoach.tdeed.output import convert_output, output_name, shift_to_match_time, write_output
from aicoach.tdeed.timecode import Segment, plan_segment
from aicoach.tdeed.workdir import prepare_workdir
from tests.fixtures.tdeed_synth import FAKE_CONFIG, raw_output


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "config" / "SoccerNet").mkdir(parents=True)
    (repo / "config" / "SoccerNet" / "SoccerNet_small.json").write_text(json.dumps(FAKE_CONFIG))
    (repo / "data" / "soccernet").mkdir(parents=True)
    (repo / "data" / "soccernet" / "class.txt").write_text("Penalty\nGoal\n")
    return repo


@pytest.mark.unit
def test_prepare_workdir_patches_batch_and_workers_and_keeps_the_rest(tmp_path: Path) -> None:
    repo, checkpoint = _repo(tmp_path), tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"weights")
    work = tmp_path / "work"

    prepare_workdir(work, repo, checkpoint)

    config = json.loads((work / "config" / "SoccerNet" / "SoccerNet_small.json").read_text())
    assert config == {**FAKE_CONFIG, "batch_size": 2, "num_workers": 1}
    assert (work / "data" / "soccernet" / "class.txt").read_text() == "Penalty\nGoal\n"
    linked = work / "checkpoints" / "SoccerNet" / "SoccerNet_small" / "checkpoint_best.pt"
    assert linked.read_bytes() == b"weights"
    # The cached copy of T-DEED is not touched.
    assert json.loads((repo / "config" / "SoccerNet" / "SoccerNet_small.json").read_text()) == (
        FAKE_CONFIG
    )


@pytest.mark.unit
def test_prepare_workdir_falls_back_to_a_copy_when_links_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, checkpoint = _repo(tmp_path), tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"weights")

    def no_links(_self: Path, _target: Path) -> None:
        msg = "no links here"
        raise OSError(msg)

    monkeypatch.setattr(Path, "hardlink_to", no_links)
    prepare_workdir(tmp_path / "work", repo, checkpoint)

    assert (
        tmp_path / "work" / "checkpoints" / "SoccerNet" / "SoccerNet_small" / "checkpoint_best.pt"
    ).read_bytes() == b"weights"


@pytest.mark.unit
def test_prepare_workdir_missing_or_broken_config_raises(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"w")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SpottingError, match="cannot prepare"):
        prepare_workdir(tmp_path / "w1", empty, checkpoint)

    repo = _repo(tmp_path)
    (repo / "config" / "SoccerNet" / "SoccerNet_small.json").write_text("{not json")
    with pytest.raises(SpottingError, match="cannot prepare"):
        prepare_workdir(tmp_path / "w2", repo, checkpoint)


@pytest.mark.unit
def test_output_name_whole_match_is_the_name_of_the_frozen_file() -> None:
    assert output_name(None) == "tdeed_SoccerNet_small_thr0.05_bs2.json"


@pytest.mark.unit
def test_output_name_segment_does_not_match_the_pattern_of_the_whole_match() -> None:
    name = output_name(plan_segment(800.0, 1392.0, 7940.0))

    assert name == "tdeedseg_SoccerNet_small_thr0.05_bs2_792-1392s.json"
    assert not name.startswith("tdeed_")


def _prediction(frame: int, label: TDeedLabel = TDeedLabel.FOUL) -> TDeedPrediction:
    return TDeedPrediction(frame=frame, label=label, confidence=0.5)


@pytest.mark.unit
def test_shift_to_match_time_adds_the_offset_and_keeps_only_the_piece() -> None:
    segment = Segment(792.0, 792.0, 1392.0, 1452.0)  # frames 19800 .. 34800
    raw = TDeedFile(
        predictions=(_prediction(0), _prediction(12), _prediction(15000), _prediction(15012))
    )  # 19800, 19812, 34800, 34812

    shifted = shift_to_match_time(raw, segment)

    assert [p.frame for p in shifted.predictions] == [19800, 19812, 34800]


@pytest.mark.unit
def test_shift_to_match_time_empty_input_gives_empty_output() -> None:
    segment = Segment(0.0, 0.0, 600.0, 660.0)

    assert shift_to_match_time(TDeedFile(predictions=()), segment).predictions == ()


@pytest.mark.unit
def test_shift_to_match_time_start_off_the_frame_grid_raises() -> None:
    with pytest.raises(SpottingError, match="whole frame"):
        shift_to_match_time(TDeedFile(predictions=()), Segment(0.01, 0.01, 600.0, 660.0))


@pytest.mark.unit
def test_write_output_round_trips_through_the_loader(tmp_path: Path) -> None:
    data = TDeedFile(predictions=(_prediction(12), _prediction(24, TDeedLabel.CORNER)))

    write_output(tmp_path / "o.json", data)

    assert load_tdeed_file(tmp_path / "o.json") == data
    assert '    "predictions"' in (tmp_path / "o.json").read_text()  # indent 4, like T-DEED


@pytest.mark.unit
def test_convert_output_whole_match_keeps_the_raw_content(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text(raw_output([(12, "Foul", 0.138427734375)]))

    count = convert_output(raw, tmp_path / "out.json", None)

    assert count == 1
    assert json.loads((tmp_path / "out.json").read_text()) == json.loads(raw.read_text())


@pytest.mark.unit
def test_convert_output_segment_writes_match_time(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text(raw_output([(12, "Foul", 0.5), (24, "Corner", 0.25)]))
    segment = plan_segment(792.0, 1392.0, 7940.0)

    count = convert_output(raw, tmp_path / "out.json", segment)

    assert count == 2
    frames = [p.frame for p in load_tdeed_file(tmp_path / "out.json").predictions]
    assert frames == [19812, 19824]


@pytest.mark.unit
@pytest.mark.parametrize("content", [None, "{not json", '{"predictions": [{"frame": 1}]}'])
def test_convert_output_missing_or_invalid_raw_file_raises(
    tmp_path: Path, content: str | None
) -> None:
    raw = tmp_path / "raw.json"
    if content is not None:
        raw.write_text(content)

    with pytest.raises(SpottingError):
        convert_output(raw, tmp_path / "out.json", None)

    assert not (tmp_path / "out.json").exists()
