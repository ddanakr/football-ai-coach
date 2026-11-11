"""Command line `spotting-eval`: a run prints a table and writes JSON, errors give exit code 2."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from aicoach.cli import main
from aicoach.eval.schemas import GroundTruth, HalfBounds
from tests.fixtures.factories import make_gt


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


def _write_truth(path: Path, *, with_bounds: bool = True) -> Path:
    truth = GroundTruth(
        video="synthetic.mp4",
        halves={"1": HalfBounds(start=100.0, end=2000.0), "2": HalfBounds(start=3000.0, end=5000.0)}
        if with_bounds
        else {},
        events=(make_gt("goal", 500.0), make_gt("goal", 3500.0, half=2)),
    )
    path.write_text(truth.model_dump_json(), encoding="utf-8")
    return path


def _write_predictions(path: Path) -> Path:
    items = [
        {"frame": 12525, "label": "Goal", "confidence": 0.8},  # 501 s, half 1
        {"frame": 87525, "label": "Goal", "confidence": 0.8},  # 3501 s, half 2
        {"frame": 5000, "label": "Throw-in", "confidence": 0.9},
    ]
    path.write_text(json.dumps({"predictions": items}), encoding="utf-8")
    return path


@pytest.mark.unit
def test_main_spotting_eval_prints_tables_and_writes_json_next_to_the_predictions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pred = _write_predictions(tmp_path / "tdeed_match.json")
    truth = _write_truth(tmp_path / "truth.json")

    code = main(
        ["spotting-eval", "--predictions", str(pred), "--ground-truth", str(truth),
         "--thresholds", "0.1", "0.5"]
    )  # fmt: skip

    assert code == 0
    out = capsys.readouterr().out
    assert "half 1" in out
    assert "half 2" not in out
    assert "3501" not in out
    result_file = tmp_path / "eval_tdeed_match_half1.json"
    assert str(result_file) in out
    data = json.loads(result_file.read_text(encoding="utf-8"))
    assert [t["threshold"] for t in data["thresholds"]] == [0.1, 0.5]
    assert data["thresholds"][0]["n_predictions"] == 1
    assert data["thresholds"][0]["tolerances"][0]["micro"]["tp"] == 1
    assert data["predictions_file"] == "tdeed_match.json"
    assert "3501" not in json.dumps(data)


@pytest.mark.unit
def test_main_spotting_eval_default_thresholds_are_the_five(tmp_path: Path) -> None:
    pred = _write_predictions(tmp_path / "tdeed_match.json")
    truth = _write_truth(tmp_path / "truth.json")

    code = main(["spotting-eval", "--predictions", str(pred), "--ground-truth", str(truth)])

    assert code == 0
    data = json.loads((tmp_path / "eval_tdeed_match_half1.json").read_text(encoding="utf-8"))
    assert [t["threshold"] for t in data["thresholds"]] == [0.05, 0.1, 0.2, 0.3, 0.5]


@pytest.mark.unit
def test_main_spotting_eval_output_option_chooses_the_file(tmp_path: Path) -> None:
    pred = _write_predictions(tmp_path / "tdeed_match.json")
    truth = _write_truth(tmp_path / "truth.json")
    out = tmp_path / "custom.json"

    code = main(
        ["spotting-eval", "--predictions", str(pred), "--ground-truth", str(truth),
         "--output", str(out)]
    )  # fmt: skip

    assert code == 0
    assert out.is_file()
    assert not (tmp_path / "eval_tdeed_match_half1.json").exists()


@pytest.mark.unit
def test_main_spotting_eval_half_two_exits_with_2_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pred = _write_predictions(tmp_path / "tdeed_match.json")
    truth = _write_truth(tmp_path / "truth.json")

    code = main(
        ["spotting-eval", "--predictions", str(pred), "--ground-truth", str(truth),
         "--half", "2"]
    )  # fmt: skip

    assert code == 2
    assert "reserved" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tdeed_match.json", "truth.json"]


@pytest.mark.unit
@pytest.mark.parametrize("half", ["0", "3"])
def test_main_spotting_eval_invalid_half_exits_with_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], half: str
) -> None:
    pred = _write_predictions(tmp_path / "tdeed_match.json")
    truth = _write_truth(tmp_path / "truth.json")

    code = main(
        ["spotting-eval", "--predictions", str(pred), "--ground-truth", str(truth),
         "--half", half]
    )  # fmt: skip

    assert code == 2
    assert "half" in capsys.readouterr().err


@pytest.mark.unit
def test_main_spotting_eval_unwritable_output_exits_with_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pred = _write_predictions(tmp_path / "tdeed_match.json")
    truth = _write_truth(tmp_path / "truth.json")
    out = tmp_path / "no_such_dir" / "result.json"

    code = main(
        ["spotting-eval", "--predictions", str(pred), "--ground-truth", str(truth),
         "--output", str(out)]
    )  # fmt: skip

    assert code == 2
    assert "cannot write" in capsys.readouterr().err


@pytest.mark.unit
def test_main_spotting_eval_missing_predictions_file_exits_with_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    truth = _write_truth(tmp_path / "truth.json")

    code = main(
        [
            "spotting-eval",
            "--predictions",
            str(tmp_path / "nope.json"),
            "--ground-truth",
            str(truth),
        ]
    )

    assert code == 2
    assert "cannot read" in capsys.readouterr().err


@pytest.mark.unit
def test_main_spotting_eval_unknown_label_exits_with_2_and_names_the_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pred = tmp_path / "tdeed_bad.json"
    pred.write_text(
        json.dumps({"predictions": [{"frame": 1, "label": "Dive", "confidence": 0.5}]}),
        encoding="utf-8",
    )
    truth = _write_truth(tmp_path / "truth.json")

    code = main(["spotting-eval", "--predictions", str(pred), "--ground-truth", str(truth)])

    assert code == 2
    assert "predictions.0.label" in capsys.readouterr().err


@pytest.mark.unit
def test_main_spotting_eval_ground_truth_without_half_bounds_exits_with_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pred = _write_predictions(tmp_path / "tdeed_match.json")
    truth = _write_truth(tmp_path / "truth.json", with_bounds=False)

    code = main(["spotting-eval", "--predictions", str(pred), "--ground-truth", str(truth)])

    assert code == 2
    assert "half 1" in capsys.readouterr().err


@pytest.mark.unit
def test_main_spotting_eval_missing_ground_truth_exits_with_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pred = _write_predictions(tmp_path / "tdeed_match.json")

    code = main(
        ["spotting-eval", "--predictions", str(pred), "--ground-truth", str(tmp_path / "no.json")]
    )

    assert code == 2
    assert "ground-truth" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.parametrize(
    ("option", "value", "name"),
    [
        ("--fps", "0", "fps"),
        ("--radius-s", "-1", "radius_s"),
        ("--thresholds", "-0.5", "threshold"),
    ],
)
def test_main_spotting_eval_invalid_numbers_exit_with_2_and_name_the_option(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], option: str, value: str, name: str
) -> None:
    pred = _write_predictions(tmp_path / "tdeed_match.json")
    truth = _write_truth(tmp_path / "truth.json")

    code = main(
        ["spotting-eval", "--predictions", str(pred), "--ground-truth", str(truth), option, value]
    )

    assert code == 2
    assert name in capsys.readouterr().err
