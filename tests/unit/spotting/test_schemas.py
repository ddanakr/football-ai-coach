"""Reading the T-DEED output file: valid files and the ways a file can be wrong."""

import json
from pathlib import Path

import pytest

from aicoach.errors import AicoachError, SpottingError
from aicoach.spotting.schemas import TDeedFile, TDeedLabel, load_tdeed_file

# The 17 class names of SoccerNet action spotting, in the order of T-DEED's class.txt.
SOCCERNET_LABELS = [
    "Penalty",
    "Kick-off",
    "Goal",
    "Substitution",
    "Offside",
    "Shots on target",
    "Shots off target",
    "Clearance",
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Direct free-kick",
    "Corner",
    "Yellow card",
    "Red card",
    "Yellow->red card",
]


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "tdeed_test.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.unit
def test_label_enum_has_exactly_the_17_soccernet_classes() -> None:
    assert [label.value for label in TDeedLabel] == SOCCERNET_LABELS


@pytest.mark.unit
def test_load_tdeed_file_valid_file_returns_predictions(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "predictions": [
                {"frame": 300, "label": "Goal", "confidence": 0.8},
                {"frame": 12, "label": "Yellow->red card", "confidence": 0.25},
            ]
        },
    )

    result = load_tdeed_file(path)

    assert isinstance(result, TDeedFile)
    assert [(p.frame, p.label, p.confidence) for p in result.predictions] == [
        (300, TDeedLabel.GOAL, 0.8),
        (12, TDeedLabel.YELLOW_RED_CARD, 0.25),
    ]


@pytest.mark.unit
def test_load_tdeed_file_empty_predictions_list_is_valid(tmp_path: Path) -> None:
    assert load_tdeed_file(_write(tmp_path, {"predictions": []})).predictions == ()


@pytest.mark.unit
def test_load_tdeed_file_tolerates_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "tdeed_bom.json"
    path.write_text('{"predictions": []}', encoding="utf-8-sig")

    assert load_tdeed_file(path).predictions == ()


@pytest.mark.unit
def test_load_tdeed_file_missing_file_raises_spotting_error(tmp_path: Path) -> None:
    with pytest.raises(SpottingError, match="cannot read"):
        load_tdeed_file(tmp_path / "nope.json")


@pytest.mark.unit
def test_load_tdeed_file_is_an_aicoach_error(tmp_path: Path) -> None:
    with pytest.raises(AicoachError):
        load_tdeed_file(tmp_path / "nope.json")


@pytest.mark.unit
@pytest.mark.parametrize("text", ["", "{", "not json", '{"predictions": [}'])
def test_load_tdeed_file_broken_json_raises_spotting_error(tmp_path: Path, text: str) -> None:
    path = tmp_path / "tdeed_broken.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(SpottingError, match="not valid JSON"):
        load_tdeed_file(path)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ([], "<root>"),
        ({}, "predictions"),
        ({"predictions": None}, "predictions"),
        ({"predictions": [{"label": "Goal", "confidence": 0.5}]}, "predictions.0.frame"),
        ({"predictions": [{"frame": 5, "confidence": 0.5}]}, "predictions.0.label"),
        ({"predictions": [{"frame": 5, "label": "Goal"}]}, "predictions.0.confidence"),
    ],
)
def test_load_tdeed_file_missing_fields_name_the_field(
    tmp_path: Path, payload: object, field: str
) -> None:
    with pytest.raises(SpottingError, match=field):
        load_tdeed_file(_write(tmp_path, payload))


@pytest.mark.unit
@pytest.mark.parametrize("label", ["goal", "Penalty kick", "", "Yellow card ", 7, None])
def test_load_tdeed_file_unknown_label_raises_and_names_the_field(
    tmp_path: Path, label: object
) -> None:
    path = _write(tmp_path, {"predictions": [{"frame": 5, "label": label, "confidence": 0.5}]})

    with pytest.raises(SpottingError, match="predictions.0.label"):
        load_tdeed_file(path)


@pytest.mark.unit
@pytest.mark.parametrize("frame", [-1, -250, 1.5, "12", None, True])
def test_load_tdeed_file_bad_frame_raises_and_names_the_field(
    tmp_path: Path, frame: object
) -> None:
    path = _write(tmp_path, {"predictions": [{"frame": frame, "label": "Goal", "confidence": 0.5}]})

    with pytest.raises(SpottingError, match="predictions.0.frame"):
        load_tdeed_file(path)


@pytest.mark.unit
def test_load_tdeed_file_frame_zero_is_valid(tmp_path: Path) -> None:
    path = _write(tmp_path, {"predictions": [{"frame": 0, "label": "Goal", "confidence": 0.5}]})

    assert load_tdeed_file(path).predictions[0].frame == 0


@pytest.mark.unit
@pytest.mark.parametrize("confidence", ["0.5", None, True, "nan"])
def test_load_tdeed_file_bad_confidence_raises(tmp_path: Path, confidence: object) -> None:
    path = _write(
        tmp_path, {"predictions": [{"frame": 5, "label": "Goal", "confidence": confidence}]}
    )

    with pytest.raises(SpottingError, match="predictions.0.confidence"):
        load_tdeed_file(path)


@pytest.mark.unit
@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_load_tdeed_file_non_finite_confidence_raises(tmp_path: Path, literal: str) -> None:
    # json.dumps cannot write these, but Python's json module reads them from a file.
    path = tmp_path / "tdeed_nan.json"
    path.write_text(
        '{"predictions": [{"frame": 5, "label": "Goal", "confidence": ' + literal + "}]}",
        encoding="utf-8",
    )

    with pytest.raises(SpottingError, match="predictions.0.confidence"):
        load_tdeed_file(path)


@pytest.mark.unit
def test_load_tdeed_file_error_names_the_file_and_the_index_of_the_bad_item(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        {
            "predictions": [
                {"frame": 1, "label": "Goal", "confidence": 0.5},
                {"frame": -3, "label": "Goal", "confidence": 0.5},
            ]
        },
    )

    with pytest.raises(SpottingError) as info:
        load_tdeed_file(path)

    assert path.name in str(info.value)
    assert "predictions.1.frame" in str(info.value)
