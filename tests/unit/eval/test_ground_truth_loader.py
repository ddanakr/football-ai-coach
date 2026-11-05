"""Loading and validating the ground-truth JSON written by the labeling tool."""

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from aicoach.domain.events import EventType
from aicoach.errors import AicoachError, EvalError
from aicoach.eval.schemas import GroundTruth, Prediction, load_ground_truth

# Any: a raw JSON payload with mixed value types (str, float, None, list, nested dict); the
# tests mutate and re-serialize it, so a narrower type would only add casts.
VALID: dict[str, Any] = {
    "video": "synthetic.mp4",
    "halves": {"1": {"start": 12.0, "end": 2800.5}, "2": {"start": None, "end": None}},
    "events": [
        {
            "type": "goal",
            "t_v": 1234.5,
            "half": 1,
            "clock": None,
            "team": None,
            "visibility": True,
            "replay": False,
            "annotator": "tester",
            "note": "",
        },
        {"type": "subst", "t_v": 3000, "half": 2},
    ],
}


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _with_event_change(**changes: Any) -> dict[str, Any]:  # noqa: ANN401  # test payload values
    data = copy.deepcopy(VALID)
    data["events"][0].update(changes)
    return data


@pytest.mark.unit
def test_load_ground_truth_valid_file_returns_typed_events(tmp_path: Path) -> None:
    gt = load_ground_truth(_write(tmp_path, VALID))

    assert gt.video == "synthetic.mp4"
    assert [e.type for e in gt.events] == [EventType.GOAL, EventType.SUBST]
    assert gt.events[0].t_v == 1234.5
    assert gt.events[1].t_v == 3000.0
    assert gt.events[1].visibility is True  # defaults for omitted optional fields
    assert gt.events[1].replay is False
    assert gt.halves["1"].end == 2800.5
    assert gt.halves["2"].start is None


@pytest.mark.unit
def test_load_ground_truth_empty_events_is_valid(tmp_path: Path) -> None:
    gt = load_ground_truth(_write(tmp_path, {"video": "v.mp4", "events": []}))

    assert gt.events == ()
    assert gt.halves == {}


@pytest.mark.unit
def test_load_ground_truth_accepts_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(VALID), encoding="utf-8-sig")

    assert len(load_ground_truth(path).events) == 2


@pytest.mark.unit
def test_load_ground_truth_broken_json_raises_readable_error(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text('{"video": "v.mp4", "events": [', encoding="utf-8")

    with pytest.raises(EvalError, match="not valid JSON") as info:
        load_ground_truth(path)

    assert "labels.json" in str(info.value)


@pytest.mark.unit
def test_load_ground_truth_missing_file_raises_eval_error(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="cannot read"):
        load_ground_truth(tmp_path / "absent.json")


@pytest.mark.unit
def test_load_ground_truth_missing_required_top_level_field_names_the_field(
    tmp_path: Path,
) -> None:
    payload = {"video": "v.mp4"}

    with pytest.raises(EvalError, match="events"):
        load_ground_truth(_write(tmp_path, payload))


@pytest.mark.unit
def test_load_ground_truth_missing_required_event_field_names_location(tmp_path: Path) -> None:
    payload = copy.deepcopy(VALID)
    del payload["events"][1]["t_v"]

    with pytest.raises(EvalError, match=r"events\.1\.t_v"):
        load_ground_truth(_write(tmp_path, payload))


@pytest.mark.unit
@pytest.mark.parametrize(
    "changes",
    [
        {"type": "penalty"},
        {"type": None},
        {"half": 3},
        {"half": 0},
        {"t_v": -1.0},
        {"t_v": "12"},
        {"t_v": None},
        {"t_v": True},
        {"team": "red"},
        {"visibility": "maybe"},
    ],
)
def test_load_ground_truth_invalid_event_value_raises_eval_error(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    with pytest.raises(EvalError, match=r"events\.0"):
        load_ground_truth(_write(tmp_path, _with_event_change(**changes)))


@pytest.mark.unit
@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_load_ground_truth_non_finite_time_raises_eval_error(tmp_path: Path, bad: float) -> None:
    # json.dumps writes NaN / Infinity, which Python's json module reads back.
    with pytest.raises(EvalError, match=r"events\.0\.t_v"):
        load_ground_truth(_write(tmp_path, _with_event_change(t_v=bad)))


@pytest.mark.unit
@pytest.mark.parametrize("payload", [[], "text", 5, None])
def test_load_ground_truth_root_is_not_an_object_raises_eval_error(
    tmp_path: Path, payload: object
) -> None:
    with pytest.raises(EvalError):
        load_ground_truth(_write(tmp_path, payload))


@pytest.mark.unit
def test_load_ground_truth_bad_half_key_or_reversed_bounds_raises_eval_error(
    tmp_path: Path,
) -> None:
    bad_key = {**VALID, "halves": {"3": {"start": 0.0, "end": 1.0}}}
    reversed_bounds = {**VALID, "halves": {"1": {"start": 10.0, "end": 5.0}}}

    with pytest.raises(EvalError, match="halves"):
        load_ground_truth(_write(tmp_path, bad_key))
    with pytest.raises(EvalError, match="halves"):
        load_ground_truth(_write(tmp_path, reversed_bounds))


@pytest.mark.unit
def test_eval_error_is_part_of_the_project_exception_hierarchy() -> None:
    assert issubclass(EvalError, AicoachError)


@pytest.mark.unit
def test_ground_truth_model_is_immutable() -> None:
    gt = GroundTruth.model_validate(VALID)

    with pytest.raises(ValueError, match="frozen"):
        gt.video = "other.mp4"


@pytest.mark.unit
def test_prediction_defaults_and_validation() -> None:
    pred = Prediction(type=EventType.SHOT, t_v=10.0)

    assert pred.score is None
    assert pred.status == "auto"
    assert pred.half is None
    with pytest.raises(ValueError, match="t_v"):
        Prediction(type=EventType.SHOT, t_v=math.nan)
    with pytest.raises(ValueError, match="score"):
        Prediction(type=EventType.SHOT, t_v=1.0, score=math.nan)
    with pytest.raises(ValueError, match="type"):
        Prediction.model_validate({"type": "offside", "t_v": 1.0})
    with pytest.raises(ValueError, match="frozen"):
        pred.t_v = 5.0
