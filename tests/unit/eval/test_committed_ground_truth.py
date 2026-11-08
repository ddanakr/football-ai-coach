"""The ground-truth file committed to the repository must pass the project loader."""

from pathlib import Path

import pytest

from aicoach.domain.events import EventType
from aicoach.eval.schemas import GroundTruth, load_ground_truth

GROUND_TRUTH_PATH = (
    Path(__file__).resolve().parents[3] / "data_labels" / "match01_ground_truth.json"
)


@pytest.fixture(scope="module")
def ground_truth() -> GroundTruth:
    """Load the committed file once for the module."""
    return load_ground_truth(GROUND_TRUTH_PATH)


@pytest.mark.unit
def test_committed_ground_truth_passes_the_loader(ground_truth: GroundTruth) -> None:
    assert ground_truth.video
    assert ground_truth.events


@pytest.mark.unit
@pytest.mark.parametrize("half", [1, 2])
def test_committed_ground_truth_has_events_in_both_halves(
    ground_truth: GroundTruth, half: int
) -> None:
    assert any(event.half == half for event in ground_truth.events)


@pytest.mark.unit
@pytest.mark.parametrize("half", ["1", "2"])
def test_committed_ground_truth_has_bounds_for_both_halves(
    ground_truth: GroundTruth, half: str
) -> None:
    bounds = ground_truth.halves[half]  # type: ignore[index]  # str key, validated by the loader
    assert bounds.start is not None
    assert bounds.end is not None


@pytest.mark.unit
def test_committed_ground_truth_uses_only_the_five_allowed_types(
    ground_truth: GroundTruth,
) -> None:
    allowed = {"goal", "shot", "corner", "foul", "subst"}
    assert {member.value for member in EventType} == allowed
    assert {event.type for event in ground_truth.events} <= allowed
