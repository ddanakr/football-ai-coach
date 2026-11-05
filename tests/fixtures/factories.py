"""Factories for synthetic ground-truth events and predictions."""

from typing import Literal

from aicoach.domain.events import EventType
from aicoach.eval.schemas import GroundTruthEvent, Prediction
from aicoach.storage import NewPrediction

Half = Literal[1, 2]


def make_gt(
    event_type: EventType | str,
    t_v: float,
    *,
    half: Half = 1,
) -> GroundTruthEvent:
    """Build a ground-truth event with neutral defaults."""
    return GroundTruthEvent(
        type=EventType(event_type),
        t_v=t_v,
        half=half,
        annotator="tester",
    )


def make_pred(
    event_type: EventType | str,
    t_v: float,
    *,
    score: float | None = None,
    status: str = "auto",
    half: Half | None = None,
) -> Prediction:
    """Build a prediction with neutral defaults."""
    return Prediction(type=EventType(event_type), t_v=t_v, score=score, status=status, half=half)


def make_new_prediction(
    event_type: EventType | str,
    t_v: float,
    *,
    source: Literal["plates", "model", "speech", "merge"] = "model",
    score: float | None = None,
    half: Half | None = None,
    team: Literal["home", "away"] | None = None,
) -> NewPrediction:
    """Build a storable prediction with neutral defaults."""
    return NewPrediction(
        type=EventType(event_type), t_v=t_v, source=source, score=score, half=half, team=team
    )
