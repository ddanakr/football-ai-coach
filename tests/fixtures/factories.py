"""Factories for synthetic ground-truth events and predictions."""

from typing import Literal

from aicoach.domain.events import EventType
from aicoach.eval.schemas import GroundTruthEvent, Prediction

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
