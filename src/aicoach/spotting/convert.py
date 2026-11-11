"""From T-DEED detections to our events: time, class mapping, duplicates, threshold, half."""

import math
from collections.abc import Mapping, Sequence
from typing import Final, Literal

from aicoach.domain.events import EventType
from aicoach.errors import SpottingError
from aicoach.eval.matching import filter_predictions_by_half
from aicoach.eval.schemas import HalfBounds, Prediction
from aicoach.spotting.schemas import TDeedFile, TDeedLabel

# T-DEED's sampling step is fixed in source frames at 25 fps (D-042), so 25 is the value of
# its files, a parameter only for the tests and for a re-encoded source.
DEFAULT_FPS: Final = 25.0
# Two shot classes at one point give a duplicate; 3 s is several steps of the model (0.48 s)
# and well inside the matching window (D-044).
DEFAULT_MERGE_RADIUS_S: Final = 3.0
# Absolute slack for float noise at the radius, as in `aicoach.eval.matching`.
_BOUNDARY_EPS: Final = 1e-9

# 6 of 17 classes are kept (D-043). Penalty is dropped on purpose: it would duplicate a shot
# or a goal. Cards are dropped: the plates stage gives them.
LABEL_TO_EVENT: Final[Mapping[TDeedLabel, EventType | None]] = {
    TDeedLabel.PENALTY: None,
    TDeedLabel.KICK_OFF: None,
    TDeedLabel.GOAL: EventType.GOAL,
    TDeedLabel.SUBSTITUTION: EventType.SUBST,
    TDeedLabel.OFFSIDE: None,
    TDeedLabel.SHOTS_ON_TARGET: EventType.SHOT,
    TDeedLabel.SHOTS_OFF_TARGET: EventType.SHOT,
    TDeedLabel.CLEARANCE: None,
    TDeedLabel.BALL_OUT_OF_PLAY: None,
    TDeedLabel.THROW_IN: None,
    TDeedLabel.FOUL: EventType.FOUL,
    TDeedLabel.INDIRECT_FREE_KICK: None,
    TDeedLabel.DIRECT_FREE_KICK: None,
    TDeedLabel.CORNER: EventType.CORNER,
    TDeedLabel.YELLOW_CARD: None,
    TDeedLabel.RED_CARD: None,
    TDeedLabel.YELLOW_RED_CARD: None,
}


def to_predictions(file: TDeedFile, fps: float = DEFAULT_FPS) -> list[Prediction]:
    """Convert detections to predictions: `t_v = frame / fps`, classes mapped, others dropped.

    Args:
        file: Validated T-DEED output.
        fps: Frame rate the `frame` field counts in (25 for T-DEED).

    Returns:
        Predictions with `status="auto"`, `score` = confidence and no `half`, in file order.

    Raises:
        SpottingError: `fps` is not a positive finite number.
    """
    if not math.isfinite(fps) or fps <= 0:
        msg = f"fps must be a positive finite number, got {fps}"
        raise SpottingError(msg)
    result: list[Prediction] = []
    for item in file.predictions:
        event_type = LABEL_TO_EVENT[item.label]
        if event_type is None:
            continue
        result.append(
            Prediction(type=event_type, t_v=item.frame / fps, score=item.confidence, status="auto")
        )
    return result


def _strength(pred: Prediction) -> float:
    """Confidence for ordering; an event without a score is the weakest."""
    return -math.inf if pred.score is None else pred.score


def merge_duplicates(
    predictions: Sequence[Prediction], radius_s: float = DEFAULT_MERGE_RADIUS_S
) -> list[Prediction]:
    """Per event type, keep the most confident event of every group closer than `radius_s`.

    Greedy: events are visited by falling confidence (ties: the earlier time; no score counts
    as the lowest), and an event is dropped if a kept event of the same type lies within
    `radius_s`, the boundary included. Suppression is not transitive: only kept events
    suppress.

    Args:
        predictions: Predictions in any order.
        radius_s: Radius in seconds, finite and not negative.

    Returns:
        The kept predictions sorted by time.

    Raises:
        SpottingError: `radius_s` is negative, NaN or infinite.
    """
    if not math.isfinite(radius_s) or radius_s < 0:
        msg = f"radius_s must be a finite non-negative number, got {radius_s}"
        raise SpottingError(msg)
    kept: list[Prediction] = []
    for pred in sorted(predictions, key=lambda p: (-_strength(p), p.t_v)):
        near = any(
            other.type is pred.type and abs(other.t_v - pred.t_v) <= radius_s + _BOUNDARY_EPS
            for other in kept
        )
        if not near:
            kept.append(pred)
    return sorted(kept, key=lambda p: p.t_v)


def filter_by_confidence(predictions: Sequence[Prediction], threshold: float) -> list[Prediction]:
    """Keep predictions with `score >= threshold` (the threshold itself stays).

    A prediction without a score is dropped.

    Raises:
        SpottingError: `threshold` is negative, NaN or infinite.
    """
    if not math.isfinite(threshold) or threshold < 0:
        msg = f"threshold must be a finite non-negative number, got {threshold}"
        raise SpottingError(msg)
    return [p for p in predictions if p.score is not None and p.score >= threshold]


def predictions_in_half(
    predictions: Sequence[Prediction],
    half: Literal[1, 2],
    halves: Mapping[Literal["1", "2"], HalfBounds],
) -> list[Prediction]:
    """Keep the predictions inside the half bounds (ends included) and set their `half`.

    Raises:
        EvalError: The bounds of `half` are missing or incomplete.
    """
    inside = filter_predictions_by_half(predictions, half, halves)
    return [p.model_copy(update={"half": half}) for p in inside]
