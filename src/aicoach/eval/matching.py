"""Greedy matching of predictions to ground-truth events.

Tolerance convention
--------------------
`tolerance_s` is the HALF-width of the matching window: an event is found when
`|t_pred - t_gt| <= tolerance_s`, the boundary included. SoccerNet's action-spotting
protocol defines `delta` as the full window width, so `tolerance_s = delta / 2`
(SoccerNet `delta = 10 s` equals `tolerance_s = 5.0` here).
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from aicoach.domain.events import EventType
from aicoach.errors import EvalError
from aicoach.eval.schemas import GroundTruthEvent, Prediction

# Absolute slack for float noise: 1239.56 - 1234.56 is 5.000000000000227, yet it is a
# 5 s error. Far below the 0.01 s resolution of labels, so it never changes a real verdict.
_BOUNDARY_EPS = 1e-9


class _HasHalf(Protocol):
    @property
    def half(self) -> int | None: ...


_T = TypeVar("_T", bound=_HasHalf)


@dataclass(frozen=True)
class MatchPair:
    """One matched pair; indices point into `MatchResult.predictions` and `.ground_truth`."""

    pred_index: int
    gt_index: int
    abs_dt_s: float


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching: the pairs plus the (normalized) inputs they index into."""

    predictions: tuple[Prediction, ...]
    ground_truth: tuple[GroundTruthEvent, ...]
    pairs: tuple[MatchPair, ...]

    @property
    def pred_matched(self) -> list[bool]:
        """For each prediction: is it a true positive."""
        matched = {pair.pred_index for pair in self.pairs}
        return [i in matched for i in range(len(self.predictions))]

    @property
    def gt_found(self) -> list[bool]:
        """For each ground-truth event: was it found by a prediction."""
        found = {pair.gt_index for pair in self.pairs}
        return [i in found for i in range(len(self.ground_truth))]


def normalize_predictions(predictions: Sequence[Prediction]) -> list[Prediction]:
    """Turn disallowed goals into shots; the inputs are not modified.

    A goal with `status == "disallowed"` was ruled out (for example by VAR), so it is
    evaluated as a shot. Order and length are preserved.
    """
    return [
        p.model_copy(update={"type": EventType.SHOT})
        if p.type is EventType.GOAL and p.status == "disallowed"
        else p
        for p in predictions
    ]


def filter_half(items: Sequence[_T], half: Literal[1, 2]) -> list[_T]:
    """Keep only items from the given half; items with an unknown half are dropped.

    Raises:
        EvalError: `half` is not 1 or 2.
    """
    if half not in (1, 2):
        msg = f"half must be 1 or 2, got {half!r}"
        raise EvalError(msg)
    return [item for item in items if item.half == half]


def match_events(
    predictions: Sequence[Prediction],
    ground_truth: Sequence[GroundTruthEvent],
    tolerance_s: float,
) -> MatchResult:
    """Greedily match predictions to ground truth, class by class.

    Disallowed goals are first turned into shots. Then, within each class, all
    (prediction, truth) pairs with `|dt| <= tolerance_s` are sorted by `|dt|` (ties by
    prediction index, then truth index, so the result is deterministic) and assigned while
    both sides are still free. Each prediction and each truth is used at most once.

    Args:
        predictions: Predictions in any order.
        ground_truth: Ground-truth events in any order.
        tolerance_s: Half-width of the window in seconds; see the module docstring for how
            it relates to SoccerNet's `delta`.

    Returns:
        The pairs, with indices into the (normalized) inputs stored in the result.

    Raises:
        EvalError: `tolerance_s` is negative, NaN or infinite.
    """
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        msg = f"tolerance_s must be a finite non-negative number, got {tolerance_s}"
        raise EvalError(msg)
    preds = tuple(normalize_predictions(predictions))
    truths = tuple(ground_truth)

    truths_by_type: dict[EventType, list[int]] = defaultdict(list)
    for gi, truth in enumerate(truths):
        truths_by_type[truth.type].append(gi)
    candidates: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(preds):
        for gi in truths_by_type.get(pred.type, ()):
            abs_dt = abs(pred.t_v - truths[gi].t_v)
            if abs_dt <= tolerance_s + _BOUNDARY_EPS:
                candidates.append((abs_dt, pi, gi))
    candidates.sort()

    used_preds: set[int] = set()
    used_truths: set[int] = set()
    pairs: list[MatchPair] = []
    for abs_dt, pi, gi in candidates:
        if pi in used_preds or gi in used_truths:
            continue
        used_preds.add(pi)
        used_truths.add(gi)
        pairs.append(MatchPair(pred_index=pi, gt_index=gi, abs_dt_s=abs_dt))
    return MatchResult(predictions=preds, ground_truth=truths, pairs=tuple(pairs))
