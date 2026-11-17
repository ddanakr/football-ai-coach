"""The review layer: model shots that are not yet events of the catalog (pure functions).

For a reviewer who wants a high Recall of shots, the layer lists every shot detection of the model
(at the threshold of the T-DEED file itself) except those that fall into the merge window of a
shot the catalog already has, and collapses duplicates inside the layer by the same rule as the
fusion stage (`merge_duplicates`). Other classes get no candidates: goals and substitutions come
from the scoreboard and the plates, and fouls gained too little (D-075). The fusion stage, its
thresholds and the `predictions` table are not touched.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from aicoach.agent.context import game_minute
from aicoach.domain.events import EventType
from aicoach.eval.schemas import Prediction
from aicoach.fusion.halves import half_at
from aicoach.fusion.schemas import Half, HalfSpan
from aicoach.review.schemas import ReviewCandidate
from aicoach.scoreboard.mapping import clock_at
from aicoach.scoreboard.schemas import OffsetSegment
from aicoach.spotting.convert import merge_duplicates

# Absolute slack for float noise at the radius, as in `aicoach.spotting.convert`.
_BOUNDARY_EPS: Final = 1e-9

# The classes that get candidates. One place to change if another class is wanted.
LAYER_TYPES: Final = frozenset({EventType.SHOT})

ExistingEvent = tuple[EventType, float]  # the type and `t_v` of an event the catalog has


@dataclass(frozen=True)
class LayerBuild:
    """The candidates and how many detections each filter removed (`n_model` counts the
    detections of the layer classes only)."""

    candidates: list[ReviewCandidate]
    n_model: int
    n_existing: int
    n_merged: int
    n_outside_halves: int


def _near_existing(pred: Prediction, existing: Sequence[ExistingEvent], radius_s: float) -> bool:
    return any(
        kind is pred.type and abs(t_v - pred.t_v) <= radius_s + _BOUNDARY_EPS
        for kind, t_v in existing
    )


def build_review_candidates(
    model: Sequence[Prediction],
    existing: Sequence[ExistingEvent],
    bounds: Mapping[Half, HalfSpan],
    segments: Sequence[OffsetSegment],
    *,
    merge_radius_s: float,
    max_extrapolation_s: float,
) -> LayerBuild:
    """Candidates from the converted model output.

    Args:
        model: All detections in our classes (`to_predictions`), with their confidence; only the
            classes of `LAYER_TYPES` are used.
        existing: Type and time of every event of the catalog (the frozen predictions).
        bounds: Half bounds of the fusion stage; detections outside both halves are dropped.
        segments: Clock segments of `scoreboard.json`, for the game minute.
        merge_radius_s: The merge window of the fusion stage (`FuseConfig.merge_radius_s`): both
            the distance to an existing event and the distance between duplicates.
        max_extrapolation_s: How far the clock mapping may extrapolate (scoreboard parameter).

    Returns:
        The candidates in time order with ids `r:1`, `r:2`, ... and the number of detections
        that each filter removed. The same input always gives the same ids.

    Raises:
        SpottingError: `merge_radius_s` is negative, NaN or infinite.
    """
    model = [p for p in model if p.type in LAYER_TYPES]
    fresh = [p for p in model if not _near_existing(p, existing, merge_radius_s)]
    merged = merge_duplicates(fresh, merge_radius_s)
    placed = [(p, half_at(p.t_v, bounds)) for p in merged]
    inside = sorted(
        ((p, h) for p, h in placed if h is not None), key=lambda ph: (ph[0].t_v, ph[0].type.value)
    )
    candidates = [
        ReviewCandidate(
            id=f"r:{number}",
            type=pred.type,
            t_v=pred.t_v,
            half=half,
            minute=game_minute(clock_at(segments, pred.t_v, max_extrapolation_s)),
            score=pred.score or 0.0,  # `to_predictions` always sets a score
        )
        for number, (pred, half) in enumerate(inside, start=1)
    ]
    return LayerBuild(
        candidates=candidates,
        n_model=len(model),
        n_existing=len(model) - len(fresh),
        n_merged=len(fresh) - len(merged),
        n_outside_halves=len(placed) - len(inside),
    )


def count_candidates(candidates: Sequence[ReviewCandidate]) -> dict[str, dict[str, int]]:
    """Candidates by type and half: `{"shot": {"1": 4, "2": 2}}`; empty types are absent."""
    counts: dict[str, Counter[str]] = {}
    for candidate in candidates:
        counts.setdefault(candidate.type.value, Counter())[str(candidate.half)] += 1
    return {kind: dict(by_half) for kind, by_half in counts.items()}
