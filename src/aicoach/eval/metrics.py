"""Precision, recall, timing error and bootstrap intervals per class and overall.

The tolerance parameter `tolerance_s` is the half-width of the matching window, not
SoccerNet's `delta` (full width); see `aicoach.eval.matching`.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from aicoach.domain.events import EventType
from aicoach.errors import EvalError
from aicoach.eval.bootstrap import DEFAULT_N_BOOT, DEFAULT_SEED, bootstrap_ci
from aicoach.eval.matching import (
    MatchResult,
    filter_half,
    filter_predictions_by_half,
    match_events,
)
from aicoach.eval.schemas import GroundTruthEvent, HalfBounds, Prediction
from aicoach.eval.stats import percentile

# Tolerances from TZ section 5: +-5 s and +-10 s.
DEFAULT_TOLERANCES_S: tuple[float, ...] = (5.0, 10.0)
_MEDIAN_Q = 50.0
_P90_Q = 90.0


@dataclass(frozen=True)
class Metrics:
    """Counts and rates for one class (or for all classes together).

    `precision` and `recall` are `None` when their denominator is zero (no predictions,
    no ground truth). The intervals are 95% bootstrap intervals, `None` for empty data.
    Errors are `|t_pred - t_gt|` in seconds over the matched pairs.
    """

    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    precision_ci: tuple[float, float] | None
    recall_ci: tuple[float, float] | None
    median_error_s: float | None
    p90_error_s: float | None


@dataclass(frozen=True)
class ToleranceResult:
    """Metrics at one tolerance: every class plus the micro average over all classes."""

    tolerance_s: float
    per_class: dict[EventType, Metrics]
    micro: Metrics


@dataclass(frozen=True)
class EvaluationReport:
    """Results for all requested tolerances, in the order requested."""

    half: Literal[1, 2] | None
    results: tuple[ToleranceResult, ...]

    def result_for(self, tolerance_s: float) -> ToleranceResult:
        """Return the result for one tolerance.

        Raises:
            EvalError: The tolerance was not evaluated.
        """
        for result in self.results:
            # isclose, not ==: 0.1 + 0.2 and 0.3 are the same tolerance.
            if math.isclose(result.tolerance_s, tolerance_s, rel_tol=1e-9, abs_tol=1e-9):
                return result
        msg = f"tolerance {tolerance_s} was not evaluated"
        raise EvalError(msg)


def _metrics(
    pred_flags: Sequence[bool],
    gt_flags: Sequence[bool],
    errors: Sequence[float],
    *,
    n_boot: int,
    seed: int,
) -> Metrics:
    """Build `Metrics` from per-prediction and per-truth flags and matched-pair errors."""
    tp = sum(gt_flags)
    n_pred, n_gt = len(pred_flags), len(gt_flags)
    return Metrics(
        tp=tp,
        fp=n_pred - tp,
        fn=n_gt - tp,
        precision=tp / n_pred if n_pred else None,
        recall=tp / n_gt if n_gt else None,
        precision_ci=bootstrap_ci(pred_flags, n_boot=n_boot, seed=seed),
        recall_ci=bootstrap_ci(gt_flags, n_boot=n_boot, seed=seed),
        median_error_s=percentile(errors, _MEDIAN_Q),
        p90_error_s=percentile(errors, _P90_Q),
    )


def _evaluate_matching(
    match: MatchResult, *, n_boot: int, seed: int
) -> tuple[dict[EventType, Metrics], Metrics]:
    """Split one matching into per-class metrics and the micro total."""
    pred_matched = match.pred_matched
    gt_found = match.gt_found
    error_by_gt = {pair.gt_index: pair.abs_dt_s for pair in match.pairs}

    per_class: dict[EventType, Metrics] = {}
    for event_type in EventType:
        pred_idx = [i for i, p in enumerate(match.predictions) if p.type is event_type]
        gt_idx = [i for i, g in enumerate(match.ground_truth) if g.type is event_type]
        per_class[event_type] = _metrics(
            [pred_matched[i] for i in pred_idx],
            [gt_found[i] for i in gt_idx],
            [error_by_gt[i] for i in gt_idx if i in error_by_gt],
            n_boot=n_boot,
            seed=seed,
        )
    micro = _metrics(
        pred_matched, gt_found, [pair.abs_dt_s for pair in match.pairs], n_boot=n_boot, seed=seed
    )
    return per_class, micro


def evaluate(
    predictions: Sequence[Prediction],
    ground_truth: Sequence[GroundTruthEvent],
    *,
    tolerances_s: Sequence[float] = DEFAULT_TOLERANCES_S,
    half: Literal[1, 2] | None = None,
    halves: Mapping[Literal["1", "2"], HalfBounds] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> EvaluationReport:
    """Evaluate predictions against ground truth at one or more tolerances.

    Args:
        predictions: System predictions; disallowed goals are evaluated as shots.
        ground_truth: Ground-truth events.
        tolerances_s: Half-widths of the matching window in seconds (SoccerNet `delta / 2`).
        half: If set, keep only events of that half before evaluating. A prediction with
            `half=None` is placed by `halves`; one outside the bounds is dropped.
        halves: Half bounds of the ground-truth file (`GroundTruth.halves`); read only when
            `half` is set and some prediction has no `half`.
        n_boot: Bootstrap resamples for the confidence intervals.
        seed: Bootstrap seed; the same inputs always give the same report.

    Returns:
        Per-class and micro metrics for every tolerance.

    Raises:
        EvalError: No tolerances given, an invalid tolerance, half or `n_boot`; or predictions
            without `half` must be placed but the bounds of the requested half are null.
    """
    if not tolerances_s:
        msg = "tolerances_s must contain at least one tolerance"
        raise EvalError(msg)
    if half is not None:
        predictions = filter_predictions_by_half(predictions, half, halves)
        ground_truth = filter_half(ground_truth, half)
    results: list[ToleranceResult] = []
    for tolerance_s in tolerances_s:
        match = match_events(predictions, ground_truth, tolerance_s)
        per_class, micro = _evaluate_matching(match, n_boot=n_boot, seed=seed)
        results.append(ToleranceResult(tolerance_s=tolerance_s, per_class=per_class, micro=micro))
    return EvaluationReport(half=half, results=tuple(results))
