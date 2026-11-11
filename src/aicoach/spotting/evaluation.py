"""Evaluation of a T-DEED file against the ground truth on the first half only.

The first half is the development half (CLAUDE.md section 7, D-008): thresholds are chosen on
it. The second half is reserved for the final numbers, so this module refuses to touch it
and its output never mentions it (D-045).
"""

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

# Any below: JSON documents of nested dicts, lists, numbers, strings and None.
from typing import Any, Final, Literal

from aicoach.errors import EvalError, SpottingError
from aicoach.eval.bootstrap import DEFAULT_N_BOOT, DEFAULT_SEED
from aicoach.eval.metrics import (
    DEFAULT_TOLERANCES_S,
    EvaluationReport,
    Metrics,
    ToleranceResult,
    evaluate,
)
from aicoach.eval.report import format_markdown
from aicoach.eval.schemas import GroundTruth
from aicoach.spotting.convert import (
    DEFAULT_FPS,
    DEFAULT_MERGE_RADIUS_S,
    filter_by_confidence,
    merge_duplicates,
    predictions_in_half,
    to_predictions,
)
from aicoach.spotting.schemas import TDeedFile

DEFAULT_THRESHOLDS: Final[tuple[float, ...]] = (0.05, 0.1, 0.2, 0.3, 0.5)
# The only half this module evaluates; 2 is kept for the final evaluation (D-045).
DEVELOPMENT_HALF: Final = 1
_EMPTY = "—"


@dataclass(frozen=True)
class ThresholdResult:
    """Metrics at one confidence threshold.

    `n_predictions` counts the predictions of the half left after merging and thresholding.
    """

    threshold: float
    n_predictions: int
    report: EvaluationReport


@dataclass(frozen=True)
class SpottingEvaluation:
    """All thresholds with the parameters that produced them.

    `n_in_half` counts the mapped events inside the half before merging, `n_merged` after.
    """

    half: Literal[1, 2]
    fps: float
    radius_s: float
    tolerances_s: tuple[float, ...]
    n_boot: int
    seed: int
    n_in_half: int
    n_merged: int
    results: tuple[ThresholdResult, ...]


def _check_thresholds(thresholds: Sequence[float]) -> None:
    if not thresholds:
        msg = "thresholds must contain at least one confidence threshold"
        raise SpottingError(msg)
    for value in thresholds:
        if not math.isfinite(value) or value < 0:
            msg = f"threshold must be a finite non-negative number, got {value}"
            raise SpottingError(msg)


def check_half(half: int) -> None:
    """Accept only half 1.

    Raises:
        SpottingError: `half` is 2 (reserved for the final evaluation) or not a half.
    """
    if half == DEVELOPMENT_HALF:
        return
    if half in (1, 2):
        msg = (
            f"half {half} is reserved for the final evaluation (D-045); "
            f"this command evaluates half {DEVELOPMENT_HALF} only"
        )
    else:
        msg = f"half must be {DEVELOPMENT_HALF}, got {half!r}"
    raise SpottingError(msg)


def evaluate_tdeed(
    tdeed: TDeedFile,
    ground_truth: GroundTruth,
    *,
    half: Literal[1, 2] = DEVELOPMENT_HALF,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    fps: float = DEFAULT_FPS,
    radius_s: float = DEFAULT_MERGE_RADIUS_S,
    tolerances_s: Sequence[float] = DEFAULT_TOLERANCES_S,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> SpottingEvaluation:
    """Evaluate the T-DEED predictions of one half at several confidence thresholds.

    Pipeline: convert to our events, keep the half by the ground-truth bounds, merge
    duplicates once, then for every threshold keep `score >= threshold` and evaluate.
    Merging before thresholding gives the same events as after it (the strongest event of a
    group always passes first), so the order is only an optimization.

    Args:
        tdeed: Validated T-DEED output.
        ground_truth: Ground truth with the bounds of the half.
        half: Must be 1; half 2 is reserved for the final evaluation.
        thresholds: Confidence thresholds, each finite and not negative.
        fps: Frame rate of the `frame` field.
        radius_s: Merge radius for duplicates, in seconds.
        tolerances_s: Half-widths of the matching window (`tolerance_s`, D-026).
        n_boot: Bootstrap resamples; `seed` is the seed.
        seed: Bootstrap seed.

    Returns:
        One report per threshold, in the order given.

    Raises:
        SpottingError: `half` is not 1 (2 is reserved), a threshold, `fps` or `radius_s` is
            invalid.
        EvalError: The ground truth has no complete bounds for the half, or a tolerance is
            invalid.
    """
    check_half(half)
    _check_thresholds(thresholds)
    bounds = ground_truth.halves.get("1")
    if bounds is None or bounds.start is None or bounds.end is None:
        msg = "the ground truth has no start and end for half 1: mark the half bounds first"
        raise EvalError(msg)
    in_half = predictions_in_half(to_predictions(tdeed, fps), half, ground_truth.halves)
    merged = merge_duplicates(in_half, radius_s)
    results: list[ThresholdResult] = []
    for threshold in thresholds:
        kept = filter_by_confidence(merged, threshold)
        report = evaluate(
            kept,
            ground_truth.events,
            tolerances_s=tolerances_s,
            half=half,
            halves=ground_truth.halves,
            n_boot=n_boot,
            seed=seed,
        )
        results.append(ThresholdResult(threshold, len(kept), report))
    return SpottingEvaluation(
        half=half,
        fps=fps,
        radius_s=radius_s,
        tolerances_s=tuple(tolerances_s),
        n_boot=n_boot,
        seed=seed,
        n_in_half=len(in_half),
        n_merged=len(merged),
        results=tuple(results),
    )


def _metrics_dict(metrics: Metrics) -> dict[str, Any]:
    return dataclasses.asdict(metrics)


def _tolerance_dict(result: ToleranceResult) -> dict[str, Any]:
    return {
        "tolerance_s": result.tolerance_s,
        "per_class": {kind.value: _metrics_dict(m) for kind, m in result.per_class.items()},
        "micro": _metrics_dict(result.micro),
    }


def evaluation_to_dict(evaluation: SpottingEvaluation) -> dict[str, Any]:
    """The evaluation as a JSON-ready dictionary (no NaN, no half-2 data).

    Metrics per threshold, per tolerance, per class and micro; every metric as in
    `aicoach.eval.metrics.Metrics` (`precision` and `recall` are `null` when undefined).
    """
    return {
        "half": evaluation.half,
        "fps": evaluation.fps,
        "merge_radius_s": evaluation.radius_s,
        "tolerances_s": list(evaluation.tolerances_s),
        "n_boot": evaluation.n_boot,
        "seed": evaluation.seed,
        "n_in_half": evaluation.n_in_half,
        "n_merged": evaluation.n_merged,
        "thresholds": [
            {
                "threshold": item.threshold,
                "n_predictions": item.n_predictions,
                "tolerances": [_tolerance_dict(r) for r in item.report.results],
            }
            for item in evaluation.results
        ],
    }


def _number(value: float | None) -> str:
    return _EMPTY if value is None else f"{value:.3f}"


def _summary(evaluation: SpottingEvaluation) -> str:
    """One row per threshold: found, precision and recall of all classes per tolerance."""
    header = ["Threshold", "Predictions"]
    for tolerance in evaluation.tolerances_s:
        header += [f"Found ±{tolerance:g}", f"Precision ±{tolerance:g}", f"Recall ±{tolerance:g}"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for item in evaluation.results:
        cells = [f"{item.threshold:g}", str(item.n_predictions)]
        for result in item.report.results:
            micro = result.micro
            cells += [str(micro.tp), _number(micro.precision), _number(micro.recall)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_evaluation(evaluation: SpottingEvaluation) -> str:
    """Console text: a summary of all thresholds, then the tables per threshold.

    "Found" is the number of ground-truth events found within the tolerance.
    """
    blocks = [
        f"## Summary, half {evaluation.half}, all classes (micro)",
        _summary(evaluation),
    ]
    for item in evaluation.results:
        blocks.append(
            f"## Confidence threshold {item.threshold:g}: {item.n_predictions} predictions"
        )
        blocks.append(format_markdown(item.report).rstrip("\n"))
    return "\n\n".join(blocks) + "\n"
