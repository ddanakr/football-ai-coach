"""The `spotting-eval` command body: read the files, evaluate, write the JSON next to the input.

Side effects (disk, logs) live here; `aicoach.spotting.evaluation` stays pure.
"""

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import structlog

from aicoach.eval.schemas import load_ground_truth
from aicoach.spotting.convert import DEFAULT_FPS, DEFAULT_MERGE_RADIUS_S
from aicoach.spotting.evaluation import (
    DEFAULT_THRESHOLDS,
    DEVELOPMENT_HALF,
    SpottingEvaluation,
    check_half,
    evaluate_tdeed,
    evaluation_to_dict,
)
from aicoach.spotting.schemas import load_tdeed_file
from aicoach.stage_common import elapsed_ms, write_atomic

STAGE = "spotting_eval"
# Default ground truth of the pilot match; relative to the working directory (repository root).
DEFAULT_GROUND_TRUTH = Path("data_labels") / "match01_ground_truth.json"


def default_output_path(predictions_path: Path, half: int) -> Path:
    """`eval_<name of the predictions file>_half<N>.json` next to the predictions file.

    The `eval_` prefix keeps the result out of a `tdeed_*.json` pattern for model outputs.
    """
    return predictions_path.with_name(f"eval_{predictions_path.stem}_half{half}.json")


def run_spotting_eval(
    predictions_path: Path,
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH,
    *,
    output_path: Path | None = None,
    half: Literal[1, 2] = DEVELOPMENT_HALF,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    fps: float = DEFAULT_FPS,
    radius_s: float = DEFAULT_MERGE_RADIUS_S,
) -> tuple[SpottingEvaluation, Path]:
    """Evaluate a T-DEED file on one half and write the result as JSON.

    Args:
        predictions_path: T-DEED `results_inference.json`.
        ground_truth_path: Ground-truth file with the half bounds.
        output_path: Result file; default `default_output_path`.
        half: Must be 1 (half 2 is reserved, D-045).
        thresholds: Confidence thresholds.
        fps: Frame rate of the `frame` field.
        radius_s: Merge radius for duplicates, in seconds.

    Returns:
        The evaluation and the path of the written file.

    Raises:
        SpottingError: The half is not 1, a parameter is invalid, or the T-DEED file is broken.
        EvalError: The ground-truth file is broken or has no bounds for the half.
        StorageError: The result file cannot be written.
    """
    started = time.perf_counter()
    check_half(half)  # before reading anything
    tdeed = load_tdeed_file(predictions_path)
    truth = load_ground_truth(ground_truth_path)
    evaluation = evaluate_tdeed(
        tdeed, truth, half=half, thresholds=thresholds, fps=fps, radius_s=radius_s
    )
    document = evaluation_to_dict(evaluation)
    document["predictions_file"] = predictions_path.name
    document["ground_truth_file"] = ground_truth_path.name
    target = output_path or default_output_path(predictions_path, half)
    write_atomic(target, json.dumps(document, indent=2, allow_nan=False) + "\n")
    structlog.get_logger().info(
        "spotting_eval_done",
        stage=STAGE,
        match_id=predictions_path.resolve().parent.name,
        half=half,
        n_thresholds=len(evaluation.results),
        duration_ms=elapsed_ms(started),
    )
    return evaluation, target
