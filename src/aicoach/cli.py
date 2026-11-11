"""Command line entry point: `python -m aicoach.cli <stage> ...` (CLAUDE.md section 4).

Thin by design: it parses arguments, calls the stage and reports. The video path is only an
argument; results are written next to the video (the match directory).
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from aicoach.errors import AicoachError, ConfigError
from aicoach.logging_setup import configure_logging
from aicoach.plates.config import PlatesConfig
from aicoach.plates.stage import run_plates
from aicoach.scoreboard.config import ScoreboardConfig
from aicoach.scoreboard.stage import (
    DEFAULT_CALIBRATION_STEP_S,
    run_calibration,
    run_labelling,
    run_scoreboard,
)
from aicoach.spotting.convert import DEFAULT_FPS, DEFAULT_MERGE_RADIUS_S
from aicoach.spotting.evaluation import DEFAULT_THRESHOLDS, DEVELOPMENT_HALF, format_evaluation
from aicoach.spotting.run import DEFAULT_GROUND_TRUTH, run_spotting_eval

_DEFAULT_WORKDIR_ROOT = Path("scratch")
_Config = TypeVar("_Config", ScoreboardConfig, PlatesConfig)


def build_parser() -> argparse.ArgumentParser:
    """The argument parser with one sub-command per stage."""
    parser = argparse.ArgumentParser(prog="aicoach", description="Football AI Coach stages")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING or ERROR")
    commands = parser.add_subparsers(dest="command", required=True)

    board = commands.add_parser("scoreboard", help="read the clock and the score of the broadcast")
    board.add_argument("--video", type=Path, required=True, help="path of the match video")
    board.add_argument("--match-id", help="default: name of the video's directory")
    board.add_argument("--force", action="store_true", help="ignore an up-to-date result")
    board.add_argument("--sample-fps", type=float, help="frames per second to read (default 1)")
    board.add_argument("--cpu", action="store_true", help="decode on the CPU (very slow for AV1)")

    plates = commands.add_parser(
        "plates", help="find substitution, goal and card plates (needs scoreboard.json)"
    )
    plates.add_argument("--video", type=Path, required=True, help="path of the match video")
    plates.add_argument("--match-id", help="default: name of the video's directory")
    plates.add_argument("--force", action="store_true", help="ignore an up-to-date result")
    plates.add_argument("--sample-fps", type=float, help="frames per second to read (default 2)")
    plates.add_argument("--cpu", action="store_true", help="decode on the CPU (very slow for AV1)")

    calib = commands.add_parser(
        "scoreboard-calibrate", help="cut digit templates from the broadcast (two steps)"
    )
    calib.add_argument("--video", type=Path, required=True, help="path of the match video")
    calib.add_argument("--match-id", help="default: name of the video's directory")
    calib.add_argument("--workdir", type=Path, help="clusters and sheets (default: scratch/...)")
    calib.add_argument("--labels", type=Path, help="step 2: labels file, makes the templates")
    calib.add_argument("--step-s", type=float, help="seconds between sampled frames (default 5)")
    calib.add_argument("--cpu", action="store_true", help="decode on the CPU (very slow for AV1)")

    spot = commands.add_parser(
        "spotting-eval", help="evaluate a T-DEED output file on the first half (D-045)"
    )
    spot.add_argument("--predictions", type=Path, required=True, help="T-DEED results JSON")
    spot.add_argument(
        "--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH, help="default: %(default)s"
    )
    spot.add_argument(
        "--half", type=int, default=DEVELOPMENT_HALF, help="1 only; half 2 is reserved"
    )
    spot.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="confidence thresholds (default: %(default)s); values below the threshold of "
        "T-DEED's own inference.py (default 0.2) add nothing",
    )
    spot.add_argument("--fps", type=float, default=DEFAULT_FPS, help="frame rate of `frame`")
    spot.add_argument(
        "--radius-s",
        type=float,
        default=DEFAULT_MERGE_RADIUS_S,
        help="merge radius for duplicates, seconds (default: %(default)s)",
    )
    spot.add_argument("--output", type=Path, help="default: eval_<name>_half1.json next to input")
    return parser


def _match_id(args: argparse.Namespace) -> str:
    """The given id, or the name of the video's directory (also for a bare `src.mp4`)."""
    given: str | None = args.match_id
    return given or args.video.resolve().parent.name


def _config(model: type[_Config], hw_accel: bool, sample_fps: float | None = None) -> _Config:
    """Build a stage config; an invalid value is a configuration error, not a traceback."""
    values: dict[str, object] = {"hw_accel": hw_accel}
    if sample_fps is not None:
        values["sample_fps"] = sample_fps
    try:
        return model.model_validate(values)
    except ValidationError as err:
        first = err.errors()[0]
        msg = f"invalid option {first['loc'][0]}: {first['msg']}"
        raise ConfigError(msg) from err


def _scoreboard(args: argparse.Namespace) -> int:
    config = _config(ScoreboardConfig, not args.cpu, args.sample_fps)
    match_id = _match_id(args)
    result = run_scoreboard(args.video, match_id, config, force=args.force)
    stats = result.stats
    sys.stdout.write(
        f"scoreboard.json: {stats.n_samples} samples, {stats.n_visible} with the board, "
        f"{stats.n_unreadable} unreadable, {stats.n_jumps} offset jumps, "
        f"{len(result.score_changes)} score changes, decode mode {result.decode_mode}\n"
    )
    return 0


def _plates(args: argparse.Namespace) -> int:
    config = _config(PlatesConfig, not args.cpu, args.sample_fps)
    result = run_plates(args.video, _match_id(args), config, force=args.force)
    counts = ", ".join(f"{kind} {n}" for kind, n in result.stats.n_by_type.items())
    sys.stdout.write(
        f"plates.json: {len(result.plates)} plates ({counts}), {result.stats.n_rejected} "
        f"rejected candidates, {len(result.logo_clusters)} logo clusters, "
        f"{result.stats.n_low_conf} low-confidence, decode mode {result.decode_mode}\n"
    )
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    match_id = _match_id(args)
    workdir = args.workdir or _DEFAULT_WORKDIR_ROOT / f"{match_id}_scoreboard_calibration"
    if args.labels is not None:
        report = run_labelling(args.video, workdir, args.labels)
        sys.stdout.write(f"templates written: {report.templates}\n")
        for kind, labels in report.missing.items():
            if labels:
                sys.stdout.write(f"  {kind}: no template for {', '.join(labels)}\n")
        return 0
    config = _config(ScoreboardConfig, not args.cpu)
    step_s = DEFAULT_CALIBRATION_STEP_S if args.step_s is None else args.step_s
    summary = run_calibration(args.video, workdir, match_id, config, step_s)
    sys.stdout.write(
        f"{summary.n_board_frames} of {summary.n_frames} frames show the board "
        f"(decode mode {summary.decode_mode}); clusters: {summary.clusters}\n"
        f"Look at {workdir}/sheet_clock.png, sheet_score.png, sheet_added.png, write a labels "
        f"file {{kind: {{cluster id: label}}}} and run again with --labels <file>\n"
    )
    return 0


def _spotting_eval(args: argparse.Namespace) -> int:
    evaluation, written = run_spotting_eval(
        args.predictions,
        args.ground_truth,
        output_path=args.output,
        half=args.half,
        thresholds=args.thresholds,
        fps=args.fps,
        radius_s=args.radius_s,
    )
    sys.stdout.write(format_evaluation(evaluation))
    sys.stdout.write(f"\nresult: {written}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command; returns the process exit code (0 ok, 2 on a handled error)."""
    args = build_parser().parse_args(argv)
    try:
        configure_logging(args.log_level)
        if args.command == "scoreboard":
            return _scoreboard(args)
        if args.command == "plates":
            return _plates(args)
        if args.command == "spotting-eval":
            return _spotting_eval(args)
        return _calibrate(args)
    except AicoachError as err:
        sys.stderr.write(f"error: {err}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
