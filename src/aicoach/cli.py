"""Command line entry point: `python -m aicoach.cli <stage> ...` (CLAUDE.md section 4).

Thin by design: it parses arguments, calls the stage and reports. The video path is only an
argument; results are written next to the video (the match directory).
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from aicoach.errors import AicoachError, ConfigError
from aicoach.logging_setup import configure_logging
from aicoach.scoreboard.config import ScoreboardConfig
from aicoach.scoreboard.stage import (
    DEFAULT_CALIBRATION_STEP_S,
    run_calibration,
    run_labelling,
    run_scoreboard,
)

_DEFAULT_WORKDIR_ROOT = Path("scratch")


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

    calib = commands.add_parser(
        "scoreboard-calibrate", help="cut digit templates from the broadcast (two steps)"
    )
    calib.add_argument("--video", type=Path, required=True, help="path of the match video")
    calib.add_argument("--match-id", help="default: name of the video's directory")
    calib.add_argument("--workdir", type=Path, help="clusters and sheets (default: scratch/...)")
    calib.add_argument("--labels", type=Path, help="step 2: labels file, makes the templates")
    calib.add_argument("--step-s", type=float, help="seconds between sampled frames (default 5)")
    calib.add_argument("--cpu", action="store_true", help="decode on the CPU (very slow for AV1)")
    return parser


def _match_id(args: argparse.Namespace) -> str:
    """The given id, or the name of the video's directory (also for a bare `src.mp4`)."""
    given: str | None = args.match_id
    return given or args.video.resolve().parent.name


def _config(hw_accel: bool, sample_fps: float | None = None) -> ScoreboardConfig:
    """Build the stage config; an invalid value is a configuration error, not a traceback."""
    values: dict[str, object] = {"hw_accel": hw_accel}
    if sample_fps is not None:
        values["sample_fps"] = sample_fps
    try:
        return ScoreboardConfig.model_validate(values)
    except ValidationError as err:
        first = err.errors()[0]
        msg = f"invalid option {first['loc'][0]}: {first['msg']}"
        raise ConfigError(msg) from err


def _scoreboard(args: argparse.Namespace) -> int:
    config = _config(hw_accel=not args.cpu, sample_fps=args.sample_fps)
    match_id = _match_id(args)
    result = run_scoreboard(args.video, match_id, config, force=args.force)
    stats = result.stats
    sys.stdout.write(
        f"scoreboard.json: {stats.n_samples} samples, {stats.n_visible} with the board, "
        f"{stats.n_unreadable} unreadable, {stats.n_jumps} offset jumps, "
        f"{len(result.score_changes)} score changes, decode mode {result.decode_mode}\n"
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
    config = _config(hw_accel=not args.cpu)
    step_s = DEFAULT_CALIBRATION_STEP_S if args.step_s is None else args.step_s
    summary = run_calibration(args.video, workdir, match_id, config, step_s)
    sys.stdout.write(
        f"{summary.n_board_frames} of {summary.n_frames} frames show the board "
        f"(decode mode {summary.decode_mode}); clusters: {summary.clusters}\n"
        f"Look at {workdir}/sheet_clock.png, sheet_score.png, sheet_added.png, write a labels "
        f"file {{kind: {{cluster id: label}}}} and run again with --labels <file>\n"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command; returns the process exit code (0 ok, 2 on a handled error)."""
    args = build_parser().parse_args(argv)
    try:
        configure_logging(args.log_level)
        if args.command == "scoreboard":
            return _scoreboard(args)
        return _calibrate(args)
    except AicoachError as err:
        sys.stderr.write(f"error: {err}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
