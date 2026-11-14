"""Command line entry point: `python -m aicoach.cli <stage> ...` (CLAUDE.md section 4).

Thin by design: it parses arguments, calls the stage and reports. The video path is only an
argument; results are written next to the video (the match directory).
"""

import argparse
import subprocess
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from aicoach.clips import DEFAULT_AFTER_S, DEFAULT_BEFORE_S, export_event_clip
from aicoach.config import load_settings
from aicoach.errors import AicoachError, ConfigError
from aicoach.fusion.config import load_fuse_config
from aicoach.fusion.rules import count_events
from aicoach.fusion.schemas import CandidatesResult
from aicoach.fusion.stage import DB_FILE, run_fuse
from aicoach.i18n import ru
from aicoach.ingest.download import VIDEO_FILE
from aicoach.ingest.stage import EXIT_WAITING_TDEED, IngestTools, run_ingest
from aicoach.logging_setup import configure_logging
from aicoach.pipeline import Stages, run_pipeline
from aicoach.plate_names.stage import STAGE as PLATE_NAMES_STAGE
from aicoach.plate_names.stage import run_plate_names
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
from aicoach.status import read_status
from aicoach.ui.launch import AppArgs, StartArgs, build_streamlit_command
from aicoach.youtube import validate_video_id

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

    names = commands.add_parser(
        "plate-names", help="read player names from the plate crops (Tesseract; needs plates.json)"
    )
    names.add_argument("--video", type=Path, required=True, help="path of the match video")
    names.add_argument("--match-id", help="default: name of the video's directory")
    names.add_argument(
        "--lineups", type=Path, help="lineups.csv (default: lineups.csv by the video, if present)"
    )
    names.add_argument("--force", action="store_true", help="ignore an up-to-date result")

    fuse = commands.add_parser(
        "fuse", help="merge scoreboard, plates and model events into the database"
    )
    _add_fuse_options(fuse)
    fuse.add_argument("--db", type=Path, help="database file (default: aicoach.db by the video)")

    run = commands.add_parser("run", help="scoreboard, plates and fuse in order (no model)")
    run.add_argument("--cpu", action="store_true", help="decode on the CPU (very slow for AV1)")
    _add_fuse_options(run, force_help="recompute all three stages (reads the video twice, slow)")

    ingest = commands.add_parser(
        "ingest", help="YouTube link to events: download, audio, scoreboard, plates, fuse"
    )
    ingest.add_argument("url", help="YouTube link of the match (watch, youtu.be or live form)")
    ingest.add_argument(
        "--data-root",
        type=Path,
        help="directory with one directory per match (default: AICOACH_DATA_ROOT)",
    )
    ingest.add_argument("--match-id", help="name of the match directory (default: the video id)")
    ingest.add_argument(
        "--templates",
        type=Path,
        help="scoreboard_templates of a calibrated match of the same broadcast (copied once)",
    )
    ingest.add_argument("--cpu", action="store_true", help="decode on the CPU (very slow for AV1)")

    ui = commands.add_parser(
        "ui", help="open the event viewer in the browser (view, edit, clips, new matches)"
    )
    ui.add_argument("--video", type=Path, help="path of the match video to open")
    ui.add_argument("--youtube-id", help="YouTube id of the same match (11 chars)")
    ui.add_argument(
        "--data-root",
        type=Path,
        help="directory for new matches, enables the 'new match' block (default: "
        "AICOACH_DATA_ROOT)",
    )
    ui.add_argument(
        "--templates", type=Path, help="scoreboard_templates given to every ingest from the viewer"
    )
    ui.add_argument("--match-id", help="default: name of the video's directory")
    ui.add_argument("--db", type=Path, help="database file (default: aicoach.db by the video)")

    clip = commands.add_parser("clip", help="cut the clip of one event with FFmpeg (FR-7)")
    clip.add_argument("--video", type=Path, required=True, help="path of the match video")
    clip.add_argument("--event-id", required=True, help="event id from events_current, e.g. p:12")
    clip.add_argument("--match-id", help="default: name of the video's directory")
    clip.add_argument("--db", type=Path, help="database file (default: aicoach.db by the video)")
    clip.add_argument(
        "--before",
        type=float,
        default=DEFAULT_BEFORE_S,
        help=f"seconds before (default: {DEFAULT_BEFORE_S:g})",
    )
    clip.add_argument(
        "--after",
        type=float,
        default=DEFAULT_AFTER_S,
        help=f"seconds after (default: {DEFAULT_AFTER_S:g})",
    )

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


def _add_fuse_options(
    parser: argparse.ArgumentParser, force_help: str = "ignore an up-to-date result"
) -> None:
    """Options shared by `fuse` and `run`."""
    parser.add_argument("--video", type=Path, required=True, help="path of the match video")
    parser.add_argument("--match-id", help="default: name of the video's directory")
    parser.add_argument(
        "--predictions", type=Path, help="T-DEED results JSON; without it only the graphics count"
    )
    parser.add_argument("--config", type=Path, help="JSON with FuseConfig fields to override")
    parser.add_argument("--force", action="store_true", help=force_help)
    parser.add_argument(
        "--discard-edits",
        action="store_true",
        help="allow a new run although the match has user edits (they leave the current view)",
    )


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


def _plate_names(args: argparse.Namespace) -> int:
    """Exit code 0 also when the step is skipped for want of Tesseract (the reason is shown)."""
    match_dir = args.video.parent
    result = run_plate_names(
        match_dir,
        _match_id(args),
        lineup_path=args.lineups,
        force=args.force,
        tesseract_cmd=load_settings().tesseract_cmd,
    )
    if result is None:
        reason = read_status(match_dir).stages[PLATE_NAMES_STAGE].error
        sys.stdout.write(f"plate_names skipped: {reason}\n")
        return 0
    stats = result.stats
    source = f"lineup {result.lineup}" if result.lineup else "no lineup (names are not checked)"
    sys.stdout.write(
        f"plate_names.json: {stats.n_plates} plates, {stats.n_names} names: "
        f"{stats.n_matched} matched, {stats.n_unchecked} unchecked, "
        f"{stats.n_unrecognized} unrecognized; {source}\n"
    )
    for text in result.warnings:
        sys.stdout.write(f"  warning: {text}\n")
    return 0


def _report_fusion(result: CandidatesResult) -> None:
    counts = ", ".join(
        f"{kind} {sum(by_half.values())} (1st half {by_half.get('1', 0)}, "
        f"2nd half {by_half.get('2', 0)})"
        for kind, by_half in sorted(count_events(result.candidates).items())
    )
    sys.stdout.write(
        f"candidates.json: {counts or 'no events'}; mode {result.mode}; run {result.run_id}; "
        f"{len(result.dropped)} dropped, {len(result.warnings)} warning(s)\n"
    )
    for text in result.warnings:
        sys.stdout.write(f"  warning: {text}\n")


def _fuse(args: argparse.Namespace) -> int:
    result = run_fuse(
        args.video.parent,
        _match_id(args),
        args.predictions,
        load_fuse_config(args.config),
        force=args.force,
        discard_edits=args.discard_edits,
        db_path=args.db,
    )
    _report_fusion(result)
    return 0


def _run(args: argparse.Namespace) -> int:
    done = run_pipeline(
        args.video,
        _match_id(args),
        predictions_path=args.predictions,
        fuse_config=load_fuse_config(args.config),
        force=args.force,
        hw_accel=not args.cpu,
        discard_edits=args.discard_edits,
    )
    sys.stdout.write(
        f"scoreboard: {len(done.scoreboard.score_changes)} score changes; "
        f"plates: {len(done.plates.plates)}\n"
    )
    _report_fusion(done.fusion)
    return 0


def _ingest(args: argparse.Namespace) -> int:
    """Exit code 0 when the events are in the database, 3 while the T-DEED file is missing."""
    settings = load_settings(data_root=args.data_root)
    if settings.aicoach_data_root is None:
        raise ConfigError(ru.NO_DATA_ROOT)
    tools = None
    if settings.tesseract_cmd is not None:
        names = partial(run_plate_names, tesseract_cmd=settings.tesseract_cmd)
        tools = IngestTools(stages=Stages(plate_names=names))
    result = run_ingest(
        args.url,
        settings.aicoach_data_root,
        match_id=args.match_id,
        templates=args.templates,
        hw_accel=not args.cpu,
        tools=tools,
    )
    if result.fusion is None:
        waiting = ru.TDEED_WAITING.format(video=result.match_dir / VIDEO_FILE, dir=result.match_dir)
        sys.stdout.write(f"{waiting}\n{ru.TDEED_RESUME_CLI}\n")
        return EXIT_WAITING_TDEED
    sys.stdout.write(f"match directory: {result.match_dir}\n")
    _report_fusion(result.fusion)
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


def _ui(args: argparse.Namespace) -> int:
    """Run the Streamlit viewer; its exit code is ours. The video id stays in the arguments."""
    if (args.video is None) != (args.youtube_id is None):
        msg = "ui: --video and --youtube-id go together"
        raise ConfigError(msg)
    data_root = load_settings(data_root=args.data_root).aicoach_data_root
    if args.video is None and data_root is None:
        msg = (
            "ui: give --video and --youtube-id to open a match, or --data-root "
            "(or AICOACH_DATA_ROOT) for new ones"
        )
        raise ConfigError(msg)
    match = None
    if args.video is not None:
        match = AppArgs(
            db=args.db or args.video.parent / DB_FILE,
            match_id=_match_id(args),
            youtube_id=validate_video_id(args.youtube_id),
            match_dir=args.video.parent,
            video=args.video,
        )
    command = build_streamlit_command(
        StartArgs(match=match, data_root=data_root, templates=args.templates)
    )
    try:
        # Fixed program and our own arguments: nothing here comes from a shell string.
        return subprocess.run(command, check=False).returncode  # noqa: S603
    except KeyboardInterrupt:
        return 0


def _clip(args: argparse.Namespace) -> int:
    path = export_event_clip(
        args.db or args.video.parent / DB_FILE,
        _match_id(args),
        args.video,
        args.event_id,
        args.before,
        args.after,
    )
    sys.stdout.write(f"clip: {path}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command; returns the process exit code (0 ok, 2 on a handled error, 3 while
    `ingest` waits for the T-DEED file)."""
    args = build_parser().parse_args(argv)
    try:
        configure_logging(args.log_level)
        if args.command == "scoreboard":
            return _scoreboard(args)
        if args.command == "plates":
            return _plates(args)
        if args.command == "plate-names":
            return _plate_names(args)
        if args.command == "spotting-eval":
            return _spotting_eval(args)
        if args.command == "fuse":
            return _fuse(args)
        if args.command == "run":
            return _run(args)
        if args.command == "ingest":
            return _ingest(args)
        if args.command == "ui":
            return _ui(args)
        if args.command == "clip":
            return _clip(args)
        return _calibrate(args)
    except AicoachError as err:
        sys.stderr.write(f"error: {err}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
