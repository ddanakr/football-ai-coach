"""Commands `tdeed` and `tdeed-compare` (D-080); registered by `cli.py`.

Kept apart from `cli.py` like `corner_cli.py`. They only parse, call the stage and report.
"""

import argparse
import sys
from pathlib import Path

from aicoach.config import load_settings
from aicoach.errors import SpottingError
from aicoach.fusion.config import FuseConfig
from aicoach.i18n import ru
from aicoach.spotting.schemas import load_tdeed_file
from aicoach.tdeed.compare import (
    DEFAULT_EDGE_MARGIN_S,
    DEFAULT_SCORE_TOL,
    DEFAULT_TIME_TOL_S,
    compare_tdeed,
    filter_by_fuse_thresholds,
    format_comparison,
    verdict,
)
from aicoach.tdeed.config import FPS
from aicoach.tdeed.device import DEVICE_CHOICES
from aicoach.tdeed.stage import TdeedResult, default_cache_dir, run_tdeed
from aicoach.tdeed.timecode import format_timecode, parse_timecode, parse_window

COMMANDS = ("tdeed", "tdeed-compare")
EXIT_DIFFERENT = 1  # `tdeed-compare` found differences (2 stays "error")


def add_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add the two sub-commands to the parser."""
    tdeed = commands.add_parser(
        "tdeed", help="run T-DEED (shots, corners, fouls) on the match video, on the video card"
    )
    tdeed.add_argument("--match-dir", type=Path, required=True, help="directory of the match")
    tdeed.add_argument("--match-id", help="default: name of the match directory")
    tdeed.add_argument("--video", type=Path, help="default: src.mp4 in the match directory")
    tdeed.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="auto: the video card, an error without one; cpu only when asked, far slower",
    )
    tdeed.add_argument("--checkpoint", type=Path, help="checkpoint file (default: downloaded)")
    tdeed.add_argument("--cache-dir", type=Path, help="default: TDEED_CACHE_DIR, then ~/.cache")
    tdeed.add_argument(
        "--from",
        dest="t_from",
        metavar="TIME",
        help="start of a piece, mm:ss or hh:mm:ss; moved down to a multiple of 24 s; the result "
        "goes to its own tdeedseg_*.json",
    )
    tdeed.add_argument(
        "--to", dest="t_to", metavar="TIME", help="end of a piece, mm:ss or hh:mm:ss"
    )
    tdeed.add_argument("--force", action="store_true", help="run although the output is there")

    compare = commands.add_parser(
        "tdeed-compare", help="compare two T-DEED output files (does a run reproduce a file?)"
    )
    compare.add_argument("a", type=Path, help="the reference file, for example the frozen one")
    compare.add_argument("b", type=Path, help="the file to check")
    compare.add_argument(
        "--window", metavar="FROM-TO", help="part of the video to compare, for example 13:12-23:12"
    )
    compare.add_argument(
        "--edge-margin",
        type=float,
        default=DEFAULT_EDGE_MARGIN_S,
        help="seconds at each end of the window not compared (default: %(default)s)",
    )
    compare.add_argument(
        "--time-tol",
        type=float,
        default=DEFAULT_TIME_TOL_S,
        help="seconds within which two predictions are a pair (default: %(default)s)",
    )
    compare.add_argument(
        "--score-tol",
        type=float,
        default=DEFAULT_SCORE_TOL,
        help="largest difference of paired confidences (default: %(default)s)",
    )
    compare.add_argument("--fps", type=float, default=FPS, help="frame rate of `frame`")
    compare.add_argument(
        "--fuse-thresholds",
        action="store_true",
        help="also compare only what `fuse` takes: shots, corners and fouls at or above the "
        "FuseConfig thresholds of their class; both verdicts are printed, exit 0 needs both",
    )


def format_result(result: TdeedResult) -> str:
    """The report of a run: the file, the piece, the times and, for a piece, the estimate."""
    name = result.output.name
    if result.cached:
        return ru.TDEED_CACHED.format(file=name, n=result.n_predictions) + "\n"
    lines = [ru.TDEED_DONE.format(file=name, n=result.n_predictions, device=result.device)]
    segment = result.segment
    if segment is not None:
        lines.append(
            ru.TDEED_SEGMENT.format(
                start=format_timecode(segment.start_s),
                end=format_timecode(segment.end_s),
                length=format_timecode(segment.length_s),
                requested=format_timecode(segment.requested_start_s),
            )
        )
    timings = result.timings
    if timings is None:
        return "\n".join(lines) + "\n"
    lines.append(
        ru.TDEED_TIMINGS.format(
            prepare=timings.prepare_s,
            ffmpeg=timings.ffmpeg_s,
            inference=timings.inference_s,
            total=timings.total_s,
        )
    )
    processed_s = segment.cut_length_s if segment else result.video_duration_s
    per_minute = 60.0 / processed_s
    lines.append(
        ru.TDEED_PER_MINUTE.format(
            ffmpeg=timings.ffmpeg_s * per_minute, inference=timings.inference_s * per_minute
        )
    )
    if segment is not None:
        scale = result.video_duration_s / processed_s
        lines.append(
            ru.TDEED_ESTIMATE.format(
                minutes=result.video_duration_s / 60,
                ffmpeg=timings.ffmpeg_s * scale,
                inference=timings.inference_s * scale,
                total=(timings.ffmpeg_s + timings.inference_s) * scale,
                total_min=(timings.ffmpeg_s + timings.inference_s) * scale / 60,
            )
        )
    return "\n".join(lines) + "\n"


def _run_tdeed(args: argparse.Namespace) -> int:
    match_dir: Path = args.match_dir
    t_from = parse_timecode(args.t_from) if args.t_from else None
    t_to = parse_timecode(args.t_to) if args.t_to else None
    settings = load_settings(tdeed_cache_dir=args.cache_dir)
    result = run_tdeed(
        match_dir,
        args.match_id or match_dir.resolve().name,
        cache_dir=settings.tdeed_cache_dir or default_cache_dir(),
        video=args.video,
        device=args.device,
        checkpoint=args.checkpoint,
        t_from=t_from,
        t_to=t_to,
        force=args.force,
    )
    if result.device == "cpu" and not result.cached:
        sys.stdout.write(f"{ru.TDEED_CPU_WARNING}\n")
    for warning in result.warnings:
        sys.stdout.write(f"warning: {warning}\n")
    sys.stdout.write(format_result(result))
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    window = parse_window(args.window) if args.window else None
    if args.fps <= 0:
        msg = "--fps must be positive"
        raise SpottingError(msg)
    first, second = load_tdeed_file(args.a), load_tdeed_file(args.b)
    options = {
        "time_tol_s": args.time_tol,
        "score_tol": args.score_tol,
        "window_s": window,
        "edge_margin_s": args.edge_margin,
        "fps": args.fps,
    }
    everything = compare_tdeed(first, second, **options)
    if not args.fuse_thresholds:
        sys.stdout.write(format_comparison(everything))
        return 0 if everything.same else EXIT_DIFFERENT
    config = FuseConfig()
    above = compare_tdeed(
        filter_by_fuse_thresholds(first, config),
        filter_by_fuse_thresholds(second, config),
        **options,
    )
    thresholds = ", ".join(f"{k} {v:g}" for k, v in config.model_thresholds.items())
    sys.stdout.write(f"== {ru.TDEED_COMPARE_ALL}\n{format_comparison(everything)}")
    sys.stdout.write(f"== {ru.TDEED_COMPARE_FUSE} ({thresholds})\n{format_comparison(above)}")
    sys.stdout.write(f"{ru.TDEED_COMPARE_ALL}: {verdict(everything)}\n")
    sys.stdout.write(f"{ru.TDEED_COMPARE_FUSE}: {verdict(above)}\n")
    return 0 if everything.same and above.same else EXIT_DIFFERENT


def run_command(args: argparse.Namespace) -> int | None:
    """Run the command of `args`; `None` when it is not one of ours.

    Raises:
        AicoachError: The stage failed (the caller prints it and exits with 2).
    """
    if args.command == "tdeed":
        return _run_tdeed(args)
    if args.command == "tdeed-compare":
        return _run_compare(args)
    return None
