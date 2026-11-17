"""Commands `teams`, `corner-tags` and `corner-eval` (D-078); registered by `cli.py`.

Kept apart from `cli.py` so that file does not grow by another screen of arguments. Like the rest
of the command line it only parses, calls the stage and reports.
"""

import argparse
import sys
from pathlib import Path

from aicoach.agent.service import make_video_model
from aicoach.config import load_settings
from aicoach.corner_tags.run import TRUTH_FILE, run_corner_eval
from aicoach.corner_tags.stage import DB_FILE, run_corner_tags
from aicoach.errors import AicoachError
from aicoach.teams.stage import run_teams

COMMANDS = ("teams", "corner-tags", "corner-eval")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--match-dir", type=Path, required=True, help="directory of the match")
    parser.add_argument("--match-id", help="default: name of the match directory")
    parser.add_argument("--db", type=Path, help="database file (default: aicoach.db in the dir)")
    parser.add_argument("--video", type=Path, help="default: src.mp4 in the match dir")
    parser.add_argument("--model", help="OpenRouter video model (default: AICOACH_VLM_MODEL)")


def add_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add the three sub-commands to the parser."""
    teams = commands.add_parser(
        "teams", help="team names and kit colours (lineup, title, video model; D-078)"
    )
    _common(teams)
    teams.add_argument("--force", action="store_true", help="recompute (earlier votes are reused)")

    tags = commands.add_parser(
        "corner-tags", help="structured tags of every corner from its clip (D-078)"
    )
    _common(tags)
    tags.add_argument("--force", action="store_true", help="ask again for every corner")

    evaluate = commands.add_parser(
        "corner-eval", help="accuracy of the corner tags per field, section of docs/metrics.md"
    )
    evaluate.add_argument("--match-dir", type=Path, required=True, help="directory of the match")
    evaluate.add_argument(
        "--truth", type=Path, help=f"filled truth file (default: {TRUTH_FILE} in the match dir)"
    )
    evaluate.add_argument("--output", type=Path, default=Path("docs") / "metrics.md")
    evaluate.add_argument(
        "--final",
        action="store_true",
        help="also score the 2nd half (the check, run once with the question unchanged)",
    )


def run_command(args: argparse.Namespace) -> int | None:
    """Run the command of `args`; `None` when it is not one of ours.

    Raises:
        AicoachError: The stage failed (the caller prints it and exits with 2).
    """
    if args.command not in COMMANDS:
        return None
    match_dir: Path = args.match_dir
    if args.command == "corner-eval":
        scores = run_corner_eval(
            match_dir, args.truth or match_dir / TRUTH_FILE, args.output, final=args.final
        )
        sys.stdout.write(f"result: {args.output} (halves: {', '.join(map(str, scores))})\n")
        return 0
    match_id = args.match_id or match_dir.resolve().name
    model = make_video_model(load_settings(), match_dir, args.model)
    db = args.db or match_dir / DB_FILE
    if args.command == "teams":
        result = run_teams(
            match_dir, match_id, db_path=db, video=args.video, force=args.force, model=model
        )
        for side in (result.home, result.away):
            sys.stdout.write(
                f"{side.name} ({side.name_source}): kit {side.kit or 'not determined'} "
                f"({side.kit_source}), votes {side.kit_votes}\n"
            )
        for note in result.notes:
            sys.stdout.write(f"note: {note}\n")
        return 0
    tagged = run_corner_tags(
        match_dir, match_id, db_path=db, video=args.video, force=args.force, model=model
    )
    if tagged is None:
        msg = f"no tags: no database {db} or no video"
        raise AicoachError(msg)
    for tag in tagged.tags:
        sys.stdout.write(f"{tag.event_id} [{tag.status}] {tag.fields.model_dump()}\n")
    return 0
