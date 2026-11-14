"""Start-up of the viewer: the arguments of the Streamlit script and the command that runs it.

The video id is only ever an argument of the process: it is not written to a file or to the
database (D-054).
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from aicoach.errors import ConfigError
from aicoach.ui.logic import validate_video_id


@dataclass(frozen=True)
class AppArgs:
    """Arguments the Streamlit script receives after `--`."""

    db: Path
    match_id: str
    youtube_id: str
    # Where `candidates.json` and `scoreboard.json` lie (the video's directory); they give the
    # half bounds and the video length that manual edits are checked against. None: next to `db`.
    match_dir: Path | None = None
    # The match video, for clip export (FR-7). None: the viewer cannot export clips.
    video: Path | None = None

    @property
    def results_dir(self) -> Path:
        """Directory with the stage results: `match_dir`, or the database's directory."""
        return self.match_dir or self.db.parent


def app_script() -> Path:
    """Path of the Streamlit script `app.py`."""
    return Path(__file__).with_name("app.py")


@dataclass(frozen=True)
class StartArgs:
    """Everything the script receives: an opened match (or none) and where new matches go."""

    match: AppArgs | None = None
    # Directory with one directory per match; None hides the "new match" block (D15).
    data_root: Path | None = None
    # `scoreboard_templates` of a calibrated match, given to every ingest the viewer starts.
    templates: Path | None = None


def build_streamlit_command(args: AppArgs | StartArgs, python: str | None = None) -> list[str]:
    """Command line that runs the viewer: `python -m streamlit run app.py -- <arguments>`."""
    start = args if isinstance(args, StartArgs) else StartArgs(match=args)
    command = [python or sys.executable, "-m", "streamlit", "run", str(app_script()), "--"]
    match = start.match
    if match is not None:
        command += [
            "--db",
            str(match.db),
            "--match-id",
            match.match_id,
            # `=` form: an id may start with "-", which argparse would take for an option.
            f"--youtube-id={match.youtube_id}",
        ]
        if match.match_dir is not None:
            command += ["--match-dir", str(match.match_dir)]
        if match.video is not None:
            command += ["--video", str(match.video)]
    if start.data_root is not None:
        command += ["--data-root", str(start.data_root)]
    if start.templates is not None:
        command += ["--templates", str(start.templates)]
    return command


def parse_start_args(argv: Sequence[str]) -> StartArgs:
    """Read and check the arguments given to the script (everything after `--`).

    `--db`, `--match-id` and `--youtube-id` come together or not at all.

    Raises:
        ConfigError: Only some of the match arguments are given, an argument is malformed, or
            the video id has a wrong form.
    """
    parser = argparse.ArgumentParser(prog="aicoach-ui", exit_on_error=False)
    for name in ("--db", "--match-dir", "--video", "--data-root", "--templates"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--match-id")
    parser.add_argument("--youtube-id")
    try:
        # Streamlit may add its own words to the command line; they are not ours to judge.
        known, _ = parser.parse_known_args(list(argv))
    except argparse.ArgumentError as err:
        msg = f"invalid viewer argument: {err}"
        raise ConfigError(msg) from err
    required = ("db", "match_id", "youtube_id")
    given = [name for name in required if getattr(known, name)]
    match = None
    if given:
        for name in required:
            if name not in given:
                msg = f"the viewer needs --{name.replace('_', '-')}"
                raise ConfigError(msg)
        match = AppArgs(
            db=known.db,
            match_id=known.match_id,
            youtube_id=validate_video_id(known.youtube_id),
            match_dir=known.match_dir,
            video=known.video,
        )
    return StartArgs(match=match, data_root=known.data_root, templates=known.templates)


def parse_app_args(argv: Sequence[str]) -> AppArgs:
    """The opened match from the script arguments (see `parse_start_args`).

    Raises:
        ConfigError: No match is given, or its arguments are incomplete or malformed.
    """
    match = parse_start_args(argv).match
    if match is None:
        msg = "the viewer needs --db"
        raise ConfigError(msg)
    return match
