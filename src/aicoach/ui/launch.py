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


def app_script() -> Path:
    """Path of the Streamlit script `app.py`."""
    return Path(__file__).with_name("app.py")


def build_streamlit_command(args: AppArgs, python: str | None = None) -> list[str]:
    """Command line that runs the viewer: `python -m streamlit run app.py -- <arguments>`."""
    return [
        python or sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_script()),
        "--",
        "--db",
        str(args.db),
        "--match-id",
        args.match_id,
        # `=` form: an id may start with "-", which argparse would take for an option.
        f"--youtube-id={args.youtube_id}",
    ]


def parse_app_args(argv: Sequence[str]) -> AppArgs:
    """Read and check the arguments given to the script (everything after `--`).

    Raises:
        ConfigError: An argument is missing or the video id has a wrong form.
    """
    parser = argparse.ArgumentParser(prog="aicoach-ui", exit_on_error=False)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--match-id")
    parser.add_argument("--youtube-id")
    try:
        # Streamlit may add its own words to the command line; they are not ours to judge.
        known, _ = parser.parse_known_args(list(argv))
    except argparse.ArgumentError as err:
        msg = f"invalid viewer argument: {err}"
        raise ConfigError(msg) from err
    for name in ("db", "match_id", "youtube_id"):
        if not getattr(known, name):
            msg = f"the viewer needs --{name.replace('_', '-')}"
            raise ConfigError(msg)
    return AppArgs(
        db=known.db, match_id=known.match_id, youtube_id=validate_video_id(known.youtube_id)
    )
