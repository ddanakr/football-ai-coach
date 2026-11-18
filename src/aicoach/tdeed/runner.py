"""The side effects of the T-DEED run: child processes, the checkpoint download, file hashes.

The stage takes these as injectable functions, so its tests run without git, FFmpeg, torch, a
video card or a network.
"""

import hashlib
import subprocess  # noqa: S404  # child processes with argument lists, never a shell
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

from aicoach.errors import SpottingError
from aicoach.tdeed.config import DOWNLOAD_TIMEOUT_S

CommandRunner = Callable[[list[str], float], "subprocess.CompletedProcess[bytes]"]
InferenceRunner = Callable[[list[str], Path, Mapping[str, str], float], int]
Downloader = Callable[[str, Path], None]
_CHUNK = 1 << 20
_TAIL_LINES = 20
_TAIL_CHARS = 2000


def run_command(command: list[str], timeout_s: float) -> "subprocess.CompletedProcess[bytes]":
    """Run a command with its output captured; the caller reads the exit code."""
    return subprocess.run(  # noqa: S603  # an argument list built by `commands`, no shell
        command, capture_output=True, check=False, timeout=timeout_s
    )


def run_inference(command: list[str], cwd: Path, env: Mapping[str, str], timeout_s: float) -> int:
    """Run `inference.py` with its output going to ours, so the progress bar is seen as it goes.

    Returns:
        The exit code.
    """
    return subprocess.run(  # noqa: S603  # an argument list built by `commands`, no shell
        command, cwd=cwd, env=dict(env), check=False, timeout=timeout_s
    ).returncode


def stderr_tail(stderr: bytes) -> str:
    """The end of an error output, short enough for a message."""
    text = stderr.decode("utf-8", errors="replace").strip()
    return "\n".join(text.splitlines()[-_TAIL_LINES:])[-_TAIL_CHARS:] or "no error output"


def checked(command: list[str], runner: CommandRunner, timeout_s: float, what: str) -> str:
    """Run a command that must succeed; return its output text.

    Raises:
        SpottingError: The program is missing, times out, or exits with a non-zero code.
    """
    try:
        done = runner(command, timeout_s)
    except FileNotFoundError as err:
        msg = f"{what}: {command[0]} not found (install it and put it in PATH)"
        raise SpottingError(msg) from err
    except subprocess.TimeoutExpired as err:
        msg = f"{what}: {command[0]} did not finish in {timeout_s:g} s"
        raise SpottingError(msg) from err
    except OSError as err:
        msg = f"{what}: cannot start {command[0]}: {err}"
        raise SpottingError(msg) from err
    if done.returncode != 0:
        msg = f"{what} failed (exit code {done.returncode}): {stderr_tail(done.stderr)}"
        raise SpottingError(msg)
    return done.stdout.decode("utf-8", errors="replace")


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    """Download `url` (https only) to `destination`; a partial file is not left behind.

    Raises:
        SpottingError: The address is not https, the server refuses, the network fails, or the file
            cannot be written.
    """
    if not url.startswith("https://"):
        msg = f"refusing to download from a non-https address: {url[:40]}"
        raise SpottingError(msg)
    partial = destination.with_name(destination.name + ".part")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (
            urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response,  # noqa: S310  # https checked above
            partial.open("wb") as out,
        ):
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                out.write(chunk)
        partial.replace(destination)
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        partial.unlink(missing_ok=True)
        msg = f"cannot download the checkpoint: {err}"
        raise SpottingError(msg) from err
