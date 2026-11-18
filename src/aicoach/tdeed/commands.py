"""The command lines of the T-DEED run, built as plain lists (no side effects, easy to check)."""

import sys
from pathlib import Path

from aicoach.tdeed.config import (
    FPS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    INPUT_CRF,
    INPUT_PRESET,
    MODEL,
    REPO_COMMIT,
    REPO_URL,
    THRESHOLD,
)

# The launcher is run by its path, not as a module: it needs nothing of this package, so it works
# from any directory and in any environment that has torch.
CPU_RUNNER = Path(__file__).with_name("cpu_runner.py")


def git_fetch_commands(repo_dir: Path, *, git: str = "git") -> list[list[str]]:
    """Fetch `REPO_COMMIT` of T-DEED into `repo_dir` and check it out, history not included."""
    where = str(repo_dir)
    return [
        [git, "init", "--quiet", where],
        [git, "-C", where, "remote", "add", "origin", REPO_URL],
        [git, "-C", where, "fetch", "--quiet", "--depth", "1", "origin", REPO_COMMIT],
        [git, "-C", where, "checkout", "--quiet", "--detach", "FETCH_HEAD"],
    ]


def git_head_command(repo_dir: Path, *, git: str = "git") -> list[str]:
    """The commit that `repo_dir` is at."""
    return [git, "-C", str(repo_dir), "rev-parse", "HEAD"]


def git_modified_command(repo_dir: Path, *, git: str = "git") -> list[str]:
    """Tracked files of `repo_dir` that differ from the commit (untracked files do not count:
    Python may leave caches there)."""
    return [git, "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"]


def build_input_command(
    video: Path,
    output: Path,
    *,
    start_s: float = 0.0,
    length_s: float | None = None,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """FFmpeg arguments for the model's input: the first video track at 25 frames per second and
    796x448, without sound, H.264 (the model reads it with OpenCV).

    `start_s` and `length_s` cut a piece; the seek before the input is exact because the video
    is re-encoded.
    """
    command = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-v", "error"]
    if start_s > 0:
        command += ["-ss", f"{start_s:.3f}"]
    command += ["-i", str(video), "-map", "0:v:0", "-an"]
    if length_s is not None:
        command += ["-t", f"{length_s:.3f}"]
    command += [
        "-vf",
        f"fps={FPS},scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
        "-c:v",
        "libx264",
        "-preset",
        INPUT_PRESET,
        "-crf",
        str(INPUT_CRF),
        "-pix_fmt",
        "yuv420p",
        "-f",
        "mp4",
        str(output),
    ]
    return command


def build_inference_command(repo_dir: Path, video: Path, *, cpu: bool) -> list[str]:
    """The command of the unmodified `inference.py`, run from the work directory.

    On the processor the script runs through `cpu_runner.py`, because it names the video card in
    its code.
    """
    python = [sys.executable, "-u"]
    script = str(repo_dir / "inference.py")
    runner = [str(CPU_RUNNER), script] if cpu else [script]
    return [
        *python,
        *runner,
        "--model",
        MODEL,
        "--video_path",
        str(video),
        "--frame_width",
        str(FRAME_WIDTH),
        "--frame_height",
        str(FRAME_HEIGHT),
        "--inference_threshold",
        str(THRESHOLD),
    ]
