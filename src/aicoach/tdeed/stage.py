"""The `tdeed` stage: the T-DEED model on a match video, on the video card (D-080).

Steps: fetch the T-DEED code at its fixed commit into a cache outside the repository, get the
checkpoint, cut the video into the model's input (25 fps, 796x448), run the unmodified
`inference.py` as a separate process in a work directory, and write the result as the file that
`fuse` reads. A whole-match run writes `tdeed_<model>_thr<threshold>_bs<batch>.json` into the match
directory and does nothing when a `tdeed_*.json` is already there; no flag replaces such a file (it
may be the one `fuse` used). A segment run (`--from`, `--to`) writes its own `tdeedseg_*.json`,
which no other stage reads, so it can be compared with the whole-match file (`tdeed-compare`).
"""

import hashlib
import os
import re
import shutil
import subprocess  # noqa: S404  # only for its exception types
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from aicoach.clips import Probe
from aicoach.errors import AicoachError, SpottingError
from aicoach.logging_setup import stage_logger
from aicoach.scoreboard.video import probe_video
from aicoach.spotting.schemas import load_tdeed_file
from aicoach.stage_common import Logger, elapsed_ms, finish_timing, record_failure, set_status
from aicoach.status import utc_now_iso
from aicoach.tdeed.commands import (
    build_inference_command,
    build_input_command,
    git_fetch_commands,
    git_head_command,
    git_modified_command,
)
from aicoach.tdeed.config import (
    CHECKPOINT_SHA256,
    CHECKPOINT_URL,
    FFMPEG_TIMEOUT_S,
    FPS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GIT_TIMEOUT_S,
    INFERENCE_TIMEOUT_S,
    INPUT_CRF,
    MODEL,
    OUTPUT_PREFIX,
    REPO_COMMIT,
)
from aicoach.tdeed.device import Device, cuda_available, resolve_device
from aicoach.tdeed.output import convert_output, output_name
from aicoach.tdeed.runner import (
    CommandRunner,
    Downloader,
    InferenceRunner,
    checked,
    download_file,
    run_command,
    run_inference,
    sha256_of,
)
from aicoach.tdeed.timecode import Segment, plan_segment
from aicoach.tdeed.workdir import RAW_OUTPUT, prepare_workdir

STAGE = "tdeed"
SEGMENT_STAGE = "tdeed_segment"  # a segment run leaves the entry of the real stage alone
REPO_DIR = "T-DEED"
CHECKPOINT_FILE = "checkpoint_best.pt"
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.IGNORECASE)


@dataclass(frozen=True)
class TdeedTools:
    """The outside effects of the stage; the real ones by default, fakes in tests."""

    command: CommandRunner = run_command
    inference: InferenceRunner = run_inference
    download: Downloader = download_file
    probe: Probe = probe_video
    cuda_available: Callable[[], bool] = cuda_available


@dataclass(frozen=True)
class Timings:
    """Seconds spent in the parts of a run that did work."""

    prepare_s: float
    ffmpeg_s: float
    inference_s: float

    @property
    def total_s(self) -> float:
        """The three parts together."""
        return self.prepare_s + self.ffmpeg_s + self.inference_s


@dataclass(frozen=True)
class TdeedResult:
    """What a run did. `timings` is None when the output was already there."""

    output: Path
    device: Device | None  # None when nothing ran
    cached: bool
    n_predictions: int
    video_duration_s: float
    segment: Segment | None = None
    timings: Timings | None = None
    warnings: list[str] = field(default_factory=list)


def default_cache_dir() -> Path:
    """`~/.cache/aicoach/tdeed`, where the T-DEED copy, the checkpoint and the model inputs live."""
    return Path.home() / ".cache" / "aicoach" / "tdeed"


def find_existing(match_dir: Path) -> list[Path]:
    """The `tdeed_*.json` files of the match directory (a segment file does not match)."""
    return sorted(p for p in match_dir.glob(f"{OUTPUT_PREFIX}*.json") if p.is_file())


def _subcommand(command: list[str]) -> str:
    """The git subcommand of `git [-C dir] <subcommand> ...`, for messages."""
    return command[3] if command[1] == "-C" else command[1]


def ensure_repo(cache_dir: Path, tools: TdeedTools, log: Logger) -> Path:
    """The T-DEED code at `REPO_COMMIT` in the cache, fetched when it is not there.

    Raises:
        SpottingError: git fails, or the copy in the cache is at another commit or was modified
            (the model would then not be the one of the frozen predictions).
    """
    repo = cache_dir / REPO_DIR
    if (repo / "inference.py").is_file() and (repo / ".git").is_dir():
        head = checked(
            git_head_command(repo), tools.command, GIT_TIMEOUT_S, "git rev-parse"
        ).strip()
        if head != REPO_COMMIT:
            msg = f"{repo} is at commit {head[:7]}, not {REPO_COMMIT[:7]}: delete it to fetch again"
            raise SpottingError(msg)
        changed = checked(git_modified_command(repo), tools.command, GIT_TIMEOUT_S, "git status")
        if changed.strip():
            msg = f"the T-DEED copy {repo} was modified: delete the directory to fetch it again"
            raise SpottingError(msg)
        log.info("tdeed_repo_reused", commit=REPO_COMMIT[:7])
        return repo
    shutil.rmtree(repo, ignore_errors=True)  # a fetch that stopped halfway leaves an empty clone
    repo.parent.mkdir(parents=True, exist_ok=True)
    for command in git_fetch_commands(repo):
        checked(command, tools.command, GIT_TIMEOUT_S, f"git {_subcommand(command)}")
    log.info("tdeed_repo_fetched", commit=REPO_COMMIT[:7])
    return repo


def ensure_checkpoint(
    cache_dir: Path, given: Path | None, tools: TdeedTools, log: Logger
) -> tuple[Path, list[str]]:
    """The checkpoint file, and warnings about it.

    A given file is used as it is, with a warning when it is not the known `SoccerNet_small`
    (its SHA-256 differs). Otherwise the cached download is used, or downloaded and checked.

    Raises:
        SpottingError: The given file is missing, or the download failed or gave another file
            (Google Drive may answer with a page instead of the file).
    """
    if given is not None:
        if not given.is_file():
            msg = f"checkpoint not found: {given}"
            raise SpottingError(msg)
        if sha256_of(given) == CHECKPOINT_SHA256:
            return given, []
        return given, [f"{given} is not the known {MODEL} checkpoint (SHA-256 differs)"]
    target = cache_dir / "checkpoints" / MODEL / CHECKPOINT_FILE
    if target.is_file() and sha256_of(target) == CHECKPOINT_SHA256:
        return target, []
    log.info("tdeed_checkpoint_download", target=str(target))
    tools.download(CHECKPOINT_URL, target)
    if sha256_of(target) != CHECKPOINT_SHA256:
        target.unlink(missing_ok=True)
        msg = (
            f"the downloaded checkpoint is not the expected {MODEL} (SHA-256 differs): download "
            "it by hand from the Drive folder of the T-DEED README and pass it with --checkpoint"
        )
        raise SpottingError(msg)
    return target, []


def _input_path(cache_dir: Path, video: Path, segment: Segment | None) -> Path:
    """The cached model input for this video file and piece (a changed file gets a new name)."""
    stat = video.stat()
    bounds = (segment.start_s, segment.cut_end_s) if segment else (0.0, 0.0)
    parts = [str(video.resolve()), str(stat.st_size), str(stat.st_mtime_ns), str(bounds)]
    parts += [f"{FRAME_WIDTH}x{FRAME_HEIGHT}@{FPS}", f"crf{INPUT_CRF}"]
    key = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return cache_dir / "inputs" / f"{key}.mp4"


def prepare_input(
    video: Path, cache_dir: Path, segment: Segment | None, tools: TdeedTools, log: Logger
) -> Path:
    """The video as the model wants it, made by FFmpeg unless it is in the cache already."""
    target = _input_path(cache_dir, video, segment)
    if target.is_file():
        log.info("tdeed_input_reused", file=target.name)
        return target
    partial = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    start = segment.start_s if segment else 0.0
    length = segment.cut_length_s if segment else None
    command = build_input_command(video, partial, start_s=start, length_s=length)
    try:
        checked(command, tools.command, FFMPEG_TIMEOUT_S, "preparing the model input (FFmpeg)")
        partial.replace(target)
    except (SpottingError, OSError):
        partial.unlink(missing_ok=True)
        raise
    return target


def _environment(cache_dir: Path) -> dict[str, str]:
    """The child's environment: the downloads of timm (the backbone weights it asks for before
    the checkpoint overwrites them) go to the cache, and Python writes no `.pyc` into the copy."""
    # Keys and tokens of the user are not for a third-party program: dropped from its environment.
    env = {k: v for k, v in os.environ.items() if not _SECRET_NAME.search(k)}
    env["HF_HOME"] = str(cache_dir / "hf")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_model(
    repo: Path,
    checkpoint: Path,
    model_input: Path,
    cache_dir: Path,
    device: Device,
    segment: Segment | None,
    target: Path,
    tools: TdeedTools,
) -> int:
    """Run `inference.py` in a fresh work directory, convert its output into `target`, remove the
    work directory; returns the number of predictions written.

    Raises:
        SpottingError: The script fails or times out, or leaves no valid output.
    """
    runs = cache_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="run_", dir=runs))
    try:
        prepare_workdir(workdir, repo, checkpoint)
        command = build_inference_command(repo, model_input, cpu=device == "cpu")
        try:
            code = tools.inference(command, workdir, _environment(cache_dir), INFERENCE_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as err:
            msg = f"cannot run inference.py: {err}"
            raise SpottingError(msg) from err
        if code != 0:
            msg = f"inference.py failed (exit code {code}); its error is in the output above"
            raise SpottingError(msg)
        return convert_output(workdir / RAW_OUTPUT, target, segment)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _plan(
    video: Path, t_from: float | None, t_to: float | None, tools: TdeedTools
) -> tuple[float, Segment | None]:
    """The duration of the video, and the piece of it when `--from` or `--to` was given."""
    try:
        duration = tools.probe(video).duration_s
    except AicoachError as err:
        msg = f"cannot read the video {video}: {err}"
        raise SpottingError(msg) from err
    if t_from is None and t_to is None:
        return duration, None
    return duration, plan_segment(t_from, t_to, duration)


def _cached_result(target: Path, duration: float) -> TdeedResult:
    return TdeedResult(target, None, True, _count(target), duration)


def _count(path: Path) -> int:
    return len(load_tdeed_file(path).predictions)


@dataclass(frozen=True)
class _Job:
    match_dir: Path
    video: Path
    cache: Path
    device: Device
    checkpoint: Path | None
    segment: Segment | None
    target: Path
    duration: float
    tools: TdeedTools
    log: Logger

    @property
    def entry(self) -> str:
        return SEGMENT_STAGE if self.segment else STAGE


def _work(job: _Job) -> TdeedResult:
    """The steps of a run, each reported in `status.json` as it starts."""
    d, log = job.match_dir, job.log
    started = time.perf_counter()
    set_status(d, job.entry, "running", 0.02, log)
    repo = ensure_repo(job.cache, job.tools, log)
    set_status(d, job.entry, "running", 0.05, log)
    checkpoint, warnings = ensure_checkpoint(job.cache, job.checkpoint, job.tools, log)
    prepared = time.perf_counter()
    set_status(d, job.entry, "running", 0.10, log)
    model_input = prepare_input(job.video, job.cache, job.segment, job.tools, log)
    encoded = time.perf_counter()
    set_status(d, job.entry, "running", 0.40, log)
    count = run_model(
        repo, checkpoint, model_input, job.cache, job.device, job.segment, job.target, job.tools
    )
    done = time.perf_counter()
    timings = Timings(prepared - started, encoded - prepared, done - encoded)
    log.info("tdeed_done", n_predictions=count, duration_ms=elapsed_ms(started))
    return TdeedResult(
        job.target, job.device, False, count, job.duration, job.segment, timings, warnings
    )


def _run(job: _Job) -> TdeedResult:
    """A run under its `status.json` entry: `running`, then `done` with the times, or `failed`."""
    started_at, started = utc_now_iso(), time.perf_counter()
    try:
        result = _work(job)
    except BaseException as err:  # any failure, even Ctrl+C, must not leave "running" behind
        record_failure(job.match_dir, job.entry, 0.0, err, job.log, started)
        raise
    set_status(job.match_dir, job.entry, "done", 1.0, job.log)
    finish_timing(job.match_dir, job.entry, started_at, started, job.log)
    return result


def run_tdeed(
    match_dir: Path,
    match_id: str,
    *,
    cache_dir: Path | None = None,
    video: Path | None = None,
    device: str = "auto",
    checkpoint: Path | None = None,
    t_from: float | None = None,
    t_to: float | None = None,
    force: bool = False,
    tools: TdeedTools | None = None,
) -> TdeedResult:
    """Run T-DEED on the video of a match and write its predictions into the match directory.

    Args:
        match_dir: The match directory: `src.mp4` is read from it, the result is written to it.
        match_id: Match id for the logs.
        cache_dir: Where the T-DEED copy, the checkpoint and the model inputs are kept (outside
            the repository); default `~/.cache/aicoach/tdeed`.
        video: The video; default `src.mp4` in the match directory.
        device: `auto` (the video card, an error without one), `cuda`, or `cpu` (explicit only,
            much slower).
        checkpoint: A checkpoint file instead of the download.
        t_from: Segment start in seconds, moved down to the 24 s grid of the model; with `t_to`
            the result goes to a `tdeedseg_*.json` and the `tdeed_*.json` files are not touched.
        t_to: Segment end in seconds.
        force: Run a segment again although its file is there. A whole-match file is never
            replaced by the command (it may be the frozen one): `force` with one there is an error.
        tools: The outside effects (injected in tests).

    Returns:
        The result; `cached` is True when the output was already there and nothing ran.

    Raises:
        SpottingError: No video card for `auto`, an unreadable video, a bad segment, a failing git,
            FFmpeg or `inference.py`, a bad checkpoint, or an output that is not a T-DEED file.
    """
    tools = tools or TdeedTools()
    cache = cache_dir or default_cache_dir()
    video = video or match_dir / "src.mp4"
    duration, segment = _plan(video, t_from, t_to, tools)
    target = match_dir / output_name(segment)
    own = [target] if target.is_file() else []  # a segment run looks only at its own file
    existing = find_existing(match_dir) if segment is None else own
    log = stage_logger(STAGE, match_id)
    if existing and not force:
        log.info("tdeed_skipped", reason="output exists", file=existing[0].name)
        # Nothing runs, so no video card is needed to say so.
        return _cached_result(existing[0], duration)
    if existing and segment is None:
        # A whole-match file may be the frozen one that `fuse` used: never replaced by a command.
        msg = (
            f"{existing[0].name} exists: delete it by hand to run T-DEED again "
            "(--force is only for segments)"
        )
        raise SpottingError(msg)
    chosen = resolve_device(device, tools.cuda_available())
    if chosen == "cpu":
        log.warning("tdeed_on_cpu", note="much slower than the video card")
    job = _Job(match_dir, video, cache, chosen, checkpoint, segment, target, duration, tools, log)
    return _run(job)
