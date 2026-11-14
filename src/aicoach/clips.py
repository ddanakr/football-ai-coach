"""Clip export with FFmpeg (FR-7): a window around an event, cut from the match video.

The event time is read from `events_current`, so a user's move or retype is respected. The
clip is re-encoded (H.264 and AAC), not stream-copied: the broadcast is AV1, and a stream copy
can only start on a key frame, so the clip would start seconds away from the asked time
(D-056). Clips are written to `<match directory>/clips/` (the video's directory, as for the
other stages) and an existing clip is returned without running FFmpeg again.
"""

import math
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import structlog

from aicoach.errors import AicoachError, ExportError
from aicoach.scoreboard.video import VideoInfo, probe_video
from aicoach.stage_common import elapsed_ms
from aicoach.storage import CurrentEvent, connect, get_current_events

CLIPS_DIR = "clips"
DEFAULT_BEFORE_S = 8.0
DEFAULT_AFTER_S = 12.0
FFMPEG_TIMEOUT_S = 300.0
# Quality and speed of the re-encode: a fast preset is enough for a short review clip.
X264_PRESET = "veryfast"
X264_CRF = "23"
AAC_BITRATE = "128k"
STDERR_TAIL_LINES = 20
STDERR_TAIL_CHARS = 2000

Runner = Callable[[list[str], float], "subprocess.CompletedProcess[bytes]"]
Probe = Callable[[Path], VideoInfo]

logger = structlog.get_logger()


def _require_finite(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value):
            msg = f"{name} must be a finite number, got {value}"
            raise ValueError(msg)


def clip_window(
    t_v: float,
    before_s: float,
    after_s: float,
    duration_s: float,
    *,
    min_length_s: float = 0.0,
) -> tuple[float, float]:
    """Window `[t_v - before_s, t_v + after_s]` cut to the video `[0, duration_s]`.

    Near the start or the end of the video the window is trimmed, not refused. A window
    shorter than `min_length_s` (for example both margins 0) is widened inside the video,
    first towards the end, so a clip has at least one frame.

    Returns:
        `(start, end)` in video seconds, `0 <= start <= end <= duration_s`.

    Raises:
        ValueError: A value is not finite, a margin or `min_length_s` is negative,
            `duration_s` is not positive, or `t_v` lies outside `[0, duration_s]`.
    """
    _require_finite(
        t_v=t_v,
        before_s=before_s,
        after_s=after_s,
        duration_s=duration_s,
        min_length_s=min_length_s,
    )
    if before_s < 0 or after_s < 0:
        msg = f"clip margins must not be negative, got before {before_s} s and after {after_s} s"
        raise ValueError(msg)
    if min_length_s < 0:
        msg = f"min_length_s must not be negative, got {min_length_s}"
        raise ValueError(msg)
    if duration_s <= 0:
        msg = f"video duration must be positive, got {duration_s} s"
        raise ValueError(msg)
    if not 0 <= t_v <= duration_s:
        msg = f"event time {t_v} s is outside the video [0, {duration_s}] s"
        raise ValueError(msg)
    start, end = max(0.0, t_v - before_s), min(duration_s, t_v + after_s)
    if end - start < min_length_s:
        end = min(duration_s, start + min_length_s)
        start = max(0.0, end - min_length_s)
    return start, end


def clip_filename(event: CurrentEvent, before_s: float, after_s: float) -> str:
    """File name `<id>_<type>_t<ms>ms_b<before>_a<after>.mp4`, e.g. `p12_goal_t200000ms_b8_a12.mp4`.

    The id loses its colon (not allowed in Windows file names). The event time in milliseconds
    is part of the name, so a clip made before a user moved the event is not returned for the
    new time. No team or player names.
    """
    event_id = event.event_id.replace(":", "")
    millis = round(event.t_v * 1000)
    return f"{event_id}_{event.type.value}_t{millis}ms_b{before_s:g}_a{after_s:g}.mp4"


def build_ffmpeg_command(
    video: Path, output: Path, start_s: float, length_s: float, *, ffmpeg: str = "ffmpeg"
) -> list[str]:
    """FFmpeg arguments that re-encode `length_s` seconds of `video` from `start_s` to `output`.

    `-ss` before `-i` seeks fast and, with re-encoding, still starts at the exact time. The
    first video stream is required, the first audio stream is optional (`0:a:0?`). The output
    is H.264 4:2:0 (plays in browsers even from a 10-bit source) with AAC sound and the index
    at the front (`+faststart`) for playback while loading. The container is set by `-f mp4`
    because the output may be a temporary name.
    """
    return [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(video),
        "-t",
        f"{length_s:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        X264_PRESET,
        "-crf",
        X264_CRF,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        AAC_BITRATE,
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(output),
    ]


def _subprocess_runner(
    command: list[str], timeout_s: float
) -> "subprocess.CompletedProcess[bytes]":
    # A list of our own arguments, no shell: the video path cannot inject anything.
    return subprocess.run(  # noqa: S603
        command, capture_output=True, timeout=timeout_s, check=False
    )


def stderr_tail(stderr: bytes | None) -> str:
    """The last lines of FFmpeg's error output, decoded leniently and shortened."""
    text = (stderr or b"").decode("utf-8", errors="replace").strip()
    tail = "\n".join(text.splitlines()[-STDERR_TAIL_LINES:])
    return tail[-STDERR_TAIL_CHARS:] or "(no output)"


def run_ffmpeg(command: list[str], timeout_s: float, runner: Runner = _subprocess_runner) -> None:
    """Run FFmpeg and turn every failure into `ExportError`.

    Raises:
        ExportError: FFmpeg is not installed, did not finish in `timeout_s`, or exited with a
            non-zero code (the message carries the tail of its error output), or could not be
            started at all.
    """
    try:
        done = runner(command, timeout_s)
    except FileNotFoundError as err:
        msg = f"{command[0]} not found: install FFmpeg and put it in PATH"
        raise ExportError(msg) from err
    except subprocess.TimeoutExpired as err:
        msg = f"FFmpeg did not finish in {timeout_s:g} s"
        raise ExportError(msg) from err
    except OSError as err:
        msg = f"cannot start {command[0]}: {err}"
        raise ExportError(msg) from err
    if done.returncode != 0:
        msg = f"FFmpeg failed (exit code {done.returncode}): {stderr_tail(done.stderr)}"
        raise ExportError(msg)


def _window_for(
    event: CurrentEvent, before_s: float, after_s: float, info: VideoInfo
) -> tuple[float, float]:
    try:
        return clip_window(
            event.t_v, before_s, after_s, info.duration_s, min_length_s=1.0 / info.fps
        )
    except ValueError as err:
        msg = f"cannot cut a clip for event {event.event_id}: {err}"
        raise ExportError(msg) from err


def _probe(video: Path, probe: Probe) -> VideoInfo:
    try:
        return probe(video)
    except AicoachError as err:
        msg = f"cannot read the video for a clip: {err}"
        raise ExportError(msg) from err


def _encode(
    command: list[str], partial: Path, output: Path, timeout_s: float, runner: Runner
) -> None:
    """Run FFmpeg into `partial` and rename it to `output`; on any failure remove `partial`."""
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        run_ffmpeg(command, timeout_s, runner)
        partial.replace(output)
    except ExportError:
        partial.unlink(missing_ok=True)
        raise
    except OSError as err:
        partial.unlink(missing_ok=True)
        msg = f"cannot write the clip {output}: {err}"
        raise ExportError(msg) from err


def export_clip(
    video: Path,
    event: CurrentEvent,
    before_s: float = DEFAULT_BEFORE_S,
    after_s: float = DEFAULT_AFTER_S,
    *,
    timeout_s: float = FFMPEG_TIMEOUT_S,
    runner: Runner = _subprocess_runner,
    probe: Probe = probe_video,
) -> Path:
    """Cut the clip of one event into `<video's directory>/clips/` and return its path.

    An existing clip with the same name is returned at once: FFmpeg is not run and the video is
    not read. A new clip is written to a temporary name and renamed when FFmpeg has finished, so
    a failure or a timeout never leaves a half-written clip under the final name.

    Raises:
        ExportError: The margins are invalid, the video cannot be read, the event lies outside
            it, or FFmpeg failed.
    """
    output = video.parent / CLIPS_DIR / clip_filename(event, before_s, after_s)
    log = logger.bind(stage="clip", match_id=event.match_id, event_id=event.event_id)
    if output.is_file():
        log.info("clip_cached", path=output.name)
        return output
    start, end = _window_for(event, before_s, after_s, _probe(video, probe))
    partial = output.with_name(output.name + ".part")
    command = build_ffmpeg_command(video, partial, start, end - start)
    started = time.perf_counter()
    try:
        _encode(command, partial, output, timeout_s, runner)
    except ExportError as err:
        log.exception("clip_failed", reason=str(err)[:200], duration_ms=elapsed_ms(started))
        raise
    log.info(
        "clip_exported",
        path=output.name,
        start_s=start,
        end_s=end,
        duration_ms=elapsed_ms(started),
    )
    return output


def find_current_event(db_path: Path, match_id: str, event_id: str) -> CurrentEvent:
    """The event as `events_current` shows it now (user edits applied, deleted ones hidden).

    Raises:
        ExportError: The database file does not exist, or the match has no such visible event
            (unknown id, deleted, or replaced by a newer fusion run).
        StorageError: SQLite failed.
    """
    if not db_path.is_file():
        msg = f"database {db_path} does not exist"
        raise ExportError(msg)
    with connect(db_path) as conn:
        events = get_current_events(conn, match_id)
    for event in events:
        if event.event_id == event_id:
            return event
    msg = (
        f"event {event_id} is not among the current events of match {match_id!r} "
        "(unknown, deleted, or from an older fusion run)"
    )
    raise ExportError(msg)


def export_event_clip(
    db_path: Path,
    match_id: str,
    video: Path,
    event_id: str,
    before_s: float = DEFAULT_BEFORE_S,
    after_s: float = DEFAULT_AFTER_S,
    *,
    timeout_s: float = FFMPEG_TIMEOUT_S,
) -> Path:
    """Look the event up in `events_current` and export its clip (see `export_clip`).

    Raises:
        ExportError: The event is not found, or the export failed.
        StorageError: SQLite failed.
    """
    event = find_current_event(db_path, match_id, event_id)
    return export_clip(video, event, before_s, after_s, timeout_s=timeout_s)
