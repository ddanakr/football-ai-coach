"""Checks and conversions of the downloaded video: frame size, and the 16 kHz mono audio track."""

from pathlib import Path

from aicoach.clips import Runner, run_ffmpeg
from aicoach.errors import ExportError, IngestError
from aicoach.scoreboard.video import VideoInfo

AUDIO_FILE = "audio.wav"
AUDIO_RATE_HZ = 16000
AUDIO_TIMEOUT_S = 1800.0  # a two-hour broadcast decodes in about a minute; this is a hard stop
# The scoreboard area and the plate layout are fractions measured on a 1280x720 broadcast.
EXPECTED_SIZE = (1280, 720)


def check_frame_size(info: VideoInfo, expected: tuple[int, int] = EXPECTED_SIZE) -> None:
    """Refuse a video whose frame size is not the one the scoreboard and plates are tuned for.

    Raises:
        IngestError: The size differs; the message says why it matters.
    """
    if (info.width, info.height) != expected:
        width, height = expected
        msg = (
            f"the video is {info.width}x{info.height}, but the scoreboard and plate detectors "
            f"are calibrated for {width}x{height} broadcasts (D-030, D-036); other sizes would "
            "give wrong or no events, so processing stops here"
        )
        raise IngestError(msg)


def build_audio_command(video: Path, output: Path, *, ffmpeg: str = "ffmpeg") -> list[str]:
    """FFmpeg arguments: the first audio track as 16 kHz mono 16-bit WAV into `output`."""
    return [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_RATE_HZ),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(output),
    ]


def extract_audio(
    video: Path, output: Path, runner: Runner | None = None, timeout_s: float = AUDIO_TIMEOUT_S
) -> Path:
    """Write the audio of `video` to `output`; a temporary file is renamed only on success.

    Raises:
        IngestError: FFmpeg is missing, fails (no audio track, broken file) or times out, or the
            file cannot be written.
    """
    partial = output.with_name(output.name + ".part")
    try:
        command = build_audio_command(video, partial)
        if runner is None:
            run_ffmpeg(command, timeout_s)
        else:
            run_ffmpeg(command, timeout_s, runner)
        partial.replace(output)
    except ExportError as err:
        partial.unlink(missing_ok=True)
        msg = f"audio extraction failed: {err}"
        raise IngestError(msg) from err
    except OSError as err:
        partial.unlink(missing_ok=True)
        msg = f"cannot write {output}: {err}"
        raise IngestError(msg) from err
    return output
