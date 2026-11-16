"""Cutting a fragment of the sound out of the match video (FFmpeg).

16 kHz mono MP3: the Gemini docs say audio is downsampled to 16 Kbps and mixed to one channel,
so a bigger file adds nothing; MP3 is on the list of supported formats; a 2-minute fragment is
well under the 20 MB inline request limit (audio docs, checked 2026-11-16).
"""

from pathlib import Path

from aicoach.clips import Runner, run_ffmpeg, subprocess_runner
from aicoach.errors import ExportError, TranscriptionError

AUDIO_MIME = "audio/mp3"
MP3_BITRATE = "32k"
SAMPLE_RATE = "16000"
FFMPEG_TIMEOUT_S = 120.0


def build_audio_command(
    video: Path, output: Path, start_s: float, length_s: float, *, ffmpeg: str = "ffmpeg"
) -> list[str]:
    """FFmpeg arguments that cut `length_s` seconds of sound from `start_s` into an MP3."""
    return [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{start_s:.3f}", "-i", str(video), "-t", f"{length_s:.3f}",
        "-vn", "-ac", "1", "-ar", SAMPLE_RATE, "-c:a", "libmp3lame", "-b:a", MP3_BITRATE,
        "-f", "mp3", str(output),
    ]  # fmt: skip


def cut_audio(
    video: Path,
    output: Path,
    start_s: float,
    length_s: float,
    runner: Runner = subprocess_runner,
) -> bytes:
    """Cut the fragment into `output` (written through a temporary name) and return its bytes.

    Raises:
        TranscriptionError: Bad bounds, FFmpeg failed, or it produced no file or an empty one.
    """
    if start_s < 0 or length_s <= 0:
        msg = f"bad fragment bounds: start {start_s}, length {length_s}"
        raise TranscriptionError(msg)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    try:
        command = build_audio_command(video, partial, start_s, length_s)
        try:
            run_ffmpeg(command, FFMPEG_TIMEOUT_S, runner)
        except ExportError as err:
            raise TranscriptionError(str(err)) from err
        if not partial.is_file() or partial.stat().st_size == 0:
            msg = "FFmpeg produced no sound (the video may have no audio track in this range)"
            raise TranscriptionError(msg)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return output.read_bytes()
