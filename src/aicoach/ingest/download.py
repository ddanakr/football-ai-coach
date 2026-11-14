"""Download of the broadcast with yt-dlp (Python API): video up to 720p and audio, one `src.mp4`.

yt-dlp keeps partial downloads (`.part`) and continues them on the next start, so a broken
connection costs only the missing bytes. The final `src.mp4` appears only after the merge, so
its presence means a complete download.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import yt_dlp
from yt_dlp.utils import YoutubeDLError

from aicoach.errors import IngestError

VIDEO_FILE = "src.mp4"
MAX_HEIGHT = 720
SOCKET_TIMEOUT_S = 30
RETRIES = 10
_ERROR_CHARS = 500

ProgressCallback = Callable[[float], None]


class Downloader(Protocol):
    """Downloads a YouTube video into `<target_dir>/src.mp4` (yt-dlp, or a fake in tests)."""

    def download(self, url: str, target_dir: Path, on_progress: ProgressCallback) -> Path:
        """Download and return the path of `src.mp4`.

        Raises:
            IngestError: The video is unavailable, still live, or the download failed.
        """
        ...


def ytdlp_options(target_dir: Path, hook: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Options of `yt_dlp.YoutubeDL`: best video not above 720p plus best audio, merged to mp4.

    Any: yt-dlp takes a plain dict of mixed values (it has no typed options).
    """
    return {
        "format": f"bv*[height<={MAX_HEIGHT}]+ba",
        "merge_output_format": "mp4",
        "paths": {"home": str(target_dir), "temp": str(target_dir)},
        "outtmpl": {"default": "src.%(ext)s"},
        "continuedl": True,
        "noplaylist": True,
        "retries": RETRIES,
        "fragment_retries": RETRIES,
        "socket_timeout": SOCKET_TIMEOUT_S,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
    }


def progress_fraction(event: dict[str, Any]) -> float | None:
    """Downloaded share of the current file from a yt-dlp progress event, if it is known.

    Any: yt-dlp progress events are plain dicts of mixed values.
    """
    if event.get("status") != "downloading":
        return None
    done = event.get("downloaded_bytes")
    total = event.get("total_bytes") or event.get("total_bytes_estimate")
    if not isinstance(done, int | float) or not isinstance(total, int | float) or total <= 0:
        return None
    return min(1.0, max(0.0, float(done) / float(total)))


def _short(err: BaseException) -> str:
    text = str(err).removeprefix("ERROR: ").strip() or type(err).__name__
    return text[:_ERROR_CHARS]


class YtDlpDownloader:
    """The real downloader: yt-dlp through its Python API, ffmpeg for the merge."""

    def download(self, url: str, target_dir: Path, on_progress: ProgressCallback) -> Path:
        """Download the video and its audio into `<target_dir>/src.mp4`.

        Raises:
            IngestError: YouTube refused or the video is unavailable (yt-dlp's reason is in the
                message), the broadcast is live or upcoming, or no file appeared.
        """

        def hook(event: dict[str, Any]) -> None:
            fraction = progress_fraction(event)
            if fraction is not None:
                on_progress(fraction)

        try:
            with yt_dlp.YoutubeDL(ytdlp_options(target_dir, hook)) as ydl:
                info = ydl.extract_info(url, download=False)
                _refuse_live(info)
                ydl.process_ie_result(info, download=True)
        except YoutubeDLError as err:
            msg = f"YouTube download failed: {_short(err)}"
            raise IngestError(msg) from err
        except OSError as err:
            msg = f"cannot write the download to {target_dir}: {_short(err)}"
            raise IngestError(msg) from err
        video = target_dir / VIDEO_FILE
        if not video.is_file():
            msg = f"the download finished but {video} is missing (no mp4 merge?)"
            raise IngestError(msg)
        return video


def _refuse_live(info: object) -> None:
    """A live or upcoming broadcast cannot be processed: it has no end yet."""
    status = info.get("live_status") if isinstance(info, dict) else None
    if status in ("is_live", "is_upcoming"):
        msg = "the broadcast is live or has not started yet: process it after it has ended"
        raise IngestError(msg)
