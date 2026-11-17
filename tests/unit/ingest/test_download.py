"""The yt-dlp downloader with a fake `YoutubeDL` (no network): options, progress, errors."""

from pathlib import Path
from types import TracebackType
from typing import Any

import pytest
from yt_dlp.utils import DownloadError

from aicoach.errors import IngestError
from aicoach.ingest.download import (
    VIDEO_FILE,
    YtDlpDownloader,
    progress_fraction,
    ytdlp_options,
)

URL = "https://www.youtube.com/watch?v=abcDEF_-123"


# Any: yt-dlp takes and returns plain dicts of mixed values.
class FakeYoutubeDL:
    """Stands in for `yt_dlp.YoutubeDL`: records the options, reports progress, writes a file."""

    created: list[dict[str, Any]] = []
    info: dict[str, Any] = {"id": "abcDEF_-123", "live_status": "was_live"}
    error: BaseException | None = None
    write_file = True

    def __init__(self, options: dict[str, Any]) -> None:
        self.options = options
        FakeYoutubeDL.created.append(options)

    def __enter__(self) -> "FakeYoutubeDL":
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def extract_info(self, url: str, download: bool) -> dict[str, Any]:
        assert url == URL
        assert download is False
        if FakeYoutubeDL.error is not None:
            raise FakeYoutubeDL.error
        return FakeYoutubeDL.info

    def process_ie_result(self, info: dict[str, Any], download: bool) -> dict[str, Any]:
        assert download is True
        hook = self.options["progress_hooks"][0]
        hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
        hook({"status": "finished"})
        if FakeYoutubeDL.write_file:
            (Path(self.options["paths"]["home"]) / VIDEO_FILE).write_bytes(b"x")
        return info


@pytest.fixture
def fake_ydl(monkeypatch: pytest.MonkeyPatch) -> type[FakeYoutubeDL]:
    FakeYoutubeDL.created = []
    FakeYoutubeDL.info = {"id": "abcDEF_-123", "live_status": "was_live"}
    FakeYoutubeDL.error = None
    FakeYoutubeDL.write_file = True
    monkeypatch.setattr("aicoach.ingest.download.yt_dlp.YoutubeDL", FakeYoutubeDL)
    return FakeYoutubeDL


@pytest.mark.unit
def test_ytdlp_options_ask_for_720p_video_plus_audio_merged_to_mp4_with_resume(
    tmp_path: Path,
) -> None:
    options = ytdlp_options(tmp_path, lambda _event: None)

    assert options["format"] == "bv*[height<=720]+ba"
    assert options["merge_output_format"] == "mp4"
    assert options["continuedl"] is True
    assert options["noplaylist"] is True
    assert options["socket_timeout"] > 0
    assert options["paths"]["home"] == str(tmp_path)
    assert options["outtmpl"]["default"] == "src.%(ext)s"
    assert options["writeinfojson"] is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"status": "downloading", "downloaded_bytes": 25, "total_bytes": 100}, 0.25),
        ({"status": "downloading", "downloaded_bytes": 10, "total_bytes_estimate": 40}, 0.25),
        ({"status": "downloading", "downloaded_bytes": 500, "total_bytes": 100}, 1.0),
        ({"status": "downloading", "downloaded_bytes": 5}, None),
        ({"status": "downloading", "downloaded_bytes": 5, "total_bytes": 0}, None),
        ({"status": "downloading", "downloaded_bytes": None, "total_bytes": 10}, None),
        ({"status": "finished", "downloaded_bytes": 10, "total_bytes": 10}, None),
        ({}, None),
    ],
)
def test_progress_fraction_known_and_unknown_totals(
    event: dict[str, Any], expected: float | None
) -> None:
    assert progress_fraction(event) == expected


@pytest.mark.unit
def test_download_success_returns_src_mp4_and_reports_progress(
    tmp_path: Path, fake_ydl: type[FakeYoutubeDL]
) -> None:
    seen: list[float] = []

    path = YtDlpDownloader().download(URL, tmp_path, seen.append)

    assert path == tmp_path / VIDEO_FILE
    assert seen == [0.5]
    assert len(fake_ydl.created) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "ERROR: [youtube] abcDEF_-123: Video unavailable",
        "ERROR: [youtube] abcDEF_-123: Sign in to confirm you're not a bot",
        "ERROR: unable to download video data: HTTP Error 403: Forbidden",
    ],
)
def test_download_youtube_error_becomes_ingest_error_with_the_reason(
    tmp_path: Path, fake_ydl: type[FakeYoutubeDL], text: str
) -> None:
    fake_ydl.error = DownloadError(text)

    with pytest.raises(IngestError, match="YouTube download failed") as caught:
        YtDlpDownloader().download(URL, tmp_path, lambda _f: None)

    assert text.removeprefix("ERROR: ") in str(caught.value)
    assert not (tmp_path / VIDEO_FILE).exists()


@pytest.mark.unit
@pytest.mark.parametrize("status", ["is_live", "is_upcoming"])
def test_download_live_broadcast_is_refused(
    tmp_path: Path, fake_ydl: type[FakeYoutubeDL], status: str
) -> None:
    fake_ydl.info = {"id": "abcDEF_-123", "live_status": status}

    with pytest.raises(IngestError, match="live"):
        YtDlpDownloader().download(URL, tmp_path, lambda _f: None)


@pytest.mark.unit
def test_download_disk_error_becomes_ingest_error(
    tmp_path: Path, fake_ydl: type[FakeYoutubeDL]
) -> None:
    fake_ydl.error = PermissionError("denied")

    with pytest.raises(IngestError, match="cannot write the download"):
        YtDlpDownloader().download(URL, tmp_path, lambda _f: None)


@pytest.mark.unit
def test_download_without_merged_file_raises(tmp_path: Path, fake_ydl: type[FakeYoutubeDL]) -> None:
    fake_ydl.write_file = False

    with pytest.raises(IngestError, match="missing"):
        YtDlpDownloader().download(URL, tmp_path, lambda _f: None)
