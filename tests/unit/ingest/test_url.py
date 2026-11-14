"""YouTube links: the video id from the usual forms, refusal of anything else."""

import pytest

from aicoach.errors import IngestError
from aicoach.ingest.url import canonical_url, parse_youtube_url

VID = "abcDEF_-123"  # invented id


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}&t=125s",
        f"http://m.youtube.com/watch?feature=share&v={VID}&list=PL1",
        f"www.youtube.com/watch?v={VID}",
        f"https://youtu.be/{VID}",
        f"https://youtu.be/{VID}?si=xyz&t=30",
        f"youtu.be/{VID}/",
        f"https://www.youtube.com/live/{VID}",
        f"https://www.youtube.com/live/{VID}?si=abc",
        f"  https://WWW.YOUTUBE.COM/watch?v={VID}  ",
        f"https://www.youtube.com/watch/?v={VID}",
    ],
)
def test_parse_youtube_url_known_forms_give_the_id(url: str) -> None:
    assert parse_youtube_url(url) == VID


@pytest.mark.unit
@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("x" * 2001, "too long"),
        ("https://example.com/watch?v=abcDEF_-123", "unknown site"),
        ("https://youtube.com.evil.org/watch?v=abcDEF_-123", "unknown site"),
        ("ftp://youtube.com/watch?v=abcDEF_-123", "http and https"),
        ("javascript:alert(1)", "unknown site|http"),
        ("https://www.youtube.com/watch?v=short", "11-character"),
        ("https://www.youtube.com/watch?v=abcDEF_-1234", "11-character"),
        ("https://www.youtube.com/watch?v=abc<DEF>123", "11-character"),
        ("https://www.youtube.com/watch", "11-character"),
        ("https://www.youtube.com/watch?v=abcDEF_-123&v=zzzzzzzzzzz", "11-character"),
        ("https://www.youtube.com/channel/UCabcdefghij", "11-character"),
        ("https://www.youtube.com/live/", "11-character"),
        ("https://youtu.be/", "11-character"),
        ("https://youtu.be/abcDEF_-123/extra", "11-character"),
        ("https://[::1/watch?v=abcDEF_-123", "cannot be parsed"),
        ("не ссылка", "unknown site"),
    ],
)
def test_parse_youtube_url_garbage_raises_ingest_error(url: str, reason: str) -> None:
    with pytest.raises(IngestError, match=reason):
        parse_youtube_url(url)


@pytest.mark.unit
def test_canonical_url_drops_everything_but_the_id() -> None:
    assert canonical_url(VID) == f"https://www.youtube.com/watch?v={VID}"


@pytest.mark.unit
def test_canonical_url_bad_id_raises() -> None:
    with pytest.raises(IngestError):
        canonical_url("bad id")
