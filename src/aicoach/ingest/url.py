"""YouTube links: the video id from the usual link forms, checked before anything is downloaded."""

from urllib.parse import parse_qs, urlsplit

from aicoach.errors import ConfigError, IngestError
from aicoach.youtube import validate_video_id, watch_url

_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com"})
_SHORT_HOST = "youtu.be"
_PATH_PREFIXES = ("/live/",)
_MAX_URL_CHARS = 2000


def parse_youtube_url(url: str) -> str:
    """The 11-character video id of a YouTube link.

    Accepted forms (with or without `https://`, extra parameters such as `&t=` are ignored):
    `youtube.com/watch?v=<id>`, `youtu.be/<id>`, `youtube.com/live/<id>`.

    Raises:
        IngestError: The text is not a YouTube video link or the id has a wrong form.
    """
    text = url.strip()
    if not text or len(text) > _MAX_URL_CHARS:
        msg = "not a YouTube link: the text is empty or too long"
        raise IngestError(msg)
    if "://" not in text:
        text = "https://" + text
    try:
        parts = urlsplit(text)
        host = (parts.hostname or "").lower()
    except ValueError as err:
        msg = "not a YouTube link: the address cannot be parsed"
        raise IngestError(msg) from err
    if parts.scheme not in ("http", "https"):
        msg = "not a YouTube link: only http and https addresses are accepted"
        raise IngestError(msg)
    return _checked_id(_raw_id(host, parts.path, parts.query))


def _raw_id(host: str, path: str, query: str) -> str:
    """The id candidate for a known host and path form; empty when the form is unknown."""
    if host == _SHORT_HOST:
        return path.strip("/")
    if host not in _HOSTS:
        msg = f"not a YouTube link: unknown site {host or '(none)'}"
        raise IngestError(msg)
    if path.rstrip("/") == "/watch":
        values = parse_qs(query).get("v", [])
        return values[0] if len(values) == 1 else ""
    for prefix in _PATH_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix) :].strip("/")
    return ""


def _checked_id(raw: str) -> str:
    try:
        return validate_video_id(raw)
    except ConfigError as err:
        msg = (
            "not a YouTube video link: expected youtube.com/watch?v=<id>, youtu.be/<id> or "
            "youtube.com/live/<id> with an 11-character id"
        )
        raise IngestError(msg) from err


def canonical_url(video_id: str) -> str:
    """The plain watch link of the video, without playlists, times or tracking parameters.

    Raises:
        IngestError: The id has a wrong form.
    """
    try:
        return watch_url(video_id)
    except ConfigError as err:
        raise IngestError(str(err)) from err
