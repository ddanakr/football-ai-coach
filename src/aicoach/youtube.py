"""YouTube video id and plain watch link: shared by `aicoach.ingest` and `aicoach.ui`."""

import re

from aicoach.errors import ConfigError

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
WATCH_URL = "https://www.youtube.com/watch?v={video_id}"


def validate_video_id(video_id: str) -> str:
    """Return the YouTube video id if it has the usual 11-character form.

    The id ends up in URLs and in HTML, so anything else is refused.

    Raises:
        ConfigError: The id is empty or has other characters or another length.
    """
    if not _VIDEO_ID.fullmatch(video_id):
        msg = "the YouTube video id must be 11 characters: letters, digits, '-' and '_'"
        raise ConfigError(msg)
    return video_id


def watch_url(video_id: str) -> str:
    """Plain watch URL of the video (for `st.video`).

    Raises:
        ConfigError: Bad video id.
    """
    return WATCH_URL.format(video_id=validate_video_id(video_id))
