"""Team names from the title of the YouTube video (D-078).

`ingest` keeps the yt-dlp info of the video as `src.info.json` next to `src.mp4`. A title such as
`Актобе - Кайрат | КПЛ 2025` names the home team first (the broadcast convention). The parser is
deliberately strict: a title that does not split into exactly two names gives nothing, and the
caller falls back to the default words, rather than guessing a wrong name.
"""

import json
import re
from pathlib import Path

import structlog

from aicoach.teams.store import MAX_NAME_CHARS, Side

logger = structlog.get_logger()

INFO_FILE = "src.info.json"
# " - ", " — ", " – ", " vs " (with or without a dot), always with spaces around: a hyphen inside
# a name ("Жетысу-М") is not a separator.
_SEPARATOR = re.compile(r"\s+(?:-|—|–|vs\.?)\s+", re.IGNORECASE)
_TAIL = re.compile(r"\s*(?:\||/|#|\(|\[).*$")
_MAX_TITLE_CHARS = 300
# A title is text from outside, and the names go into the system prompt of the agent: only
# letters, digits, spaces and a few marks of club names, at most this many words.
_SAFE_NAME = re.compile(r"[\w .'’&-]+")
_MAX_NAME_WORDS = 4


def _clean(part: str) -> str:
    return " ".join(part.split()).strip(" .,:;")


def _is_safe_name(name: str) -> bool:
    return (
        bool(name)
        and len(name) <= MAX_NAME_CHARS
        and len(name.split()) <= _MAX_NAME_WORDS
        and _SAFE_NAME.fullmatch(name) is not None
    )


def parse_title_teams(title: str) -> dict[Side, str]:
    """The two team names of a match title, home first; empty when the title is not of the form
    `<home> <separator> <away>` after the trailing `| ...`, `(...)`, `#...` parts are cut off and,
    in the first part, a prefix before `: `.

    The separators are ` - `, ` — `, ` – ` and ` vs `.
    """
    head = _TAIL.sub("", title[:_MAX_TITLE_CHARS])
    parts = _SEPARATOR.split(head)
    if len(parts) != 2:  # none, or three: "league - home - away" is not guessed at
        return {}
    home, away = (_clean(p) for p in parts)
    if ": " in home:
        home = _clean(home.rsplit(": ", 1)[1])
    if not all(_is_safe_name(name) for name in (home, away)):
        return {}
    if home.casefold() == away.casefold():
        return {}
    return {"home": home, "away": away}


def read_title(match_dir: Path) -> str | None:
    """The `title` of `src.info.json`; `None` when the file is missing, broken or has none."""
    path = match_dir / INFO_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as err:
        logger.warning("info_json_unreadable", stage="teams", reason=type(err).__name__)
        return None
    title = data.get("title") if isinstance(data, dict) else None
    return title if isinstance(title, str) and title.strip() else None
