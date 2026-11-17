"""What is known about the two teams of a match: names and kit colours (D-078).

`teams.json` is written by the `teams` stage (names from the lineup, the video title or the
defaults; kit colours voted from video clips). `teams_override.json` holds what the user typed in
the viewer; it is a separate file, so a new run of the stage never erases it. `load_effective`
joins both for the agent, the viewer and the corner tags: override, then the stage, then the
lineup names, then the defaults. Nothing here touches the event database.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aicoach.errors import StorageError
from aicoach.i18n import ru
from aicoach.plate_names.lineup import LINEUP_FILE, parse_team_names
from aicoach.stage_common import write_atomic

logger = structlog.get_logger()

RESULT_FILE = "teams.json"
OVERRIDE_FILE = "teams_override.json"
SCHEMA_VERSION = 1
MAX_NAME_CHARS = 60

Side = Literal["home", "away"]
SIDES: tuple[Side, ...] = ("home", "away")
# The normalised kit colours a model may answer with, and the word for "cannot tell".
KIT_COLORS = (
    "white",
    "black",
    "red",
    "blue",
    "light_blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "other",
)
KIT_UNCLEAR = "unclear"
Kit = Literal[
    "white",
    "black",
    "red",
    "blue",
    "light_blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "other",
]
NameSource = Literal["lineup", "title", "default", "manual"]
KitSource = Literal["vlm", "manual", "undetermined"]
Role = Literal["goal", "card"]
VoteStatus = Literal["ok", "retried", "invalid"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KitVote(_Model):
    """One clip's answer about a kit colour; `kit` is a colour or `unclear` (an unusable answer
    after the retry is `unclear` with status `invalid`)."""

    event_id: str
    t_v: float
    role: Role
    side: Side
    kit: str
    status: VoteStatus
    model: str
    prompt_version: int
    before_s: float
    after_s: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = None


class TeamEntry(_Model):
    """One side: the name and where it came from, the kit colour (`None`: not determined) and
    the count of valid votes per colour."""

    name: str
    name_source: NameSource
    kit: Kit | None = None
    kit_source: KitSource = "undetermined"
    kit_votes: dict[str, int] = Field(default_factory=dict)


class TeamsResult(_Model):
    """`teams.json`."""

    schema_version: int = SCHEMA_VERSION
    match_id: str
    cache_key: str
    model: str | None = None
    home: TeamEntry
    away: TeamEntry
    votes: list[KitVote] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    # Why the kit colours were not asked for at all (no key, no video, no events); `None` when
    # they were.
    kits_skipped: str | None = None

    def entry(self, side: Side) -> TeamEntry:
        """The entry of a side."""
        return self.home if side == "home" else self.away


class SideOverride(_Model):
    """What the user set for one side; `None` leaves the field to the stage."""

    name: str | None = Field(default=None, max_length=MAX_NAME_CHARS)
    kit: Kit | None = None


class TeamsOverride(_Model):
    """`teams_override.json`."""

    home: SideOverride = Field(default_factory=SideOverride)
    away: SideOverride = Field(default_factory=SideOverride)


class EffectiveTeam(_Model):
    """A side as everyone downstream sees it."""

    name: str
    name_source: NameSource
    kit: Kit | None = None
    kit_source: KitSource = "undetermined"


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as err:
        logger.warning("teams_file_unreadable", stage="teams", file=path.name, reason=str(err))
        return None


def read_teams(match_dir: Path) -> TeamsResult | None:
    """`teams.json` of the match; `None` when it is missing, unreadable or invalid (a warning is
    logged: the files only decorate, they must not stop the agent or the viewer)."""
    text = _read(match_dir / RESULT_FILE)
    if text is None:
        return None
    try:
        return TeamsResult.model_validate_json(text)
    except (ValidationError, ValueError) as err:
        logger.warning("teams_file_invalid", stage="teams", reason=type(err).__name__)
        return None


def read_override(match_dir: Path) -> TeamsOverride:
    """The user's override; empty when there is none or the file is unusable (a warning)."""
    text = _read(match_dir / OVERRIDE_FILE)
    if text is None:
        return TeamsOverride()
    try:
        return TeamsOverride.model_validate_json(text)
    except (ValidationError, ValueError) as err:
        logger.warning("teams_override_invalid", stage="teams", reason=type(err).__name__)
        return TeamsOverride()


def save_override(match_dir: Path, override: TeamsOverride) -> None:
    """Write the override atomically; an override with nothing set removes the file.

    Raises:
        StorageError: The file cannot be written or removed.
    """
    path = match_dir / OVERRIDE_FILE
    empty = all(side.name is None and side.kit is None for side in (override.home, override.away))
    if empty:
        try:
            path.unlink(missing_ok=True)
        except OSError as err:
            msg = f"cannot remove {path}"
            raise StorageError(msg) from err
        return
    write_atomic(path, override.model_dump_json(indent=2))


def _clean_name(name: str | None) -> str | None:
    cleaned = " ".join((name or "").split())
    return cleaned or None


def _lineup_names(match_dir: Path) -> dict[str, str]:
    try:
        data = (match_dir / LINEUP_FILE).read_bytes()
    except OSError:
        return {}
    return {str(side): name for side, name in parse_team_names(data).items()}


def load_effective(
    match_dir: Path, lineup_names: Mapping[str, str] | None = None
) -> dict[Side, EffectiveTeam]:
    """Both sides with the user's override applied over `teams.json`.

    Args:
        match_dir: The match directory.
        lineup_names: Names from `lineups.csv` (used when `teams.json` is not there); `None`
            reads the file of the match directory if there is one.
    """
    if lineup_names is None:
        lineup_names = _lineup_names(match_dir)
    stage = read_teams(match_dir)
    override = read_override(match_dir)
    result: dict[Side, EffectiveTeam] = {}
    for side in SIDES:
        manual = getattr(override, side)
        if stage is not None:
            entry = stage.entry(side)
            name, name_source = entry.name, entry.name_source
            kit, kit_source = entry.kit, entry.kit_source
        else:
            from_lineup = _clean_name((lineup_names or {}).get(side))
            name = from_lineup or ru.TEAM_LABELS[side]
            name_source = "lineup" if from_lineup else "default"
            kit, kit_source = None, "undetermined"
        manual_name = _clean_name(manual.name)
        if manual_name is not None:
            name, name_source = manual_name, "manual"
        if manual.kit is not None:
            kit, kit_source = manual.kit, "manual"
        result[side] = EffectiveTeam(
            name=name, name_source=name_source, kit=kit, kit_source=kit_source
        )
    return result


def side_of_kit(kit: str | None, teams: Mapping[Side, EffectiveTeam]) -> Side | None:
    """The side whose kit colour is `kit`; `None` when the colour is unknown, matches no team or
    both (two teams in one colour cannot be told apart)."""
    if kit is None:
        return None
    fitting = [side for side in SIDES if teams[side].kit == kit]
    return fitting[0] if len(fitting) == 1 else None
