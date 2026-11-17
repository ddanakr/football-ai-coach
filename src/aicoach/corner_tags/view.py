"""Corner tags as the viewer and the agent read them: matched to the current events, the attacking
team worked out from the kit colour, filters and the Russian text (D-078).

Pure functions plus one reader. Nothing is written. The tags are a model's reading of a clip, so
every text made here says so ("по клипу, не проверено") where it is shown to a person.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from aicoach.corner_tags.schemas import (
    ATTACKING_KIT,
    CHOICES,
    FIELDS,
    IS_CORNER,
    UNCLEAR,
    ZONE,
    CornerTag,
    CornerTagsFile,
    read_tags_file,
)
from aicoach.domain.events import EventType
from aicoach.i18n import ru
from aicoach.storage import CurrentEvent
from aicoach.teams.store import EffectiveTeam, Side, side_of_kit

UNKNOWN_TEAM = "unknown"
TEAM_CHOICES = ("home", "away", UNKNOWN_TEAM)


@dataclass(frozen=True)
class CornerView:
    """The tags of one corner and the side the kit colour points to (`None`: not known)."""

    tag: CornerTag
    side: Side | None

    def value(self, field_name: str) -> str:
        """The code of one of the six fields."""
        return str(getattr(self.tag.fields, field_name))


@dataclass(frozen=True)
class CornerFilter:
    """What the user or the agent asks for: the attacking team (`home`, `away` or `unknown`) and
    chosen codes of the fields. Empty means no condition."""

    team: str | None = None
    fields: Mapping[str, str] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        """Whether any condition is set."""
        return self.team is not None or bool(self.fields)


def build_views(
    tags: CornerTagsFile | None,
    events: Iterable[CurrentEvent],
    teams: Mapping[Side, EffectiveTeam],
) -> dict[str, CornerView]:
    """The views of the corners among `events`, by event id.

    A tag belongs to an event only when the time is the same: a corner the user moved is not the
    clip the model looked at, so it has no tags until the stage runs again.
    """
    if tags is None:
        return {}
    by_id = {t.event_id: t for t in tags.tags}
    views: dict[str, CornerView] = {}
    for event in events:
        tag = by_id.get(event.event_id)
        if tag is None or round(tag.t_v * 1000) != round(event.t_v * 1000):
            continue
        views[event.event_id] = CornerView(tag, _attacking_side(tag, teams))
    return views


def _attacking_side(tag: CornerTag, teams: Mapping[Side, EffectiveTeam]) -> Side | None:
    kit = tag.fields.attacking_kit
    return None if kit == UNCLEAR else side_of_kit(kit, teams)


def load_views(
    match_dir: Path, events: Iterable[CurrentEvent], teams: Mapping[Side, EffectiveTeam]
) -> dict[str, CornerView]:
    """`build_views` with the tags read from `corner_tags.json` of the match directory."""
    return build_views(read_tags_file(match_dir), events, teams)


def matches(view: CornerView | None, flt: CornerFilter) -> bool:
    """Whether a corner passes the filter; a corner without tags passes only an empty filter."""
    if not flt.active:
        return True
    if view is None:
        return False
    if flt.team is not None:
        side = UNKNOWN_TEAM if view.side is None else view.side
        if side != flt.team:
            return False
    return all(view.value(name) == code for name, code in flt.fields.items())


def filter_choices(field_name: str) -> tuple[str, ...]:
    """The allowed codes of a field (for a selector)."""
    return CHOICES[field_name]


def value_label(field_name: str, code: str) -> str:
    """Russian name of a code of a field."""
    return ru.CORNER_VALUE_LABELS[field_name].get(code, code)


def team_label(side: Side | None, teams: Mapping[Side, EffectiveTeam]) -> str:
    """The attacking team for a person: its name, or "команда не определена"."""
    return ru.CORNER_TEAM_UNKNOWN if side is None else teams[side].name


def describe(view: CornerView, teams: Mapping[Side, EffectiveTeam]) -> str:
    """One line for the events table: team, then the fields that are known, then the mark that the
    tags are the model's reading of a clip. A clip with no corner in it says so."""
    fields = view.tag.fields
    if fields.is_corner == "no":
        return f"{ru.CORNER_NOT_A_CORNER} ({ru.CORNER_UNVERIFIED})"
    parts = [f"{ru.CORNER_ATTACKING}: {team_label(view.side, teams)}"]
    for name in FIELDS:
        if name in (IS_CORNER, ATTACKING_KIT):
            continue
        code = view.value(name)
        if code == UNCLEAR:
            continue
        text = f"{ru.CORNER_FIELD_SHORT[name]}: {value_label(name, code)}"
        parts.append(f"{text} ({ru.CORNER_LOW_RELIABILITY})" if name == ZONE else text)
    return f"{' · '.join(parts)} ({ru.CORNER_UNVERIFIED})"


def apply_filter(
    events: Iterable[CurrentEvent], views: Mapping[str, CornerView], flt: CornerFilter
) -> list[CurrentEvent]:
    """The events that pass the corner filter: corners by their tags, every other event as it is
    (the filter is about corners and hides nothing else)."""
    return [e for e in events if e.type != EventType.CORNER or matches(views.get(e.event_id), flt)]


def texts(views: Mapping[str, CornerView], teams: Mapping[Side, EffectiveTeam]) -> dict[str, str]:
    """The table text of every corner that has tags, by event id."""
    return {event_id: describe(view, teams) for event_id, view in views.items()}
