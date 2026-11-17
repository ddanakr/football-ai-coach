"""Corner filters and the team block of the viewer's sidebar (D-078).

The filters narrow the corners of the events table by the tags a video model read from their
clips (always shown as "by the clip, not verified"); they hide no other event. The team block
shows the names and kit colours of both sides and lets the user set them by hand: the values go to
`teams_override.json` and win over what the stage found. Files are written only in the button
callback, never while the page draws.
"""

from collections.abc import Mapping
from functools import partial
from pathlib import Path

import streamlit as st
import structlog

from aicoach.corner_tags.schemas import FIELDS
from aicoach.corner_tags.view import (
    TEAM_CHOICES,
    UNKNOWN_TEAM,
    CornerFilter,
    filter_choices,
    value_label,
)
from aicoach.errors import StorageError
from aicoach.i18n import ru
from aicoach.teams.store import (
    KIT_COLORS,
    SIDES,
    EffectiveTeam,
    Side,
    SideOverride,
    TeamsOverride,
    read_override,
    save_override,
)

logger = structlog.get_logger()
_AUTO = "auto"
_SAVED = "teams_saved"  # (level, text) of the last save, shown once


def _team_option(code: str | None, teams: Mapping[Side, EffectiveTeam]) -> str:
    if code is None:
        return ru.CORNER_FILTER_ANY
    if code == UNKNOWN_TEAM:
        return ru.CORNER_FILTER_TEAM_UNKNOWN
    return teams["home" if code == "home" else "away"].name


def _value_text(field_name: str, code: str | None) -> str:
    return ru.CORNER_FILTER_ANY if code is None else value_label(field_name, code)


def render_corner_filters(*, has_tags: bool, teams: Mapping[Side, EffectiveTeam]) -> CornerFilter:
    """Draw the corner filters in the sidebar and return what the user chose.

    Args:
        has_tags: Whether any corner has tags; without them the block only says how to make them.
        teams: The two sides (the names label the team selector).
    """
    with st.sidebar, st.expander(ru.CORNER_FILTERS_HEADER):
        if not has_tags:
            st.caption(ru.CORNER_FILTERS_NO_TAGS)
            return CornerFilter()
        team_options: list[str | None] = [None, *TEAM_CHOICES]
        team = st.selectbox(
            ru.CORNER_FILTER_TEAM,
            team_options,
            format_func=lambda code: _team_option(code, teams),
            key="corner_filter_team",
        )
        chosen: dict[str, str] = {}
        for name in FIELDS:
            field_options: list[str | None] = [None, *filter_choices(name)]
            code = st.selectbox(
                ru.CORNER_FIELD_LABELS[name],
                field_options,
                format_func=partial(_value_text, name),
                key=f"corner_filter_{name}",
            )
            if code is not None:
                chosen[name] = code
        st.caption(ru.CORNER_FILTER_NOTE)
    return CornerFilter(team=team, fields=chosen)


def _on_save(match_dir: Path, keys: Mapping[Side, tuple[str, str]]) -> None:
    """Write the override from the widgets of the block. Runs in a callback only."""
    sides: dict[Side, SideOverride] = {}
    for side, (name_key, kit_key) in keys.items():
        name = str(st.session_state.get(name_key, "")).strip()
        kit = st.session_state.get(kit_key, _AUTO)
        sides[side] = SideOverride(
            name=name or None,
            kit=kit if kit in KIT_COLORS else None,
        )
    try:
        save_override(match_dir, TeamsOverride(home=sides["home"], away=sides["away"]))
    except (StorageError, ValueError) as err:
        logger.warning("teams_override_failed", stage="ui", reason=str(err)[:200])
        st.session_state[_SAVED] = ("error", ru.TEAMS_SAVE_FAILED.format(error=err))
        return
    st.session_state[_SAVED] = ("ok", ru.TEAMS_SAVED)


def _on_reset(match_dir: Path, keys: Mapping[Side, tuple[str, str]]) -> None:
    """Remove the override and clear the widgets. Runs in a callback only."""
    try:
        save_override(match_dir, TeamsOverride())
    except StorageError as err:
        st.session_state[_SAVED] = ("error", ru.TEAMS_SAVE_FAILED.format(error=err))
        return
    for name_key, kit_key in keys.values():
        st.session_state[name_key] = ""
        st.session_state[kit_key] = _AUTO
    st.session_state[_SAVED] = ("ok", ru.TEAMS_SAVED)


def _kit_text(team: EffectiveTeam) -> str:
    return ru.KIT_LABELS[team.kit] if team.kit else ru.KIT_UNDETERMINED


def render_teams(match_dir: Path, teams: Mapping[Side, EffectiveTeam]) -> None:
    """Draw the team block: what is known, and fields to set the name and the kit by hand."""
    override = read_override(match_dir)
    keys: dict[Side, tuple[str, str]] = {
        side: (f"teams_name_{side}", f"teams_kit_{side}") for side in SIDES
    }
    with st.sidebar, st.expander(ru.TEAMS_HEADER):
        st.caption(ru.TEAMS_HELP)
        for side in SIDES:
            team = teams[side]
            st.text(
                ru.TEAMS_LINE.format(
                    side=ru.TEAM_LABELS[side],
                    name=team.name,
                    name_source=ru.TEAMS_SOURCE_LABELS[team.name_source],
                    kit=_kit_text(team),
                    kit_source=ru.TEAMS_SOURCE_LABELS[team.kit_source],
                )
            )
            name_key, kit_key = keys[side]
            manual: SideOverride = getattr(override, side)
            st.text_input(
                ru.TEAMS_NAME_LABEL.format(side=ru.TEAM_LABELS[side]),
                value=manual.name or "",
                key=name_key,
            )
            st.selectbox(
                ru.TEAMS_KIT_LABEL.format(side=ru.TEAM_LABELS[side]),
                [_AUTO, *KIT_COLORS],
                index=([_AUTO, *KIT_COLORS].index(manual.kit) if manual.kit else 0),
                format_func=lambda c: ru.TEAMS_KIT_AUTO if c == _AUTO else ru.KIT_LABELS[c],
                key=kit_key,
            )
        st.button(ru.TEAMS_SAVE, key="btn_teams_save", on_click=_on_save, args=(match_dir, keys))
        st.button(ru.TEAMS_RESET, key="btn_teams_reset", on_click=_on_reset, args=(match_dir, keys))
        flash: tuple[str, str] | None = st.session_state.pop(_SAVED, None)
        if flash is not None:
            (st.success if flash[0] == "ok" else st.error)(flash[1])
