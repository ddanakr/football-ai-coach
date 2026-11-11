"""Streamlit viewer of the event catalog (D12a, view only). Thin: the logic is in `logic.py`.

Started by `python -m aicoach.cli ui`; the arguments after `--` are read by `parse_app_args`.
"""

import sys

import streamlit as st
import streamlit.components.v1 as components

from aicoach.errors import ConfigError, StorageError
from aicoach.i18n import ru
from aicoach.storage import CurrentEvent
from aicoach.ui.data import load_events
from aicoach.ui.launch import AppArgs, parse_app_args
from aicoach.ui.logic import (
    EventFilter,
    EventRow,
    HalfFilter,
    available_types,
    filter_events,
    iframe_html,
    player_start,
    to_rows,
    watch_url,
    youtube_link,
)

LIST_HEIGHT_PX = 620
PLAYER_HEIGHT_PX = 380
_SELECTED = "selected_row"
_CLICKS = "play_clicks"
_HALF_OPTIONS: list[HalfFilter] = [None, 1, 2]


def _select(row: EventRow) -> None:
    """Button callback: remember the chosen event; runs before the script reruns."""
    st.session_state[_SELECTED] = row
    st.session_state[_CLICKS] = st.session_state.get(_CLICKS, 0) + 1


def _half_text(half: HalfFilter) -> str:
    return ru.HALF_ALL if half is None else ru.HALF_TEXT.format(half=half)


def _sidebar(events: list[CurrentEvent]) -> tuple[EventFilter, str]:
    """Draw the sidebar; return the chosen filter and the player variant."""
    with st.sidebar:
        st.header(ru.SIDEBAR_PLAYER)
        variant = st.radio(ru.PLAYER_VARIANT, [ru.PLAYER_IFRAME, ru.PLAYER_ST_VIDEO])
        st.header(ru.SIDEBAR_FILTERS)
        present = available_types(events)
        types = st.multiselect(
            ru.FILTER_TYPES, present, default=present, format_func=lambda t: ru.TYPE_LABELS[t]
        )
        half = st.radio(ru.FILTER_HALF, _HALF_OPTIONS, format_func=_half_text)
        min_score = st.slider(
            ru.FILTER_MIN_SCORE, 0.0, 1.0, 0.0, 0.05, help=ru.FILTER_MIN_SCORE_HELP
        )
    return EventFilter(types=frozenset(types), half=half, min_score=min_score), str(variant)


def _render_list(rows: list[EventRow]) -> None:
    widths = [1, 2, 3, 1, 2, 3, 2]
    with st.container(height=LIST_HEIGHT_PX):
        header = st.columns(widths)
        names = ("", ru.COLUMN_TIME, ru.COLUMN_TYPE, ru.COLUMN_HALF, ru.COLUMN_SCORE)
        for col, name in zip(header, (*names, ru.COLUMN_SOURCE, ru.COLUMN_TEAM), strict=True):
            col.markdown(f"**{name}**" if name else "")
        for row in rows:
            cols = st.columns(widths)
            cols[0].button(
                ru.PLAY_BUTTON,
                key=f"play_{row.event_id}",
                help=ru.PLAY_HELP,
                on_click=_select,
                args=(row,),
            )
            texts = (row.time_text, row.type_text, row.half_text, row.score_text)
            for col, text in zip(cols[1:], (*texts, row.source_text, row.team_text), strict=True):
                col.write(text)


def _render_player(args: AppArgs, variant: str) -> None:
    selected: EventRow | None = st.session_state.get(_SELECTED)
    t_v = 0.0 if selected is None else selected.t_v
    # Nothing chosen yet: show the start of the video and do not play by itself.
    start = 0 if selected is None else player_start(t_v)
    autoplay = selected is not None
    if selected is not None:
        st.markdown(f"**{ru.PLAYING.format(type=selected.type_label, time=selected.time_text)}**")
    if variant == ru.PLAYER_IFRAME:
        html = iframe_html(
            args.youtube_id,
            start,
            autoplay=autoplay,
            nonce=st.session_state.get(_CLICKS, 0),
            height=PLAYER_HEIGHT_PX,
        )
        components.html(html, height=PLAYER_HEIGHT_PX + 10)
    else:
        st.video(watch_url(args.youtube_id), start_time=start, autoplay=autoplay)
    link = (
        youtube_link(args.youtube_id, t_v) if selected is not None else watch_url(args.youtube_id)
    )
    st.markdown(f"{ru.LINK_TEXT}: [{link}]({link})")
    st.caption(ru.TIME_NOTE)


def main() -> None:
    """Draw the page; every user action reruns it."""
    st.set_page_config(page_title=ru.TITLE, layout="wide")
    st.title(ru.TITLE)
    try:
        args = parse_app_args(sys.argv[1:])
    except ConfigError as err:
        st.error(ru.BAD_ARGS.format(error=err))
        st.stop()
    if not args.db.is_file():
        st.error(ru.NO_DB.format(path=args.db))
        st.stop()
    try:
        events = load_events(args.db, args.match_id)
    except StorageError as err:
        st.error(ru.DB_ERROR.format(path=args.db, error=err))
        st.stop()
    if not events:
        st.warning(ru.NO_EVENTS.format(match_id=args.match_id))
        st.stop()

    flt, variant = _sidebar(events)
    shown = filter_events(events, flt)
    list_col, player_col = st.columns([3, 2])
    with list_col:
        st.caption(ru.SHOWN.format(shown=len(shown), total=len(events)))
        if shown:
            _render_list(to_rows(shown))
        else:
            st.info(ru.NOTHING_MATCHES)
    with player_col:
        _render_player(args, variant)


main()
