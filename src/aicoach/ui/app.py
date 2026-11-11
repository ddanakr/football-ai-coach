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
    selected_row,
    table_records,
    to_rows,
    watch_url,
    youtube_link,
)

# The page gives the player frame this height until the frame fits itself to the 16:9 width.
PLAYER_HEIGHT_PX = 540
PLAYER_COLUMNS = [1, 4, 1]  # the player sits in the middle one, it is not stretched over the page
TABLE_HEIGHT_PX = 380
TABLE_KEY = "events_table"
_SELECTED = "selected_row"
_CLICKS = "play_clicks"
_HALF_OPTIONS: list[HalfFilter] = [None, 1, 2]


def _select(row: EventRow) -> None:
    """Button callback: remember the chosen event; runs before the script reruns."""
    st.session_state[_SELECTED] = row
    st.session_state[_CLICKS] = st.session_state.get(_CLICKS, 0) + 1


def _current_event_id() -> str | None:
    chosen: EventRow | None = st.session_state.get(_SELECTED)
    return None if chosen is None else chosen.event_id


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


def _render_table(rows: list[EventRow]) -> EventRow | None:
    """Draw the events table over the full width; return the row the user picked, if any."""
    state = st.dataframe(
        table_records(rows),
        key=TABLE_KEY,
        hide_index=True,
        height=TABLE_HEIGHT_PX,
        on_select="rerun",
        selection_mode="single-row",
        column_config={ru.COLUMN_SCORE: st.column_config.NumberColumn(format="%.2f")},
    )
    return selected_row(rows, state.selection.rows)


def _render_player(args: AppArgs, variant: str) -> None:
    selected: EventRow | None = st.session_state.get(_SELECTED)
    t_v = 0.0 if selected is None else selected.t_v
    # Nothing chosen yet: show the start of the video and do not play by itself.
    start = 0 if selected is None else player_start(t_v)
    autoplay = selected is not None
    if selected is not None:
        st.markdown(f"**{ru.PLAYING.format(type=selected.type_label, time=selected.time_text)}**")
    else:
        st.caption(ru.TABLE_HINT)
    if variant == ru.PLAYER_IFRAME:
        html = iframe_html(
            args.youtube_id,
            start,
            autoplay=autoplay,
            nonce=st.session_state.get(_CLICKS, 0),
        )
        components.html(html, height=PLAYER_HEIGHT_PX)
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
    player_slot = st.container()  # filled below the table code: the pick is known only then
    st.caption(ru.SHOWN.format(shown=len(shown), total=len(events)))
    if shown:
        rows = to_rows(shown)
        picked = _render_table(rows)
        # Only a new pick restarts the player; other reruns (filters, variant) leave it alone.
        if picked is not None and picked.event_id != _current_event_id():
            _select(picked)
    else:
        st.info(ru.NOTHING_MATCHES)
    with player_slot:
        _, middle, _ = st.columns(PLAYER_COLUMNS)
        with middle:
            _render_player(args, variant)


main()
