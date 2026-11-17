"""The "Reel by category" block of the viewer (FR-10): a kind, a half, a side and the margins, a
button that cuts and joins every matching event, and the last reel with a download button.

Drawn into the sidebar by `app.py`. Nothing is written to the database: the reel is a file under
`clips/reels/`. The work is done while the page draws, inside a progress bar, once per click; the
finished reel is kept in `st.session_state`, so a rerun does not cut it again (and the file cache of
`aicoach.reel` makes a repeated click free).
"""

from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import structlog

from aicoach.agent.context import EventFacts, load_context
from aicoach.clips import DEFAULT_AFTER_S, DEFAULT_BEFORE_S
from aicoach.config import load_settings
from aicoach.errors import AicoachError
from aicoach.i18n import ru
from aicoach.reel import (
    CARD,
    DEFAULT_MAX_EVENTS,
    KINDS,
    ReelFilter,
    ReelOptions,
    ReelResult,
    build_reel,
    select_events,
)
from aicoach.ui.launch import AppArgs
from aicoach.ui.logic import format_mmss

BUTTON_KEY = "btn_reel"
_KIND = "reel_kind"
_HALF = "reel_half"
_SIDE = "reel_side"
_BEFORE = "reel_before"
_AFTER = "reel_after"
_RESULT = "reel_result"  # (path, caption, notes) of the last reel; kept across reruns
_ERROR = "reel_error"  # why the last attempt failed; cleared by the next one

logger = structlog.get_logger()


@dataclass(frozen=True)
class _Choice:
    """What the user picked in the block, ready for the selection and the build."""

    flt: ReelFilter
    options: ReelOptions


def _kind_text(kind: str) -> str:
    if kind == CARD:
        return ru.REEL_KIND_CARD
    return next(label for event_type, label in ru.TYPE_LABELS.items() if event_type.value == kind)


def _half_text(half: int | None) -> str:
    return ru.HALF_ALL if half is None else ru.HALF_TEXT.format(half=half)


def _side_text(side: str | None) -> str:
    return ru.REEL_SIDE_ALL if side is None else ru.TEAM_LABELS[side]


def _draw_choice() -> _Choice:
    """The fields of the block; returns what they say now."""
    kind = st.selectbox(ru.REEL_KIND, KINDS, format_func=_kind_text, key=_KIND)
    half = st.selectbox(ru.REEL_HALF, [None, 1, 2], format_func=_half_text, key=_HALF)
    side = st.selectbox(ru.REEL_SIDE, [None, "home", "away"], format_func=_side_text, key=_SIDE)
    before_col, after_col = st.columns(2)
    before = before_col.number_input(
        ru.CLIP_BEFORE, min_value=0.0, value=DEFAULT_BEFORE_S, step=1.0, key=_BEFORE
    )
    after = after_col.number_input(
        ru.CLIP_AFTER, min_value=0.0, value=DEFAULT_AFTER_S, step=1.0, key=_AFTER
    )
    flt = ReelFilter(str(kind), half, side)
    options = ReelOptions(
        before_s=float(before),
        after_s=float(after),
        max_events=DEFAULT_MAX_EVENTS,
        font_file=load_settings().aicoach_font_file,
    )
    return _Choice(flt, options)


def _load_facts(args: AppArgs) -> tuple[EventFacts, ...] | None:
    """The current events with their minutes, or None (with a message) when they cannot be read."""
    try:
        return load_context(args.db, args.match_id, args.results_dir).events
    except AicoachError as err:
        st.error(ru.REEL_FAILED.format(error=err))
        return None


def _notes(result: ReelResult) -> list[str]:
    notes = []
    if not result.captioned:
        notes.append(ru.REEL_NO_CAPTIONS)
    if result.reencoded:
        notes.append(ru.REEL_REENCODED)
    if result.team_unknown:
        notes.append(ru.REEL_TEAM_UNKNOWN.format(count=result.team_unknown))
    return notes


def _build(args: AppArgs, video: Path, facts: tuple[EventFacts, ...], choice: _Choice) -> None:
    """Cut and join the reel; keep the result or the reason of the failure."""
    st.session_state.pop(_ERROR, None)
    bar = st.progress(0.0)

    def on_progress(done: int, total: int) -> None:
        bar.progress(done / total, text=ru.REEL_PROGRESS.format(done=done, total=total))

    try:
        result = build_reel(
            video, args.results_dir, facts, choice.flt, choice.options, progress=on_progress
        )
    except AicoachError as err:
        logger.warning("reel_failed", stage="ui", match_id=args.match_id, reason=str(err)[:200])
        st.session_state[_ERROR] = ru.REEL_FAILED.format(error=err)
        return
    finally:
        bar.empty()
    length = (
        ru.REEL_DURATION_UNKNOWN if result.duration_s is None else format_mmss(result.duration_s)
    )
    caption = ru.REEL_READY.format(count=result.n_events, duration=length)
    if result.cached:
        caption = f"{caption} {ru.REEL_FROM_CACHE}."
    st.session_state[_RESULT] = (str(result.path), caption, _notes(result))


def _show_result() -> None:
    error: str | None = st.session_state.get(_ERROR)
    if error is not None:
        st.error(error)
    kept: tuple[str, str, list[str]] | None = st.session_state.get(_RESULT)
    if kept is None:
        return
    path = Path(kept[0])
    try:
        data = path.read_bytes()
    except OSError:
        st.session_state.pop(_RESULT, None)
        st.error(ru.REEL_FILE_GONE.format(name=path.name))
        return
    st.caption(kept[1])
    for note in kept[2]:
        st.caption(note)
    st.video(data, format="video/mp4")
    st.download_button(
        ru.REEL_DOWNLOAD, data=data, file_name=path.name, mime="video/mp4", key="btn_reel_download"
    )


def _selection_line(facts: tuple[EventFacts, ...], choice: _Choice) -> bool:
    """Say how many events the reel would hold; return True when a reel can be built."""
    selection = select_events(facts, choice.flt)
    count = len(selection.events)
    if count == 0:
        st.caption(ru.REEL_EMPTY)
    elif count > choice.options.max_events:
        st.caption(ru.REEL_TOO_MANY.format(count=count, limit=choice.options.max_events))
    else:
        st.caption(ru.REEL_COUNT.format(count=count))
    if selection.team_unknown:
        st.caption(ru.REEL_TEAM_UNKNOWN.format(count=selection.team_unknown))
    return 0 < count <= choice.options.max_events


def render_reel(args: AppArgs) -> None:
    """Draw the block into the sidebar: fields, the count, the button, the last reel."""
    with st.sidebar:
        st.header(ru.REEL_HEADER)
        choice = _draw_choice()
        video = args.video
        facts = _load_facts(args) if video is not None else None
        if video is None:
            st.caption(ru.REEL_NO_VIDEO)
        can_build = facts is not None and _selection_line(facts, choice)
        clicked = st.button(
            ru.REEL_BUILD, key=BUTTON_KEY, disabled=not can_build, help=ru.REEL_BUILD_HELP
        )
        if clicked and video is not None and facts is not None:
            _build(args, video, facts, choice)
        _show_result()
