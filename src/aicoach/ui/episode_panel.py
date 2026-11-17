"""The "Describe the episode" button of the viewer (FR-11, D-073): the description of the chosen
event from its clip, kept in `st.session_state`, always marked "by the clip, not verified".

Drawn into the sidebar by `app.py`, next to the clip export. The request is made while the page
draws, inside a spinner, once per click (a rerun does not repeat it: the answer is kept, and the
cache of `aicoach.episode` makes a repeated click free).
"""

import streamlit as st
import structlog

from aicoach.agent.service import make_video_model
from aicoach.config import load_settings
from aicoach.episode import EpisodeDescription, describe_episode
from aicoach.errors import AicoachError
from aicoach.i18n import ru
from aicoach.storage import CurrentEvent
from aicoach.ui.launch import AppArgs
from aicoach.ui.logic import to_row

_RESULT = "episode_result"  # (header, EpisodeDescription) of the last description
_ERROR = "episode_error"  # why the last attempt failed; cleared by the next one
BUTTON_KEY = "btn_episode"

logger = structlog.get_logger()


def meta_text(result: EpisodeDescription) -> str:
    """The line under the text: the model, the tokens, the cost, or "from the cache"."""
    if result.cached:
        return ru.EPISODE_META_CACHED
    cost = ru.EPISODE_COST_UNKNOWN if result.cost_usd is None else f"{result.cost_usd:.4f} USD"
    return ru.EPISODE_META.format(
        model=result.model, tokens_in=result.tokens_in, tokens_out=result.tokens_out, cost=cost
    )


def _describe(args: AppArgs, event: CurrentEvent, before_s: float, after_s: float) -> None:
    """Ask for the description and keep the result or the reason of the failure."""
    st.session_state.pop(_ERROR, None)
    video = args.video
    if video is None:  # the button is disabled without a video; kept for a direct call
        return
    try:
        model = make_video_model(load_settings(), args.results_dir)
        with st.spinner(ru.EPISODE_SPINNER):
            result = describe_episode(video, event, model, before_s=before_s, after_s=after_s)
    except AicoachError as err:
        logger.warning("episode_failed", stage="ui", match_id=args.match_id, reason=str(err)[:200])
        st.session_state[_ERROR] = ru.EPISODE_FAILED.format(error=err)
        return
    row = to_row(event)
    header = ru.EPISODE_HEADER.format(type=row.type_label, time=row.time_text)
    st.session_state[_RESULT] = (header, result)


def _show_result() -> None:
    error: str | None = st.session_state.get(_ERROR)
    if error is not None:
        st.error(error)
    kept: tuple[str, EpisodeDescription] | None = st.session_state.get(_RESULT)
    if kept is None:
        return
    header, result = kept
    st.caption(header)
    st.info(result.text)
    st.caption(f"{ru.EPISODE_UNVERIFIED} {meta_text(result)}")


def render_describe(
    args: AppArgs, selected: CurrentEvent | None, before_s: float, after_s: float
) -> None:
    """Draw the button for the chosen event and the last description (in the sidebar).

    Args:
        args: The open match (the video is needed).
        selected: The chosen event; without one the button is disabled.
        before_s: Seconds before the event, the same fields as the clip export.
        after_s: Seconds after.
    """
    if args.video is None:
        st.caption(ru.EPISODE_NO_VIDEO)
    elif selected is None:
        st.caption(ru.EPISODE_PICK_FIRST)
    clicked = st.button(
        ru.EPISODE_DESCRIBE,
        key=BUTTON_KEY,
        disabled=args.video is None or selected is None,
        help=ru.EPISODE_DESCRIBE_HELP,
    )
    if clicked and selected is not None:
        _describe(args, selected, before_s, after_s)
    _show_result()
