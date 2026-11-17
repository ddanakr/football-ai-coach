"""The "Agent" panel of the viewer: chat history in `st.session_state`, a question box, link
buttons that put the player on the event. Drawn by `app.py`; the logic is in `agent_chat.py`.
"""

from collections.abc import Callable
from pathlib import Path

import streamlit as st

from aicoach.agent.loop import AgentAnswer
from aicoach.agent.service import AgentRuntime, build_runtime, make_llm, optional_video_model
from aicoach.config import load_settings
from aicoach.errors import AicoachError
from aicoach.i18n import ru
from aicoach.ui.agent_chat import answer_markdown, meta_line, unique_refs
from aicoach.ui.launch import AppArgs

HISTORY_KEY = "agent_history"  # list of (question, answer or error text)
_RUNTIME_KEY = "agent_runtime"  # (identity of the match, AgentRuntime)
_INPUT_KEY = "agent_input"

# One place for the size and the look of the panel (D-062).
# The height of the scrolling history. The video next to it is as high as its column is wide
# divided by 16/9 (about 400 px in a 1300 px wide window with the 3:2 columns of `app.py`), plus a
# title and a link line: change this one number to match another window.
AGENT_PANEL_HEIGHT_PX = 420
# The user's messages on the right, the agent's on the left. It relies on the test ids of
# Streamlit's chat message and its user avatar and on `:has()`; a Streamlit update can change
# them, then the messages just stay on one side (nothing breaks).
_CSS = (
    "<style>"
    '[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) '
    "{flex-direction: row-reverse; text-align: right;}"
    "</style>"
)

History = list[tuple[str, AgentAnswer | str]]


def _runtime(args: AppArgs) -> AgentRuntime:
    """The agent of the open match; built once per match and kept in the session.

    Raises:
        ConfigError: No key is set for the chosen provider.
        StorageError: The database does not exist.
    """
    identity = (args.db, args.match_id, args.video)
    cached = st.session_state.get(_RUNTIME_KEY)
    if cached is not None and cached[0] == identity:
        runtime: AgentRuntime = cached[1]
        return runtime
    st.session_state[HISTORY_KEY] = []  # another match: the old questions do not belong to it
    settings = load_settings()
    runtime = build_runtime(
        make_llm(settings, args.results_dir),
        args.db,
        args.match_id,
        args.results_dir,
        args.video,
        optional_video_model(settings, args.results_dir),
    )
    st.session_state[_RUNTIME_KEY] = (identity, runtime)
    return runtime


def _show_answer(index: int, answer: AgentAnswer, select: Callable[[str], None]) -> None:
    st.markdown(answer_markdown(answer))
    for ref in unique_refs(answer):
        columns = st.columns([2, 3])
        columns[0].button(
            ru.AGENT_OPEN.format(label=ref.label),
            key=f"agent_open_{index}_{ref.event_id}",
            on_click=select,
            args=(ref.event_id,),
        )
        if ref.url:
            columns[1].markdown(f"[{ru.LINK_TEXT}]({ref.url})")
    for clip in answer.clips:
        _show_clip(clip.path)
    st.caption(meta_line(answer))


def _show_clip(path: Path) -> None:
    try:
        data = path.read_bytes()
    except OSError:
        st.caption(ru.CLIP_FILE_GONE.format(name=path.name))
        return
    st.caption(ru.AGENT_CLIP.format(name=path.name))
    st.video(data)


def render_agent_panel(args: AppArgs, select: Callable[[str], None]) -> None:
    """Draw the chat: a scrolling history of fixed height, the question box under it (both in
    the column the caller gives), and answer the new question.

    Args:
        args: The open match.
        select: Makes an event the chosen one (the player next to the panel shows it).
    """
    st.markdown(_CSS, unsafe_allow_html=True)
    st.subheader(ru.AGENT_TAB)
    try:
        runtime = _runtime(args)
    except AicoachError as err:
        st.caption(ru.AGENT_NO_KEY.format(error=err))
        runtime = None
    history: History = st.session_state.setdefault(HISTORY_KEY, [])
    # A fixed height makes the container scroll, and it scrolls to the newest message by itself
    # (`autoscroll` is on for a fixed-height container that holds chat messages).
    with st.container(height=AGENT_PANEL_HEIGHT_PX, border=True):
        if not history:
            st.caption(ru.AGENT_HINT)
        for index, (question, reply) in enumerate(history):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                if isinstance(reply, str):
                    st.error(reply)
                else:
                    _show_answer(index, reply, select)
    asked = st.chat_input(ru.AGENT_INPUT, key=_INPUT_KEY, disabled=runtime is None)
    if st.button(ru.AGENT_CLEAR, disabled=not history, key="agent_clear"):
        history.clear()
        st.rerun()
    if asked and runtime is not None:
        _ask(runtime, asked, history, args.youtube_id)
        st.rerun()  # draws the new entry through the same code as the old ones


def _ask(runtime: AgentRuntime, question: str, history: History, youtube_id: str) -> None:
    with st.spinner(ru.AGENT_THINKING):
        try:
            reply: AgentAnswer | str = runtime.agent.ask(
                question, runtime.box_factory(), youtube_id
            )
        except AicoachError as err:
            reply = ru.AGENT_FAILED.format(error=err)
    history.append((question, reply))
