"""The "Agent" panel of the viewer: chat history in `st.session_state`, a question box, link
buttons that put the player on the event. Drawn by `app.py`; the logic is in `agent_chat.py`.
"""

from collections.abc import Callable
from pathlib import Path

import streamlit as st

from aicoach.agent.loop import AgentAnswer
from aicoach.agent.service import AgentRuntime, build_runtime, gemini_client
from aicoach.config import load_settings
from aicoach.errors import AicoachError
from aicoach.i18n import ru
from aicoach.ui.agent_chat import answer_markdown, meta_line, unique_refs
from aicoach.ui.launch import AppArgs

HISTORY_KEY = "agent_history"  # list of (question, answer or error text)
_RUNTIME_KEY = "agent_runtime"  # (identity of the match, AgentRuntime)
_INPUT_KEY = "agent_input"

History = list[tuple[str, AgentAnswer | str]]


def _runtime(args: AppArgs) -> AgentRuntime:
    """The agent of the open match; built once per match and kept in the session.

    Raises:
        ConfigError: No Gemini key is set.
        StorageError: The database does not exist.
    """
    identity = (args.db, args.match_id, args.video)
    cached = st.session_state.get(_RUNTIME_KEY)
    if cached is not None and cached[0] == identity:
        runtime: AgentRuntime = cached[1]
        return runtime
    st.session_state[HISTORY_KEY] = []  # another match: the old questions do not belong to it
    runtime = build_runtime(
        gemini_client(load_settings()),
        args.db,
        args.match_id,
        args.results_dir,
        args.video,
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
    """Draw the chat: history, the question box, and answer the new question.

    Args:
        args: The open match.
        select: Makes an event the chosen one (the player above the panel shows it).
    """
    st.subheader(ru.AGENT_TAB)
    st.caption(ru.AGENT_HINT)
    try:
        runtime = _runtime(args)
    except AicoachError as err:
        st.caption(ru.AGENT_NO_KEY.format(error=err))
        runtime = None
    history: History = st.session_state.setdefault(HISTORY_KEY, [])
    for index, (question, reply) in enumerate(history):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            if isinstance(reply, str):
                st.error(reply)
            else:
                _show_answer(index, reply, select)
    if st.button(ru.AGENT_CLEAR, disabled=not history, key="agent_clear"):
        history.clear()
        st.rerun()
    asked = st.chat_input(ru.AGENT_INPUT, key=_INPUT_KEY, disabled=runtime is None)
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
