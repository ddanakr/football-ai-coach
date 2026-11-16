"""The "Agent" panel through the Streamlit script (`AppTest`) with a scripted model."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.fixtures.agent_synth import FakeLLM, turn
from tests.fixtures.ui_synth import write_match_dir
from tests.unit.ui.test_app_edits import MATCH, VID

from aicoach.errors import ExternalServiceError
from aicoach.i18n import ru
from aicoach.storage import connect, get_current_events
from aicoach.ui.agent_panel import HISTORY_KEY
from aicoach.ui.launch import AppArgs, app_script, build_streamlit_command

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1"])

GOALS = ("query_events", {"type": "goal"})


def goal_id(db: Path) -> str:
    """The id of the goal of the synthetic match."""
    with connect(db) as conn:
        return next(e.event_id for e in get_current_events(conn, MATCH) if e.type == "goal")


def start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, llm: FakeLLM | None) -> AppTest:
    """Run the page over a synthetic match; `llm` None means no Gemini key is set."""
    db = tmp_path / "aicoach.db"
    args = AppArgs(db=db, match_id=MATCH, youtube_id=VID, match_dir=tmp_path)
    monkeypatch.setattr("sys.argv", [str(app_script()), *build_streamlit_command(args)[6:]])
    if llm is not None:
        monkeypatch.setattr("aicoach.ui.agent_panel.gemini_client", lambda _settings: llm)
        monkeypatch.setenv("GEMINI_API_KEY", "key-for-the-fake-client")
    return AppTest.from_file(str(app_script()), default_timeout=20).run()


@pytest.mark.unit
def test_without_a_key_the_panel_explains_and_the_question_box_is_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_match_dir(tmp_path, MATCH)
    app = start(monkeypatch, tmp_path, None)

    assert not app.exception
    assert any("GEMINI_API_KEY" in w.value for w in app.caption)
    assert app.chat_input[0].disabled


@pytest.mark.unit
def test_question_is_answered_with_a_link_button_that_puts_the_player_on_the_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    event_id = goal_id(write_match_dir(tmp_path, MATCH))
    llm = FakeLLM([turn("", GOALS), turn(f"Гол: [E:{event_id}].")])
    app = start(monkeypatch, tmp_path, llm)

    app.chat_input[0].set_value("Кто забил?").run()

    assert not app.exception
    ((question, answer),) = app.session_state[HISTORY_KEY]
    assert question == "Кто забил?"
    assert answer.outcome == "first_try"
    assert any(m.value.startswith("Гол: **[⚽ ") for m in app.markdown)
    buttons = [b for b in app.button if b.key and b.key.startswith("agent_open_")]
    assert len(buttons) == 1
    assert not [t for t in app.markdown if t.value.startswith(f"**{ru.PLAYING[:6]}")]  # not yet

    buttons[0].click().run()

    assert any(m.value.startswith("**Сейчас: Гол") for m in app.markdown)


@pytest.mark.unit
def test_service_failure_is_shown_in_the_chat_and_the_page_survives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_match_dir(tmp_path, MATCH)
    llm = FakeLLM([ExternalServiceError("Gemini request failed (429)")])
    app = start(monkeypatch, tmp_path, llm)

    app.chat_input[0].set_value("Привет?").run()

    assert not app.exception
    assert any("429" in e.value for e in app.error)


@pytest.mark.unit
def test_clear_button_empties_the_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_match_dir(tmp_path, MATCH)
    app = start(monkeypatch, tmp_path, FakeLLM([turn("Не знаю.")]))
    app.chat_input[0].set_value("Что-нибудь?").run()
    assert len(app.session_state[HISTORY_KEY]) == 1

    app.button(key="agent_clear").click().run()

    assert app.session_state[HISTORY_KEY] == []


@pytest.mark.unit
def test_the_panel_has_the_alignment_style_once_and_one_question_box(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_match_dir(tmp_path, MATCH)
    app = start(monkeypatch, tmp_path, FakeLLM([]))

    styles = [m for m in app.markdown if "stChatMessageAvatarUser" in m.value]

    assert len(styles) == 1
    assert len(app.chat_input) == 1
    assert not app.exception
