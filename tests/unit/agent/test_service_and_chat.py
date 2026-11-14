"""Wiring of the agent for a match, the console summary, and the helpers of the chat panel."""

from pathlib import Path

import pytest

from aicoach.agent import service
from aicoach.agent.evaluation import EvalRow, Question, format_summary, summarize
from aicoach.agent.loop import AgentAnswer
from aicoach.agent.messages import Usage
from aicoach.agent.render import EventRef
from aicoach.agent.service import build_runtime
from aicoach.config import load_settings
from aicoach.errors import ConfigError, StorageError
from aicoach.ui.agent_chat import answer_markdown, meta_line, unique_refs
from tests.fixtures.agent_synth import FakeLLM, turn
from tests.fixtures.ui_synth import write_match_dir


@pytest.mark.unit
def test_build_runtime_without_video_offers_no_clip_tool(tmp_path: Path) -> None:
    db = write_match_dir(tmp_path, "m1")

    runtime = build_runtime(FakeLLM([]), db, "m1", tmp_path)

    assert "make_clip" not in [s.name for s in runtime.box_factory().specs()]


@pytest.mark.unit
def test_build_runtime_with_video_cuts_clips_through_the_clip_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = write_match_dir(tmp_path, "m1")
    calls: list[tuple[Path, str, str, float, float]] = []

    def fake_export(
        _db_path: Path, match_id: str, video: Path, event_id: str, before_s: float, after_s: float
    ) -> Path:
        calls.append((video, match_id, event_id, before_s, after_s))
        return tmp_path / "clip.mp4"

    monkeypatch.setattr(service, "export_event_clip", fake_export)
    runtime = build_runtime(FakeLLM([]), db, "m1", tmp_path, tmp_path / "src.mp4")
    box = runtime.box_factory()
    event_id = box.ctx.events[0].event.event_id

    result = box.call("make_clip", {"event_id": event_id})

    assert result["status"] == "clip_ready"
    assert calls == [(tmp_path / "src.mp4", "m1", event_id, 8.0, 12.0)]


@pytest.mark.unit
def test_build_runtime_sees_edits_made_after_it_was_built(tmp_path: Path) -> None:
    db = write_match_dir(tmp_path, "m1")
    runtime = build_runtime(FakeLLM([]), db, "m1", tmp_path)
    first = runtime.box_factory().ctx.events

    from aicoach.storage import add_edit, connect

    with connect(db) as conn:
        add_edit(conn, "m1", "delete", author="test", target=first[0].event.event_id)

    assert len(runtime.box_factory().ctx.events) == len(first) - 1


@pytest.mark.unit
def test_build_runtime_without_database_is_a_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError):
        build_runtime(FakeLLM([]), tmp_path / "none.db", "m1", tmp_path)


@pytest.mark.unit
def test_gemini_client_without_a_key_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        service.gemini_client(load_settings())


def answer(text: str = "Гол [E:p:2] и снова [E:p:2], удар [E:p:7].") -> AgentAnswer:
    from aicoach.agent.render import render_answer
    from tests.fixtures.agent_synth import make_context

    segments = render_answer(text, make_context(), "abcDEF_-123")
    return AgentAnswer(
        question="?",
        segments=segments,
        outcome="retry",
        n_links=3,
        tool_calls=[{"name": "match_info", "args": {}}],
        usage=Usage(120, 30),
        duration_ms=2500,
        model="m",
    )


@pytest.mark.unit
def test_answer_markdown_shows_labels_in_bold_and_no_raw_ids() -> None:
    text = answer_markdown(answer())

    assert "**[Гол, 1-й тайм, 23', видео 25:00]**" in text
    assert "[E:" not in text


@pytest.mark.unit
def test_unique_refs_lists_each_event_once_in_order() -> None:
    refs = unique_refs(answer())

    assert [r.event_id for r in refs] == ["p:2", "p:7"]
    assert all(isinstance(r, EventRef) for r in refs)


@pytest.mark.unit
def test_meta_line_has_outcome_calls_tokens_and_seconds() -> None:
    line = meta_line(answer())

    assert "со второй попытки" in line
    assert "120 вход, 30 выход" in line
    assert "2.5 с" in line


@pytest.mark.unit
def test_format_summary_counts_outcomes_and_marks_an_unknown_price() -> None:
    rows = [EvalRow(Question("1", "a", "q", "", ""), None, "boom")]

    text = format_summary(summarize(rows))

    assert "questions: 1" in text
    assert "service errors: 1" in text
    assert "n/a" in text


@pytest.mark.unit
def test_format_summary_shows_the_cost_when_the_model_has_a_price(tmp_path: Path) -> None:
    from aicoach.agent.evaluation import run_questions
    from aicoach.agent.loop import Agent
    from aicoach.agent.tools import ToolBox
    from tests.fixtures.agent_synth import make_context

    question = Question("1", "a", "q?", "", "")
    summary = run_questions(
        Agent(FakeLLM([turn("Не знаю.", tokens=(1000, 500))]), "s"),
        [question],
        lambda: ToolBox(make_context()),
        tmp_path / "a.csv",
    )

    assert f"{(1000 * 0.30 + 500 * 2.50) / 1_000_000:.4f}" in format_summary(summary)
