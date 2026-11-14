"""Rendering of links, and the `agent-eval` question file, answers file and summary."""

import csv
import io
from collections.abc import Sequence
from pathlib import Path

import pytest

from aicoach.agent.evaluation import (
    ANSWER_COLUMNS,
    MAX_CONSECUTIVE_ERRORS,
    EvalSummary,
    read_questions,
    run_questions,
    summarize,
)
from aicoach.agent.loop import Agent
from aicoach.agent.messages import ModelTurn
from aicoach.agent.render import EventRef, count_links, plain_text, render_answer
from aicoach.agent.tools import ToolBox
from aicoach.errors import EvalError, ExternalServiceError, StorageError
from tests.fixtures.agent_synth import FakeLLM, make_context, turn

GOALS = ("query_events", {"type": "goal"})
VID = "abcDEF_-123"


@pytest.mark.unit
def test_render_splits_text_and_links_in_order() -> None:
    segments = render_answer("Раз [E:p:2], два [E:p:6]!", make_context(), VID)

    assert [s if isinstance(s, str) else s.event_id for s in segments] == [
        "Раз ",
        "p:2",
        ", два ",
        "p:6",
        "!",
    ]
    assert count_links(segments) == 2


@pytest.mark.unit
def test_render_label_leaves_out_the_unknown_minute_and_half() -> None:
    (ref,) = [s for s in render_answer("[E:p:7]", make_context(), None) if isinstance(s, EventRef)]

    assert ref.label == "Удар, 2-й тайм, видео 116:40"
    assert ref.url is None


@pytest.mark.unit
def test_render_label_writes_added_time() -> None:
    (ref,) = [s for s in render_answer("[E:p:4]", make_context(), VID) if isinstance(s, EventRef)]

    assert ref.label.startswith("Фол, 1-й тайм, 45+1'")


@pytest.mark.unit
def test_render_keeps_a_link_to_an_unknown_event_as_text() -> None:
    assert render_answer("Гол [E:p:99].", make_context(), VID) == ["Гол [E:p:99]."]


@pytest.mark.unit
def test_render_of_empty_text_is_empty() -> None:
    assert render_answer("", make_context(), VID) == []


@pytest.mark.unit
def test_plain_text_shows_label_and_url() -> None:
    text = plain_text(render_answer("Гол [E:p:2].", make_context(), VID))

    assert text.startswith("Гол [Гол, 1-й тайм, 23', видео 25:00: https://www.youtube.com/")
    assert plain_text(render_answer("Гол [E:p:2].", make_context(), None)) == (
        "Гол [Гол, 1-й тайм, 23', видео 25:00]."
    )


QUESTIONS = (
    "id,type,question,expected,what_we_check\n"
    "1,count,Сколько голов?,2,число\n"
    '2,list,"Кто, где забил?",,\n'
)


def write_questions(tmp_path: Path, text: str = QUESTIONS, bom: bool = False) -> Path:
    path = tmp_path / "q.csv"
    path.write_bytes((("﻿" if bom else "") + text).encode("utf-8"))
    return path


@pytest.mark.unit
@pytest.mark.parametrize("bom", [False, True])
def test_read_questions_parses_rows_with_commas_in_quotes(tmp_path: Path, bom: bool) -> None:
    questions = read_questions(write_questions(tmp_path, bom=bom))

    assert [(q.id, q.type, q.question, q.expected) for q in questions] == [
        ("1", "count", "Сколько голов?", "2"),
        ("2", "list", "Кто, где забил?", ""),
    ]


@pytest.mark.unit
def test_read_questions_accepts_semicolons(tmp_path: Path) -> None:
    path = write_questions(tmp_path, "id;type;question\n1;a;Привет?\n")

    assert read_questions(path)[0].question == "Привет?"


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "",
        "id,type\n1,a\n",
        "id,question\n",
        "id,question\n1,\n",
        "id,question\n,Вопрос?\n",
        "id,question\n1,А?\n1,Б?\n",
    ],
)
def test_read_questions_bad_files_are_eval_errors(tmp_path: Path, text: str) -> None:
    with pytest.raises(EvalError):
        read_questions(write_questions(tmp_path, text))


@pytest.mark.unit
def test_read_questions_missing_or_not_utf8_file_is_an_eval_error(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="cannot read"):
        read_questions(tmp_path / "none.csv")
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"id,question\n1,\xff\xfe\n")
    with pytest.raises(EvalError, match="cannot read"):
        read_questions(bad)


def run(
    tmp_path: Path, script: Sequence[ModelTurn | Exception], n_questions: int = 2
) -> EvalSummary:
    llm = FakeLLM(script)
    questions = read_questions(write_questions(tmp_path))[:n_questions]
    return run_questions(
        Agent(llm, "s"), questions, lambda: ToolBox(make_context()), tmp_path / "answers.csv", VID
    )


@pytest.mark.unit
def test_run_questions_writes_answers_with_an_empty_grade_and_counts_outcomes(
    tmp_path: Path,
) -> None:
    summary = run(
        tmp_path,
        [
            turn("", GOALS),
            turn("Два гола [E:p:2] [E:p:6]."),
            turn("Гол на 12:34."),
            turn("Не знаю."),
        ],
    )

    assert (summary.total, summary.first_try, summary.retry, summary.refused, summary.errors) == (
        2,
        1,
        1,
        0,
        0,
    )
    assert summary.links == 2
    assert summary.tokens_in == 40
    raw = (tmp_path / "answers.csv").read_bytes().decode("utf-8")
    assert raw.startswith("﻿")
    rows = list(csv.DictReader(io.StringIO(raw.lstrip("﻿"))))
    assert list(rows[0]) == list(ANSWER_COLUMNS)
    assert [r["grade"] for r in rows] == ["", ""]
    assert [r["validator"] for r in rows] == ["first_try", "retry"]
    assert rows[0]["tool_calls"] == "query_events"
    assert rows[0]["expected"] == "2"
    assert "Гол, 1-й тайм, 23'" in rows[0]["answer"]


@pytest.mark.unit
def test_run_questions_a_failed_question_is_recorded_and_the_run_goes_on(tmp_path: Path) -> None:
    summary = run(tmp_path, [ExternalServiceError("Gemini request failed (503)"), turn("Не знаю.")])

    assert (summary.first_try, summary.errors) == (1, 1)
    rows = list(csv.DictReader(io.StringIO((tmp_path / "answers.csv").read_text("utf-8-sig"))))
    assert rows[0]["validator"] == "error"
    assert "503" in rows[0]["answer"]


@pytest.mark.unit
def test_run_questions_stops_after_repeated_service_failures_but_keeps_the_rows(
    tmp_path: Path,
) -> None:
    questions = [f"{i},a,Вопрос {i}?\n" for i in range(5)]
    path = write_questions(tmp_path, "id,type,question\n" + "".join(questions))
    llm = FakeLLM([ExternalServiceError("down")] * MAX_CONSECUTIVE_ERRORS)

    with pytest.raises(ExternalServiceError, match="down"):
        run_questions(
            Agent(llm, "s"),
            read_questions(path),
            lambda: ToolBox(make_context()),
            tmp_path / "answers.csv",
        )

    rows = list(csv.DictReader(io.StringIO((tmp_path / "answers.csv").read_text("utf-8-sig"))))
    assert len(rows) == MAX_CONSECUTIVE_ERRORS


@pytest.mark.unit
def test_summarize_of_nothing_is_zero_without_a_cost() -> None:
    summary = summarize([])

    assert (summary.total, summary.links, summary.seconds) == (0, 0, 0.0)
    assert summary.cost_usd is None


@pytest.mark.unit
def test_run_questions_a_project_error_other_than_the_service_ends_only_that_question(
    tmp_path: Path,
) -> None:
    summary = run(tmp_path, [StorageError("journal is not writable"), turn("Не знаю.")])

    assert (summary.first_try, summary.errors) == (1, 1)
    rows = list(csv.DictReader(io.StringIO((tmp_path / "answers.csv").read_text("utf-8-sig"))))
    assert "journal is not writable" in rows[0]["answer"]
