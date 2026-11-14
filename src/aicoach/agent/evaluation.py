"""`agent-eval`: run a file of questions through the agent and write `answers.csv`.

The answers are graded by a person afterwards (the `grade` column stays empty). The summary
counts how the validator saw the answers; it says nothing about whether they are right.
"""

import csv
import io
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from aicoach.agent.loop import Agent, AgentAnswer
from aicoach.agent.render import plain_text
from aicoach.agent.tools import ToolBox
from aicoach.errors import AicoachError, EvalError, ExternalServiceError
from aicoach.stage_common import write_atomic

ANSWERS_FILE = "answers.csv"
QUESTION_COLUMNS = ("id", "type", "question", "expected", "what_we_check")
ANSWER_COLUMNS = (
    *QUESTION_COLUMNS,
    "answer",
    "validator",
    "n_links",
    "tool_calls",
    "tokens_in",
    "tokens_out",
    "seconds",
    "grade",
)
MAX_CONSECUTIVE_ERRORS = 3  # then the service is down for good: stop instead of hammering it

logger = structlog.get_logger()


@dataclass(frozen=True)
class Question:
    """One row of the questions file."""

    id: str
    type: str
    question: str
    expected: str
    what_we_check: str


@dataclass(frozen=True)
class EvalRow:
    """The outcome for one question: an answer, or the error that stopped it."""

    question: Question
    answer: AgentAnswer | None
    error: str | None = None


@dataclass(frozen=True)
class EvalSummary:
    """Counts over all questions; `first_try`, `retry`, `refused` and `errors` add up to `total`."""

    total: int
    first_try: int
    retry: int
    refused: int
    errors: int
    links: int
    tokens_in: int
    tokens_out: int
    seconds: float
    cost_usd: float | None
    rows: list[EvalRow] = field(default_factory=list)


def read_questions(path: Path) -> list[Question]:
    """Read the questions CSV (UTF-8, `,` or `;`, columns `id,type,question,expected,
    what_we_check`; `expected` and `what_we_check` may be empty).

    Raises:
        EvalError: The file is unreadable, a needed column is missing, a question is empty, an
            id repeats, or there are no questions. The message names the line, not the content.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as err:
        msg = f"cannot read the questions file {path}: {type(err).__name__}"
        raise EvalError(msg) from err
    header = text.splitlines()[0] if text.strip() else ""
    delimiter = ";" if header.count(";") > header.count(",") else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter, skipinitialspace=True))
    columns = {(name or "").strip().lower() for name in (rows[0] if rows else {})}
    missing = [c for c in ("id", "question") if c not in columns]
    if missing:
        msg = f"questions file has no column {', '.join(missing)}"
        raise EvalError(msg)
    return _questions(rows)


def _questions(rows: Iterable[dict[str, str]]) -> list[Question]:
    found: list[Question] = []
    seen: set[str] = set()
    for line, raw in enumerate(rows, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
        if not any(row.values()):
            continue
        if not row.get("question"):
            msg = f"questions file, line {line}: the question is empty"
            raise EvalError(msg)
        question_id = row.get("id", "")
        if not question_id or question_id in seen:
            msg = f"questions file, line {line}: the id is empty or repeated"
            raise EvalError(msg)
        seen.add(question_id)
        found.append(
            Question(
                question_id,
                row.get("type", ""),
                row["question"],
                row.get("expected", ""),
                row.get("what_we_check", ""),
            )
        )
    if not found:
        msg = "the questions file has no questions"
        raise EvalError(msg)
    return found


def _safe_cell(text: str) -> str:
    """A text that a spreadsheet will not run as a formula (a leading `=`, `+`, `-`, `@`)."""
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def _row_cells(row: EvalRow) -> list[str]:
    q, a = row.question, row.answer
    head = [q.id, q.type, q.question, q.expected, q.what_we_check]
    if a is None:
        return [*head, _safe_cell(f"ERROR: {row.error}"), "error", "0", "", "0", "0", "0", ""]
    return [
        *head,
        _safe_cell(plain_text(a.segments)),
        a.outcome,
        str(a.n_links),
        ";".join(str(c["name"]) for c in a.tool_calls),
        str(a.usage.input_tokens),
        str(a.usage.output_tokens),
        f"{a.duration_ms / 1000:.1f}",
        "",  # grade: filled in by hand
    ]


def format_answers(rows: Iterable[EvalRow]) -> str:
    """The `answers.csv` text (UTF-8 with BOM so that a spreadsheet opens Cyrillic correctly)."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(ANSWER_COLUMNS)
    writer.writerows(_row_cells(r) for r in rows)
    return "\N{BYTE ORDER MARK}" + out.getvalue()


def summarize(rows: list[EvalRow]) -> EvalSummary:
    """Count the outcomes."""
    answers = [r.answer for r in rows if r.answer is not None]
    costs = [a.cost_usd for a in answers]
    return EvalSummary(
        total=len(rows),
        first_try=sum(a.outcome == "first_try" for a in answers),
        retry=sum(a.outcome == "retry" for a in answers),
        refused=sum(a.outcome == "refused" for a in answers),
        errors=sum(r.answer is None for r in rows),
        links=sum(a.n_links for a in answers),
        tokens_in=sum(a.usage.input_tokens for a in answers),
        tokens_out=sum(a.usage.output_tokens for a in answers),
        seconds=sum(a.duration_ms for a in answers) / 1000,
        cost_usd=None if not costs or None in costs else sum(c for c in costs if c is not None),
        rows=rows,
    )


def run_questions(
    agent: Agent,
    questions: list[Question],
    box_factory: Callable[[], ToolBox],
    output: Path,
    youtube_id: str | None = None,
) -> EvalSummary:
    """Ask every question and write `answers.csv` (also when it stops early).

    A service failure on one question is recorded in its row and the run goes on; after
    `MAX_CONSECUTIVE_ERRORS` failures in a row the run stops with `ExternalServiceError`, the rows
    so far are written.

    Raises:
        ExternalServiceError: The service kept failing.
        StorageError: A file cannot be written.
    """
    rows: list[EvalRow] = []
    streak = 0
    failure: AicoachError | None = None
    for question in questions:
        try:
            answer = agent.ask(question.question, box_factory(), youtube_id)
        except AicoachError as err:
            # Any project error ends only this question: the answers already paid for are kept.
            logger.warning("agent_eval_error", stage="agent", question=question.id, error=str(err))
            rows.append(EvalRow(question, None, str(err)))
            streak = streak + 1 if isinstance(err, ExternalServiceError) else 0
            if streak >= MAX_CONSECUTIVE_ERRORS:
                failure = err
                break
        else:
            streak = 0
            rows.append(EvalRow(question, answer))
    write_atomic(output, format_answers(rows))
    if failure is not None:
        raise failure
    return summarize(rows)


def format_summary(summary: EvalSummary) -> str:
    """The summary for the console: how the validator saw the answers, tokens, time, cost."""
    cost = "n/a (model not in the price table)"
    if summary.cost_usd is not None:
        cost = f"{summary.cost_usd:.4f}"
    lines = [
        f"questions: {summary.total}",
        f"validator passed on the first try: {summary.first_try}",
        f"validator passed on the second try: {summary.retry}",
        f"refused (could not confirm with data): {summary.refused}",
        f"service errors: {summary.errors}",
        f"event links in the answers: {summary.links}",
        f"tokens: {summary.tokens_in} in, {summary.tokens_out} out",
        f"time in the agent: {summary.seconds:.1f} s",
        f"estimated cost at paid-tier prices, USD: {cost}",
    ]
    return "\n".join(lines) + "\n"
