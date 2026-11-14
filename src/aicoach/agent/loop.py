"""The agent loop: question, tool calls, validated answer (FR-8, D-061).

Each question stands alone (no memory of earlier questions): what an answer may refer to is
exactly what the tools returned for it, so the validator can check it. The model gets at most
`max_tool_calls` calls per question; a rejected answer gets one more try with the reason; a second
rejection ends in a refusal without links.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal  # Any: JSON for and from the model

import structlog

from aicoach.agent.context import MatchContext
from aicoach.agent.journal import JournalEntry, append_entry
from aicoach.agent.llm import LLMClient
from aicoach.agent.messages import (
    HistoryItem,
    ToolCall,
    ToolResult,
    ToolResults,
    Usage,
    UserMessage,
)
from aicoach.agent.pricing import cost_usd
from aicoach.agent.render import Segment, count_links, plain_text, render_answer
from aicoach.agent.tools import ClipMade, ToolBox
from aicoach.agent.validator import Violation, describe, find_violations
from aicoach.errors import ExternalServiceError, StorageError
from aicoach.i18n import ru

logger = structlog.get_logger()

MAX_TOOL_CALLS = 6
MAX_TEAM_NAME_CHARS = 40
MAX_ATTEMPTS = 2  # the first answer and one retry after a validator complaint
SYSTEM_PROMPT_FILE = Path(__file__).with_name("system_prompt.md")

Outcome = Literal["first_try", "retry", "refused"]


@dataclass(frozen=True)
class AgentAnswer:
    """The result for one question. `segments` are what the user sees (text and event links);
    `outcome` is how the validator saw the answer."""

    question: str
    segments: list[Segment]
    outcome: Outcome
    n_links: int
    tool_calls: list[dict[str, Any]]
    usage: Usage
    duration_ms: int
    model: str
    violations: list[str] = field(default_factory=list)
    clips: list[ClipMade] = field(default_factory=list)
    cost_usd: float | None = None

    @property
    def text(self) -> str:
        """The answer as plain text (links shown as `[label]`)."""
        return plain_text(self.segments)


@dataclass
class _Run:
    """The state of one question."""

    history: list[HistoryItem]
    usage: Usage = field(default_factory=Usage)
    calls: list[dict[str, Any]] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)


def load_system_prompt(ctx: MatchContext) -> str:
    """The system prompt file with the team names put in; it holds no match data itself.

    Raises:
        OSError: The prompt file is missing (a broken installation).
    """
    text = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    return text.replace("{home}", _plain_name(ctx.team_names["home"])).replace(
        "{away}", _plain_name(ctx.team_names["away"])
    )


def _plain_name(name: str) -> str:
    """A team name from `lineups.csv` as one short plain line: it goes into the prompt, so it
    must not be able to carry text of its own (line breaks, brackets)."""
    cleaned = "".join(ch for ch in " ".join(name.split()) if ch.isalnum() or ch in " .-'")
    return " ".join(cleaned.split())[:MAX_TEAM_NAME_CHARS] or "?"


class Agent:
    """Answers questions with the tools of a `ToolBox`.

    Args:
        llm: The language-model client.
        system_prompt: The prompt (see `load_system_prompt`).
        match_dir: Where the journal is written; `None` writes no journal.
        max_tool_calls: Tool calls allowed per question.
        clock: A monotonic clock in seconds (a fake in tests).
    """

    def __init__(
        self,
        llm: LLMClient,
        system_prompt: str,
        *,
        match_dir: Path | None = None,
        max_tool_calls: int = MAX_TOOL_CALLS,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._llm = llm
        self._system = system_prompt
        self._match_dir = match_dir
        self._max_calls = max_tool_calls
        self._clock = clock

    def ask(self, question: str, box: ToolBox, youtube_id: str | None = None) -> AgentAnswer:
        """Answer one question.

        Args:
            question: The user's question (never part of the system prompt).
            box: The tools for this question (fresh: it remembers what it handed out).
            youtube_id: YouTube id of the match, for the links of the answer.

        Returns:
            The answer; a refusal (`outcome="refused"`) has no links.

        Raises:
            ExternalServiceError: The model service failed for good; the journal records it.
            StorageError: The journal cannot be written.
        """
        started = self._clock()
        run = _Run([UserMessage(question)])
        try:
            text, outcome = self._converse(run, box)
        except ExternalServiceError:
            self._journal_quietly(question, run, self._duration_ms(started), "error")
            raise
        if outcome == "refused":
            segments: list[Segment] = [ru.AGENT_REFUSAL]
        else:
            segments = render_answer(text, box.ctx, youtube_id)
        n_links = count_links(segments)
        duration_ms = self._duration_ms(started)
        self._journal(question, run, duration_ms, outcome, n_links)
        answer = AgentAnswer(
            question=question,
            segments=segments,
            outcome=outcome,
            n_links=n_links,
            tool_calls=run.calls,
            usage=run.usage,
            duration_ms=duration_ms,
            model=self._llm.model,
            violations=run.codes,
            clips=[] if outcome == "refused" else list(box.clips),
            cost_usd=cost_usd(self._llm.model, run.usage),
        )
        logger.info(
            "agent_answered",
            stage="agent",
            match_id=box.ctx.match_id,
            outcome=outcome,
            n_links=n_links,
            tool_calls=len(run.calls),
            tokens_in=run.usage.input_tokens,
            tokens_out=run.usage.output_tokens,
            duration_ms=duration_ms,
        )
        return answer

    def _duration_ms(self, started: float) -> int:
        return round((self._clock() - started) * 1000)

    def _converse(self, run: _Run, box: ToolBox) -> tuple[str, Outcome]:
        """Requests until an answer passes the validator or the attempts run out."""
        attempt = 1
        while True:
            allow_tools = len(run.calls) < self._max_calls
            turn = self._llm.generate(
                self._system, run.history, box.specs(), allow_tools=allow_tools
            )
            run.usage += turn.usage
            run.history.append(turn)
            if turn.calls:
                if not allow_tools:  # asked for tools although they were switched off
                    run.codes.append("tool_limit")
                    return "", "refused"
                run.history.append(self._run_calls(turn.calls, run, box))
                continue
            problems = self._check(turn.text, box, run.history[0], turn.truncated)
            if not problems:
                return turn.text.strip(), "first_try" if attempt == 1 else "retry"
            run.codes.extend(v.code for v in problems)
            if attempt >= MAX_ATTEMPTS:
                return "", "refused"
            attempt += 1
            run.history.append(UserMessage(describe(problems)))

    def _run_calls(self, calls: Sequence[ToolCall], run: _Run, box: ToolBox) -> ToolResults:
        results: list[ToolResult] = []
        for call in calls:
            if len(run.calls) >= self._max_calls:
                result: dict[str, Any] = {"error": "лимит вызовов инструментов исчерпан"}
            else:
                run.calls.append({"name": call.name, "args": call.args})
                result = box.call(call.name, call.args)
            results.append(ToolResult(call.name, result, call.call_id))
        return ToolResults(tuple(results))

    @staticmethod
    def _check(text: str, box: ToolBox, asked: HistoryItem, truncated: bool) -> list[Violation]:
        if not text.strip():
            return [Violation("empty", "ответ пустой")]
        if truncated:
            return [Violation("truncated", "ответ оборван по длине: сделай его короче")]
        # A name the user wrote in the question may be repeated ("Выдумова в матче нет").
        question = asked.text if isinstance(asked, UserMessage) else ""
        return find_violations(
            text,
            seen_ids=box.seen_event_ids,
            existing_ids=set(box.ctx.by_id()),
            allowed_names=[*box.allowed_names, question],
        )

    def _journal_quietly(self, question: str, run: _Run, duration_ms: int, outcome: str) -> None:
        """Journal an error outcome without letting a journal failure hide the real error."""
        try:
            self._journal(question, run, duration_ms, outcome, 0)
        except StorageError as err:
            logger.warning("agent_journal_failed", stage="agent", error=str(err))

    def _journal(
        self, question: str, run: _Run, duration_ms: int, outcome: str, n_links: int
    ) -> None:
        if self._match_dir is None:
            return
        append_entry(
            self._match_dir,
            JournalEntry(
                question=question,
                tools=run.calls,
                tokens_in=run.usage.input_tokens,
                tokens_out=run.usage.output_tokens,
                duration_ms=duration_ms,
                model=self._llm.model,
                validator=outcome,
                violations=run.codes,
                n_links=n_links,
                cost_usd=cost_usd(self._llm.model, run.usage),
            ),
        )
