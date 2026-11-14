"""Pure helpers of the "Agent" panel: how an answer becomes text and buttons."""

from aicoach.agent.loop import AgentAnswer
from aicoach.agent.render import EventRef
from aicoach.i18n import ru


def answer_markdown(answer: AgentAnswer) -> str:
    """The answer text with every event link shown as a bold label.

    The label (type, half, game minute, video time) comes from the database, not from the
    model. Plain Markdown only: raw HTML is not enabled in the page.
    """
    parts = [s if isinstance(s, str) else f"**[{s.label}]**" for s in answer.segments]
    return "".join(parts)


def unique_refs(answer: AgentAnswer) -> list[EventRef]:
    """The event links of the answer, each event once, in order of appearance."""
    seen: dict[str, EventRef] = {}
    for segment in answer.segments:
        if isinstance(segment, EventRef):
            seen.setdefault(segment.event_id, segment)
    return list(seen.values())


def meta_line(answer: AgentAnswer) -> str:
    """One line under the answer: how the validator saw it, the tool calls, tokens, time."""
    return ru.AGENT_META.format(
        outcome=ru.AGENT_OUTCOME_LABELS.get(answer.outcome, answer.outcome),
        calls=len(answer.tool_calls),
        tokens_in=answer.usage.input_tokens,
        tokens_out=answer.usage.output_tokens,
        seconds=answer.duration_ms / 1000,
    )
