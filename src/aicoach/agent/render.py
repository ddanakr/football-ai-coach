"""Turns the links `[E:<id>]` of a validated answer into what the user reads and clicks.

The model writes only ids. The label (type, half, game minute, video time) and the YouTube link
come from the database and the scoreboard, never from the model (CLAUDE.md section 5, rule 6).
"""

from dataclasses import dataclass

from aicoach.agent.context import EventFacts, MatchContext
from aicoach.agent.validator import LINK
from aicoach.i18n import ru
from aicoach.ui.logic import format_mmss, type_label, youtube_link


@dataclass(frozen=True)
class EventRef:
    """A link of the answer: `label` for the text, `t_v` for the player, `url` for YouTube."""

    event_id: str
    label: str
    t_v: float
    url: str | None


Segment = str | EventRef


def event_label(facts: EventFacts) -> str:
    """`Гол, 1-й тайм, 23', видео 38:12`: parts that are unknown are left out."""
    event = facts.event
    parts = [type_label(event.type)]
    if event.half is not None:
        parts.append(ru.HALF_TEXT.format(half=event.half))
    if facts.minute_text is not None:
        parts.append(facts.minute_text)
    parts.append(ru.AGENT_VIDEO_TIME.format(time=format_mmss(event.t_v)))
    return ", ".join(parts)


def render_answer(text: str, ctx: MatchContext, youtube_id: str | None = None) -> list[Segment]:
    """Split the answer into text and event links.

    A link to an event that is not in the context stays as it was written (the validator
    rejects such answers before they get here).

    Args:
        text: The validated answer.
        ctx: The match snapshot the answer was checked against.
        youtube_id: YouTube id of the match for the `?t=` links; without it there is no URL.

    Raises:
        ConfigError: `youtube_id` is not a valid id.
    """
    by_id = ctx.by_id()
    segments: list[Segment] = []
    position = 0
    for match in LINK.finditer(text):
        facts = by_id.get(match.group(1))
        if facts is None:
            continue
        if match.start() > position:
            segments.append(text[position : match.start()])
        url = youtube_link(youtube_id, facts.event.t_v) if youtube_id else None
        segments.append(EventRef(facts.event.event_id, event_label(facts), facts.event.t_v, url))
        position = match.end()
    if position < len(text):
        segments.append(text[position:])
    return segments


def count_links(segments: list[Segment]) -> int:
    """How many event links the rendered answer has."""
    return sum(isinstance(s, EventRef) for s in segments)


def plain_text(segments: list[Segment]) -> str:
    """The answer as one string (for the CSV of `agent-eval`): `[label: url]` for each link."""
    parts: list[str] = []
    for segment in segments:
        if isinstance(segment, str):
            parts.append(segment)
        else:
            parts.append(
                f"[{segment.label}: {segment.url}]" if segment.url else f"[{segment.label}]"
            )
    return "".join(parts)
