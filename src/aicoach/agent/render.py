"""Turns the links `[E:<id>]` of a validated answer into what the user reads and clicks.

The model writes only ids. The label (type, half, game minute, video time) and the YouTube link
come from the database and the scoreboard, never from the model (CLAUDE.md section 5, rule 6).
"""

import re
from dataclasses import dataclass

from aicoach.agent.context import EventFacts, MatchContext, SegmentFacts
from aicoach.agent.validator import LINK
from aicoach.i18n import ru
from aicoach.ui.logic import format_mmss, youtube_link


@dataclass(frozen=True)
class EventRef:
    """A link of the answer: `label` for the text, `t_v` for the player, `url` for YouTube."""

    event_id: str
    label: str
    t_v: float
    url: str | None


Segment = str | EventRef
# The two phrases side by side (a comma, a colon or a dash may stand between them).
_EPISODE_MARK = re.compile(r"по клипу\W{0,5}не проверено")


def event_label(facts: EventFacts) -> str:
    """`⚽ 23' · видео 38:12`: an icon, the game minute and the video time.

    The icon is the card's for a foul with a card plate. Without a game minute the half stands
    in its place (if known).
    """
    event = facts.event
    icon = ru.CARD_ICONS.get(facts.card or "") or ru.EVENT_ICONS[event.type]
    parts: list[str] = []
    if facts.minute_text is not None:
        parts.append(facts.minute_text)
    elif event.half is not None:
        parts.append(ru.HALF_TEXT.format(half=event.half))
    parts.append(ru.AGENT_VIDEO_TIME.format(time=format_mmss(event.t_v)))
    return f"{icon} {' · '.join(parts)}"


def segment_label(facts: SegmentFacts) -> str:
    """`🎙 23' · видео 38:12` for a stretch of the commentary: the same form as an event's."""
    seg = facts.segment
    parts: list[str] = []
    if facts.minute_text is not None:
        parts.append(facts.minute_text)
    elif seg.half is not None:
        parts.append(ru.HALF_TEXT.format(half=seg.half))
    parts.append(ru.AGENT_VIDEO_TIME.format(time=format_mmss(seg.t_start)))
    return f"{ru.SEGMENT_ICON} {' · '.join(parts)}"


def _target(
    ref_id: str, events: dict[str, EventFacts], speech: dict[str, SegmentFacts]
) -> tuple[str, str, float] | None:
    """Id, label and video time of an event or a commentary segment; None for an unknown id."""
    facts = events.get(ref_id)
    if facts is not None:
        return facts.event.event_id, event_label(facts), facts.event.t_v
    spoken = speech.get(ref_id)
    if spoken is not None:
        return spoken.segment.id, segment_label(spoken), spoken.segment.t_start
    return None


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
    events, speech = ctx.by_id(), ctx.segment_by_id()
    segments: list[Segment] = []
    position = 0
    for match in LINK.finditer(text):
        target = _target(match.group(1), events, speech)
        if target is None:
            continue
        if match.start() > position:
            segments.append(text[position : match.start()])
        event_id, label, t_v = target
        url = youtube_link(youtube_id, t_v) if youtube_id else None
        segments.append(EventRef(event_id, label, t_v, url))
        position = match.end()
    if position < len(text):
        segments.append(text[position:])
    return segments


def has_episode_mark(segments: list[Segment]) -> bool:
    """Whether the answer already says "by the clip, not verified" (any case, side by side)."""
    text = "".join(s for s in segments if isinstance(s, str)).casefold()
    return _EPISODE_MARK.search(text) is not None


def with_episode_mark(segments: list[Segment]) -> list[Segment]:
    """The answer with the mark of an unverified description, added by code (D-074): the prompt
    asks the model to write it, but a model may forget, and the mark must not depend on that."""
    if has_episode_mark(segments):
        return segments
    return [*segments, "\n\n", ru.EPISODE_ANSWER_MARK]


def with_corner_mark(segments: list[Segment]) -> list[Segment]:
    """The answer with the mark of unverified corner tags, added by code (D-078, as D-074):
    the tags are a model's reading of a clip, and the mark must not depend on the answering
    model remembering it."""
    if has_episode_mark(segments):
        return segments
    return [*segments, "\n\n", ru.CORNER_TAGS_ANSWER_MARK]


_ZONE_CAVEAT = re.compile(r"низк\w*\s+надёжн|низк\w*\s+надежн|надёжн\w*\s+низк|надежн\w*\s+низк")


def with_zone_caveat(segments: list[Segment]) -> list[Segment]:
    """The answer with the remark that the zone tag is of low reliability, added by code when
    the model did not say so (D-078): the zone was right in 4 of 8 corners."""
    text = "".join(s for s in segments if isinstance(s, str)).casefold()
    if _ZONE_CAVEAT.search(text):
        return segments
    return [*segments, "\n", ru.CORNER_ZONE_ANSWER_MARK]


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
