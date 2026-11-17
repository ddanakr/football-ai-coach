"""The tools of the agent: pure functions over a `MatchContext`, answers as plain JSON.

The model never sees video times: events carry their id (`event_id`, the only thing the answer
may refer to), the half, the game minute and, when known, the team and the player names.
Filtering by game minute, half, side and player happens here, in code, not in the model.
`ToolBox` remembers which event ids and names it handed out, so the validator can check an
answer against exactly what the tools returned (D-061).
"""

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal  # Any: JSON for and from the model

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aicoach.agent.context import EventFacts, MatchContext, SegmentFacts
from aicoach.agent.messages import ToolSpec
from aicoach.agent.transcript_tools import (
    COMMENTARY_NOTE,
    DEFAULT_WINDOW_S,
    MAX_SEARCH_LIMIT,
    MAX_SEGMENT_LIMIT,
    GetTranscriptArgs,
    SearchTranscriptArgs,
    around,
    in_minutes,
    search,
    segment_json,
)
from aicoach.agent.validator import numbers_in
from aicoach.clips import DEFAULT_AFTER_S, DEFAULT_BEFORE_S
from aicoach.corner_tags.schemas import CHOICES, FIELDS, normalize_value
from aicoach.corner_tags.view import TEAM_CHOICES, CornerFilter, CornerView, matches
from aicoach.domain.events import EventType
from aicoach.episode import EpisodeDescription
from aicoach.errors import ExportError, ExternalServiceError
from aicoach.namematch import name_key, name_matches
from aicoach.plate_names.schemas import PlayerRead

logger = structlog.get_logger()

DEFAULT_LIMIT = 20
MAX_LIMIT = 50
MAX_PLAYERS = 5
MAX_CLIP_SIDE_S = 60.0
ACCURACY_NOTE = "Это события, найденные системой автоматически, а не официальная статистика."
CORNER_TAGS_NOTE = (
    "Теги углового (corner_tags) прочитала видеомодель по клипу: это данные, а не факт, "
    "не проверено. unclear значит «не видно». Команда подачи определена по цвету формы. "
    "Поле zone (куда прилетел мяч) низкой надёжности: так и говори."
)

ClipMaker = Callable[[str, float, float], Path]
EpisodeDescriber = Callable[[str], EpisodeDescription]
Result = dict[str, Any]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryEventsArgs(_Args):
    """Arguments of `query_events`; every filter is optional."""

    type: EventType | None = None
    half: Literal[1, 2] | None = None
    minute_from: int | None = Field(default=None, ge=0, le=200)
    minute_to: int | None = Field(default=None, ge=0, le=200)
    side: Literal["home", "away", "unknown"] | None = None
    player: str | None = Field(default=None, max_length=100)
    card: bool | None = None
    # Filters over the tags a video model read from the clip of a corner (D-078); any of them
    # selects corners that have tags (a code, or its Russian name; unclear is 'не видно').
    corner_team: str | None = Field(default=None, max_length=40)
    corner_kit: str | None = Field(default=None, max_length=40)
    corner_delivery: str | None = Field(default=None, max_length=40)
    corner_zone: str | None = Field(default=None, max_length=40)
    corner_first_contact: str | None = Field(default=None, max_length=40)
    corner_outcome: str | None = Field(default=None, max_length=40)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)


class GetEventArgs(_Args):
    """Arguments of `get_event`."""

    event_id: str = Field(max_length=32)


class FindPlayerArgs(_Args):
    """Arguments of `find_player`: a name and/or a shirt number."""

    query: str | None = Field(default=None, max_length=100)
    number: int | None = Field(default=None, ge=0, le=999)
    side: Literal["home", "away"] | None = None


class MakeClipArgs(_Args):
    """Arguments of `make_clip`."""

    event_id: str = Field(max_length=32)
    before_s: float = Field(default=DEFAULT_BEFORE_S, ge=0, le=MAX_CLIP_SIDE_S)
    after_s: float = Field(default=DEFAULT_AFTER_S, ge=0, le=MAX_CLIP_SIDE_S)


class DescribeEpisodeArgs(_Args):
    """Arguments of `describe_episode`."""

    event_id: str = Field(max_length=32)


NO_TRANSCRIPT = "расшифровки комментария для этого матча нет"
_EVENT_TYPES = [str(t) for t in EventType]
_SIDES = ["home", "away"]


def _spec(name: str, description: str, properties: dict[str, Any], required: list[str]) -> ToolSpec:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return ToolSpec(name, description, schema)


MATCH_INFO = _spec(
    "match_info",
    "Обзор матча: названия команд, игровые минуты начала и конца таймов, число найденных "
    "системой событий по видам и таймам. Без аргументов.",
    {},
    [],
)
QUERY_EVENTS = _spec(
    "query_events",
    "Список событий матча по порядку времени с фильтрами. Минуты игровые, как в трансляции "
    "(добавленное время идёт дальше 45 и 90). Возвращает total (сколько всего подошло) и первые "
    "limit событий.",
    {
        "type": {"type": "string", "enum": _EVENT_TYPES, "description": "вид события"},
        "half": {"type": "integer", "enum": [1, 2], "description": "тайм"},
        "minute_from": {"type": "integer", "description": "игровая минута от (включительно)"},
        "minute_to": {"type": "integer", "description": "игровая минута до (включительно)"},
        "side": {
            "type": "string",
            "enum": [*_SIDES, "unknown"],
            "description": "home (хозяева), away (гости) или unknown (команда не определена)",
        },
        "player": {"type": "string", "description": "фамилия игрока (как её пишет пользователь)"},
        "card": {
            "type": "boolean",
            "description": "true: только фолы, за которые показана карточка (по плашке)",
        },
        "corner_team": {
            "type": "string",
            "enum": list(TEAM_CHOICES),
            "description": "угловые, которые подаёт эта сторона (по цвету формы на клипе, "
            "не проверено); unknown: команда подачи не определена",
        },
        "corner_kit": {
            "type": "string",
            "enum": list(CHOICES["attacking_kit"]),
            "description": "угловые, где подающие в форме этого цвета (по клипу, не проверено)",
        },
        "corner_delivery": {
            "type": "string",
            "enum": list(CHOICES["delivery"]),
            "description": "угловые с такой подачей: cross навес, short короткий, "
            "low_ball прострел низом, unclear не видно (по клипу, не проверено)",
        },
        "corner_zone": {
            "type": "string",
            "enum": list(CHOICES["zone"]),
            "description": "угловые, где мяч прилетел в эту зону (по клипу, не проверено)",
        },
        "corner_first_contact": {
            "type": "string",
            "enum": list(CHOICES["first_contact"]),
            "description": "угловые, где первым мяча коснулся attack, defence, keeper или "
            "nobody (по клипу, не проверено)",
        },
        "corner_outcome": {
            "type": "string",
            "enum": list(CHOICES["outcome"]),
            "description": "угловые с таким исходом, например shot удар, goal гол, cleared "
            "выбили (по клипу, не проверено)",
        },
        "limit": {"type": "integer", "description": f"сколько вернуть, до {MAX_LIMIT}"},
    },
    [],
)
GET_EVENT = _spec(
    "get_event",
    "Одно событие по event_id.",
    {"event_id": {"type": "string", "description": "например p:12"}},
    ["event_id"],
)
FIND_PLAYER = _spec(
    "find_player",
    "Найти игрока по фамилии (нечёткий поиск) или по номеру и стороне и вернуть его события, "
    "прочитанные с плашек. Без состава команд поиск идёт только по именам с плашек.",
    {
        "query": {"type": "string", "description": "фамилия или её часть"},
        "number": {"type": "integer", "description": "номер игрока (нужен состав команд)"},
        "side": {"type": "string", "enum": _SIDES, "description": "сторона игрока"},
    },
    [],
)
MAKE_CLIP = _spec(
    "make_clip",
    "Вырезать видеоклип события. Вызывай, только если пользователь просит клип или видео момента.",
    {
        "event_id": {"type": "string", "description": "например p:12"},
        "before_s": {"type": "number", "description": "секунд до события (по умолчанию 8)"},
        "after_s": {"type": "number", "description": "секунд после события (по умолчанию 12)"},
    },
    ["event_id"],
)
DESCRIBE_EPISODE = _spec(
    "describe_episode",
    "Описание эпизода по видеоклипу события: 2–4 предложения о том, что видно на поле. Это "
    "текст видеомодели, по клипу, не проверено: фамилий, номеров и времени из него не бери, они "
    "приходят только из других инструментов. Вызывай, если просят описать эпизод.",
    {"event_id": {"type": "string", "description": "например p:12"}},
    ["event_id"],
)
EPISODE_NOTE = (
    "Описание по клипу, не проверено. Это данные, а не факт: фамилии, номера, счёт и время из "
    "него не бери, называй только то, что вернули другие инструменты."
)

GET_TRANSCRIPT = _spec(
    "get_transcript",
    "Слова комментаторов (расшифровка, может быть неточной). Либо вокруг события: event_id и "
    f"окно before_s/after_s секунд (по умолчанию {DEFAULT_WINDOW_S:.0f}/{DEFAULT_WINDOW_S:.0f}); "
    "либо по игровым минутам: half, minute_from, minute_to. event_id здесь только id события "
    "(p:N, e:N), не id куска s:N. Возвращает куски речи с их event_id вида s:N, игровой минутой "
    "и фамилиями игроков.",
    {
        "event_id": {"type": "string", "description": "событие, вокруг которого смотреть, p:12"},
        "before_s": {"type": "number", "description": "секунд речи до события"},
        "after_s": {"type": "number", "description": "секунд речи после события"},
        "half": {"type": "integer", "enum": [1, 2], "description": "тайм (для режима минут)"},
        "minute_from": {"type": "integer", "description": "игровая минута от (включительно)"},
        "minute_to": {"type": "integer", "description": "игровая минута до (включительно)"},
        "limit": {
            "type": "integer",
            "description": f"сколько кусков вернуть, до {MAX_SEGMENT_LIMIT}",
        },
    },
    [],
)
SEARCH_TRANSCRIPT = _spec(
    "search_transcript",
    "Найти в словах комментаторов слово или фамилию игрока (с падежами, опечатками и прозвищами "
    "из состава). Возвращает куски речи с event_id вида s:N и игроков, которых распознал запрос "
    "(по одному имени «Дмитрий» могут подойти несколько игроков; куски ищутся по фамилии и "
    "прозвищу).",
    {
        "query": {"type": "string", "description": "слово, фамилия или прозвище"},
        "half": {"type": "integer", "enum": [1, 2], "description": "тайм"},
        "limit": {"type": "integer", "description": f"сколько вернуть, до {MAX_SEARCH_LIMIT}"},
    },
    ["query"],
)


@dataclass(frozen=True)
class ClipMade:
    """A clip a tool call produced (the interface shows it under the answer)."""

    event_id: str
    path: Path


# Keys of a result whose text is not a fact to state: the digits of an id, and what a video model
# wrote (it may contain any number or name, so its words must not make a number "seen").
_NOT_FACT_KEYS = frozenset({"event_id", "description"})

# Arguments that name a fact the model asked about (a half, a minute, a shirt number). Others
# (limit, clip window, event id) are not facts: they must not make a number "seen".
_ASKED_NUMBERS = frozenset({"half", "minute_from", "minute_to", "number"})


def _without_ids(value: Any) -> Any:  # noqa: ANN401 - walks the JSON of a result
    """The result without the values that are not facts to state: `event_id` (the digits of an
    id) and `description` (the text of a video model)."""
    if isinstance(value, dict):
        return {k: _without_ids(v) for k, v in value.items() if k not in _NOT_FACT_KEYS}
    if isinstance(value, list):
        return [_without_ids(v) for v in value]
    return value


def event_json(facts: EventFacts, team_names: Mapping[str, str]) -> Result:
    """One event as the model sees it. No video time: the interface adds it from the id."""
    event = facts.event
    named = [p for p in facts.players if p.name]
    result: Result = {
        "event_id": event.event_id,
        "type": str(event.type),
        "half": event.half,
        "minute": facts.minute,
        "minute_text": facts.minute_text,
        "team": event.team,
        "team_name": team_names.get(event.team) if event.team else None,
        "status": event.status,
        "score": None if event.score is None else round(event.score, 2),
        "source": event.source,
        "verified": event.verified,
        "added_by_user": event.origin == "user",
    }
    if named:
        result["players"] = [
            {"role": p.role, "name": p.name, "checked_against_lineup": p.status == "matched"}
            for p in named
        ]
    if len(named) < len(facts.players):
        result["unrecognized_names"] = len(facts.players) - len(named)
    if facts.card is not None:
        result["card"] = facts.card  # "yellow" or "red"
    if facts.corner is not None:
        result["corner_tags"] = corner_tags_json(facts.corner, team_names)
    if event.type == EventType.SUBST:
        result["substitution"] = _substitution(facts.players)
    return result


def corner_tags_json(view: CornerView, team_names: Mapping[str, str]) -> Result:
    """The tags of a corner for the model: codes (`unclear` is "не видно"), the attacking side
    worked out from the kit colour, and the source, so that the answer can say "by the clip"."""
    tags: Result = {name: view.value(name) for name in FIELDS}
    tags["attacking_team"] = view.side
    tags["attacking_team_name"] = team_names.get(view.side) if view.side else None
    tags["source"] = "clip_model_unverified"
    tags["zone_reliability"] = "low"  # the zone was right in half of the corners (D-078)
    return tags


def _substitution(players: tuple[PlayerRead, ...]) -> Result:
    """Who came on and who went off, from the two lines of the substitution plate; a name that
    could not be read is `None`."""
    names: Result = {"came_on": None, "went_off": None}
    for p in players:
        key = "came_on" if p.role == "in" else "went_off" if p.role == "out" else None
        if key is not None:
            names[key] = p.name
    return names


_CORNER_ARGS = {
    "corner_kit": "attacking_kit",
    "corner_delivery": "delivery",
    "corner_zone": "zone",
    "corner_first_contact": "first_contact",
    "corner_outcome": "outcome",
}


class CornerFilterError(ValueError):
    """A corner filter value is not an allowed code or Russian name of its field."""


def corner_filter(a: QueryEventsArgs) -> CornerFilter:
    """The corner filter of the query arguments.

    Raises:
        CornerFilterError: A value is not allowed for its field.
    """
    fields: dict[str, str] = {}
    for arg, field_name in _CORNER_ARGS.items():
        raw: str | None = getattr(a, arg)
        if raw is None:
            continue
        code = normalize_value(field_name, raw)
        if code is None:
            allowed = ", ".join(CHOICES[field_name])
            msg = f"{arg}: недопустимое значение {raw!r}, допустимы {allowed}"
            raise CornerFilterError(msg)
        fields[field_name] = code
    team = a.corner_team
    if team is not None and team not in TEAM_CHOICES:
        msg = f"corner_team: допустимы {', '.join(TEAM_CHOICES)}"
        raise CornerFilterError(msg)
    return CornerFilter(team=team, fields=fields)


def _has_player(facts: EventFacts, query: str) -> bool:
    return any(p.name and name_matches(query, p.name) for p in facts.players)


def _count_by(events: tuple[EventFacts, ...], key: Callable[[EventFacts], str]) -> dict[str, int]:
    return dict(sorted(Counter(key(f) for f in events).items()))


@dataclass
class ToolBox:
    """The tools over one match snapshot, for one question.

    Keeps `seen_event_ids`, `seen_names` and `seen_numbers` (what the answer may refer to or
    state) and `clips`.
    `clip_maker` makes `make_clip` available and `episode_describer` makes `describe_episode`
    available; without them the tools are not offered.
    """

    ctx: MatchContext
    clip_maker: ClipMaker | None = None
    episode_describer: EpisodeDescriber | None = None
    # Set when a description reached the model: the answer must then carry the mark (D-074).
    episode_described: bool = False
    # Set when corner tags reached the model: the answer must then carry the mark (D-078).
    corner_tags_shown: bool = False
    # Set when a `zone` value (or a zone filter) reached the model: the answer must then
    # say that this field is of low reliability (D-078).
    zone_shown: bool = False
    seen_event_ids: set[str] = field(default_factory=set)
    seen_names: set[str] = field(default_factory=set)
    seen_numbers: set[float] = field(default_factory=set)
    clips: list[ClipMade] = field(default_factory=list)

    def specs(self) -> tuple[ToolSpec, ...]:
        """The tools to offer to the model."""
        base = (MATCH_INFO, QUERY_EVENTS, GET_EVENT, FIND_PLAYER, GET_TRANSCRIPT, SEARCH_TRANSCRIPT)
        optional = (
            (MAKE_CLIP, self.clip_maker is not None),
            (DESCRIBE_EPISODE, self.episode_describer is not None),
        )
        return (*base, *(spec for spec, offered in optional if offered))

    @property
    def allowed_names(self) -> set[str]:
        """Names the answer may use: handed out by the tools, the nicknames of those players
        (the lineup `aliases`, D-077) and the team names."""
        return self.seen_names | self._seen_aliases() | set(self.ctx.team_names.values())

    def _seen_aliases(self) -> set[str]:
        """The nicknames of the lineup players whose full name a tool handed out. A nickname of
        a player the tools did not return is not here: the answer may not use it."""
        seen = {name_key(n) for n in self.seen_names}
        return {a for p in self.ctx.lineup if name_key(p.name) in seen for a in p.aliases}

    def call(self, name: str, args: Mapping[str, Any]) -> Result:
        """Run a tool; a wrong name or wrong arguments give `{"error": ...}` for the model."""
        handlers: dict[str, Callable[[Mapping[str, Any]], Result]] = {
            MATCH_INFO.name: self._match_info,
            QUERY_EVENTS.name: self._query_events,
            GET_EVENT.name: self._get_event,
            FIND_PLAYER.name: self._find_player,
            GET_TRANSCRIPT.name: self._get_transcript,
            SEARCH_TRANSCRIPT.name: self._search_transcript,
        }
        if self.clip_maker is not None:
            handlers[MAKE_CLIP.name] = self._make_clip
        if self.episode_describer is not None:
            handlers[DESCRIBE_EPISODE.name] = self._describe_episode
        handler = handlers.get(name)
        if handler is None:
            return {"error": f"неизвестный инструмент {name!r}"}
        try:
            result = handler(args)
        except ValidationError as err:
            # Field names and rules only: the model must fix its arguments.
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in err.errors()
            )
            return {"error": f"неверные аргументы: {problems}"}
        if "error" not in result:
            # What the answer may state: the numbers the tool returned, and those of the call
            # (a half or a minute the model asked about; a count it made up is not among them).
            self.seen_numbers |= numbers_in(json.dumps(_without_ids(result), ensure_ascii=False))
            asked = {k: v for k, v in args.items() if k in _ASKED_NUMBERS}
            self.seen_numbers |= numbers_in(json.dumps(asked))
        return result

    def _emit(self, facts: EventFacts) -> Result:
        """The event for the model; remembers its id and names as handed out."""
        self.seen_event_ids.add(facts.event.event_id)
        self.seen_names.update(p.name for p in facts.players if p.name)
        if facts.corner is not None:
            self.corner_tags_shown = True
            self.zone_shown = self.zone_shown or facts.corner.value("zone") != "unclear"
        return event_json(facts, self.ctx.team_names)

    def _emit_segment(self, facts: SegmentFacts) -> Result:
        """The segment for the model; remembers its id, the players and the capitalized words of
        the text as handed out (a quotation of the commentary may repeat them)."""
        seg = facts.segment
        self.seen_event_ids.add(seg.id)
        self.seen_names.update(m.name for m in seg.players)
        self.seen_names.update(w for w in seg.text.split() if w[:1].isupper())
        return segment_json(facts)

    def _segments_result(self, rows: list[SegmentFacts], limit: int, **extra: Any) -> Result:  # noqa: ANN401
        shown = rows[:limit]
        result: Result = {
            "total": len(rows),
            "returned": len(shown),
            "segments": [self._emit_segment(f) for f in shown],
            "note": COMMENTARY_NOTE,
            **extra,
        }
        if len(rows) > len(shown):
            result["truncated"] = True
        return result

    def _get_transcript(self, args: Mapping[str, Any]) -> Result:
        a = GetTranscriptArgs.model_validate(args)
        if not self.ctx.segments:
            return {"error": NO_TRANSCRIPT}
        by_minutes = a.half is not None or a.minute_from is not None or a.minute_to is not None
        if a.event_id is not None and by_minutes:
            return {"error": "нужен либо event_id, либо half/minute_from/minute_to, не оба"}
        if a.event_id is None and not by_minutes:
            return {"error": "нужен event_id или half/minute_from/minute_to"}
        if a.minute_from is not None and a.minute_to is not None and a.minute_from > a.minute_to:
            return {"error": "minute_from больше minute_to"}
        if a.event_id is not None:
            facts = self.ctx.by_id().get(a.event_id)
            if facts is None:
                return {"error": f"события {a.event_id!r} нет в данных матча"}
            rows = around(self.ctx.segments, facts.event.t_v, a.before_s, a.after_s)
            return self._segments_result(rows, a.limit)
        rows, unknown = in_minutes(self.ctx.segments, a.half, a.minute_from, a.minute_to)
        extra = {"without_known_minute": unknown} if unknown else {}
        return self._segments_result(rows, a.limit, **extra)

    def _search_transcript(self, args: Mapping[str, Any]) -> Result:
        a = SearchTranscriptArgs.model_validate(args)
        if not self.ctx.segments:
            return {"error": NO_TRANSCRIPT}
        rows, players = search(self.ctx.segments, self.ctx.lineup, a.query, a.half)
        resolved = [{"name": p.name, "side": p.side, "number": p.number} for p in players]
        self.seen_names.update(p.name for p in players)
        return self._segments_result(rows, a.limit, resolved_players=resolved)

    def _match_info(self, args: Mapping[str, Any]) -> Result:
        if args:
            return {"error": "match_info не принимает аргументов"}
        events = self.ctx.events
        halves = {
            str(half): {"first_minute": first, "last_minute": last}
            for half, (first, last) in sorted(self.ctx.half_minutes.items())
        }
        return {
            "teams": {"home": self.ctx.team_names["home"], "away": self.ctx.team_names["away"]},
            "lineup_available": self.ctx.has_lineup,
            "halves": halves,
            "events_total": len(events),
            "events_by_type": _count_by(events, lambda f: str(f.event.type)),
            "events_by_half": _count_by(events, lambda f: str(f.event.half or "unknown")),
            "cards": {
                "yellow": sum(f.card == "yellow" for f in events),
                "red": sum(f.card == "red" for f in events),
            },
            "note": ACCURACY_NOTE,
        }

    def _query_events(self, args: Mapping[str, Any]) -> Result:
        a = QueryEventsArgs.model_validate(args)
        if a.minute_from is not None and a.minute_to is not None and a.minute_from > a.minute_to:
            return {"error": "minute_from больше minute_to"}
        try:
            corners = corner_filter(a)
        except CornerFilterError as err:
            return {"error": str(err)}
        if "zone" in corners.fields:
            self.zone_shown = True
        # Every filter except the side first: the events whose team is unknown must be counted
        # when the side is asked for, so that "no shots" is not said about "shots of unknown team".
        fitting = [f for f in self.ctx.events if self._passes(f, a, corners)]
        by_minute = a.minute_from is not None or a.minute_to is not None
        without_minute = 0
        if by_minute:
            # Counted for the asked side only, like `total`.
            without_minute = sum(f.minute is None for f in fitting if self._side_fits(f, a.side))
        if by_minute:
            fitting = [f for f in fitting if f.minute is not None and self._in_minutes(f.minute, a)]
        rows = [f for f in fitting if self._side_fits(f, a.side)]
        shown = rows[: a.limit]
        result: Result = {
            "total": len(rows),
            "returned": len(shown),
            "events": [self._emit(f) for f in shown],
            "note": ACCURACY_NOTE,
        }
        if len(rows) > len(shown):
            result["truncated"] = True
        if corners.active or any(f.corner is not None for f in shown):
            result["corner_tags_note"] = CORNER_TAGS_NOTE
        if without_minute:
            result["without_known_minute"] = without_minute  # not in `total`: no clock there
        unknown_team = sum(f.event.team is None for f in fitting)
        if a.side in _SIDES and unknown_team:
            result["team_unknown"] = unknown_team
            result["team_unknown_note"] = (
                f"у {unknown_team} подходящих событий команда не определена: они не вошли в total "
                "(вызови query_events с side=unknown, чтобы увидеть их)"
            )
        return result

    @staticmethod
    def _side_fits(facts: EventFacts, side: str | None) -> bool:
        if side is None:
            return True
        return facts.event.team is None if side == "unknown" else facts.event.team == side

    @staticmethod
    def _passes(facts: EventFacts, a: QueryEventsArgs, corners: CornerFilter) -> bool:
        """All filters except the side and the minutes."""
        event = facts.event
        if a.type is not None and event.type != a.type:
            return False
        if a.half is not None and event.half != a.half:
            return False
        if a.card is not None and (facts.card is not None) != a.card:
            return False
        if not matches(facts.corner, corners):
            return False
        return a.player is None or _has_player(facts, a.player)

    @staticmethod
    def _in_minutes(minute: int, a: QueryEventsArgs) -> bool:
        return (a.minute_from is None or minute >= a.minute_from) and (
            a.minute_to is None or minute <= a.minute_to
        )

    def _get_event(self, args: Mapping[str, Any]) -> Result:
        a = GetEventArgs.model_validate(args)
        facts = self.ctx.by_id().get(a.event_id)
        if facts is None:
            return {"error": f"события {a.event_id!r} нет в данных матча"}
        return {"event": self._emit(facts)}

    def _find_player(self, args: Mapping[str, Any]) -> Result:
        a = FindPlayerArgs.model_validate(args)
        if a.query is None and a.number is None:
            return {"error": "нужна фамилия (query) или номер (number)"}
        notes: list[str] = []
        if self.ctx.has_lineup:
            players, total = self._players_from_lineup(a)
        elif a.query is None:
            return {"error": "номер можно искать только при наличии состава команд"}
        else:
            players, total = self._players_from_plates(a.query, a.side)
            if a.number is not None:
                notes.append("номер не учтён: состава команд нет")
        result: Result = {"players": players, "lineup_available": self.ctx.has_lineup}
        if total > len(players):
            notes.append(f"подошло игроков: {total}, показаны первые {len(players)}")
        if not total:
            notes.append("игрок не найден")
        if notes:
            result["note"] = "; ".join(notes)
        return result

    def _players_from_lineup(self, a: FindPlayerArgs) -> tuple[list[Result], int]:
        """Up to `MAX_PLAYERS` lineup players that fit; only the returned ones are remembered
        as handed out (the validator allows what the model really saw)."""
        fitting = [
            p
            for p in self.ctx.lineup
            if (a.side is None or p.side == a.side)
            and (a.number is None or p.number == a.number)
            and (a.query is None or name_matches(a.query, p.name))
        ]
        found: list[Result] = []
        for player in fitting[:MAX_PLAYERS]:
            self.seen_names.add(player.name)
            events = [
                self._emit(f)
                for f in self.ctx.events
                if f.event.team in (None, player.side)
                and any(p.name and name_matches(p.name, player.name) for p in f.players)
            ]
            found.append(
                {
                    "name": player.name,
                    "side": player.side,
                    "team_name": self.ctx.team_names[player.side],
                    "number": player.number,
                    "yellow_cards": sum(e.get("card") == "yellow" for e in events),
                    "red_cards": sum(e.get("card") == "red" for e in events),
                    "events": events,
                }
            )
        return found, len(fitting)

    def _players_from_plates(self, query: str, side: str | None) -> tuple[list[Result], int]:
        """Without a lineup: the names read from plates that match the query, one per name."""
        grouped: dict[str, list[EventFacts]] = {}
        for facts in self.ctx.events:
            if side is not None and facts.event.team not in (None, side):
                continue
            for p in facts.players:
                if p.name and name_matches(query, p.name):
                    grouped.setdefault(p.name, []).append(facts)
        found: list[Result] = [
            {
                "name": name,
                "side": None,
                "number": None,
                "from_plates_only": True,
                "yellow_cards": sum(f.card == "yellow" for f in dict.fromkeys(events)),
                "red_cards": sum(f.card == "red" for f in dict.fromkeys(events)),
                "events": [self._emit(f) for f in dict.fromkeys(events)],
            }
            for name, events in list(grouped.items())[:MAX_PLAYERS]
        ]
        return found, len(grouped)

    def _make_clip(self, args: Mapping[str, Any]) -> Result:
        a = MakeClipArgs.model_validate(args)
        if self.clip_maker is None:  # not offered, but a model may ask anyway
            return {"error": "клипы недоступны: видео матча не передано"}
        if a.event_id not in self.ctx.by_id():
            return {"error": f"события {a.event_id!r} нет в данных матча"}
        try:
            path = self.clip_maker(a.event_id, a.before_s, a.after_s)
        except ExportError as err:
            # The details (paths, FFmpeg output with video times) stay in the log, not in the
            # model's context.
            logger.warning("agent_clip_failed", stage="agent", error=str(err))
            return {"error": "клип не сделан"}
        self.clips.append(ClipMade(a.event_id, path))
        return {
            "event_id": a.event_id,
            "status": "clip_ready",
            "before_s": a.before_s,
            "after_s": a.after_s,
        }

    def _describe_episode(self, args: Mapping[str, Any]) -> Result:
        a = DescribeEpisodeArgs.model_validate(args)
        if self.episode_describer is None:  # not offered, but a model may ask anyway
            return {"error": "описание эпизода недоступно: нет видео матча или ключа видеомодели"}
        if a.event_id not in self.ctx.by_id():
            return {"error": f"события {a.event_id!r} нет в данных матча"}
        try:
            description = self.episode_describer(a.event_id)
        except (ExportError, ExternalServiceError) as err:
            # A budget refusal is not caught: it stops the question (as for the model itself).
            logger.warning("agent_episode_failed", stage="agent", error=str(err)[:200])
            return {"error": "описание эпизода не получено"}
        self.episode_described = True
        return {
            "event_id": a.event_id,
            "status": "described_from_clip_unverified",
            "description": description.text,
            "note": EPISODE_NOTE,
        }
