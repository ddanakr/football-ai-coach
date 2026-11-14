"""The tools of the agent: pure functions over a `MatchContext`, answers as plain JSON.

The model never sees video times: events carry their id (`event_id`, the only thing the answer
may refer to), the half, the game minute and, when known, the team and the player names.
Filtering by game minute, half, side and player happens here, in code, not in the model.
`ToolBox` remembers which event ids and names it handed out, so the validator can check an
answer against exactly what the tools returned (D-061).
"""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal  # Any: JSON for and from the model

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aicoach.agent.context import EventFacts, MatchContext
from aicoach.agent.messages import ToolSpec
from aicoach.agent.names import name_matches
from aicoach.clips import DEFAULT_AFTER_S, DEFAULT_BEFORE_S
from aicoach.domain.events import EventType
from aicoach.errors import ExportError

logger = structlog.get_logger()

DEFAULT_LIMIT = 20
MAX_LIMIT = 50
MAX_PLAYERS = 5
MAX_CLIP_SIDE_S = 60.0
ACCURACY_NOTE = "Это события, найденные системой автоматически, а не официальная статистика."

ClipMaker = Callable[[str, float, float], Path]
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


@dataclass(frozen=True)
class ClipMade:
    """A clip a tool call produced (the interface shows it under the answer)."""

    event_id: str
    path: Path


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
    return result


def _has_player(facts: EventFacts, query: str) -> bool:
    return any(p.name and name_matches(query, p.name) for p in facts.players)


def _count_by(events: tuple[EventFacts, ...], key: Callable[[EventFacts], str]) -> dict[str, int]:
    return dict(sorted(Counter(key(f) for f in events).items()))


@dataclass
class ToolBox:
    """The tools over one match snapshot, for one question.

    Keeps `seen_event_ids` and `seen_names` (what the answer may refer to) and `clips`.
    `clip_maker` makes `make_clip` available; without it the tool is not offered.
    """

    ctx: MatchContext
    clip_maker: ClipMaker | None = None
    seen_event_ids: set[str] = field(default_factory=set)
    seen_names: set[str] = field(default_factory=set)
    clips: list[ClipMade] = field(default_factory=list)

    def specs(self) -> tuple[ToolSpec, ...]:
        """The tools to offer to the model."""
        base = (MATCH_INFO, QUERY_EVENTS, GET_EVENT, FIND_PLAYER)
        return (*base, MAKE_CLIP) if self.clip_maker is not None else base

    @property
    def allowed_names(self) -> set[str]:
        """Names the answer may use: handed out by the tools, plus the team names."""
        return self.seen_names | set(self.ctx.team_names.values())

    def call(self, name: str, args: Mapping[str, Any]) -> Result:
        """Run a tool; a wrong name or wrong arguments give `{"error": ...}` for the model."""
        handlers: dict[str, Callable[[Mapping[str, Any]], Result]] = {
            MATCH_INFO.name: self._match_info,
            QUERY_EVENTS.name: self._query_events,
            GET_EVENT.name: self._get_event,
            FIND_PLAYER.name: self._find_player,
        }
        if self.clip_maker is not None:
            handlers[MAKE_CLIP.name] = self._make_clip
        handler = handlers.get(name)
        if handler is None:
            return {"error": f"неизвестный инструмент {name!r}"}
        try:
            return handler(args)
        except ValidationError as err:
            # Field names and rules only: the model must fix its arguments.
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in err.errors()
            )
            return {"error": f"неверные аргументы: {problems}"}

    def _emit(self, facts: EventFacts) -> Result:
        """The event for the model; remembers its id and names as handed out."""
        self.seen_event_ids.add(facts.event.event_id)
        self.seen_names.update(p.name for p in facts.players if p.name)
        return event_json(facts, self.ctx.team_names)

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
            "note": ACCURACY_NOTE,
        }

    def _query_events(self, args: Mapping[str, Any]) -> Result:
        a = QueryEventsArgs.model_validate(args)
        if a.minute_from is not None and a.minute_to is not None and a.minute_from > a.minute_to:
            return {"error": "minute_from больше minute_to"}
        rows = [f for f in self.ctx.events if self._passes(f, a)]
        by_minute = a.minute_from is not None or a.minute_to is not None
        without_minute = sum(f.minute is None for f in rows) if by_minute else 0
        if by_minute:
            rows = [f for f in rows if f.minute is not None and self._in_minutes(f.minute, a)]
        shown = rows[: a.limit]
        result: Result = {
            "total": len(rows),
            "returned": len(shown),
            "events": [self._emit(f) for f in shown],
            "note": ACCURACY_NOTE,
        }
        if len(rows) > len(shown):
            result["truncated"] = True
        if without_minute:
            result["without_known_minute"] = without_minute  # not in `total`: no clock there
        return result

    @staticmethod
    def _passes(facts: EventFacts, a: QueryEventsArgs) -> bool:
        event = facts.event
        if a.type is not None and event.type != a.type:
            return False
        if a.half is not None and event.half != a.half:
            return False
        if a.side == "unknown" and event.team is not None:
            return False
        if a.side in _SIDES and event.team != a.side:
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
            return {"error": f"события {a.event_id!r} нет в каталоге"}
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
            return {"error": f"события {a.event_id!r} нет в каталоге"}
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
