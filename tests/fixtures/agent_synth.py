"""Synthetic match, scripted language model for the agent tests. Names are invented."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from aicoach.agent.context import EventFacts, MatchContext, SegmentFacts, minute_text
from aicoach.agent.messages import (
    HistoryItem,
    ModelTurn,
    ToolCall,
    ToolSpec,
    Usage,
)
from aicoach.domain.events import EventType
from aicoach.plate_names.lineup import Player
from aicoach.plate_names.schemas import PlayerRead
from aicoach.speech.transcript import Mention, TranscriptSegment
from aicoach.storage import CurrentEvent

MATCH_ID = "m1"
HOME, AWAY = "Альфа", "Бета"
TESTOV = Player("home", "Тестов Иван", 9)
PROBNY = Player("home", "Пробный Сергей", 10)
OBRAZTSOV = Player("away", "Образцов Марат", 7)
NURLANOV = Player("away", "Нұрланов Ұлан", 11)


def read(name: str | None, role: str = "player", status: str = "matched") -> PlayerRead:
    """A name read from a plate."""
    return PlayerRead.model_validate(
        {
            "role": role,
            "raw_text": name or "",
            "name": name,
            "source": "ocr+lineup",
            "status": status if name else "unrecognized",
        }
    )


def make_event(
    number: int,
    kind: EventType,
    t_v: float,
    half: int | None,
    minute: int | None,
    team: str | None = None,
    players: Sequence[PlayerRead] = (),
    *,
    origin: str = "prediction",
    status: str = "auto",
    score: float | None = 1.0,
    card: str | None = None,
) -> EventFacts:
    """One event with its game minute and plate names."""
    event = CurrentEvent.model_validate(
        {
            "event_id": f"p:{number}",
            "match_id": MATCH_ID,
            "type": kind,
            "t_v": t_v,
            "half": half,
            "score": score,
            "status": status,
            "team": team,
            "source": "plates",
            "origin": origin,
            "verified": False,
            "deleted": False,
            "n_edits": 0,
        }
    )
    return EventFacts(event, minute, minute_text(minute, half), tuple(players), card)


def make_context(*, lineup: bool = True) -> MatchContext:
    """Seven events: two goals, a shot, a corner, a foul with a card name, two substitutions."""
    events = (
        make_event(1, EventType.SHOT, 600.0, 1, 5, "home"),
        make_event(2, EventType.GOAL, 1500.0, 1, 23, "home", [read(TESTOV.name)]),
        make_event(3, EventType.CORNER, 1900.0, 1, 30, "away"),
        make_event(
            4, EventType.FOUL, 2600.0, 1, 46, None, [read(OBRAZTSOV.name)], card="yellow"
        ),  # 45+1, yellow card by a plate, team unknown
        make_event(
            5,
            EventType.SUBST,
            4700.0,
            2,
            60,
            "away",
            [read(NURLANOV.name, "in"), read(None, "out")],
        ),
        make_event(
            6, EventType.GOAL, 6000.0, 2, 81, "away", [read("Ұлан Нұрланов", status="unchecked")]
        ),
        make_event(7, EventType.SHOT, 7000.0, 2, None, None),  # no clock there
    )
    return MatchContext(
        match_id=MATCH_ID,
        events=events,
        team_names={"home": HOME if lineup else "хозяева", "away": AWAY if lineup else "гости"},
        has_lineup=lineup,
        lineup=(TESTOV, PROBNY, OBRAZTSOV, NURLANOV) if lineup else (),
        half_minutes={1: (1, 47), 2: (46, 93)},
    )


def turn(
    text: str = "", *calls: tuple[str, dict[str, Any]], tokens: tuple[int, int] = (10, 5)
) -> ModelTurn:
    """A scripted model turn: text and/or tool calls."""
    tool_calls = tuple(ToolCall(name, args, f"c{i}") for i, (name, args) in enumerate(calls))
    return ModelTurn(text, tool_calls, Usage(*tokens))


class FakeLLM:
    """Plays a script of turns (or exceptions) and records what it was asked."""

    def __init__(
        self, script: Sequence[ModelTurn | Exception], model: str = "gemini-3.5-flash-lite"
    ) -> None:
        self._script = list(script)
        self.model = model
        self.requests: list[tuple[int, bool, list[str]]] = []
        self.histories: list[list[HistoryItem]] = []

    def generate(
        self,
        system: str,  # noqa: ARG002
        history: Sequence[HistoryItem],
        tools: Sequence[ToolSpec],
        *,
        allow_tools: bool = True,
    ) -> ModelTurn:
        """The next scripted turn."""
        self.histories.append(list(history))
        self.requests.append((len(history), allow_tools, [t.name for t in tools]))
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


ALIASED = Player("away", "Образцов Марат", 7, aliases=("Образец",))


def make_segment(
    number: int,
    t_start: float,
    text: str,
    half: int | None,
    minute: int | None,
    players: Sequence[Player] = (),
    *,
    speaker: str = "s0",
    language: str = "ru",
) -> SegmentFacts:
    """One stretch of commentary (5 s long) with its game minute."""
    mentions = [
        Mention(
            name=p.name,
            word=p.name.split()[0],
            side=p.side,
            number=p.number,
            via="name",
            similarity=1.0,
        )
        for p in players
    ]
    segment = TranscriptSegment(
        id=f"s:{number}",
        t_start=t_start,
        t_end=t_start + 5.0,
        half=half,  # type: ignore[arg-type]
        speaker=speaker,
        language=language,
        text=text,
        players=mentions,
    )
    return SegmentFacts(segment, minute, minute_text(minute, half))


def make_commentary_context() -> MatchContext:
    """The synthetic match with commentary around its first goal, a corner and a 2nd-half line."""
    base = make_context()
    segments = (
        make_segment(0, 1480.0, "Тестов выходит один на один", 1, 24, [TESTOV]),
        make_segment(1, 1502.0, "Гол! Тестов забивает, какой удар", 1, 25, [TESTOV]),
        make_segment(2, 1900.0, "Угловой у гостей, подача", 1, 31),
        make_segment(3, 5000.0, "Образец отдал пас на фланг", 2, 64, [ALIASED], speaker="s1"),
        make_segment(4, 6500.0, "Ұзақ соққы", 2, None, language="kk"),  # no clock there
    )
    lineup = (TESTOV, PROBNY, ALIASED, NURLANOV)
    return replace(base, segments=segments, lineup=lineup)
