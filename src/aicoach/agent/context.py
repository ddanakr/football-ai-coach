"""The facts of one match as the agent tools see them (read-only snapshot).

Built from `events_current` (user edits applied), `scoreboard.json` (game minutes),
`candidates.json` with `plate_names.json` (names of players) and the optional `lineups.csv`
(team names, players). Every file except the database is optional: what is missing only makes
the matching fields empty (`None`), it never stops the agent.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import structlog

from aicoach.domain.events import EventType
from aicoach.errors import AicoachError, StorageError
from aicoach.fusion.schemas import CandidatesResult
from aicoach.fusion.stage import RESULT_FILE as CANDIDATES_FILE
from aicoach.plate_names.lineup import LINEUP_FILE, Player, load_lineup, parse_team_names
from aicoach.plate_names.schemas import PlateNamesResult, PlayerRead
from aicoach.plate_names.stage import RESULT_FILE as PLATE_NAMES_FILE
from aicoach.scoreboard.config import ScoreboardConfig
from aicoach.scoreboard.mapping import clock_at
from aicoach.scoreboard.schemas import OffsetSegment, ScoreboardResult
from aicoach.scoreboard.stage import RESULT_FILE as SCOREBOARD_FILE
from aicoach.storage import CurrentEvent, connect, get_current_events
from aicoach.ui.data import load_time_bounds
from aicoach.ui.edits import with_derived_half

logger = structlog.get_logger()

# A broadcast writes minute N for the clock reading N-1:00..N-1:59 (a goal at 22:40 on the clock
# is "23'"), see the facts about the broadcast in MEMORY.md.
FIRST_HALF_MINUTES = 45
SECOND_HALF_MINUTES = 90
DEFAULT_TEAM_NAMES = {"home": "хозяева", "away": "гости"}

PlayerKey = tuple[EventType, float]


@dataclass(frozen=True)
class EventFacts:
    """One event with what the agent may say about it.

    `minute` is the game minute from the scoreboard clock (None where the clock is unknown);
    `players` are the names read from the plate the event came from (empty when none).
    The video time stays in `event.t_v` and is shown to the user by the interface only.
    """

    event: CurrentEvent
    minute: int | None
    minute_text: str | None
    players: tuple[PlayerRead, ...] = ()


@dataclass(frozen=True)
class MatchContext:
    """Everything the tools read; build it with `load_context` or by hand in tests."""

    match_id: str
    events: tuple[EventFacts, ...]
    team_names: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_TEAM_NAMES))
    has_lineup: bool = False
    lineup: tuple[Player, ...] = ()
    half_minutes: Mapping[int, tuple[int | None, int | None]] = field(default_factory=dict)

    def by_id(self) -> dict[str, EventFacts]:
        """The events by `event_id`."""
        return {facts.event.event_id: facts for facts in self.events}


def game_minute(clock_s: float | None) -> int | None:
    """Broadcast minute for a clock reading: `floor(clock / 60) + 1`; None without a clock."""
    if clock_s is None or not math.isfinite(clock_s) or clock_s < 0:
        return None
    return int(clock_s // 60) + 1


def minute_text(minute: int | None, half: int | None) -> str | None:
    """`23'`, and `45+2'` / `90+4'` for the stoppage time of a half (the clock runs on)."""
    if minute is None:
        return None
    if half == 1 and minute > FIRST_HALF_MINUTES:
        return f"{FIRST_HALF_MINUTES}+{minute - FIRST_HALF_MINUTES}'"
    if half == 2 and minute > SECOND_HALF_MINUTES:
        return f"{SECOND_HALF_MINUTES}+{minute - SECOND_HALF_MINUTES}'"
    return f"{minute}'"


@lru_cache(maxsize=4)
def _read_segments(path: Path, stamp: int) -> tuple[OffsetSegment, ...]:  # noqa: ARG001
    """Clock segments of `scoreboard.json`; `stamp` (modification time) only keys the cache."""
    board = ScoreboardResult.model_validate_json(path.read_text(encoding="utf-8"))
    return tuple(board.offset_segments)


def _segments(match_dir: Path) -> tuple[OffsetSegment, ...]:
    path = match_dir / SCOREBOARD_FILE
    try:
        return _read_segments(path, path.stat().st_mtime_ns)
    except (OSError, ValueError) as err:  # ValueError: bad JSON, bad schema, bad bytes
        logger.warning(
            "agent_no_clock", stage="agent", reason=type(err).__name__, path=str(match_dir)
        )
        return ()


@lru_cache(maxsize=4)
def _read_players(
    candidates: Path,
    names: Path,
    stamp: tuple[int, int],  # noqa: ARG001 - only part of the cache key
) -> dict[PlayerKey, tuple[PlayerRead, ...]]:
    events = CandidatesResult.model_validate_json(candidates.read_text(encoding="utf-8"))
    read = PlateNamesResult.model_validate_json(names.read_text(encoding="utf-8"))
    by_plate = {n.plate_id: n for n in read.names}
    kind_of_event = {EventType.GOAL: "goal", EventType.FOUL: "card", EventType.SUBST: "subst"}
    found: dict[PlayerKey, tuple[PlayerRead, ...]] = {}
    for candidate in events.candidates:
        for evidence in candidate.evidence:
            plate = by_plate.get(evidence.ref) if evidence.kind == "plate" else None
            # Same plate only if its time is the same in both files (ids are renumbered when
            # the plates stage runs again).
            if (
                plate is not None
                and plate.type == kind_of_event.get(candidate.type)
                and math.isclose(plate.t_v, evidence.t_v, abs_tol=1e-6)
            ):
                found[(candidate.type, candidate.t_v)] = tuple(plate.players)
                break
    return found


def _players_by_event(match_dir: Path) -> dict[PlayerKey, tuple[PlayerRead, ...]]:
    candidates, names = match_dir / CANDIDATES_FILE, match_dir / PLATE_NAMES_FILE
    try:
        stamp = (candidates.stat().st_mtime_ns, names.stat().st_mtime_ns)
        return _read_players(candidates, names, stamp)
    except (OSError, ValueError) as err:
        logger.warning(
            "agent_no_names", stage="agent", reason=type(err).__name__, path=str(match_dir)
        )
        return {}


def _lineup(match_dir: Path) -> tuple[tuple[Player, ...], dict[str, str]]:
    """The lineup players and team names; empty when there is no (valid) `lineups.csv`."""
    path = match_dir / LINEUP_FILE
    if not path.is_file():
        return (), {}
    try:
        players = tuple(load_lineup(path))
        names = parse_team_names(path.read_bytes())
    except (AicoachError, OSError) as err:
        logger.warning("agent_bad_lineup", stage="agent", reason=type(err).__name__)
        return (), {}
    return players, {str(side): name for side, name in names.items()}


def _half_minutes(
    segments: tuple[OffsetSegment, ...],
    max_extrapolation_s: float,
    match_dir: Path,
) -> dict[int, tuple[int | None, int | None]]:
    """Game minutes at the start and at the end of each half (from `candidates.json` bounds)."""
    try:
        bounds = CandidatesResult.model_validate_json(
            (match_dir / CANDIDATES_FILE).read_text(encoding="utf-8")
        ).half_bounds
    except (OSError, ValueError) as err:
        logger.warning("agent_no_bounds", stage="agent", reason=type(err).__name__)
        return {}
    result: dict[int, tuple[int | None, int | None]] = {}
    for key, span in bounds.items():
        first = game_minute(clock_at(segments, span.start, max_extrapolation_s))
        last = game_minute(clock_at(segments, span.end, max_extrapolation_s))
        result[int(key)] = (first, last)
    return result


def build_facts(
    events: list[CurrentEvent],
    segments: tuple[OffsetSegment, ...],
    players: Mapping[PlayerKey, tuple[PlayerRead, ...]],
    max_extrapolation_s: float,
) -> tuple[EventFacts, ...]:
    """Attach the game minute and the plate names to the events (pure)."""
    facts: list[EventFacts] = []
    for event in events:
        minute = game_minute(clock_at(segments, event.t_v, max_extrapolation_s))
        # A user-added event never gets a name; a moved or retyped one has another key.
        names = () if event.origin == "user" else players.get((event.type, event.t_v), ())
        facts.append(EventFacts(event, minute, minute_text(minute, event.half), names))
    return tuple(facts)


def load_context(db_path: Path, match_id: str, match_dir: Path) -> MatchContext:
    """Read the current state of the match.

    Raises:
        StorageError: The database file is missing or SQLite failed.
    """
    if not db_path.is_file():
        msg = f"database {db_path} does not exist"
        raise StorageError(msg)
    with connect(db_path) as conn:
        stored = get_current_events(conn, match_id)
    # Edited and added events have no half in the store: derive it as the viewer does.
    bounds = load_time_bounds(match_dir)
    events = [with_derived_half(e, bounds) for e in stored]
    max_extrapolation_s = ScoreboardConfig().timeline.max_extrapolation_s
    segments = _segments(match_dir)
    facts = build_facts(events, segments, _players_by_event(match_dir), max_extrapolation_s)
    lineup, team_names = _lineup(match_dir)
    return MatchContext(
        match_id=match_id,
        events=facts,
        team_names={**DEFAULT_TEAM_NAMES, **team_names},
        has_lineup=bool(lineup),
        lineup=lineup,
        half_minutes=_half_minutes(segments, max_extrapolation_s, match_dir),
    )
