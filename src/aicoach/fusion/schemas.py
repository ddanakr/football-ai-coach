"""Pydantic models of `candidates.json` (the output of the fusion stage)."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from aicoach.domain.events import EventType
from aicoach.fusion.config import FuseConfig

SCHEMA_VERSION = 1
VideoTime = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Half = Literal[1, 2]
Team = Literal["home", "away"]
# Where the time of a candidate comes from. `plates_only` is a plate without the signal that
# normally makes the event (a goal plate without a score change, a card plate without a foul).
Source = Literal[
    "plates", "scoreboard", "scoreboard+plates", "plates_only", "model", "model+plates"
]
EvidenceKind = Literal["score_change", "plate", "model"]
DropReason = Literal["outside_halves", "shot_near_goal"]
Mode = Literal["with_model", "no_model"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Evidence(_Frozen):
    """One signal behind a candidate: `ref` is `score_changes[i]`, a plate id or `model:<class>`."""

    kind: EvidenceKind
    ref: str
    t_v: VideoTime
    score: float | None = None


class Candidate(_Frozen):
    """One event of the catalog before it is stored. `rule` names the rule that made it."""

    type: EventType
    t_v: VideoTime
    half: Half | None = None
    team: Team | None = None
    score: float = Field(ge=0, allow_inf_nan=False)
    status: str = "auto"
    source: Source
    rule: str
    evidence: list[Evidence]


class Dropped(_Frozen):
    """A candidate that was thrown away, kept for the owner's review."""

    type: EventType
    t_v: VideoTime
    source: Source
    reason: DropReason


class HalfSpan(_Frozen):
    """Bounds of a half in video seconds (both ends included)."""

    start: VideoTime
    end: VideoTime


class CandidatesResult(_Frozen):
    """The whole `candidates.json`. `run_id` is the run written to SQLite (None: nothing was
    stored, the result is empty). `counts` is events by type and by half."""

    schema_version: int = SCHEMA_VERSION
    match_id: str
    cache_key: str
    run_id: str | None
    mode: Mode
    scoreboard_version: str
    plates_version: str
    model_version: str | None
    model_file: str | None
    config: FuseConfig
    half_bounds: dict[Literal["1", "2"], HalfSpan]
    candidates: list[Candidate]
    dropped: list[Dropped]
    warnings: list[str]
    counts: dict[str, dict[str, int]]
