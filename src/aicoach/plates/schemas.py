"""Pydantic models of `plates.json` (the output of the plates stage)."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from aicoach.scoreboard.schemas import DecodeMode, Roi

VideoTime = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PlateType = Literal["subst", "goal", "card"]
CardColor = Literal["yellow", "red"]
Team = Literal["home", "away"]
RejectReason = Literal["no_panel", "no_marker", "too_short"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class LogoCluster(_Frozen):
    """A group of similar logos: one team. `team` comes only from goal plates matched to score
    changes (`home_votes`: the score of the left team grew, `away_votes`: the other one's)."""

    id: int
    n_plates: int
    team: Team | None
    home_votes: int
    away_votes: int


class Plate(_Frozen):
    """One plate on screen. `t_first_seen` and `t_last_seen` are the first and last sampled
    second where it was seen. `team` is None if the logo could not be tied to a side.
    `scoreboard_visible_frac` is the share of the plate's seconds with the scoreboard visible
    (None if the scoreboard has no sample there). `crop` and `logo_crop` are paths relative to
    the match directory. `score_change` is the index into scoreboard.json's `score_changes` of
    the change a goal plate was matched to."""

    id: str
    type: PlateType
    card_color: CardColor | None = None
    t_first_seen: VideoTime
    t_last_seen: VideoTime
    half: Literal[1, 2] | None = None
    team: Team | None = None
    logo_cluster: int | None = None
    conf: float = Field(ge=0, le=1)
    n_frames: int
    scoreboard_visible_frac: float | None = Field(default=None, ge=0, le=1)
    score_change: int | None = None
    crop: str
    logo_crop: str


class RejectedCandidate(_Frozen):
    """Something that looked like a plate but was not accepted; kept for the owner's review."""

    id: str
    reason: RejectReason
    t_first_seen: VideoTime
    t_last_seen: VideoTime
    anchor: float = Field(ge=0, le=1)
    n_frames: int
    crop: str


class PlatesStats(_Frozen):
    """Counters that the report quotes; every number comes from the run."""

    n_samples: int
    n_candidate_frames: int
    n_by_type: dict[str, int]
    n_low_conf: int
    n_rejected: int
    # Pairs of plates of different types that overlap in time (should be 0).
    n_overlaps: int
    warnings: list[str] = Field(default_factory=list)
    duration_warning: str | None = None


class PlatesResult(_Frozen):
    """The whole `plates.json`."""

    video: str
    decode_mode: DecodeMode
    cache_key: str = ""
    scoreboard_version: str
    sample_fps: float
    roi: Roi
    logo_clusters: list[LogoCluster]
    plates: list[Plate]
    rejected: list[RejectedCandidate]
    stats: PlatesStats
