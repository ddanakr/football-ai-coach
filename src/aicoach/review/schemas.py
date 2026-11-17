"""Pydantic models of `review_candidates.json` (the review layer, D-075)."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from aicoach.domain.events import EventType
from aicoach.fusion.schemas import Half, HalfSpan

SCHEMA_VERSION = 1
VideoTime = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class ReviewCandidate(_Frozen):
    """One extra shot detection of the model for the person who reviews the catalog.

    `id` is `r:N` (N from 1 in time order), `minute` the game minute from the scoreboard clock
    (None where the clock is unknown), `score` the confidence of the model.
    """

    id: str = Field(pattern=r"^r:[0-9]+$")
    type: EventType
    t_v: VideoTime
    half: Half
    minute: int | None = None
    score: float = Field(ge=0, allow_inf_nan=False)


class ReviewCandidatesResult(_Frozen):
    """The whole `review_candidates.json`.

    The three `*_version` fields are the sha256 of the input files; `n_model` counts the shot
    detections of the model, `n_existing` those left out because the catalog already has a shot
    within the merge radius, `n_merged` the ones that collapsed into a stronger neighbour and
    `n_outside_halves` the ones in the break or the studio. `counts` is candidates by type and half.
    """

    schema_version: int = SCHEMA_VERSION
    match_id: str
    cache_key: str
    events_version: str
    model_version: str
    scoreboard_version: str
    model_file: str
    merge_radius_s: float
    n_model: int
    n_existing: int
    n_merged: int
    n_outside_halves: int
    half_bounds: dict[Literal["1", "2"], HalfSpan]
    candidates: list[ReviewCandidate]
    counts: dict[str, dict[str, int]]
