"""Pydantic models of `plate_names.json` and the stage parameters."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from aicoach.plate_names.layout import NameLayout
from aicoach.plates.schemas import PlateType, Team

SCHEMA_VERSION = 1
# Rules and parameters changed in a way the cache key does not see: raise this number.
RULES_VERSION = 1
VideoTime = Annotated[float, Field(ge=0, allow_inf_nan=False)]
# `player`: the only name of a goal or card plate; `in`/`out`: who came on / who left.
Role = Literal["player", "in", "out"]
# matched: found in the lineup; unchecked: cleaned text, not compared with a lineup; unrecognized:
# nothing readable, or the lineup has no player like it.
Status = Literal["matched", "unchecked", "unrecognized"]
Source = Literal["ocr", "ocr+lineup"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class PlateNamesConfig(_Frozen):
    """Parameters of the stage; they are part of the cache key.

    `min_similarity` was chosen on the first half of one match only (D-059).
    """

    scale: int = Field(default=3, ge=1, le=8)
    pad_px: int = Field(default=20, ge=0, le=100)
    min_similarity: float = Field(default=0.88, gt=0, le=1)
    layout: NameLayout = NameLayout()


class PlayerRead(_Frozen):
    """One name read from a plate. `name` is the lineup spelling when `matched`, the cleaned
    OCR text when `unchecked`, and None when `unrecognized`. `similarity` is the best one found
    in the lineup (also when it was too low), None when no lineup was used."""

    role: Role
    raw_text: str
    name: str | None
    source: Source
    similarity: float | None = Field(default=None, ge=0, le=1)
    status: Status


class PlateNames(_Frozen):
    """The names of one plate (one for a goal or card, two for a substitution)."""

    plate_id: str
    type: PlateType
    t_v: VideoTime
    half: Literal[1, 2] | None = None
    side: Team | None = None
    players: list[PlayerRead]


class PlateNamesStats(_Frozen):
    """Counters; shares in the report come from these."""

    n_plates: int
    n_names: int
    n_matched: int
    n_unchecked: int
    n_unrecognized: int


class PlateNamesResult(_Frozen):
    """The whole `plate_names.json`. `lineup` is the file name used (None: no lineup)."""

    schema_version: int = SCHEMA_VERSION
    cache_key: str
    plates_version: str
    lineup: str | None
    lineup_version: str | None
    config: PlateNamesConfig
    names: list[PlateNames]
    stats: PlateNamesStats
    warnings: list[str] = Field(default_factory=list)
