"""The schema of the corner tags (D-078): the six fields, their values and `corner_tags.json`.

Values are English codes in the files and in the model's answer. A value is also accepted in its
Russian name (`ru.CORNER_VALUE_LABELS`), so the person who fills the truth file may write either.
`unclear` ("не видно") is a legal value of every field: a model that cannot tell must say so.
"""

from pathlib import Path
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from aicoach.i18n import ru
from aicoach.teams.store import KIT_COLORS

RESULT_FILE: Final = "corner_tags.json"
SCHEMA_VERSION: Final = 1
UNCLEAR: Final = "unclear"

IS_CORNER: Final = "is_corner"
ZONE: Final = "zone"
ATTACKING_KIT: Final = "attacking_kit"
FIELDS: Final = (IS_CORNER, ATTACKING_KIT, "delivery", "zone", "first_contact", "outcome")
CHOICES: Final[dict[str, tuple[str, ...]]] = {
    IS_CORNER: ("yes", "no", UNCLEAR),
    ATTACKING_KIT: (*KIT_COLORS, UNCLEAR),
    "delivery": ("cross", "short", "low_ball", UNCLEAR),
    "zone": ("near_post", "center", "far_post", "edge_of_box", UNCLEAR),
    "first_contact": ("attack", "defence", "keeper", "nobody", UNCLEAR),
    "outcome": (
        "shot",
        "goal",
        "cleared",
        "keeper_caught",
        "corner_or_out",
        "attack_continues",
        UNCLEAR,
    ),
}


def _spelling(text: str) -> str:
    return "_".join(text.strip().casefold().replace("-", " ").split())


def _russian_names() -> dict[str, dict[str, str]]:
    """For each field: spelling of a Russian name (and of the code) -> code."""
    table: dict[str, dict[str, str]] = {}
    for field_name, codes in CHOICES.items():
        names = {_spelling(code): code for code in codes}
        for code, label in ru.CORNER_VALUE_LABELS[field_name].items():
            names[_spelling(label)] = code
        table[field_name] = names
    return table


_NAMES: Final = _russian_names()


def normalize_value(field_name: str, raw: str | None) -> str | None:
    """The code of a value written as a code or as its Russian name, any case; `None` when it
    is empty or not an allowed value of the field."""
    if raw is None or not raw.strip():
        return None
    return _NAMES[field_name].get(_spelling(raw))


class CornerFields(BaseModel):
    """The six fields of one corner; each is an allowed code (`unclear` when not visible)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    is_corner: str
    attacking_kit: str
    delivery: str
    zone: str
    first_contact: str
    outcome: str

    @field_validator(*FIELDS, mode="before")
    @classmethod
    def _allowed(cls, value: object, info: ValidationInfo) -> str:
        field_name = info.field_name or ""
        code = normalize_value(field_name, value) if isinstance(value, str) else None
        if code is None:
            msg = f"{field_name}: not an allowed value"
            raise ValueError(msg)
        return code

    @classmethod
    def all_unclear(cls) -> Self:
        """Every field `unclear`: what an unusable answer becomes."""
        return cls(**dict.fromkeys(FIELDS, UNCLEAR))


class CornerTag(BaseModel):
    """The tags of one corner event. `status`: `ok` (first answer), `retried` (the second answer
    was the usable one), `invalid` (no usable answer, every field `unclear`)."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    t_v: float
    half: int | None = None
    fields: CornerFields
    status: str
    model: str
    prompt_version: int
    before_s: float
    after_s: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = None
    created_at: str = ""


class CornerTagsFile(BaseModel):
    """`corner_tags.json`: the tags of the corners of a match, written after every new answer so
    that a stopped run goes on from there."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    match_id: str
    tags: list[CornerTag] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def read_tags_file(match_dir: Path) -> CornerTagsFile | None:
    """`corner_tags.json` of the match directory, or `None` when it is missing or not a valid
    file of this schema."""
    try:
        return CornerTagsFile.model_validate_json(
            (match_dir / RESULT_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
