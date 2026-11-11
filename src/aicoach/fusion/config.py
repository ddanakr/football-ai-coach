"""Parameters of the fusion rules. Every default is a starting value, not tuned on data,
except the corner threshold (D-053).

All numbers of the rules live here (CLAUDE.md section 6), not in the rule code. Thresholds and
windows are to be chosen on the first half only (D-008); until then the values below are guesses
taken from the owner's observations of the broadcast and from the task text.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from aicoach.errors import ConfigError

_Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
_Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
ModelClass = Literal["shot", "corner", "foul"]
MODEL_CLASSES: tuple[ModelClass, ...] = ("shot", "corner", "foul")


def _default_thresholds() -> dict[ModelClass, float]:
    return {"shot": 0.2, "corner": 0.05, "foul": 0.2}


class FuseConfig(BaseModel):
    """Parameters of the rules; all defaults are "starting, not tuned" (D-047)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Goal: a score growth is the event; a goal plate within the window confirms it.
    goal_plate_window_s: _Seconds = 120.0
    goal_confirmed_score: _Score = 1.0
    goal_unconfirmed_score: _Score = 0.8
    # A goal plate without a score change: the plate comes late (11.5-46 s on one match).
    goal_plate_only_lead_s: _Seconds = 30.0
    goal_plate_only_score: _Score = 0.5

    # Model events: only these classes, each with its own confidence threshold. The corner value
    # is chosen on half 1 of one match (D-053, 4 corners: 4 of 4 found at 0.05 against 2 at 0.2);
    # shot and foul kept the starting 0.2 (no gain of 2 true positives on the grid).
    model_thresholds: dict[ModelClass, _Score] = Field(default_factory=_default_thresholds)
    model_fps: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 25.0
    merge_radius_s: _Seconds = 3.0
    # A model shot this close to a goal is the goal itself (one mark per goal by the protocol).
    shot_goal_radius_s: _Seconds = 10.0

    # Card plate: a model foul in [plate - before, plate + after] is lifted; otherwise a foul
    # is added `lead` seconds before the plate. The 22 s is measured, not tuned on the metric: on
    # 3 card plates of half 1 of one match the nearest earlier foul lies 17.7-25.9 s before the
    # plate (D-053).
    card_foul_window_before_s: _Seconds = 40.0
    card_foul_window_after_s: _Seconds = 5.0
    card_foul_boost_score: _Score = 0.9
    card_foul_lead_s: _Seconds = 22.0
    card_foul_plate_only_score: _Score = 0.5

    # Half bounds come from the clock segments of scoreboard.json: the first clock reading of a
    # half minus the start margin, the last one plus the end margin.
    half_start_margin_s: _Seconds = 15.0
    half_end_margin_s: _Seconds = 15.0

    @model_validator(mode="after")
    def _every_model_class_has_a_threshold(self) -> "FuseConfig":
        if set(self.model_thresholds) != set(MODEL_CLASSES):
            msg = f"model_thresholds needs exactly the classes {list(MODEL_CLASSES)}"
            raise ValueError(msg)
        return self


def load_fuse_config(path: Path | None) -> FuseConfig:
    """Read a JSON file with `FuseConfig` fields (the rest keep their defaults).

    A nested field is replaced whole: `model_thresholds` must name all three classes.

    Args:
        path: The file, or None for the defaults.

    Raises:
        ConfigError: the file cannot be read, is not JSON, or holds an unknown or invalid value.
    """
    if path is None:
        return FuseConfig()
    try:
        return FuseConfig.model_validate_json(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError) as err:
        msg = f"cannot read the fusion config {path}: {err}"
        raise ConfigError(msg) from err
    except ValidationError as err:
        first = err.errors()[0]
        loc = ".".join(str(part) for part in first["loc"]) or "<root>"
        msg = f"invalid fusion config {path}: {loc}: {first['msg']}"
        raise ConfigError(msg) from err
