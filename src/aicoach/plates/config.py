"""Thresholds and windows of the plates stage.

Debt (as in the scoreboard stage): CLAUDE.md wants thresholds in `configs/*.yaml` read through
`Settings`; until `Settings` exists they live in these pydantic models. The colour bounds
(blue, yellow, orange, green, red, header hue and saturation) and the sizes of the small pictures
used for comparison are still constants in `detect.py` and `logos.py`: the same debt as the
scoreboard's colour bounds (D-031).
"""

from pydantic import BaseModel, ConfigDict, Field

from aicoach.plates.layout import PlateLayout


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class DetectParams(_Frozen):
    """Per-frame thresholds: shares of matching pixels in the regions of the layout.

    The `*_ref` values are the shares seen on clean plates; they only scale the confidence.
    """

    # A frame is a candidate if this share of the block's frame is the plate's blue.
    min_anchor: float = Field(default=0.3, gt=0, le=1)
    anchor_ref: float = Field(default=0.95, gt=0, le=1)
    # Light pixels in the panel: a candidate without a panel is a blue object, not a plate.
    min_panel: float = Field(default=0.5, gt=0, le=1)
    panel_ref: float = Field(default=0.8, gt=0, le=1)
    panel_light: int = Field(default=180, ge=1, le=255)  # darkest channel above this is "light"
    # Goal: orange tag share.
    min_tag: float = Field(default=0.4, gt=0, le=1)
    tag_ref: float = Field(default=0.65, gt=0, le=1)
    # Card: yellow header and yellow tab shares.
    min_header_yellow: float = Field(default=0.4, gt=0, le=1)
    header_yellow_ref: float = Field(default=0.7, gt=0, le=1)
    min_tab: float = Field(default=0.5, gt=0, le=1)
    tab_ref: float = Field(default=0.9, gt=0, le=1)
    # Substitution: both triangles must be present.
    min_arrow: float = Field(default=0.10, gt=0, le=1)
    arrow_ref: float = Field(default=0.18, gt=0, le=1)
    # Shape of the triangles: the green one points up (more pixels in its lower half), the red
    # one points down (more in its upper half), by at least this factor. A circle gives ~1.
    arrow_balance: float = Field(default=1.5, gt=1)
    # A header of another strong colour (not yellow) is a card of another colour.
    min_header_other: float = Field(default=0.4, gt=0, le=1)
    header_other_ref: float = Field(default=0.6, gt=0, le=1)


class PlatesConfig(_Frozen):
    """Everything the stage needs besides the video path."""

    # Frames per second to read: 1 / sample_fps seconds between samples.
    sample_fps: float = Field(default=2.0, gt=0)
    hw_accel: bool = True
    layout: PlateLayout = PlateLayout()
    detect: DetectParams = DetectParams()
    # Detections of one type closer than this (seconds between two hits) are one plate.
    merge_gap_s: float = Field(default=3.0, ge=0)
    # A change of the panel picture above this (mean grey difference, 0..1) starts a new plate:
    # two plates of one type in a row must not merge. Within one plate the difference is ~0.003.
    split_diff: float = Field(default=0.02, gt=0, le=1)
    # A run with fewer hits is a flicker, not a plate.
    min_hits: int = Field(default=2, ge=1)
    # Logos closer than this (mean absolute difference of 24x24 colour patches) are one team.
    logo_max_dist: float = Field(default=0.07, gt=0, le=1)
    # The goal plate is matched to the last score change at most this long before it.
    max_goal_lag_s: float = Field(default=120.0, gt=0)
    # A plate whose confidence is below this is reported as low-confidence.
    low_conf: float = Field(default=0.8, ge=0, le=1)
    # Half boundaries: a plate within this of the first/last scoreboard clock belongs to a half.
    half_margin_s: float = Field(default=300.0, ge=0)
    duration_tol_s: float = Field(default=1.0, ge=0)
