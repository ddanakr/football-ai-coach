"""Windows and thresholds of the timeline logic.

Debt: CLAUDE.md wants thresholds in `configs/*.yaml` read through `Settings`; until `Settings`
exists they live in these pydantic models (see MEMORY.md, decision on scoreboard parameters).
"""

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class TimelineParams(_Frozen):
    """Windows and thresholds of the pure timeline logic (seconds of video)."""

    # A clock is "running" if the offset t_v - clock_s does not drift inside this window.
    run_window_s: float = Field(default=6.0, gt=0)
    run_tol_s: float = Field(default=1.0, ge=0)
    run_min_points: int = Field(default=4, ge=2)
    # Rolling median over the running readings; the median keeps real values (no averaging).
    offset_window_s: float = Field(default=15.0, gt=0)
    outlier_tol_s: float = Field(default=1.5, gt=0)
    # A change of the smoothed offset above this is recorded as a jump and never corrected.
    jump_tol_s: float = Field(default=2.0, gt=0)
    # A new score must hold this long to count (filters misread digits).
    score_hold_s: float = Field(default=3.0, gt=0)
    # Gaps in visibility shorter than this do not split a visible segment.
    segment_gap_s: float = Field(default=2.0, ge=0)
    # How far the time mapping may extrapolate beyond the first and last reading of a segment.
    max_extrapolation_s: float = Field(default=30.0, ge=0)
