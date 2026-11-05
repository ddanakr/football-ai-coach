"""Pydantic models of `scoreboard.json` (the output of the scoreboard stage)."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Video seconds: finite and non-negative.
VideoTime = Annotated[float, Field(ge=0, allow_inf_nan=False)]
ScoreDirection = Literal["up", "down", "other"]
DecodeMode = Literal["opencv_gpu", "opencv_cpu"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Roi(_Frozen):
    """A rectangle in fractions of the frame size, so that it does not depend on resolution."""

    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)


class Sample(_Frozen):
    """What the scoreboard showed at one sampled second.

    `clock_s` is minutes*60+seconds of the match clock. `added` is N of the red "+N" block
    and `added_visible` says that the block is on screen (N may stay unknown). `offset_s` is
    the smoothed `t_v - clock_s`, set only where the clock was running. `clock_outlier`
    marks a reading that disagrees with its neighbours (a digit misread).
    """

    t_v: VideoTime
    visible: bool
    clock_s: int | None = None
    added: int | None = None
    added_visible: bool = False
    score: tuple[int, int] | None = None
    conf: float = Field(default=0.0, ge=0, le=1)
    offset_s: float | None = None
    clock_outlier: bool = False


class VisibleSegment(_Frozen):
    """A stretch of video where the scoreboard is on screen (short gaps are bridged)."""

    start: VideoTime
    end: VideoTime


class ScoreChange(_Frozen):
    """A confirmed change of the score, as a window: the scoreboard often disappears on
    close-ups, so the exact second is unknown. The change happened after `t_old_last_seen`
    (the last second with the old score on screen) and no later than `t_new_first_seen`
    (the first second with the new score on screen).
    """

    t_old_last_seen: VideoTime
    t_new_first_seen: VideoTime
    old: tuple[int, int]
    new: tuple[int, int]
    direction: ScoreDirection


class OffsetJump(_Frozen):
    """A jump of the smoothed offset between two neighbouring readings. Never corrected."""

    t_before: VideoTime
    t_after: VideoTime
    offset_before: float
    offset_after: float


class OffsetSegment(_Frozen):
    """A stretch between jumps where one offset holds: the basis of the time mapping."""

    start: VideoTime
    end: VideoTime
    clock_start_s: int
    clock_end_s: int
    offset_s: float
    n_readings: int


class Stats(_Frozen):
    """Counters that the report quotes; every number comes from the run."""

    n_samples: int
    n_visible: int
    n_unreadable: int
    n_clock_unreadable: int
    n_score_unreadable: int
    n_outliers: int
    n_running: int
    n_jumps: int
    video_duration_s: float
    visible_share_between_first_and_last: float | None
    unreadable_share_in_visible: float | None
    median_offset_by_segment: list[float]
    duration_warning: str | None = None


class ScoreboardResult(_Frozen):
    """The whole `scoreboard.json`."""

    video: str
    decode_mode: DecodeMode
    cache_key: str = ""
    sample_fps: float
    roi: Roi
    samples: list[Sample]
    visible_segments: list[VisibleSegment]
    score_changes: list[ScoreChange]
    offset_jumps: list[OffsetJump]
    offset_segments: list[OffsetSegment]
    stats: Stats
