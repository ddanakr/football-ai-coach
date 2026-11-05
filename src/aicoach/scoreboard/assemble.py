"""Assemble `ScoreboardResult` from per-second raw readings. Pure: no disk, no video."""

import math
from dataclasses import dataclass

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.config import TimelineParams
from aicoach.scoreboard.mapping import build_offset_segments
from aicoach.scoreboard.offsets import OffsetAnalysis, analyze_offsets, detect_jumps
from aicoach.scoreboard.schemas import (
    DecodeMode,
    OffsetJump,
    OffsetSegment,
    Roi,
    Sample,
    ScoreboardResult,
    Stats,
)
from aicoach.scoreboard.score import detect_score_changes, merge_visible_segments


@dataclass(frozen=True)
class RawReading:
    """What image reading found in one frame, before any time logic."""

    visible: bool
    clock_s: int | None = None
    score: tuple[int, int] | None = None
    conf: float = 0.0
    added_visible: bool = False
    added: int | None = None


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _visible_share_between_first_and_last(visible: list[bool]) -> float | None:
    if not any(visible):
        return None
    first = visible.index(True)
    last = len(visible) - 1 - visible[::-1].index(True)
    span = visible[first : last + 1]
    return sum(span) / len(span)


def _build_samples(
    t_v: list[float], readings: list[RawReading], analysis: OffsetAnalysis
) -> list[Sample]:
    """One `Sample` per second; a hidden board carries no clock, score or added time."""
    return [
        Sample(
            t_v=tv,
            visible=r.visible,
            clock_s=r.clock_s if r.visible else None,
            added=r.added if r.visible else None,
            added_visible=r.added_visible and r.visible,
            score=r.score if r.visible else None,
            conf=r.conf if r.visible else 0.0,
            offset_s=analysis.smoothed[i],
            clock_outlier=analysis.outlier[i],
        )
        for i, (tv, r) in enumerate(zip(t_v, readings, strict=True))
    ]


def _build_stats(
    readings: list[RawReading],
    analysis: OffsetAnalysis,
    jumps: list[OffsetJump],
    segments: list[OffsetSegment],
    video_duration_s: float,
    duration_warning: str | None,
) -> Stats:
    shown = [r for r in readings if r.visible]
    unreadable = sum(1 for r in shown if r.clock_s is None or r.score is None)
    return Stats(
        n_samples=len(readings),
        n_visible=len(shown),
        n_unreadable=unreadable,
        n_clock_unreadable=sum(1 for r in shown if r.clock_s is None),
        n_score_unreadable=sum(1 for r in shown if r.score is None),
        n_outliers=sum(analysis.outlier),
        n_running=sum(analysis.running),
        n_jumps=len(jumps),
        video_duration_s=video_duration_s,
        visible_share_between_first_and_last=_visible_share_between_first_and_last(
            [r.visible for r in readings]
        ),
        unreadable_share_in_visible=_share(unreadable, len(shown)),
        median_offset_by_segment=[seg.offset_s for seg in segments],
        duration_warning=duration_warning,
    )


def assemble_result(
    video: str,
    decode_mode: DecodeMode,
    sample_fps: float,
    roi: Roi,
    t_v: list[float],
    readings: list[RawReading],
    params: TimelineParams,
    video_duration_s: float,
    duration_warning: str | None = None,
) -> ScoreboardResult:
    """Turn raw readings into the full result: offsets, jumps, score changes, segments, stats.

    Seconds where the board is hidden carry no clock and no score. Nothing is corrected
    silently: misreads are flagged and offset jumps are listed. A score change is confirmed
    by visible seconds in the sequence of readings, not by wall time (see
    `detect_score_changes`), so a board that returns only rarely can confirm a value late.

    Raises:
        ScoreboardParseError: lengths differ, `t_v` is not finite or decreases, or `sample_fps`
            is not a positive finite number.
    """
    if len(t_v) != len(readings):
        msg = "t_v and readings must have the same length"
        raise ScoreboardParseError(msg)
    if not (math.isfinite(sample_fps) and sample_fps > 0):
        msg = "sample_fps must be positive and finite"
        raise ScoreboardParseError(msg)
    dt = 1.0 / sample_fps
    visible = [r.visible for r in readings]
    clock = [r.clock_s if r.visible else None for r in readings]
    score = [r.score if r.visible else None for r in readings]
    analysis = analyze_offsets(t_v, clock, params)
    jumps = detect_jumps(t_v, analysis.smoothed, params.jump_tol_s)
    segments = build_offset_segments(t_v, analysis.smoothed, jumps)
    return ScoreboardResult(
        video=video,
        decode_mode=decode_mode,
        sample_fps=sample_fps,
        roi=roi,
        samples=_build_samples(t_v, readings, analysis),
        visible_segments=merge_visible_segments(t_v, visible, params.segment_gap_s, dt),
        score_changes=detect_score_changes(t_v, score, params.score_hold_s, dt),
        offset_jumps=jumps,
        offset_segments=segments,
        stats=_build_stats(readings, analysis, jumps, segments, video_duration_s, duration_warning),
    )
