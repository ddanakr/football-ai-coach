"""Score history and visibility segments. Pure functions over plain lists."""

from dataclasses import dataclass

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.schemas import ScoreChange, ScoreDirection, VisibleSegment

_EPS = 1e-9  # float slack for comparisons of seconds


@dataclass
class _Run:
    score: tuple[int, int]
    first: float
    last: float
    count: int  # number of readings (visible seconds) in the run


def _direction(old: tuple[int, int], new: tuple[int, int]) -> ScoreDirection:
    delta = sum(new) - sum(old)
    if delta > 0:
        return "up"
    return "down" if delta < 0 else "other"


def _runs(t_v: list[float], scores: list[tuple[int, int] | None]) -> list[_Run]:
    """Consecutive readings with the same score; unreadable or hidden seconds are skipped."""
    runs: list[_Run] = []
    for tv, score in zip(t_v, scores, strict=True):
        if score is None:
            continue
        if runs and runs[-1].score == score:
            runs[-1].last = tv
            runs[-1].count += 1
        else:
            runs.append(_Run(score, tv, tv, 1))
    return runs


def detect_score_changes(
    t_v: list[float],
    scores: list[tuple[int, int] | None],
    hold_s: float,
    sample_dt: float,
) -> list[ScoreChange]:
    """Find confirmed score changes and report each as a window.

    A new value is confirmed when it is read on at least `hold_s` seconds of visible video in
    a row (`count * sample_dt >= hold_s`, counting only seconds where the score was read).
    Shorter runs (misread digits) are dropped and the runs around them are merged. Hidden
    seconds do not break a run, so a board that disappears for a minute and returns with the
    same score changes nothing. Goals (`up`) and reversals such as a VAR cancellation (`down`)
    are both reported. A change is `t_old_last_seen` (last reading of the old score) to
    `t_new_first_seen` (first reading of the new one).

    Raises:
        ScoreboardParseError: lengths differ or `hold_s` / `sample_dt` is not positive.
    """
    if len(t_v) != len(scores):
        msg = "t_v and scores must have the same length"
        raise ScoreboardParseError(msg)
    if not (hold_s > 0 and sample_dt > 0):
        msg = "hold_s and sample_dt must be positive"
        raise ScoreboardParseError(msg)
    confirmed: list[_Run] = []
    for run in _runs(t_v, scores):
        if run.count * sample_dt + _EPS < hold_s:
            continue
        if confirmed and confirmed[-1].score == run.score:
            confirmed[-1].last = run.last
            confirmed[-1].count += run.count
        else:
            confirmed.append(run)
    return [
        ScoreChange(
            t_old_last_seen=old.last,
            t_new_first_seen=new.first,
            old=old.score,
            new=new.score,
            direction=_direction(old.score, new.score),
        )
        for old, new in zip(confirmed, confirmed[1:], strict=False)
    ]


def merge_visible_segments(
    t_v: list[float], visible: list[bool], gap_s: float, sample_dt: float
) -> list[VisibleSegment]:
    """Join visible samples into segments; a gap of at most `gap_s` seconds does not split one.

    `end` is the time of the last visible sample of the segment. Visibility is only a
    measurement of the board: it is not a replay flag.

    Raises:
        ScoreboardParseError: lengths differ.
    """
    if len(t_v) != len(visible):
        msg = "t_v and visible must have the same length"
        raise ScoreboardParseError(msg)
    segments: list[VisibleSegment] = []
    start: float | None = None
    last = 0.0
    for tv, shown in zip(t_v, visible, strict=True):
        if not shown:
            continue
        if start is not None and tv - last > gap_s + sample_dt + _EPS:
            segments.append(VisibleSegment(start=start, end=last))
            start = None
        if start is None:
            start = tv
        last = tv
    if start is not None:
        segments.append(VisibleSegment(start=start, end=last))
    return segments
