"""Comparison of two T-DEED output files: the check that a run reproduces the frozen predictions.

Both files are read through the schema of `aicoach.spotting`. Predictions are matched per class by
time (within a tolerance), then their confidences are compared (within another). A window and an
edge margin say which part of the video is compared: a segment run lacks the frames before its start
and after its end, so its first and last seconds are not expected to agree.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

from aicoach.errors import SpottingError
from aicoach.fusion.config import FuseConfig
from aicoach.i18n import ru
from aicoach.spotting.convert import LABEL_TO_EVENT
from aicoach.spotting.schemas import TDeedFile
from aicoach.tdeed.config import FPS
from aicoach.tdeed.timecode import format_timecode

DEFAULT_TIME_TOL_S = 0.1
DEFAULT_SCORE_TOL = 0.01
DEFAULT_EDGE_MARGIN_S = 10.0
_EPS = 1e-9  # a tolerance is inclusive; this keeps 0.51 - 0.50 within 0.01 despite float rounding


@dataclass(frozen=True)
class ClassComparison:
    """One class: the counts, the pairs found, and how the pairs differ."""

    label: str
    count_a: int
    count_b: int
    matched: int  # pairs within the time tolerance
    score_differs: int  # of those, pairs whose confidences differ by more than the tolerance
    max_score_diff: float  # over the matched pairs (0.0 when there are none)

    @property
    def same(self) -> bool:
        """All predictions paired, and all confidences within the tolerance."""
        return self.count_a == self.count_b == self.matched and self.score_differs == 0


@dataclass(frozen=True)
class Comparison:
    """The result: the window really compared and one entry per class that occurs."""

    window_s: tuple[float, float]
    classes: list[ClassComparison] = field(default_factory=list)

    @property
    def same(self) -> bool:
        """True when every class agrees (also when nothing was compared)."""
        return all(c.same for c in self.classes)

    def differences(self) -> list[str]:
        """What differs, one short phrase per finding; empty when `same`."""
        found: list[str] = []
        for c in self.classes:
            if c.count_a != c.count_b:
                found.append(f"{c.label}: {c.count_a} против {c.count_b} срабатываний")
            unmatched = max(c.count_a, c.count_b) - c.matched
            if unmatched and c.count_a == c.count_b:
                found.append(f"{c.label}: {unmatched} без пары по времени")
            if c.score_differs:
                found.append(
                    f"{c.label}: {c.score_differs} с разницей оценки больше допуска "
                    f"(наибольшая {c.max_score_diff:.4f})"
                )
        return found


def filter_by_fuse_thresholds(data: TDeedFile, config: FuseConfig | None = None) -> TDeedFile:
    """The predictions that `fuse` would take: only shots, corners and fouls, each with a
    confidence at or above the threshold of its class (`FuseConfig.model_thresholds`).

    Goals, substitutions and the classes without a counterpart are not used by `fuse` from the
    model, so they are dropped. Duplicates are not merged: this is a filter, not the whole rule.
    """
    thresholds = {str(k): v for k, v in (config or FuseConfig()).model_thresholds.items()}
    kept = []
    for p in data.predictions:
        event_type = LABEL_TO_EVENT[p.label]
        if event_type is not None and p.confidence >= thresholds.get(event_type.value, math.inf):
            kept.append(p)
    return TDeedFile(predictions=tuple(kept))


def verdict(result: "Comparison") -> str:
    """One line: `совпадает`, or `различия: ...`."""
    if result.same:
        return ru.TDEED_COMPARE_SAME
    return f"{ru.TDEED_COMPARE_DIFF}: " + "; ".join(result.differences())


def _times(
    data: TDeedFile, fps: float, window: tuple[float, float]
) -> dict[str, list[tuple[float, float]]]:
    """`(time, confidence)` per class inside the window (bounds included), by time."""
    low, high = window
    by_class: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for p in data.predictions:
        t = p.frame / fps
        if low <= t <= high:
            by_class[p.label.value].append((t, p.confidence))
    for items in by_class.values():
        items.sort()
    return by_class


def _pair(
    a: list[tuple[float, float]], b: list[tuple[float, float]], time_tol: float
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Pairs within `time_tol`, the closest first; each prediction is used once."""
    candidates = sorted(
        (abs(x[0] - y[0]), i, j)
        for i, x in enumerate(a)
        for j, y in enumerate(b)
        if abs(x[0] - y[0]) <= time_tol + _EPS
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs = []
    for _, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((a[i], b[j]))
    return pairs


def _intersection(a: TDeedFile, b: TDeedFile, fps: float) -> tuple[float, float]:
    spans = [
        (min(p.frame for p in f.predictions) / fps, max(p.frame for p in f.predictions) / fps)
        for f in (a, b)
        if f.predictions
    ]
    if not spans:
        return (0.0, 0.0)
    # One empty file: the span of the other, so its predictions count as unmatched.
    return (max(s[0] for s in spans), min(s[1] for s in spans))


def compare_tdeed(
    a: TDeedFile,
    b: TDeedFile,
    *,
    time_tol_s: float = DEFAULT_TIME_TOL_S,
    score_tol: float = DEFAULT_SCORE_TOL,
    window_s: tuple[float, float] | None = None,
    edge_margin_s: float = DEFAULT_EDGE_MARGIN_S,
    fps: float = FPS,
) -> Comparison:
    """Compare two outputs class by class.

    Args:
        a: The reference (for example the frozen whole-match file).
        b: The file to check (for example a segment run).
        time_tol_s: Predictions of one class within this many seconds are a pair (inclusive).
        score_tol: Paired confidences may differ by this much (inclusive).
        window_s: The part of the video to compare, in seconds; default the overlap of the time
            ranges of the two files.
        edge_margin_s: Seconds at each end of the window that are left out.
        fps: The frame rate that `frame` counts at.

    Raises:
        SpottingError: A tolerance, the margin or the frame rate is negative, not finite or not
            positive (for the frame rate), or the margin leaves no window.
    """
    for name, value in (
        ("time tolerance", time_tol_s),
        ("score tolerance", score_tol),
        ("edge margin", edge_margin_s),
    ):
        if not math.isfinite(value) or value < 0:
            msg = f"{name} must be a finite number, not negative"
            raise SpottingError(msg)
    if not math.isfinite(fps) or fps <= 0:
        msg = "fps must be a positive number"
        raise SpottingError(msg)
    if window_s is None and not a.predictions and not b.predictions:
        return Comparison((0.0, 0.0))  # nothing in either file: nothing to compare
    low, high = window_s if window_s is not None else _intersection(a, b, fps)
    low, high = low + edge_margin_s, high - edge_margin_s
    if high < low:
        msg = "the edge margin leaves no window to compare"
        raise SpottingError(msg)
    ta, tb = _times(a, fps, (low, high)), _times(b, fps, (low, high))
    classes = []
    for label in sorted(set(ta) | set(tb)):
        left, right = ta.get(label, []), tb.get(label, [])
        pairs = _pair(left, right, time_tol_s)
        diffs = [abs(x[1] - y[1]) for x, y in pairs]
        classes.append(
            ClassComparison(
                label,
                len(left),
                len(right),
                len(pairs),
                sum(1 for d in diffs if d > score_tol + _EPS),
                max(diffs, default=0.0),
            )
        )
    return Comparison((low, high), classes)


def format_comparison(result: Comparison) -> str:
    """A table per class and the verdict line: `совпадает` or `различия: ...`."""
    low, high = result.window_s
    lines = [
        f"окно сравнения {format_timecode(low)}-{format_timecode(high)} (после отступа от краёв)",
        f"{'класс':<20}{'A':>6}{'B':>6}{'пар':>6}{'оц.разн.':>10}{'макс.разн.':>12}",
    ]
    lines += [
        f"{c.label:<20}{c.count_a:>6}{c.count_b:>6}{c.matched:>6}{c.score_differs:>10}"
        f"{c.max_score_diff:>12.4f}"
        for c in result.classes
    ]
    lines.append(verdict(result))
    return "\n".join(lines) + "\n"
