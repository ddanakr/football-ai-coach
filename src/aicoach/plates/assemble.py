"""Build `plates.json` from tracks, logo labels and `scoreboard.json` (pure, no disk, no video)."""

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Literal

from aicoach.errors import PlatesError
from aicoach.plates.config import PlatesConfig
from aicoach.plates.schemas import (
    LogoCluster,
    Plate,
    PlatesResult,
    PlatesStats,
    RejectedCandidate,
    Team,
)
from aicoach.plates.teams import TeamDecision, decide_teams, goal_side, match_score_change
from aicoach.plates.tracks import Track, card_color_of, count_overlaps
from aicoach.scoreboard.schemas import (
    DecodeMode,
    OffsetSegment,
    Roi,
    ScoreboardResult,
    ScoreChange,
)

CROPS_DIR = "plate_crops"
_EPS = 1e-9


def half_of(t_v: float, segments: Sequence[OffsetSegment], margin_s: float) -> Literal[1, 2] | None:
    """The half of the match a video time belongs to, from the clock segments of scoreboard.json.

    The break is the widest gap between two clock segments. Times up to `margin_s` before the
    first clock and after the last one still count; further out (pre-match, studio) and a result
    with fewer than two segments give None.
    """
    if len(segments) < 2:
        return None
    ordered = sorted(segments, key=lambda s: s.start)
    gaps = [(ordered[i + 1].start - ordered[i].end, i) for i in range(len(ordered) - 1)]
    _, widest = max(gaps)
    middle = (ordered[widest].end + ordered[widest + 1].start) / 2
    if t_v < ordered[0].start - margin_s or t_v > ordered[-1].end + margin_s:
        return None
    return 1 if t_v < middle else 2


def visible_fraction(
    times: Sequence[float], board_times: Sequence[float], board_visible: Sequence[bool]
) -> float | None:
    """Share of `times` at which the scoreboard was visible, from its samples (the sample at or
    before each time). None if no time has a sample (times before the first or after the last
    sample plus its step are skipped)."""
    if not board_times:
        return None
    step = board_times[1] - board_times[0] if len(board_times) > 1 else 1.0
    seen: list[bool] = []
    for t in times:
        index = bisect_right(board_times, t + _EPS) - 1
        if index < 0 or t > board_times[-1] + step:
            continue
        seen.append(board_visible[index])
    return sum(seen) / len(seen) if seen else None


def _plate_id(index: int) -> str:
    return f"plate_{index + 1:03d}"


def _reject_id(index: int) -> str:
    return f"rejected_{index + 1:03d}"


def crop_path(name: str) -> str:
    """Path of a crop file relative to the match directory."""
    return f"{CROPS_DIR}/{name}"


def _rejected(rejects: Sequence[Track]) -> list[RejectedCandidate]:
    items: list[RejectedCandidate] = []
    for i, track in enumerate(rejects):
        reason = track.reject
        if reason is None:
            msg = "a rejected track must have a reason"
            raise PlatesError(msg)
        items.append(
            RejectedCandidate(
                id=_reject_id(i),
                reason=reason,
                t_first_seen=track.t_first,
                t_last_seen=track.t_last,
                anchor=round(max(h.reading.anchor for h in track.hits), 4),
                n_frames=len(track.hits),
                crop=crop_path(f"{_reject_id(i)}.png"),
            )
        )
    return items


def _team_evidence(
    labels: Sequence[int], matched: Sequence[int | None], changes: Sequence[ScoreChange]
) -> list[tuple[int, Team]]:
    """`(cluster, side)` for every goal plate that was matched to a score change with a side."""
    evidence: list[tuple[int, Team]] = []
    for label, change_index in zip(labels, matched, strict=True):
        side = goal_side(changes[change_index]) if change_index is not None else None
        if side is not None:
            evidence.append((label, side))
    return evidence


def _clusters(labels: Sequence[int], decision: TeamDecision) -> list[LogoCluster]:
    return [
        LogoCluster(
            id=c,
            n_plates=sum(1 for label in labels if label == c),
            team=decision.teams.get(c),
            home_votes=decision.home_votes[c],
            away_votes=decision.away_votes[c],
        )
        for c in range(len(set(labels)))
    ]


def _stats(
    items: Sequence[Plate],
    plates: Sequence[Track],
    rejects: Sequence[Track],
    n_samples: int,
    warnings: Sequence[str],
    duration_warning: str | None,
    low_conf: float,
) -> PlatesStats:
    by_type = {kind: sum(1 for p in items if p.type == kind) for kind in ("subst", "goal", "card")}
    return PlatesStats(
        n_samples=n_samples,
        n_candidate_frames=sum(len(t.hits) for t in (*plates, *rejects)),
        n_by_type=by_type,
        n_low_conf=sum(1 for p in items if p.conf < low_conf),
        n_rejected=len(rejects),
        n_overlaps=count_overlaps(list(plates)),
        warnings=list(warnings),
        duration_warning=duration_warning,
    )


@dataclass(frozen=True)
class _BoardContext:
    """What a plate needs from scoreboard.json: the result and its sample times and flags."""

    board: ScoreboardResult
    times: Sequence[float]
    visible: Sequence[bool]


def assemble_result(
    *,
    video: str,
    decode_mode: DecodeMode,
    sample_fps: float,
    roi: Roi,
    n_samples: int,
    plates: Sequence[Track],
    rejects: Sequence[Track],
    logo_labels: Sequence[int],
    board: ScoreboardResult,
    scoreboard_version: str,
    config: PlatesConfig,
    duration_warning: str | None = None,
) -> PlatesResult:
    """Make the result. `logo_labels[i]` is the logo cluster of `plates[i]`.

    Goal plates are matched to score changes, the matches decide which logo cluster is the home
    team (see `teams.decide_teams`); plates of a cluster without a decided side get no team.

    Raises:
        PlatesError: a track has no type (a refused candidate was passed as a plate).
    """
    changes = board.score_changes
    matched: list[int | None] = [
        match_score_change(t.t_first, changes, config.max_goal_lag_s) if t.kind == "goal" else None
        for t in plates
    ]
    decision = decide_teams(len(set(logo_labels)), _team_evidence(logo_labels, matched, changes))
    context = _BoardContext(
        board, [s.t_v for s in board.samples], [s.visible for s in board.samples]
    )
    items = [
        _plate(i, track, logo_labels[i], matched[i], decision, context, config)
        for i, track in enumerate(plates)
    ]
    stats = _stats(
        items, plates, rejects, n_samples, decision.warnings, duration_warning, config.low_conf
    )
    return PlatesResult(
        video=video,
        decode_mode=decode_mode,
        scoreboard_version=scoreboard_version,
        sample_fps=sample_fps,
        roi=roi,
        logo_clusters=_clusters(logo_labels, decision),
        plates=items,
        rejected=_rejected(rejects),
        stats=stats,
    )


def _plate(
    index: int,
    track: Track,
    label: int,
    change_index: int | None,
    decision: TeamDecision,
    context: _BoardContext,
    config: PlatesConfig,
) -> Plate:
    kind = track.kind
    if kind is None:
        msg = "a plate track must have a type"
        raise PlatesError(msg)
    pid = _plate_id(index)
    return Plate(
        id=pid,
        type=kind,
        card_color=card_color_of(track),
        t_first_seen=track.t_first,
        t_last_seen=track.t_last,
        half=half_of(track.t_first, context.board.offset_segments, config.half_margin_s),
        team=decision.teams.get(label),
        logo_cluster=label,
        conf=round(median(h.reading.conf for h in track.hits), 4),
        n_frames=len(track.hits),
        scoreboard_visible_frac=visible_fraction(
            [h.t_v for h in track.hits], context.times, context.visible
        ),
        score_change=change_index,
        crop=crop_path(f"{pid}.png"),
        logo_crop=crop_path(f"{pid}_logo.png"),
    )
