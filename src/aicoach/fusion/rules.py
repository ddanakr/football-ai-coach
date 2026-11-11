"""Fusion rules: scoreboard, plates and (optionally) model events become catalog candidates.

Pure functions: no disk, no database. Rules (D-047):

1. A substitution plate is a `subst` at its first second.
2. A score growth is a `goal` at the first second of the new score; a goal plate within a
   window after it confirms it; a goal plate without a growth gives a lagged `goal`; growth that
   is taken back later (VAR) is a `shot` with status `disallowed`.
3. Model events: only `shot`, `corner`, `foul`, each over its threshold; a shot near a goal is
   the goal itself and is dropped.
4. A card plate lifts a model foul before it or adds a foul by itself.
5. Events outside the halves are dropped; the half is set from the bounds.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from aicoach.domain.events import EventType
from aicoach.eval.schemas import Prediction
from aicoach.fusion.config import FuseConfig
from aicoach.fusion.halves import half_at, half_bounds
from aicoach.fusion.schemas import Candidate, Dropped, Evidence, Half, HalfSpan
from aicoach.plates.schemas import Plate, PlatesResult
from aicoach.plates.teams import goal_side
from aicoach.scoreboard.schemas import ScoreboardResult, ScoreChange
from aicoach.spotting.convert import merge_duplicates

# Bump when a rule changes, so that a cached candidates.json made by older rules is not reused.
RULES_VERSION = 1

NO_MODEL_WARNING = (
    "no model predictions file: only goals, substitutions and card-based fouls are produced "
    "(no shots, corners or other fouls)"
)


@dataclass(frozen=True)
class FusionOutput:
    """What the rules produce: the events, what was thrown away, the warnings and the bounds."""

    candidates: list[Candidate]
    dropped: list[Dropped]
    warnings: list[str]
    half_bounds: dict[Half, HalfSpan]


def _plate_evidence(plate: Plate) -> Evidence:
    return Evidence(kind="plate", ref=plate.id, t_v=plate.t_first_seen, score=plate.conf)


def _cancelled_growths(changes: Sequence[ScoreChange], warnings: list[str]) -> set[int]:
    """Positions of growths that a later decrease took back (the score went up, then back)."""
    cancelled: set[int] = set()
    growths: list[int] = []
    for pos, change in enumerate(changes):
        if change.direction == "up":
            growths.append(pos)
        elif change.direction == "down":
            undone = next(
                (
                    g
                    for g in reversed(growths)
                    if g not in cancelled
                    and changes[g].old == change.new
                    and changes[g].new == change.old
                ),
                None,
            )
            if undone is None:
                warnings.append(
                    f"score fell from {change.old} to {change.new} without a matching growth: "
                    "ignored"
                )
            else:
                cancelled.add(undone)
        else:
            warnings.append(f"score changed from {change.old} to {change.new}: ignored")
    return cancelled


def _claim_plate(
    change: ScoreChange, plates: Sequence[Plate], claimed: set[str], window_s: float
) -> Plate | None:
    """The earliest unclaimed goal plate from the start of the change window up to `window_s`
    after its end; one plate confirms one goal."""
    for plate in plates:
        if plate.id in claimed:
            continue
        if change.t_old_last_seen <= plate.t_first_seen <= change.t_new_first_seen + window_s:
            claimed.add(plate.id)
            return plate
    return None


def _goal_from_change(
    index: int,
    change: ScoreChange,
    plate: Plate | None,
    cancelled: bool,
    config: FuseConfig,
    warnings: list[str],
) -> Candidate:
    side = goal_side(change)
    if side is None:
        warnings.append(f"score grew from {change.old} to {change.new}: team is null")
    evidence = [
        Evidence(kind="score_change", ref=f"score_changes[{index}]", t_v=change.t_new_first_seen)
    ]
    if plate is not None:
        evidence.append(_plate_evidence(plate))
        if side is not None and plate.team is not None and plate.team != side:
            warnings.append(
                f"plate {plate.id} shows team {plate.team} but the score grew for {side} "
                "(own goal?): team taken from the scoreboard"
            )
    return Candidate(
        type=EventType.SHOT if cancelled else EventType.GOAL,
        t_v=change.t_new_first_seen,
        team=side,
        score=config.goal_confirmed_score if plate else config.goal_unconfirmed_score,
        status="disallowed" if cancelled else "auto",
        source="scoreboard+plates" if plate else "scoreboard",
        rule="shot_disallowed" if cancelled else "goal_score_change",
        evidence=evidence,
    )


def _goal_plate_only(plate: Plate, config: FuseConfig, warnings: list[str]) -> Candidate:
    warnings.append(f"goal plate {plate.id} has no score change: goal added from the plate alone")
    return Candidate(
        type=EventType.GOAL,
        t_v=max(0.0, plate.t_first_seen - config.goal_plate_only_lead_s),
        team=plate.team,
        score=config.goal_plate_only_score,
        source="plates_only",
        rule="goal_plates_only",
        evidence=[_plate_evidence(plate)],
    )


def goal_candidates(
    board: ScoreboardResult, goal_plates: Sequence[Plate], config: FuseConfig, warnings: list[str]
) -> list[Candidate]:
    """Goals (and disallowed shots) from score growths, then lagged goals from unclaimed plates."""
    indexed = sorted(enumerate(board.score_changes), key=lambda item: item[1].t_new_first_seen)
    cancelled = _cancelled_growths([change for _, change in indexed], warnings)
    plates = sorted(goal_plates, key=lambda p: p.t_first_seen)
    claimed: set[str] = set()
    result: list[Candidate] = []
    for pos, (index, change) in enumerate(indexed):
        if change.direction != "up":
            continue
        plate = _claim_plate(change, plates, claimed, config.goal_plate_window_s)
        result.append(_goal_from_change(index, change, plate, pos in cancelled, config, warnings))
    result.extend(_goal_plate_only(p, config, warnings) for p in plates if p.id not in claimed)
    return result


def subst_candidates(plates: Sequence[Plate]) -> list[Candidate]:
    """One substitution per substitution plate, at the first second of the plate."""
    return [
        Candidate(
            type=EventType.SUBST,
            t_v=p.t_first_seen,
            team=p.team,
            score=p.conf,
            source="plates",
            rule="subst_plate",
            evidence=[_plate_evidence(p)],
        )
        for p in plates
    ]


def _model_candidate(pred: Prediction) -> Candidate:
    evidence = Evidence(
        kind="model", ref=f"model:{pred.type.value}", t_v=pred.t_v, score=pred.score
    )
    return Candidate(
        type=pred.type,
        t_v=pred.t_v,
        score=pred.score or 0.0,  # callers keep only events that have a score
        source="model",
        rule="model",
        evidence=[evidence],
    )


def model_candidates(model: Sequence[Prediction], config: FuseConfig) -> list[Candidate]:
    """Model shots, corners and fouls over their thresholds, duplicates merged.

    Model goals and substitutions are ignored on purpose: the scoreboard and the plates make
    them. Duplicates of one class closer than `merge_radius_s` collapse to the strongest.
    """
    thresholds: dict[str, float] = {str(k): v for k, v in config.model_thresholds.items()}
    kept = [
        p
        for p in model
        if p.score is not None and p.score >= thresholds.get(p.type.value, float("inf"))
    ]
    return [_model_candidate(p) for p in merge_duplicates(kept, config.merge_radius_s)]


def split_shots_near_goals(
    model: Sequence[Candidate], goals: Sequence[Candidate], radius_s: float
) -> tuple[list[Candidate], list[Candidate]]:
    """Split model events into kept and dropped: a model shot within `radius_s` of a goal (or of
    a disallowed goal) is dropped, the protocol has one mark per goal."""
    goal_times = [g.t_v for g in goals if g.type is EventType.GOAL or g.status == "disallowed"]
    kept: list[Candidate] = []
    dropped: list[Candidate] = []
    for event in model:
        near = event.type is EventType.SHOT and any(
            abs(event.t_v - t) <= radius_s for t in goal_times
        )
        (dropped if near else kept).append(event)
    return kept, dropped


def _pick_foul(plate: Plate, fouls: Sequence[Candidate], config: FuseConfig) -> Candidate | None:
    """The model foul in the window around the plate that is nearest to where a foul is
    expected (`card_foul_lead_s` before the plate); ties go to the earlier one."""
    low = plate.t_first_seen - config.card_foul_window_before_s
    high = plate.t_first_seen + config.card_foul_window_after_s
    expected = plate.t_first_seen - config.card_foul_lead_s
    inside = [f for f in fouls if low <= f.t_v <= high]
    return min(inside, key=lambda f: (abs(f.t_v - expected), f.t_v), default=None)


def _foul_from_plate(plate: Plate, config: FuseConfig) -> Candidate:
    return Candidate(
        type=EventType.FOUL,
        t_v=max(0.0, plate.t_first_seen - config.card_foul_lead_s),
        team=plate.team,
        score=config.card_foul_plate_only_score,
        source="plates_only",
        rule="foul_plates_only",
        evidence=[_plate_evidence(plate)],
    )


def _lift_foul(foul: Candidate, plate: Plate, config: FuseConfig) -> Candidate:
    return foul.model_copy(
        update={
            "score": max(foul.score, config.card_foul_boost_score),
            "team": plate.team,
            "source": "model+plates",
            "rule": "foul_plate_boost",
            "evidence": [*foul.evidence, _plate_evidence(plate)],
        }
    )


def card_foul_candidates(
    card_plates: Sequence[Plate], fouls: Sequence[Candidate], config: FuseConfig
) -> tuple[list[Candidate], list[Candidate]]:
    """Fouls made by card plates and the model fouls that no plate used.

    A model foul near a card plate is lifted to `card_foul_boost_score` (never lowered) and gets
    the plate's team; without one a foul is added before the plate. One foul serves one plate.
    """
    remaining = list(fouls)
    made: list[Candidate] = []
    for plate in sorted(card_plates, key=lambda p: p.t_first_seen):
        target = _pick_foul(plate, remaining, config)
        if target is None:
            made.append(_foul_from_plate(plate, config))
        else:
            remaining.remove(target)
            made.append(_lift_foul(target, plate, config))
    return made, remaining


def place_in_halves(
    candidates: Sequence[Candidate], bounds: dict[Half, HalfSpan], warnings: list[str]
) -> tuple[list[Candidate], list[Dropped]]:
    """Set `half` from the bounds; events outside both halves are dropped with one warning per
    class (a substitution in the break is not an event of the match)."""
    placed: list[Candidate] = []
    dropped: list[Dropped] = []
    for event in candidates:
        half = half_at(event.t_v, bounds)
        if half is None:
            dropped.append(
                Dropped(
                    type=event.type, t_v=event.t_v, source=event.source, reason="outside_halves"
                )
            )
        else:
            placed.append(event.model_copy(update={"half": half}))
    for kind, n in Counter(d.type.value for d in dropped).items():
        warnings.append(f"dropped {n} {kind} event(s) outside the halves")
    return placed, dropped


def count_events(candidates: Sequence[Candidate]) -> dict[str, dict[str, int]]:
    """Events by type and half: `{"goal": {"1": 3, "2": 1}}`; types without events are absent."""
    counts: dict[str, dict[str, int]] = {}
    for event in candidates:
        by_half = counts.setdefault(event.type.value, {})
        key = str(event.half)
        by_half[key] = by_half.get(key, 0) + 1
    return counts


def fuse_signals(
    board: ScoreboardResult,
    plates: PlatesResult,
    model: Sequence[Prediction] | None,
    config: FuseConfig,
) -> FusionOutput:
    """Apply all rules.

    Args:
        board: Validated `scoreboard.json`; its clock segments give the half bounds.
        plates: Validated `plates.json`.
        model: Model predictions already converted to our classes (`to_predictions`), or None
            for the "no model" mode.
        config: Rule parameters.

    Raises:
        FusionError: the scoreboard has no two clock segments, so no half bounds.
    """
    warnings: list[str] = []
    bounds = half_bounds(
        board.offset_segments, config.half_start_margin_s, config.half_end_margin_s
    )
    if model is None:
        warnings.append(NO_MODEL_WARNING)
    by_type: dict[str, list[Plate]] = {"goal": [], "subst": [], "card": []}
    for plate in plates.plates:
        by_type[plate.type].append(plate)
    goals = goal_candidates(board, by_type["goal"], config, warnings)
    modelled, near_goal = split_shots_near_goals(
        model_candidates(model or [], config), goals, config.shot_goal_radius_s
    )
    fouls = [e for e in modelled if e.type is EventType.FOUL]
    card_fouls, plain_fouls = card_foul_candidates(by_type["card"], fouls, config)
    rest = [e for e in modelled if e.type is not EventType.FOUL]
    ordered = sorted(
        [*goals, *subst_candidates(by_type["subst"]), *rest, *card_fouls, *plain_fouls],
        key=lambda e: (e.t_v, e.type.value),
    )
    placed, dropped = place_in_halves(ordered, bounds, warnings)
    dropped.extend(
        Dropped(type=e.type, t_v=e.t_v, source=e.source, reason="shot_near_goal") for e in near_goal
    )
    return FusionOutput(placed, sorted(dropped, key=lambda d: d.t_v), warnings, bounds)
