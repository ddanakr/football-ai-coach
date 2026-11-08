"""Which team a plate belongs to: from goal plates and score changes (pure functions).

The score on the scoreboard is "home - away": when the first number grows, the home team scored.
A goal plate shows the logo of the team that scored, so the logo group of a goal plate is the
home team if the score of the home side grew just before it, the away team otherwise.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from aicoach.errors import PlatesError
from aicoach.plates.schemas import Team
from aicoach.scoreboard.schemas import ScoreChange


def goal_side(change: ScoreChange) -> Team | None:
    """Which side scored in `change`: `home` if only the first number grew, `away` if only the
    second grew, None for a decrease (a goal taken back) or any other change."""
    (old_home, old_away), (new_home, new_away) = change.old, change.new
    if new_home == old_home + 1 and new_away == old_away:
        return "home"
    if new_away == old_away + 1 and new_home == old_home:
        return "away"
    return None


def match_score_change(
    t_plate: float, changes: Sequence[ScoreChange], max_lag_s: float
) -> int | None:
    """Index of the score change a goal plate seen at `t_plate` belongs to, or None.

    A change counts as "before" the plate if its window started before the plate appeared
    (`t_old_last_seen <= t_plate`): the scoreboard is often hidden, so the goal may have happened
    anywhere in the window. The lag is measured from the end of the window and is zero while
    the plate is inside it. The change with the smallest lag wins (the nearest one before the
    plate); on a tie the later one. Changes more than `max_lag_s` back are ignored.
    """
    best: tuple[float, float, int] | None = None
    for index, change in enumerate(changes):
        if change.t_old_last_seen > t_plate:
            continue
        lag = max(0.0, t_plate - change.t_new_first_seen)
        if lag > max_lag_s:
            continue
        key = (lag, -change.t_old_last_seen, index)
        if best is None or key < best:
            best = key
    return None if best is None else best[2]


@dataclass(frozen=True)
class TeamDecision:
    """Team of every logo cluster that could be decided, the votes and the warnings."""

    teams: Mapping[int, Team]
    home_votes: Mapping[int, int]
    away_votes: Mapping[int, int]
    warnings: Sequence[str] = field(default_factory=tuple)


def decide_teams(n_clusters: int, evidence: Sequence[tuple[int, Team]]) -> TeamDecision:
    """Tie logo clusters to sides.

    `evidence` has one `(cluster, side)` per goal plate that was matched to a score change.
    Exactly two clusters are expected (two teams); with any other number nothing is decided.
    A cluster gets a side if its votes are not tied; if both clusters got the same side the
    evidence contradicts itself and nothing is decided. With no clusters at all (no plates)
    there is nothing to decide and no warning.

    Raises:
        PlatesError: `evidence` names a cluster outside `0..n_clusters-1`.
    """
    for cluster, _ in evidence:
        if not 0 <= cluster < n_clusters:
            msg = f"evidence for logo cluster {cluster}, but there are {n_clusters} clusters"
            raise PlatesError(msg)
    home = dict.fromkeys(range(n_clusters), 0)
    away = dict.fromkeys(range(n_clusters), 0)
    for cluster, side in evidence:
        (home if side == "home" else away)[cluster] += 1
    if n_clusters == 0:
        return TeamDecision({}, home, away)
    if n_clusters != 2:
        warning = f"expected 2 logo clusters, found {n_clusters}: teams are not set"
        return TeamDecision({}, home, away, (warning,))
    teams: dict[int, Team] = {}
    for cluster in range(n_clusters):
        if home[cluster] > away[cluster]:
            teams[cluster] = "home"
        elif away[cluster] > home[cluster]:
            teams[cluster] = "away"
    warnings: list[str] = []
    if len(set(teams.values())) < len(teams):
        warnings.append("both logo clusters were tied to the same side: teams are not set")
        teams = {}
    undecided = [c for c in range(n_clusters) if c not in teams]
    if undecided:
        warnings.append(f"no goal evidence for logo cluster(s) {undecided}: team is null")
    return TeamDecision(teams, home, away, tuple(warnings))
