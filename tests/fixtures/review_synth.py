"""A synthetic match for the review layer: the fusion stage with a model file, no pictures."""

from pathlib import Path

from aicoach.fusion.stage import run_fuse
from aicoach.review.schemas import ReviewCandidatesResult
from aicoach.review.stage import run_review_candidates
from tests.fixtures.fusion_synth import plate, write_inputs, write_tdeed
from tests.fixtures.plates_synth import change

TDEED_NAME = "tdeed_synth.json"
FPS = 25

# (video second, class, confidence). Halves of the synthetic match: 85..1015 and 1485..2515.
# With the starting parameters of the fusion stage the events are a goal at 200 s, a shot at
# 400 s, a substitution at 600 s and a foul before the card plate. The layer holds shots only:
# the detections of the other classes must not become candidates.
DETECTIONS: list[tuple[float, str, float]] = [
    (210.0, "Shots on target", 0.4),  # a candidate (the goal at 200 s is another type)
    (400.0, "Shots on target", 0.5),  # the shot event itself
    (402.0, "Shots off target", 0.1),  # 2 s from the shot event: left out
    (600.0, "Shots on target", 0.08),  # a candidate (the substitution at 600 s is another type)
    (601.0, "Shots off target", 0.06),  # 1 s from the candidate above: merged into it
    (700.0, "Shots off target", 0.03),  # a candidate
    (1200.0, "Shots on target", 0.3),  # in the break between the halves: dropped
    (2000.0, "Shots on target", 0.1),  # a candidate of the 2nd half
    (201.5, "Goal", 0.9),  # not a shot: no candidate
    (212.0, "Goal", 0.4),  # a goal the scoreboard did not show: still no candidate
    (600.5, "Substitution", 0.2),  # not a shot: no candidate
    (700.0, "Corner", 0.03),  # under the corner threshold of fuse, not a shot: no candidate
    (2100.0, "Foul", 0.1),  # under the foul threshold of fuse, not a shot: no candidate
    (610.0, "Yellow card", 0.9),  # a class that is not mapped: ignored
]


def write_review_match(match_dir: Path, match_id: str = "m1") -> tuple[Path, Path]:
    """Run the fusion stage over synthetic inputs with the model file `DETECTIONS`.

    Returns the database and the T-DEED file.
    """
    write_inputs(
        match_dir,
        [change((0, 0), (1, 0), 199.0, 200.0)],
        [
            plate("p1", "goal", 230.0),
            plate("p2", "subst", 600.0),
            plate("p3", "card", 700.0),
        ],
        n_samples=3000,
    )
    tdeed = match_dir / TDEED_NAME
    write_tdeed(tdeed, [(round(t * FPS), label, conf) for t, label, conf in DETECTIONS])
    run_fuse(match_dir, match_id, tdeed, run_id="run1")
    return match_dir / "aicoach.db", tdeed


def write_review_layer(match_dir: Path, match_id: str = "m1") -> ReviewCandidatesResult:
    """`write_review_match` and the review layer on top of it."""
    _, tdeed = write_review_match(match_dir, match_id)
    return run_review_candidates(match_dir, match_id, tdeed)
