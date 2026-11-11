"""Synthetic inputs of the fusion stage: plates and plate results without any picture."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from aicoach.plates.schemas import Plate, PlatesResult, PlatesStats, PlateType, Team
from aicoach.scoreboard.schemas import Roi, ScoreChange
from tests.fixtures.plates_synth import scoreboard_result

# Two halves, as clock segments: 100..1000 and 1500..2500 (video seconds).
HALF_SPANS = [(100.0, 1000.0), (1500.0, 2500.0)]


def plate(
    plate_id: str,
    kind: PlateType,
    t: float,
    *,
    team: Team | None = "home",
    conf: float = 1.0,
) -> Plate:
    """A plate first seen at `t`."""
    return Plate(
        id=plate_id,
        type=kind,
        card_color="yellow" if kind == "card" else None,
        t_first_seen=t,
        t_last_seen=t + 4.0,
        team=team,
        conf=conf,
        n_frames=9,
        crop=f"plate_crops/{plate_id}.png",
        logo_crop=f"plate_crops/{plate_id}_logo.png",
    )


def plates_result(plates: Sequence[Plate], scoreboard_version: str = "v") -> PlatesResult:
    """A plates.json with the given plates."""
    return PlatesResult(
        video="match.mp4",
        decode_mode="opencv_cpu",
        scoreboard_version=scoreboard_version,
        sample_fps=2.0,
        roi=Roi(x0=0.1, y0=0.7, x1=0.6, y1=0.9),
        logo_clusters=[],
        plates=list(plates),
        rejected=[],
        stats=PlatesStats(
            n_samples=0,
            n_candidate_frames=0,
            n_by_type={},
            n_low_conf=0,
            n_rejected=0,
            n_overlaps=0,
        ),
    )


def write_inputs(
    match_dir: Path,
    changes: list[ScoreChange] | None = None,
    plates: Sequence[Plate] = (),
    segments: list[tuple[float, float]] | None = None,
    n_samples: int = 12,
) -> None:
    """Write `scoreboard.json` and a matching `plates.json` (same version) into `match_dir`.

    `n_samples` is the length of the synthetic video in seconds (`video_duration_s`).
    """
    board = scoreboard_result(changes, n_samples=n_samples, segments=segments or HALF_SPANS)
    raw = board.model_dump_json().encode()
    version = hashlib.sha256(raw).hexdigest()
    (match_dir / "scoreboard.json").write_bytes(raw)
    result = plates_result(plates, scoreboard_version=version)
    (match_dir / "plates.json").write_bytes(result.model_dump_json().encode())


def write_tdeed(path: Path, detections: Sequence[tuple[int, str, float]]) -> None:
    """Write a T-DEED output file with `(frame, label, confidence)` detections."""
    items = [{"frame": f, "label": label, "confidence": c} for f, label, c in detections]
    path.write_text(json.dumps({"predictions": items}), encoding="utf-8")
