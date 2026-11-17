"""Synthetic match for the team and corner-tag tests: a small database, scripted answers of the
video model. No network, no FFmpeg; the clips are written by `FakeClipper`."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from aicoach.domain.events import EventType
from aicoach.storage import NewPrediction, add_predictions, connect, init_db
from aicoach.vlm import VideoReply
from tests.fixtures.episode_synth import reply

MATCH_ID = "m1"
Side = Literal["home", "away"] | None
Spec = tuple[EventType, float, int, Side]

# Two goals for each side and three corners: the votes of both kits and the tags to read.
DEFAULT_SPECS: tuple[Spec, ...] = (
    (EventType.GOAL, 600.0, 1, "home"),
    (EventType.GOAL, 900.0, 1, "away"),
    (EventType.CORNER, 1000.0, 1, None),
    (EventType.GOAL, 4000.0, 2, "home"),
    (EventType.CORNER, 4100.0, 2, None),
    (EventType.GOAL, 4500.0, 2, "away"),
    (EventType.CORNER, 4800.0, 2, None),
)


def write_db(match_dir: Path, specs: Sequence[Spec] = DEFAULT_SPECS) -> Path:
    """Create `aicoach.db` in `match_dir` with one run of predictions, and return its path."""
    match_dir.mkdir(parents=True, exist_ok=True)
    path = match_dir / "aicoach.db"
    init_db(path)
    rows = [
        NewPrediction(type=t, t_v=t_v, source="plates", half=half, team=team)  # type: ignore[arg-type]
        for t, t_v, half, team in specs
    ]
    with connect(path) as conn:
        add_predictions(conn, MATCH_ID, "run1", rows)
    return path


def write_video(match_dir: Path) -> Path:
    """An empty stand-in for `src.mp4` (the fake clipper never reads it)."""
    video = match_dir / "src.mp4"
    video.write_bytes(b"video")
    return video


def kit_reply(kit: str, cost: float | None = 0.003) -> VideoReply:
    """The model's answer to the kit question."""
    return reply(json.dumps({"kit": kit}), cost)


CORNER_JSON = {
    "is_corner": "yes",
    "attacking_kit": "white",
    "delivery": "cross",
    "zone": "far_post",
    "first_contact": "defence",
    "outcome": "cleared",
}


def corner_reply(cost: float | None = 0.004, **changes: str) -> VideoReply:
    """The model's answer to the corner question; `changes` replace fields of `CORNER_JSON`."""
    return reply(json.dumps({**CORNER_JSON, **changes}), cost)


def write_lineup(match_dir: Path, home: str = "Актобе", away: str = "Кайрат") -> None:
    """A lineup file with only the columns the team names need."""
    text = f"team,side,number,name\n{home},home,1,Иванов Иван\n{away},away,1,Петров Пётр\n"
    (match_dir / "lineups.csv").write_text(text, encoding="utf-8")


def write_tag_files(match_dir: Path, home_kit: str | None = "white") -> None:
    """`corner_tags.json` for the three corners of `DEFAULT_SPECS` (ids p:3, p:5, p:7) and a
    `teams.json` with the kits: the first corner is the home team's, the second the away team's,
    the third has an unclear kit."""
    from aicoach.corner_tags.schemas import CornerFields, CornerTag, CornerTagsFile
    from aicoach.teams.store import RESULT_FILE, TeamEntry, TeamsResult

    kits = ["white", "red", "unclear"]
    outcomes = ["cleared", "shot", "unclear"]
    times = [1000.0, 4100.0, 4800.0]
    halves = [1, 2, 2]
    tags = [
        CornerTag(
            event_id=f"p:{number}",
            t_v=t_v,
            half=half,
            fields=CornerFields.model_validate(
                {**CORNER_JSON, "attacking_kit": kit, "outcome": outcome}
            ),
            status="ok",
            model="test/vlm-1",
            prompt_version=1,
            before_s=4.0,
            after_s=12.0,
            cost_usd=0.004,
        )
        for number, t_v, half, kit, outcome in zip(
            (3, 5, 7), times, halves, kits, outcomes, strict=True
        )
    ]
    (match_dir / "corner_tags.json").write_text(
        CornerTagsFile(match_id=MATCH_ID, tags=tags).model_dump_json(), encoding="utf-8"
    )
    home = TeamEntry(
        name="Актобе",
        name_source="lineup",
        kit=home_kit,  # type: ignore[arg-type]
        kit_source="vlm" if home_kit else "undetermined",
    )
    away = TeamEntry(name="Кайрат", name_source="lineup", kit="red", kit_source="vlm")
    teams = TeamsResult(match_id=MATCH_ID, cache_key="k", home=home, away=away)
    (match_dir / RESULT_FILE).write_text(teams.model_dump_json(), encoding="utf-8")
