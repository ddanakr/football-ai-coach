"""Synthetic events and a small database for the viewer tests."""

from pathlib import Path
from typing import Literal

from aicoach.domain.events import EventType
from aicoach.storage import CurrentEvent, NewPrediction, add_predictions, connect, init_db


def make_event(
    event_id: str = "p:1",
    type: EventType = EventType.GOAL,
    t_v: float = 100.0,
    half: Literal[1, 2] | None = 1,
    score: float | None = 0.8,
    status: str = "auto",
    team: Literal["home", "away"] | None = "home",
    source: Literal["plates", "model", "speech", "merge"] | None = "model",
) -> CurrentEvent:
    """A current event as the store returns it (no database needed)."""
    return CurrentEvent(
        event_id=event_id,
        match_id="m",
        type=type,
        t_v=t_v,
        half=half,
        score=score,
        status=status,
        team=team,
        source=source,
        origin="prediction" if source else "user",
        verified=False,
        deleted=False,
        n_edits=0,
    )


def write_events_db(path: Path, match_id: str = "m1") -> None:
    """Create a database with five predictions of one run: both halves, with and without score."""
    init_db(path)
    predictions = [
        NewPrediction(type=EventType.GOAL, t_v=600.0, source="plates", half=1, team="home"),
        NewPrediction(type=EventType.SHOT, t_v=900.0, source="model", half=1, score=0.30),
        NewPrediction(type=EventType.FOUL, t_v=1200.0, source="merge", half=1, score=0.90),
        NewPrediction(type=EventType.CORNER, t_v=4000.0, source="model", half=2, score=0.60),
        NewPrediction(type=EventType.SUBST, t_v=4300.0, source="plates", half=2, team="away"),
    ]
    with connect(path) as conn:
        add_predictions(conn, match_id, "run1", predictions)
