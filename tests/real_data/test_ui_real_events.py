"""The viewer's reading and formatting on the real `aicoach.db` of a finished `run`.

Needs the match video path in AICOACH_REAL_MATCH_VIDEO and `aicoach.db` beside it (outside the
repository, never committed). Skipped if either is missing. The video id is not needed: the
test uses a fixed fake id. Only counts and formats are checked; nothing is stored.
"""

import os
import re
from pathlib import Path

import pytest

from aicoach.fusion.stage import DB_FILE
from aicoach.storage import CurrentEvent
from aicoach.ui.data import load_events
from aicoach.ui.logic import EventFilter, available_types, filter_events, to_rows, youtube_link

FAKE_VIDEO_ID = "abcDEF_-123"


def _match_dir() -> Path:
    env = os.environ.get("AICOACH_REAL_MATCH_VIDEO")
    if not env:
        pytest.skip("AICOACH_REAL_MATCH_VIDEO is not set")
    return Path(env).parent


@pytest.fixture(scope="module")
def events() -> list[CurrentEvent]:
    match_dir = _match_dir()
    db = match_dir / DB_FILE
    if not db.is_file():
        pytest.skip(f"{db} not found: run the pipeline first")
    found = load_events(db, match_dir.name)
    if not found:
        pytest.skip(f"no events of match {match_dir.name!r} in {db}")
    return found


@pytest.mark.real_data
def test_real_events_are_in_time_order_and_format_to_rows(events: list[CurrentEvent]) -> None:
    rows = to_rows(events)

    assert [r.t_v for r in rows] == sorted(r.t_v for r in rows)
    assert all(re.fullmatch(r"\d{2,}:\d{2}", r.time_text) for r in rows)
    assert all(r.type_label for r in rows)


@pytest.mark.real_data
def test_real_events_filter_by_every_present_type_adds_up(events: list[CurrentEvent]) -> None:
    per_type = [
        len(filter_events(events, EventFilter(types=frozenset({t}))))
        for t in available_types(events)
    ]

    assert sum(per_type) == len(events)


@pytest.mark.real_data
def test_real_events_all_get_a_link_before_the_event(events: list[CurrentEvent]) -> None:
    for e in events:
        link = youtube_link(FAKE_VIDEO_ID, e.t_v)
        start = int(link.rsplit("&t=", 1)[1].removesuffix("s"))
        assert 0 <= start <= e.t_v
