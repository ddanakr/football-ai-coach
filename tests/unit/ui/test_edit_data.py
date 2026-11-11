"""Writing edits from the viewer: the data layer, the time bounds, and the fusion guard."""

import os
from pathlib import Path

import pytest
from tests.fixtures.ui_synth import make_event, write_match_dir

from aicoach.domain.events import EventType
from aicoach.errors import FusionError, StorageError
from aicoach.fusion.stage import run_fuse
from aicoach.ui.data import (
    EDIT_AUTHOR,
    apply_edit,
    count_edits,
    load_events,
    load_time_bounds,
    undo_last,
)
from aicoach.ui.edits import EditPlan, plan_add, plan_delete, plan_move

MATCH = "m1"


@pytest.fixture
def match_db(tmp_path: Path) -> Path:
    """The database of a synthetic match: a goal, a substitution and a plate foul."""
    return write_match_dir(tmp_path, MATCH)


@pytest.mark.unit
@pytest.mark.usefixtures("match_db")
def test_time_bounds_come_from_the_fusion_files(tmp_path: Path) -> None:
    bounds = load_time_bounds(tmp_path)

    assert bounds is not None
    assert bounds.video_duration_s == 3000.0
    # The clock segments are 100..1000 and 1500..2500; the stage adds 15 s on both sides.
    assert (bounds.halves[1].start, bounds.halves[1].end) == (85.0, 1015.0)
    assert (bounds.halves[2].start, bounds.halves[2].end) == (1485.0, 2515.0)


@pytest.mark.unit
def test_time_bounds_without_files_are_none(tmp_path: Path) -> None:
    assert load_time_bounds(tmp_path) is None


@pytest.mark.unit
@pytest.mark.usefixtures("match_db")
def test_time_bounds_with_a_broken_file_are_none(tmp_path: Path) -> None:
    (tmp_path / "candidates.json").write_text("{not json", encoding="utf-8")

    assert load_time_bounds(tmp_path) is None


@pytest.mark.unit
@pytest.mark.usefixtures("match_db")
def test_time_bounds_are_read_again_when_a_file_changes(tmp_path: Path) -> None:
    first = load_time_bounds(tmp_path)
    path = tmp_path / "candidates.json"
    path.write_text("{not json", encoding="utf-8")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))

    assert first is not None
    assert load_time_bounds(tmp_path) is None


@pytest.mark.unit
def test_apply_edit_writes_one_journal_row_and_leaves_predictions_alone(
    match_db: Path,
) -> None:
    before = load_events(match_db, MATCH)
    goal = before[0]
    assert goal.type == EventType.GOAL

    record = apply_edit(match_db, MATCH, plan_delete(goal))

    assert (record.action, record.target, record.author) == ("delete", goal.event_id, EDIT_AUTHOR)
    assert [e.event_id for e in load_events(match_db, MATCH)] == [e.event_id for e in before[1:]]
    assert count_edits(match_db, MATCH) == 1


@pytest.mark.unit
def test_apply_edit_move_and_add_change_the_current_view(tmp_path: Path, match_db: Path) -> None:
    bounds = load_time_bounds(tmp_path)
    assert bounds is not None
    goal = load_events(match_db, MATCH)[0]

    apply_edit(match_db, MATCH, plan_move(goal, "05:00", bounds))
    added = apply_edit(match_db, MATCH, plan_add(EventType.CORNER, "10:00", bounds))

    current = {e.event_id: e for e in load_events(match_db, MATCH)}
    assert current[goal.event_id].t_v == 300.0
    assert current[goal.event_id].n_edits == 1
    assert added.event_id is not None
    assert (current[added.event_id].type, current[added.event_id].t_v) == (EventType.CORNER, 600.0)
    assert current[added.event_id].origin == "user"


@pytest.mark.unit
def test_undo_last_takes_back_edits_one_by_one(match_db: Path) -> None:
    events = load_events(match_db, MATCH)
    apply_edit(match_db, MATCH, plan_delete(events[0]))
    apply_edit(match_db, MATCH, plan_delete(events[1]))

    undo_last(match_db, MATCH)
    assert [e.event_id for e in load_events(match_db, MATCH)] == [
        events[1].event_id,
        events[2].event_id,
    ]
    undo_last(match_db, MATCH)

    assert len(load_events(match_db, MATCH)) == 3
    assert count_edits(match_db, MATCH) == 0
    with pytest.raises(StorageError, match="no edit to undo"):
        undo_last(match_db, MATCH)


@pytest.mark.unit
@pytest.mark.parametrize("action", ["count", "undo", "edit"])
def test_missing_database_is_refused_and_not_created(tmp_path: Path, action: str) -> None:
    db = tmp_path / "none.db"

    with pytest.raises(StorageError, match="does not exist"):
        if action == "count":
            count_edits(db, MATCH)
        elif action == "undo":
            undo_last(db, MATCH)
        else:
            apply_edit(db, MATCH, plan_delete(make_event()))

    assert not db.exists()


@pytest.mark.unit
def test_an_edit_of_an_event_that_is_not_in_the_latest_run_is_refused(match_db: Path) -> None:
    stale = EditPlan("delete", "message", target="p:9999")

    with pytest.raises(StorageError, match="does not exist"):
        apply_edit(match_db, MATCH, stale)

    assert count_edits(match_db, MATCH) == 0


@pytest.mark.unit
def test_fuse_refuses_to_replace_a_run_that_has_edits_made_in_the_viewer(
    tmp_path: Path, match_db: Path
) -> None:
    apply_edit(match_db, MATCH, plan_delete(load_events(match_db, MATCH)[0]))

    with pytest.raises(FusionError, match="--discard-edits"):
        run_fuse(tmp_path, MATCH, force=True, run_id="run2")

    # The edit is still in force and the first run is still the one that is shown.
    assert count_edits(match_db, MATCH) == 1
    assert len(load_events(match_db, MATCH)) == 2
