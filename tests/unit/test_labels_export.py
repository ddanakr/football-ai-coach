"""Export of the current events as a ground-truth file (`aicoach.labels_export`): the layout, the
reviewed-only filter, an empty match, and the round trip through the loader of the ground truth and
through `spotting-eval`."""

import json
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest
import structlog

from aicoach.agent.context import EventFacts
from aicoach.domain.events import EventType
from aicoach.errors import StorageError
from aicoach.eval.schemas import HalfBounds, load_ground_truth
from aicoach.labels_export import (
    ConfirmedRow,
    build_ground_truth,
    confirmed_rows,
    export_labels,
    ground_truth_json,
    is_reviewed,
    read_half_bounds,
    write_labels,
)
from aicoach.review.log import append_entry, make_entry, read_log
from aicoach.spotting.run import run_spotting_eval
from aicoach.storage import add_edit, connect, undo_edit
from aicoach.ui.data import load_events
from tests.fixtures.fusion_synth import write_tdeed
from tests.fixtures.reel_synth import make_facts
from tests.fixtures.ui_synth import write_match_dir

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
HALVES: dict[Literal["1", "2"], HalfBounds] = {
    "1": HalfBounds(start=100.0, end=1000.0),
    "2": HalfBounds(start=1500.0, end=2500.0),
}


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    yield
    structlog.reset_defaults()


def with_flags(
    facts: EventFacts, *, verified: bool = False, n_edits: int = 0, source: bool = True
) -> EventFacts:
    """The same event as a person's work leaves it."""
    update: dict[str, object] = {"verified": verified, "n_edits": n_edits}
    if not source:
        update.update(origin="user", source=None, status="manual")
    return EventFacts(
        facts.event.model_copy(update=update), facts.minute, facts.minute_text, (), facts.card
    )


# --- build_ground_truth --------------------------------------------------------------------


@pytest.mark.unit
def test_build_ground_truth_writes_the_layout_the_loader_reads(tmp_path: Path) -> None:
    facts = [
        make_facts("p:2", EventType.FOUL, 700.0, team="away", minute=9),
        make_facts("p:1", EventType.GOAL, 600.0, team="home", minute=5),
        make_facts("p:3", EventType.CORNER, 1600.0, half=2, team=None, minute=None),
    ]

    exported = build_ground_truth(facts, video="src.mp4", halves=HALVES)
    path = tmp_path / "labels.json"
    write_labels(path, exported.ground_truth)
    loaded = load_ground_truth(path)

    assert [e.type for e in loaded.events] == [EventType.GOAL, EventType.FOUL, EventType.CORNER]
    goal = loaded.events[0]
    assert (goal.t_v, goal.half, goal.clock, goal.team) == (600.0, 1, "5'", "home")
    assert (goal.visibility, goal.replay, goal.annotator) == (True, False, "aicoach")
    assert goal.note == "source=model; status=auto"
    assert loaded.events[2].clock is None and loaded.events[2].team is None
    assert loaded.video == "src.mp4"
    assert loaded.halves["2"].start == 1500.0
    assert exported.counts_by_type() == {"goal": 1, "foul": 1, "corner": 1}


@pytest.mark.unit
def test_build_ground_truth_leaves_out_events_without_a_half_and_counts_them() -> None:
    facts = [
        make_facts("p:1", EventType.GOAL, 600.0),
        make_facts("e:2", EventType.GOAL, 1200.0, half=None),  # in the break
    ]

    exported = build_ground_truth(facts, video="src.mp4", halves=HALVES)

    assert len(exported.ground_truth.events) == 1
    assert exported.skipped_no_half == 1


@pytest.mark.unit
def test_build_ground_truth_no_events_is_a_valid_empty_file(tmp_path: Path) -> None:
    exported = build_ground_truth([], video="src.mp4", halves=HALVES)
    path = tmp_path / "labels.json"
    write_labels(path, exported.ground_truth)

    loaded = load_ground_truth(path)

    assert loaded.events == ()
    assert exported.counts_by_type() == {}
    assert json.loads(path.read_text(encoding="utf-8"))["events"] == []


@pytest.mark.unit
def test_build_ground_truth_without_half_bounds_has_no_halves() -> None:
    exported = build_ground_truth(
        [make_facts("p:1", EventType.GOAL, 600.0)], video="src.mp4", halves={}
    )

    assert exported.ground_truth.halves == {}


@pytest.mark.unit
def test_build_ground_truth_user_event_is_named_ui_and_its_status_is_manual() -> None:
    added = with_flags(make_facts("e:7", EventType.SHOT, 650.0), source=False)

    exported = build_ground_truth([added], video="src.mp4", halves=HALVES)

    (event,) = exported.ground_truth.events
    assert event.annotator == "ui"
    assert event.note == "source=user; status=manual"


@pytest.mark.unit
def test_ground_truth_json_keeps_letters_and_ends_with_a_newline() -> None:
    exported = build_ground_truth([], video="матч.mp4", halves={})

    text = ground_truth_json(exported.ground_truth)

    assert "матч.mp4" in text and text.endswith("\n")


# --- reviewed only -------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("facts", "confirmed", "expected"),
    [
        (make_facts("p:1", EventType.GOAL, 10.0), frozenset[ConfirmedRow](), False),
        (make_facts("p:1", EventType.GOAL, 10.0), frozenset({("p:1", EventType.GOAL, 10.0)}), True),
        (
            make_facts("p:1", EventType.GOAL, 10.0),
            frozenset({("p:2", EventType.GOAL, 10.0)}),
            False,
        ),
        # the same id from another numbering: another type or another time is another event
        (
            make_facts("p:1", EventType.GOAL, 10.0),
            frozenset({("p:1", EventType.SHOT, 10.0)}),
            False,
        ),
        (
            make_facts("p:1", EventType.GOAL, 10.0),
            frozenset({("p:1", EventType.GOAL, 99.0)}),
            False,
        ),
        (with_flags(make_facts("p:1", EventType.GOAL, 10.0), n_edits=1), frozenset[str](), True),
        (
            with_flags(make_facts("p:1", EventType.GOAL, 10.0), verified=True),
            frozenset[ConfirmedRow](),
            True,
        ),
        (
            with_flags(make_facts("e:1", EventType.GOAL, 10.0), source=False),
            frozenset[str](),
            True,
        ),
    ],
)
def test_is_reviewed_follows_edits_verification_and_confirmation(
    facts: EventFacts, confirmed: frozenset[ConfirmedRow], expected: bool
) -> None:
    assert is_reviewed(facts, confirmed) is expected


@pytest.mark.unit
def test_build_ground_truth_reviewed_only_keeps_what_a_person_touched() -> None:
    facts = [
        make_facts("p:1", EventType.GOAL, 600.0),  # untouched
        make_facts("p:2", EventType.SHOT, 610.0),  # confirmed in the review mode
        with_flags(make_facts("p:3", EventType.CORNER, 620.0), n_edits=1),  # moved
        with_flags(make_facts("e:4", EventType.FOUL, 630.0), source=False),  # added
    ]

    exported = build_ground_truth(
        facts,
        video="src.mp4",
        halves=HALVES,
        reviewed_only=True,
        confirmed=frozenset({("p:2", EventType.SHOT, 610.0)}),
    )

    assert [e.type.value for e in exported.ground_truth.events] == ["shot", "corner", "foul"]
    assert exported.skipped_unreviewed == 1


@pytest.mark.unit
def test_build_ground_truth_reviewed_only_with_nothing_reviewed_is_an_empty_file() -> None:
    exported = build_ground_truth(
        [make_facts("p:1", EventType.GOAL, 600.0)],
        video="src.mp4",
        halves=HALVES,
        reviewed_only=True,
    )

    assert exported.ground_truth.events == ()
    assert exported.skipped_unreviewed == 1


@pytest.mark.unit
def test_confirmed_rows_the_last_decision_on_a_row_counts(tmp_path: Path) -> None:
    decisions: list[tuple[str, Literal["confirm", "delete"]]] = [
        ("p:1", "confirm"),
        ("p:2", "confirm"),
        ("p:2", "delete"),
        ("p:3", "confirm"),
    ]
    for row, action in decisions:
        append_entry(tmp_path, make_entry(row, action, 10.0, EventType.SHOT, now=NOW))

    assert confirmed_rows(read_log(tmp_path)) == frozenset(
        {("p:1", EventType.SHOT, 10.0), ("p:3", EventType.SHOT, 10.0)}
    )
    assert confirmed_rows([]) == frozenset()


@pytest.mark.unit
def test_read_half_bounds_broken_candidates_file_gives_no_halves(tmp_path: Path) -> None:
    (tmp_path / "candidates.json").write_text("{not json", encoding="utf-8")

    assert read_half_bounds(tmp_path) == {}


# --- from the database ---------------------------------------------------------------------


@pytest.mark.unit
def test_export_labels_reads_events_halves_and_minutes_of_a_match(tmp_path: Path) -> None:
    db = write_match_dir(tmp_path)

    exported = export_labels(db, "m1", tmp_path)

    truth = exported.ground_truth
    assert exported.counts_by_type() == {"goal": 1, "subst": 1, "foul": 1}
    assert truth.halves["1"].start == 85.0 and truth.halves["2"].end == 2515.0
    assert [e.half for e in truth.events] == [1, 1, 1]
    assert all(e.clock is not None for e in truth.events)


@pytest.mark.unit
def test_export_labels_without_candidates_file_exports_without_halves(tmp_path: Path) -> None:
    db = write_match_dir(tmp_path)
    (tmp_path / "candidates.json").unlink()

    exported = export_labels(db, "m1", tmp_path)

    assert exported.ground_truth.halves == {}
    assert len(exported.ground_truth.events) == 3  # the store still knows the halves


@pytest.mark.unit
def test_export_labels_reviewed_only_uses_edits_and_the_review_journal(tmp_path: Path) -> None:
    db = write_match_dir(tmp_path)
    events = {e.type: e for e in load_events(db, "m1")}
    with connect(db) as conn:
        add_edit(conn, "m1", "move", author="ui", target=events[EventType.GOAL].event_id, t_v=205.0)
        add_edit(conn, "m1", "add", author="ui", type=EventType.CORNER, t_v=300.0)
    subst = events[EventType.SUBST]
    append_entry(tmp_path, make_entry(subst.event_id, "confirm", subst.t_v, subst.type, now=NOW))

    exported = export_labels(db, "m1", tmp_path, reviewed_only=True)

    assert exported.counts_by_type() == {"goal": 1, "corner": 1, "subst": 1}  # the foul is left out
    assert exported.skipped_unreviewed == 1
    goal = next(e for e in exported.ground_truth.events if e.type == EventType.GOAL)
    assert goal.t_v == 205.0  # the edit, not the prediction


@pytest.mark.unit
def test_export_labels_reviewed_only_counts_verify_and_forgets_an_undone_edit(
    tmp_path: Path,
) -> None:
    db = write_match_dir(tmp_path)
    events = {e.type: e for e in load_events(db, "m1")}
    with connect(db) as conn:
        add_edit(conn, "m1", "verify", author="ui", target=events[EventType.SUBST].event_id)
        moved = add_edit(
            conn, "m1", "move", author="ui", target=events[EventType.GOAL].event_id, t_v=205.0
        )
        undo_edit(conn, moved.id, "ui")

    exported = export_labels(db, "m1", tmp_path, reviewed_only=True)

    assert exported.counts_by_type() == {"subst": 1}  # the undone move leaves the goal unreviewed
    assert exported.skipped_unreviewed == 2


@pytest.mark.unit
def test_export_labels_match_dir_may_differ_from_the_database_directory(tmp_path: Path) -> None:
    match_dir = tmp_path / "match"
    match_dir.mkdir()
    write_match_dir(match_dir)
    other_db = tmp_path / "copy.db"
    shutil.copy(match_dir / "aicoach.db", other_db)

    exported = export_labels(other_db, "m1", match_dir)

    assert exported.ground_truth.halves["1"].start == 85.0  # read from the match dir
    assert len(exported.ground_truth.events) == 3


@pytest.mark.unit
def test_export_labels_broken_review_journal_raises_only_when_it_is_needed(tmp_path: Path) -> None:
    db = write_match_dir(tmp_path)
    (tmp_path / "review_log.jsonl").write_text("not json\n", encoding="utf-8")

    assert export_labels(db, "m1", tmp_path).ground_truth.events
    with pytest.raises(StorageError, match="line 1"):
        export_labels(db, "m1", tmp_path, reviewed_only=True)


@pytest.mark.unit
def test_export_labels_missing_database_raises_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="does not exist"):
        export_labels(tmp_path / "none.db", "m1", tmp_path)


@pytest.mark.unit
def test_write_labels_creates_the_directory_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    exported = build_ground_truth([], video="src.mp4", halves={})
    path = tmp_path / "out" / "labels.json"

    write_labels(path, exported.ground_truth)

    assert sorted(p.name for p in path.parent.iterdir()) == ["labels.json"]


# --- the round trip ------------------------------------------------------------------------


@pytest.mark.unit
def test_exported_file_goes_through_spotting_eval_without_errors(tmp_path: Path) -> None:
    db = write_match_dir(tmp_path)
    labels = tmp_path / "labels.json"
    write_labels(labels, export_labels(db, "m1", tmp_path).ground_truth)
    tdeed = tmp_path / "tdeed_synthetic.json"
    write_tdeed(tdeed, [(5000, "Goal", 0.9), (15000, "Yellow card", 0.8)])  # 200 s and 600 s

    _, result_path = run_spotting_eval(tdeed, labels, output_path=tmp_path / "eval.json")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    goal = result["thresholds"][0]["tolerances"][0]["per_class"]["goal"]
    assert (goal["tp"], goal["fp"], goal["fn"]) == (1, 0, 0)  # the exported goal is found


@pytest.mark.unit
def test_exported_file_is_the_same_after_one_more_round(tmp_path: Path) -> None:
    db = write_match_dir(tmp_path)
    first = tmp_path / "first.json"
    write_labels(first, export_labels(db, "m1", tmp_path).ground_truth)

    loaded = load_ground_truth(first)
    second = tmp_path / "second.json"
    write_labels(second, loaded)

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
