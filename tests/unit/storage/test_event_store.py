"""Event store: immutable predictions, append-only edits, undo, the events_current view."""

import math
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from aicoach.domain.events import EventType
from aicoach.errors import AicoachError, StorageError
from aicoach.storage import (
    CurrentEvent,
    NewPrediction,
    add_edit,
    add_predictions,
    connect,
    count_effective_edits,
    get_current_events,
    init_db,
    latest_run_id,
    transaction,
    undo_edit,
)
from tests.fixtures.factories import make_new_prediction

MATCH = "m1"
OTHER = "m2"


def _seed(conn: sqlite3.Connection, match_id: str = MATCH) -> list[int]:
    """Three predictions: a shot, a corner and a goal (with a score)."""
    return add_predictions(
        conn,
        match_id,
        "run-1",
        [
            make_new_prediction("shot", 100.0, score=0.9, half=1),
            make_new_prediction("corner", 200.0, score=0.4, half=1),
            make_new_prediction("goal", 300.0, source="plates", half=1, team="home"),
        ],
    )


def _event(conn: sqlite3.Connection, event_id: str, *, match_id: str = MATCH) -> CurrentEvent:
    found = [
        e
        for e in get_current_events(conn, match_id, include_deleted=True)
        if e.event_id == event_id
    ]
    assert len(found) == 1
    return found[0]


def _table_rows(conn: sqlite3.Connection, table: str) -> list[tuple[object, ...]]:
    return [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY id")]  # noqa: S608


# --- predictions are immutable ---------------------------------------------------------


@pytest.mark.unit
def test_edits_do_not_change_the_predictions_table(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)
    before = _table_rows(conn, "predictions")

    add_edit(conn, MATCH, "move", author="a", target=f"p:{ids[0]}", t_v=150.0)
    add_edit(conn, MATCH, "retype", author="a", target=f"p:{ids[1]}", type=EventType.FOUL)
    add_edit(conn, MATCH, "delete", author="a", target=f"p:{ids[2]}")
    add_edit(conn, MATCH, "verify", author="a", target=f"p:{ids[0]}")

    assert _table_rows(conn, "predictions") == before


@pytest.mark.unit
@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE predictions SET t_v = 1.0",
        "DELETE FROM predictions",
        "UPDATE edits SET author = 'x'",
        "DELETE FROM edits",
    ],
)
def test_append_only_tables_reject_update_and_delete(conn: sqlite3.Connection, sql: str) -> None:
    _seed(conn)
    add_edit(conn, MATCH, "add", author="a", type=EventType.SHOT, t_v=5.0)

    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute(sql)


@pytest.mark.unit
@pytest.mark.parametrize(
    "values",
    [
        ("move", None, None, 1.0, None),  # move without a target
        ("move", "p:1", None, None, None),  # move without a time
        ("add", "p:1", "shot", 1.0, None),  # add with a target
        ("add", None, None, 1.0, None),  # add without a type
        ("retype", "p:1", None, None, None),  # retype without a type
        ("delete", "p:1", "shot", None, None),  # delete with extra fields
        ("undo", None, None, None, None),  # undo without undo_of
        ("move", "x:1", None, 1.0, None),  # malformed target
        ("move", "p:1", None, -1.0, None),  # negative time
        ("jump", "p:1", None, 1.0, None),  # unknown action
    ],
)
def test_edits_table_check_rejects_malformed_rows(
    conn: sqlite3.Connection, values: tuple[object, ...]
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO edits (match_id, action, target, type, t_v, undo_of, author) "
            "VALUES ('m1', ?, ?, ?, ?, ?, 'a')",
            values,
        )


@pytest.mark.unit
def test_predictions_table_check_rejects_unknown_type(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO predictions (match_id, type, t_v, source, run_id) "
            "VALUES ('m1', 'offside', 1.0, 'model', 'r')"
        )


# --- reading ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_current_events_without_edits_returns_predictions_in_time_order(
    conn: sqlite3.Connection,
) -> None:
    add_predictions(
        conn,
        MATCH,
        "r",
        [make_new_prediction("corner", 200.0), make_new_prediction("shot", 100.0)],
    )

    events = get_current_events(conn, MATCH)

    assert [(e.type, e.t_v) for e in events] == [(EventType.SHOT, 100.0), (EventType.CORNER, 200.0)]
    assert all(e.origin == "prediction" and not e.verified and e.n_edits == 0 for e in events)
    assert events[0].event_id.startswith("p:")


@pytest.mark.unit
def test_get_current_events_equal_times_are_ordered_deterministically(
    conn: sqlite3.Connection,
) -> None:
    ids = add_predictions(conn, MATCH, "r", [make_new_prediction("shot", 50.0) for _ in range(12)])

    events = get_current_events(conn, MATCH)

    assert [e.event_id for e in events] == [f"p:{i}" for i in ids]  # numeric, not "p:10" < "p:2"


@pytest.mark.unit
def test_get_current_events_empty_database_and_unknown_match_give_empty_list(
    conn: sqlite3.Connection,
) -> None:
    assert get_current_events(conn, MATCH) == []
    _seed(conn)
    assert get_current_events(conn, "no-such-match") == []
    assert count_effective_edits(conn, MATCH) == 0
    assert count_effective_edits(conn, "no-such-match") == 0


@pytest.mark.unit
def test_get_current_events_filters_by_type_and_score(conn: sqlite3.Connection) -> None:
    _seed(conn)

    only_shots = get_current_events(conn, MATCH, types=[EventType.SHOT])
    shots_and_goals = get_current_events(conn, MATCH, types={EventType.SHOT, EventType.GOAL})
    confident = get_current_events(conn, MATCH, min_score=0.5)

    assert [e.type for e in only_shots] == [EventType.SHOT]
    assert [e.type for e in shots_and_goals] == [EventType.SHOT, EventType.GOAL]
    # The 0.4 corner is cut; the plate goal has no score and stays.
    assert [e.type for e in confident] == [EventType.SHOT, EventType.GOAL]
    assert get_current_events(conn, MATCH, types=[]) == []


@pytest.mark.unit
def test_get_current_events_min_score_boundary_is_inclusive(conn: sqlite3.Connection) -> None:
    _seed(conn)

    assert len(get_current_events(conn, MATCH, min_score=0.4)) == 3


@pytest.mark.unit
@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_get_current_events_non_finite_min_score_raises(
    conn: sqlite3.Connection, bad: float
) -> None:
    with pytest.raises(StorageError, match="min_score"):
        get_current_events(conn, MATCH, min_score=bad)


@pytest.mark.unit
def test_get_current_events_keeps_team_half_and_source(conn: sqlite3.Connection) -> None:
    _seed(conn)

    goal = get_current_events(conn, MATCH, types=[EventType.GOAL])[0]

    assert (goal.team, goal.half, goal.source, goal.score) == ("home", 1, "plates", None)
    assert goal.status == "auto"


# --- edits -----------------------------------------------------------------------------


@pytest.mark.unit
def test_move_changes_time_and_counts_the_edit(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)

    add_edit(conn, MATCH, "move", author="a", target=f"p:{ids[0]}", t_v=111.5)

    event = _event(conn, f"p:{ids[0]}")
    assert event.t_v == 111.5
    assert event.n_edits == 1
    assert event.type is EventType.SHOT


@pytest.mark.unit
def test_retype_changes_class(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)

    add_edit(conn, MATCH, "retype", author="a", target=f"p:{ids[0]}", type=EventType.GOAL)

    assert _event(conn, f"p:{ids[0]}").type is EventType.GOAL
    assert len(get_current_events(conn, MATCH, types=[EventType.GOAL])) == 2


@pytest.mark.unit
def test_several_moves_last_wins_and_undo_of_last_restores_previous(
    conn: sqlite3.Connection,
) -> None:
    ids = _seed(conn)
    target = f"p:{ids[0]}"
    add_edit(conn, MATCH, "move", author="a", target=target, t_v=110.0)
    add_edit(conn, MATCH, "move", author="a", target=target, t_v=120.0)
    last = add_edit(conn, MATCH, "move", author="a", target=target, t_v=130.0)
    assert _event(conn, target).t_v == 130.0

    undo_edit(conn, last.id, "a")

    assert _event(conn, target).t_v == 120.0
    assert _event(conn, target).n_edits == 2


@pytest.mark.unit
def test_undo_of_an_earlier_move_keeps_the_later_one(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)
    target = f"p:{ids[0]}"
    first = add_edit(conn, MATCH, "move", author="a", target=target, t_v=110.0)
    add_edit(conn, MATCH, "move", author="a", target=target, t_v=120.0)

    undo_edit(conn, first.id, "a")

    assert _event(conn, target).t_v == 120.0


@pytest.mark.unit
def test_undo_of_the_only_move_returns_the_original_event(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)
    before = _event(conn, f"p:{ids[0]}")
    move = add_edit(conn, MATCH, "move", author="a", target=f"p:{ids[0]}", t_v=999.0)

    undo_edit(conn, move.id, "a")

    assert _event(conn, f"p:{ids[0]}") == before


@pytest.mark.unit
def test_undo_of_retype_returns_the_original_class(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)
    before = _event(conn, f"p:{ids[0]}")
    retype = add_edit(conn, MATCH, "retype", author="a", target=f"p:{ids[0]}", type=EventType.FOUL)

    undo_edit(conn, retype.id, "a")

    assert _event(conn, f"p:{ids[0]}") == before


@pytest.mark.unit
def test_delete_hides_the_event_but_include_deleted_shows_it(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)
    target = f"p:{ids[1]}"

    add_edit(conn, MATCH, "delete", author="a", target=target)

    assert target not in [e.event_id for e in get_current_events(conn, MATCH)]
    shown = get_current_events(conn, MATCH, include_deleted=True)
    assert [e.deleted for e in shown if e.event_id == target] == [True]
    assert len(shown) == 3
    assert len(_table_rows(conn, "predictions")) == 3  # hidden, not erased


@pytest.mark.unit
def test_undo_of_delete_brings_the_event_back(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)
    before = _event(conn, f"p:{ids[1]}")
    delete = add_edit(conn, MATCH, "delete", author="a", target=f"p:{ids[1]}")

    undo_edit(conn, delete.id, "a")

    assert get_current_events(conn, MATCH)[1] == before


@pytest.mark.unit
def test_add_creates_a_user_event_with_edit_based_id(conn: sqlite3.Connection) -> None:
    record = add_edit(conn, MATCH, "add", author="analyst", type=EventType.FOUL, t_v=42.0)

    assert record.event_id == f"e:{record.id}"
    event = _event(conn, f"e:{record.id}")
    assert (event.type, event.t_v, event.origin) == (EventType.FOUL, 42.0, "user")
    assert (event.status, event.half, event.score, event.team, event.source) == (
        "manual",
        None,
        None,
        None,
        None,
    )
    assert event.n_edits == 0  # the add is the event itself, not an edit of it


@pytest.mark.unit
def test_undo_of_add_removes_the_added_event(conn: sqlite3.Connection) -> None:
    _seed(conn)
    added = add_edit(conn, MATCH, "add", author="a", type=EventType.FOUL, t_v=42.0)

    undo_edit(conn, added.id, "a")

    assert [e.origin for e in get_current_events(conn, MATCH, include_deleted=True)] == [
        "prediction"
    ] * 3


@pytest.mark.unit
def test_added_event_can_be_moved_retyped_verified_and_deleted(conn: sqlite3.Connection) -> None:
    added = add_edit(conn, MATCH, "add", author="a", type=EventType.SHOT, t_v=10.0)
    target = f"e:{added.id}"

    add_edit(conn, MATCH, "move", author="a", target=target, t_v=20.0)
    add_edit(conn, MATCH, "retype", author="a", target=target, type=EventType.CORNER)
    add_edit(conn, MATCH, "verify", author="a", target=target)

    event = _event(conn, target)
    assert (event.t_v, event.type, event.verified, event.n_edits) == (
        20.0,
        EventType.CORNER,
        True,
        3,
    )
    add_edit(conn, MATCH, "delete", author="a", target=target)
    assert get_current_events(conn, MATCH) == []


@pytest.mark.unit
def test_verify_sets_flag_and_undo_clears_it(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)
    target = f"p:{ids[0]}"
    assert _event(conn, target).verified is False

    verify = add_edit(conn, MATCH, "verify", author="a", target=target)
    assert _event(conn, target).verified is True

    undo_edit(conn, verify.id, "a")
    assert _event(conn, target).verified is False


@pytest.mark.unit
def test_verified_event_can_still_be_edited(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)
    target = f"p:{ids[0]}"
    add_edit(conn, MATCH, "verify", author="a", target=target)

    add_edit(conn, MATCH, "move", author="a", target=target, t_v=77.0)

    event = _event(conn, target)
    assert (event.t_v, event.verified) == (77.0, True)


@pytest.mark.unit
@pytest.mark.parametrize("target", ["p:9999", "e:9999", "p:0"])
def test_edit_of_a_missing_event_is_rejected_and_not_written(
    conn: sqlite3.Connection, target: str
) -> None:
    _seed(conn)

    with pytest.raises(StorageError, match="does not exist"):
        add_edit(conn, MATCH, "move", author="a", target=target, t_v=1.0)

    assert _table_rows(conn, "edits") == []


@pytest.mark.unit
def test_edit_of_an_event_from_another_match_is_rejected(conn: sqlite3.Connection) -> None:
    ids = _seed(conn, OTHER)

    with pytest.raises(StorageError, match="does not exist"):
        add_edit(conn, MATCH, "delete", author="a", target=f"p:{ids[0]}")


@pytest.mark.unit
def test_edit_of_a_deleted_event_is_rejected_until_the_delete_is_undone(
    conn: sqlite3.Connection,
) -> None:
    ids = _seed(conn)
    target = f"p:{ids[0]}"
    delete = add_edit(conn, MATCH, "delete", author="a", target=target)

    with pytest.raises(StorageError, match="deleted"):
        add_edit(conn, MATCH, "move", author="a", target=target, t_v=1.0)
    with pytest.raises(StorageError, match="deleted"):
        add_edit(conn, MATCH, "delete", author="a", target=target)

    undo_edit(conn, delete.id, "a")
    add_edit(conn, MATCH, "move", author="a", target=target, t_v=1.0)


@pytest.mark.unit
def test_edit_of_an_added_event_whose_add_was_undone_is_rejected(
    conn: sqlite3.Connection,
) -> None:
    added = add_edit(conn, MATCH, "add", author="a", type=EventType.SHOT, t_v=1.0)
    undo_edit(conn, added.id, "a")

    with pytest.raises(StorageError, match="does not exist"):
        add_edit(conn, MATCH, "move", author="a", target=f"e:{added.id}", t_v=2.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs"),
    [
        {"action": "add", "t_v": 1.0},  # no type
        {"action": "add", "type": EventType.SHOT},  # no time
        {"action": "add", "type": EventType.SHOT, "t_v": 1.0, "target": "p:1"},
        {"action": "move", "t_v": 1.0},  # no target
        {"action": "move", "target": "p:1"},  # no time
        {"action": "retype", "target": "p:1"},  # no type
        {"action": "delete"},  # no target
        {"action": "verify", "target": "p:1", "t_v": 1.0},  # extra field
        {"action": "move", "target": "p:1", "t_v": -1.0},
        {"action": "move", "target": "p:1", "t_v": math.nan},
        {"action": "move", "target": "p:1", "t_v": math.inf},
        {"action": "move", "target": "bogus", "t_v": 1.0},
        {"action": "move", "target": "p:", "t_v": 1.0},
        {"action": "undo", "target": "p:1"},  # undo has its own function
        {"action": "add", "type": EventType.SHOT, "t_v": 1.0, "author": ""},
    ],
)
def test_add_edit_invalid_arguments_raise_storage_error(
    conn: sqlite3.Connection, kwargs: dict[str, object]
) -> None:
    _seed(conn)
    params: dict[str, object] = {"author": "a", **kwargs}

    with pytest.raises(StorageError, match="invalid edit"):
        add_edit(conn, MATCH, **params)  # type: ignore[arg-type]  # deliberately invalid

    assert _table_rows(conn, "edits") == []


@pytest.mark.unit
def test_add_edit_empty_match_id_raises_storage_error(conn: sqlite3.Connection) -> None:
    with pytest.raises(StorageError):
        add_edit(conn, "", "add", author="a", type=EventType.SHOT, t_v=1.0)


# --- undo ------------------------------------------------------------------------------


@pytest.mark.unit
def test_double_undo_of_the_same_edit_is_rejected(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)
    move = add_edit(conn, MATCH, "move", author="a", target=f"p:{ids[0]}", t_v=1.0)
    undo_edit(conn, move.id, "a")

    with pytest.raises(StorageError, match="already undone"):
        undo_edit(conn, move.id, "a")


@pytest.mark.unit
def test_undo_of_an_undo_is_rejected(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)
    move = add_edit(conn, MATCH, "move", author="a", target=f"p:{ids[0]}", t_v=1.0)
    undo = undo_edit(conn, move.id, "a")

    with pytest.raises(StorageError, match="cannot be undone"):
        undo_edit(conn, undo.id, "a")

    assert _event(conn, f"p:{ids[0]}").t_v == 100.0  # state unchanged


@pytest.mark.unit
def test_undo_of_unknown_edit_or_empty_author_is_rejected(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)
    move = add_edit(conn, MATCH, "move", author="a", target=f"p:{ids[0]}", t_v=1.0)

    with pytest.raises(StorageError, match="does not exist"):
        undo_edit(conn, 9999, "a")
    with pytest.raises(StorageError, match="author"):
        undo_edit(conn, move.id, "")


@pytest.mark.unit
def test_undo_record_points_to_the_edit_and_keeps_the_match(conn: sqlite3.Connection) -> None:
    ids = _seed(conn, OTHER)
    move = add_edit(conn, OTHER, "move", author="a", target=f"p:{ids[0]}", t_v=1.0)

    undo = undo_edit(conn, move.id, "b")

    assert (undo.action, undo.undo_of, undo.match_id, undo.author) == ("undo", move.id, OTHER, "b")
    assert undo.event_id is None


# --- counting --------------------------------------------------------------------------


@pytest.mark.unit
def test_count_effective_edits_skips_undone_edits_and_undo_records(
    conn: sqlite3.Connection,
) -> None:
    ids = _seed(conn)
    move = add_edit(conn, MATCH, "move", author="a", target=f"p:{ids[0]}", t_v=1.0)
    add_edit(conn, MATCH, "verify", author="a", target=f"p:{ids[1]}")
    add_edit(conn, MATCH, "add", author="a", type=EventType.SHOT, t_v=5.0)
    assert count_effective_edits(conn, MATCH) == 3

    undo_edit(conn, move.id, "a")

    assert count_effective_edits(conn, MATCH) == 2


@pytest.mark.unit
def test_count_effective_edits_ignores_edits_of_an_undone_add(conn: sqlite3.Connection) -> None:
    added = add_edit(conn, MATCH, "add", author="a", type=EventType.SHOT, t_v=5.0)
    add_edit(conn, MATCH, "move", author="a", target=f"e:{added.id}", t_v=6.0)
    assert count_effective_edits(conn, MATCH) == 2

    undo_edit(conn, added.id, "a")

    assert count_effective_edits(conn, MATCH) == 0


@pytest.mark.unit
def test_count_effective_edits_counts_edits_of_deleted_events(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)
    add_edit(conn, MATCH, "move", author="a", target=f"p:{ids[0]}", t_v=1.0)
    add_edit(conn, MATCH, "delete", author="a", target=f"p:{ids[0]}")

    assert count_effective_edits(conn, MATCH) == 2


@pytest.mark.unit
def test_two_matches_in_one_database_do_not_mix(conn: sqlite3.Connection) -> None:
    ids_1 = _seed(conn, MATCH)
    _seed(conn, OTHER)
    add_edit(conn, MATCH, "delete", author="a", target=f"p:{ids_1[0]}")
    add_edit(conn, OTHER, "add", author="a", type=EventType.FOUL, t_v=9.0)
    add_edit(conn, OTHER, "add", author="a", type=EventType.FOUL, t_v=19.0)

    assert len(get_current_events(conn, MATCH)) == 2
    assert len(get_current_events(conn, OTHER)) == 5
    assert {e.match_id for e in get_current_events(conn, OTHER)} == {OTHER}
    assert count_effective_edits(conn, MATCH) == 1
    assert count_effective_edits(conn, OTHER) == 2


# --- add_predictions -------------------------------------------------------------------


@pytest.mark.unit
def test_add_predictions_returns_ids_in_order_and_stores_run_id(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)

    assert ids == sorted(ids) and len(set(ids)) == 3
    rows = conn.execute("SELECT run_id, created_at FROM predictions").fetchall()
    assert {row["run_id"] for row in rows} == {"run-1"}
    assert all(row["created_at"] for row in rows)


@pytest.mark.unit
def test_add_predictions_empty_batch_is_a_noop(conn: sqlite3.Connection) -> None:
    assert add_predictions(conn, MATCH, "r", []) == []
    assert get_current_events(conn, MATCH) == []


@pytest.mark.unit
@pytest.mark.parametrize(("match_id", "run_id"), [("", "r"), ("m", "")])
def test_add_predictions_empty_identifiers_raise(
    conn: sqlite3.Connection, match_id: str, run_id: str
) -> None:
    with pytest.raises(StorageError):
        add_predictions(conn, match_id, run_id, [make_new_prediction("shot", 1.0)])


@pytest.mark.unit
def test_add_predictions_error_in_the_middle_leaves_nothing(conn: sqlite3.Connection) -> None:
    _seed(conn)
    # model_construct skips validation, so the bad row is caught only by the table CHECK.
    broken = NewPrediction.model_construct(type="offside", t_v=3.0, source="model")
    batch = [make_new_prediction("shot", 1.0), make_new_prediction("shot", 2.0), broken]

    with pytest.raises(StorageError):
        add_predictions(conn, MATCH, "run-2", batch)

    assert len(_table_rows(conn, "predictions")) == 3  # only the seed
    assert not conn.in_transaction


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"t_v": -1.0},
        {"t_v": math.nan},
        {"t_v": math.inf},
        {"t_v": "12"},
        {"score": math.nan},
        {"half": 3},
        {"team": "red"},
        {"source": "guess"},
        {"type": "penalty"},
    ],
)
def test_new_prediction_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {"type": "shot", "t_v": 1.0, "source": "model"}

    with pytest.raises(ValueError, match="validation error"):
        NewPrediction.model_validate({**base, **kwargs})


# --- runs: only the latest run of a match is current (D-028) ---------------------------


def _run(conn: sqlite3.Connection, run_id: str, *times: float, match_id: str = MATCH) -> list[int]:
    """Write one run of shots at the given video times."""
    return add_predictions(
        conn, match_id, run_id, [make_new_prediction("shot", t, score=0.5) for t in times]
    )


@pytest.mark.unit
def test_events_current_shows_only_the_latest_run_and_keeps_the_old_one(
    conn: sqlite3.Connection,
) -> None:
    _run(conn, "run-1", 10.0, 20.0, 30.0)
    new_ids = _run(conn, "run-2", 15.0, 25.0)

    events = get_current_events(conn, MATCH)

    assert [e.event_id for e in events] == [f"p:{i}" for i in new_ids]
    assert [e.t_v for e in events] == [15.0, 25.0]
    assert latest_run_id(conn, MATCH) == "run-2"
    assert len(_table_rows(conn, "predictions")) == 5  # the old run stays for audit


@pytest.mark.unit
def test_latest_run_is_the_one_written_last_not_the_one_sorted_last(
    conn: sqlite3.Connection,
) -> None:
    _run(conn, "b", 10.0)
    _run(conn, "a", 20.0)

    assert latest_run_id(conn, MATCH) == "a"
    assert [e.t_v for e in get_current_events(conn, MATCH)] == [20.0]


@pytest.mark.unit
def test_edit_of_a_prediction_from_an_old_run_does_not_touch_the_current_run(
    conn: sqlite3.Connection,
) -> None:
    old_ids = _run(conn, "run-1", 10.0, 20.0)
    add_edit(conn, MATCH, "move", author="a", target=f"p:{old_ids[0]}", t_v=11.0)
    add_edit(conn, MATCH, "delete", author="a", target=f"p:{old_ids[1]}")
    new_ids = _run(conn, "run-2", 10.0, 20.0)  # the same moments, new predictions

    events = get_current_events(conn, MATCH, include_deleted=True)

    assert [(e.event_id, e.t_v, e.deleted, e.n_edits) for e in events] == [
        (f"p:{new_ids[0]}", 10.0, False, 0),
        (f"p:{new_ids[1]}", 20.0, False, 0),
    ]
    assert len(_table_rows(conn, "edits")) == 2  # the journal keeps them


@pytest.mark.unit
def test_edit_of_a_prediction_from_an_old_run_is_rejected(conn: sqlite3.Connection) -> None:
    old_ids = _run(conn, "run-1", 10.0)
    _run(conn, "run-2", 10.0)

    with pytest.raises(StorageError, match="does not exist"):
        add_edit(conn, MATCH, "verify", author="a", target=f"p:{old_ids[0]}")

    assert _table_rows(conn, "edits") == []


@pytest.mark.unit
def test_add_predictions_rejects_a_run_id_that_already_exists_for_the_match(
    conn: sqlite3.Connection,
) -> None:
    _run(conn, "run-1", 10.0)
    _run(conn, "run-2", 20.0)

    with pytest.raises(StorageError, match="already has predictions"):
        _run(conn, "run-1", 30.0, 40.0)  # an old run id, not only the latest one

    assert [e.t_v for e in get_current_events(conn, MATCH)] == [20.0]  # still run-2
    assert len(_table_rows(conn, "predictions")) == 2
    assert not conn.in_transaction


@pytest.mark.unit
def test_add_predictions_allows_the_same_run_id_in_another_match(
    conn: sqlite3.Connection,
) -> None:
    _run(conn, "run-1", 10.0)

    _run(conn, "run-1", 99.0, match_id=OTHER)

    assert [e.t_v for e in get_current_events(conn, OTHER)] == [99.0]
    assert [e.t_v for e in get_current_events(conn, MATCH)] == [10.0]


@pytest.mark.unit
def test_match_without_predictions_has_no_latest_run_and_no_events(
    conn: sqlite3.Connection,
) -> None:
    assert latest_run_id(conn, MATCH) is None
    assert get_current_events(conn, MATCH) == []
    assert count_effective_edits(conn, MATCH) == 0


@pytest.mark.unit
def test_empty_batch_does_not_start_a_run(conn: sqlite3.Connection) -> None:
    _run(conn, "run-1", 10.0)

    assert add_predictions(conn, MATCH, "run-2", []) == []

    assert latest_run_id(conn, MATCH) == "run-1"
    assert [e.t_v for e in get_current_events(conn, MATCH)] == [10.0]


@pytest.mark.unit
def test_events_added_by_the_user_stay_visible_in_every_run(conn: sqlite3.Connection) -> None:
    added = add_edit(conn, MATCH, "add", author="a", type=EventType.GOAL, t_v=500.0)
    assert added.event_id is not None
    assert [e.event_id for e in get_current_events(conn, MATCH)] == [added.event_id]  # no run yet
    _run(conn, "run-1", 10.0)
    add_edit(conn, MATCH, "move", author="a", target=added.event_id, t_v=501.0)

    _run(conn, "run-2", 20.0)

    events = get_current_events(conn, MATCH)
    assert [(e.origin, e.t_v) for e in events] == [("prediction", 20.0), ("user", 501.0)]
    assert events[1].n_edits == 1  # its move is still applied
    assert add_edit(conn, MATCH, "verify", author="a", target=added.event_id).action == "verify"


@pytest.mark.unit
def test_new_run_of_one_match_does_not_touch_another_match(conn: sqlite3.Connection) -> None:
    _run(conn, "run-1", 10.0, 20.0)
    _run(conn, "run-1", 5.0, match_id=OTHER)

    _run(conn, "run-2", 30.0)

    assert [e.t_v for e in get_current_events(conn, MATCH)] == [30.0]
    assert [e.t_v for e in get_current_events(conn, OTHER)] == [5.0]
    assert latest_run_id(conn, OTHER) == "run-1"


@pytest.mark.unit
def test_count_effective_edits_after_a_new_run_counts_only_current_events(
    conn: sqlite3.Connection,
) -> None:
    old_ids = _run(conn, "run-1", 10.0, 20.0)
    add_edit(conn, MATCH, "move", author="a", target=f"p:{old_ids[0]}", t_v=11.0)
    add_edit(conn, MATCH, "verify", author="a", target=f"p:{old_ids[1]}")
    added = add_edit(conn, MATCH, "add", author="a", type=EventType.GOAL, t_v=500.0)
    removed = add_edit(conn, MATCH, "add", author="a", type=EventType.FOUL, t_v=600.0)
    undo_edit(conn, removed.id, "a")
    assert count_effective_edits(conn, MATCH) == 3

    new_ids = _run(conn, "run-2", 10.0)
    assert count_effective_edits(conn, MATCH) == 1  # only the user's add survives

    add_edit(conn, MATCH, "delete", author="a", target=f"p:{new_ids[0]}")
    add_edit(conn, MATCH, "move", author="a", target=f"e:{added.id}", t_v=501.0)
    assert count_effective_edits(conn, MATCH) == 3  # deleted events still count


@pytest.mark.unit
def test_init_db_replaces_an_outdated_view_definition(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    init_db(path)
    with connect(path) as conn:
        _run(conn, "run-1", 10.0)
        _run(conn, "run-2", 20.0)
        conn.execute("DROP VIEW events_current")
        conn.execute(  # the pre-D-028 shape: every run is shown
            "CREATE VIEW events_current AS SELECT 'p:' || id AS event_id, match_id, type, t_v, "
            "half, score, status, team, source, 'prediction' AS origin, 0 AS verified, "
            "0 AS deleted, 0 AS n_edits FROM predictions"
        )
        assert len(get_current_events(conn, MATCH)) == 2

    init_db(path)

    with connect(path) as conn:
        assert [e.t_v for e in get_current_events(conn, MATCH)] == [20.0]


# --- connection, schema, transactions --------------------------------------------------


@pytest.mark.unit
def test_connect_sets_wal_foreign_keys_and_busy_timeout(db_path: Path) -> None:
    with connect(db_path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")  # closed by the context manager


@pytest.mark.unit
def test_foreign_key_rejects_undo_of_a_missing_edit(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO edits (match_id, action, undo_of, author) VALUES ('m', 'undo', 77, 'a')"
        )


@pytest.mark.unit
def test_unique_index_allows_one_undo_per_edit_even_with_raw_sql(
    conn: sqlite3.Connection,
) -> None:
    ids = _seed(conn)
    move = add_edit(conn, MATCH, "move", author="a", target=f"p:{ids[0]}", t_v=1.0)
    undo_edit(conn, move.id, "a")

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO edits (match_id, action, undo_of, author) VALUES (?, 'undo', ?, 'a')",
            (MATCH, move.id),
        )


@pytest.mark.unit
def test_init_db_is_idempotent_and_keeps_data(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "events.db"  # parent directories are created
    init_db(path)
    with connect(path) as conn:
        _seed(conn)

    init_db(path)

    with connect(path) as conn:
        assert len(get_current_events(conn, MATCH)) == 3


@pytest.mark.unit
def test_init_db_rejects_a_foreign_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    init_db(path)
    with connect(path) as conn:
        conn.execute("PRAGMA user_version = 99")

    with pytest.raises(StorageError, match="schema version 99"):
        init_db(path)


@pytest.mark.unit
def test_init_db_and_connect_on_a_file_that_is_not_a_database_raise_storage_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "garbage.db"
    path.write_bytes(b"this is not a sqlite database" * 100)

    with pytest.raises(StorageError):
        init_db(path)
    with pytest.raises(StorageError), connect(path):
        pass


@pytest.mark.unit
def test_reading_an_uninitialized_database_raises_storage_error(tmp_path: Path) -> None:
    with connect(tmp_path / "empty.db") as conn, pytest.raises(StorageError):
        get_current_events(conn, MATCH)
    with connect(tmp_path / "empty.db") as conn, pytest.raises(StorageError):
        count_effective_edits(conn, MATCH)


@pytest.mark.unit
def test_storage_error_is_part_of_the_project_hierarchy() -> None:
    assert issubclass(StorageError, AicoachError)


@pytest.mark.unit
def test_transaction_rolls_back_on_error_and_rejects_nesting(conn: sqlite3.Connection) -> None:
    with pytest.raises(RuntimeError, match="boom"), transaction(conn):
        conn.execute(
            "INSERT INTO predictions (match_id, type, t_v, source, run_id) "
            "VALUES ('m', 'shot', 1.0, 'model', 'r')"
        )
        raise RuntimeError("boom")  # noqa: EM101, TRY003
    assert _table_rows(conn, "predictions") == []

    with transaction(conn), pytest.raises(StorageError, match="already open"), transaction(conn):
        pass


class _RollbackFails(sqlite3.Connection):
    """A connection whose ROLLBACK always fails, like a connection that died mid-transaction."""

    # Any: mirrors the private `_Parameters` alias of the typeshed signature.
    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        if sql == "ROLLBACK":
            raise sqlite3.OperationalError("disk I/O error")  # noqa: EM101, TRY003
        return super().execute(sql, parameters)


@pytest.mark.unit
def test_transaction_keeps_original_error_when_rollback_fails(db_path: Path) -> None:
    conn = sqlite3.connect(db_path, isolation_level=None, factory=_RollbackFails)
    try:
        with (
            capture_logs() as logs,
            pytest.raises(RuntimeError, match="boom"),
            transaction(conn),
        ):
            raise RuntimeError("boom")  # noqa: EM101, TRY003
    finally:
        conn.close()

    assert [(entry["event"], entry["stage"]) for entry in logs] == [("rollback_failed", "storage")]


# --- concurrency (WAL) -----------------------------------------------------------------


@pytest.mark.unit
def test_reader_is_not_blocked_by_an_open_write_transaction(db_path: Path) -> None:
    with connect(db_path) as writer, connect(db_path) as reader:
        _seed(writer)
        with transaction(writer):
            writer.execute(
                "INSERT INTO predictions (match_id, type, t_v, source, run_id) "
                "VALUES (?, 'shot', 1.0, 'model', 'run-1')",
                (MATCH,),
            )
            # Uncommitted: the reader sees the old state, immediately and without an error.
            during = get_current_events(reader, MATCH)
        after = get_current_events(reader, MATCH)

    assert len(during) == 3
    assert len(after) == 4


@pytest.mark.unit
def test_reader_never_sees_a_half_written_batch(db_path: Path) -> None:
    with connect(db_path) as writer, connect(db_path) as reader:
        broken = NewPrediction.model_construct(type="offside", t_v=3.0, source="model")
        with pytest.raises(StorageError):
            add_predictions(writer, MATCH, "r", [make_new_prediction("shot", 1.0), broken])

        assert get_current_events(reader, MATCH) == []


@pytest.mark.unit
def test_second_writer_waits_then_fails_with_storage_error_when_the_lock_is_held(
    db_path: Path,
) -> None:
    with connect(db_path) as first, connect(db_path) as second:
        second.execute("PRAGMA busy_timeout = 50")  # keep the test fast
        with transaction(first), pytest.raises(StorageError, match="locked"):
            add_predictions(second, MATCH, "r", [make_new_prediction("shot", 1.0)])


@pytest.mark.unit
def test_replace_into_cannot_overwrite_append_only_rows(conn: sqlite3.Connection) -> None:
    ids = _seed(conn)

    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute(
            "INSERT OR REPLACE INTO predictions (id, match_id, type, t_v, source, run_id) "
            "VALUES (?, 'm1', 'shot', 1.0, 'model', 'r')",
            (ids[0],),
        )


@pytest.mark.unit
def test_get_current_events_unknown_type_in_filter_raises_storage_error(
    conn: sqlite3.Connection,
) -> None:
    with pytest.raises(StorageError, match="unknown event type"):
        get_current_events(conn, MATCH, types=["bad"])  # type: ignore[list-item]  # invalid


@pytest.mark.unit
def test_failed_commit_rolls_back_so_the_connection_stays_usable(db_path: Path) -> None:
    with connect(db_path) as conn:
        # A deferred foreign key violation is detected only at COMMIT.
        conn.execute("PRAGMA defer_foreign_keys = ON")
        with pytest.raises(StorageError, match="cannot commit"), transaction(conn):
            conn.execute(
                "INSERT INTO edits (match_id, action, undo_of, author) "
                "VALUES ('m', 'undo', 77, 'a')"
            )

        assert not conn.in_transaction
        assert _table_rows(conn, "edits") == []
