"""The review journal: entries, the file, and the state folded from it."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aicoach.domain.events import EventType
from aicoach.errors import StorageError
from aicoach.review.log import (
    LOG_FILE,
    Action,
    ReviewLogEntry,
    append_entry,
    deleted_ids,
    entries_for_layer,
    make_entry,
    read_log,
    review_state,
)

T0 = datetime(2026, 11, 18, 12, 0, 0, tzinfo=UTC)


def entry(
    row_id: str,
    action: Action,
    minutes: int = 0,
    event_id: str | None = None,
    t_v: float = 100.0,
) -> ReviewLogEntry:
    return make_entry(
        row_id,
        action,
        t_v,
        EventType.SHOT,
        event_id=event_id,
        now=T0 + timedelta(minutes=minutes),
    )


@pytest.mark.unit
def test_make_entry_stamps_utc_time_by_default() -> None:
    made = make_entry("r:1", "reject", 12.5, EventType.FOUL)

    assert made.ts.utcoffset() == timedelta(0)
    assert (made.id, made.action, made.t_v, made.type) == ("r:1", "reject", 12.5, EventType.FOUL)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("row_id", "action", "event_id"),
    [
        ("p:1", "accept", "e:2"),  # accept decides a candidate
        ("r:1", "confirm", None),  # confirm decides an event
        ("r:1", "delete", None),
        ("p:1", "reject", None),
        ("r:1", "accept", None),  # an accept names the event it created
        ("r:1", "reject", "e:2"),  # nothing else does
        ("x:1", "reject", None),  # not a row id
        ("r:1", "accept", "p:2"),  # the created event is a user event
    ],
)
def test_entry_with_an_id_that_does_not_fit_the_action_is_refused(
    row_id: str, action: Action, event_id: str | None
) -> None:
    with pytest.raises(ValueError, match=r"action|event_id|string"):
        make_entry(row_id, action, 1.0, EventType.SHOT, event_id=event_id)


@pytest.mark.unit
@pytest.mark.parametrize("t_v", [-1.0, float("nan"), float("inf")])
def test_entry_with_a_bad_time_is_refused(t_v: float) -> None:
    with pytest.raises(ValueError, match="t_v|input"):
        make_entry("r:1", "reject", t_v, EventType.SHOT)


@pytest.mark.unit
def test_entries_round_trip_through_the_file_in_order(tmp_path: Path) -> None:
    written = [
        entry("r:1", "accept", 0, event_id="e:5"),
        entry("p:3", "confirm", 1),
        entry("r:2", "reject", 2),
        entry("p:4", "delete", 3),
    ]

    for item in written:
        append_entry(tmp_path, item)

    assert read_log(tmp_path) == written
    lines = (tmp_path / LOG_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert '"id":"r:1"' in lines[0]
    assert "2026-11-18T12:00:00Z" in lines[0]  # UTC time in the line


@pytest.mark.unit
def test_missing_journal_is_empty_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    assert read_log(tmp_path) == []
    append_entry(tmp_path, entry("r:1", "reject"))
    with (tmp_path / LOG_FILE).open("a", encoding="utf-8") as handle:
        handle.write("\n   \n")

    assert len(read_log(tmp_path)) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "{}",
        '{"ts": "2026-11-18T12:00:00", "id": "r:1", "action": "reject", "t_v": 1, "type": "shot"}',
        '{"ts": "2026-11-18T12:00:00Z", "id": "r:1", "action": "boom", "t_v": 1, "type": "shot"}',
    ],
)
def test_bad_line_is_a_storage_error_naming_the_line_but_not_its_text(
    tmp_path: Path, line: str
) -> None:
    append_entry(tmp_path, entry("r:1", "reject"))
    with (tmp_path / LOG_FILE).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")

    with pytest.raises(StorageError, match="line 2") as caught:
        read_log(tmp_path)

    assert line not in str(caught.value)


@pytest.mark.unit
def test_unreadable_journal_is_a_storage_error(tmp_path: Path) -> None:
    (tmp_path / LOG_FILE).write_bytes(b"\xff\xfe\x00bad")

    with pytest.raises(StorageError, match="cannot read"):
        read_log(tmp_path)


@pytest.mark.unit
def test_append_to_a_missing_directory_is_a_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="cannot write"):
        append_entry(tmp_path / "nope", entry("r:1", "reject"))


@pytest.mark.unit
def test_state_of_an_empty_journal_has_nothing_decided() -> None:
    state = review_state([], ["p:1"])

    assert state.decided == frozenset()
    assert state.accepted == frozenset()


@pytest.mark.unit
def test_rejection_and_confirmation_always_stand() -> None:
    entries = [entry("r:1", "reject"), entry("p:2", "confirm")]

    state = review_state(entries, ["p:2"])

    assert state.decided == {"r:1", "p:2"}
    assert state.accepted == frozenset()


@pytest.mark.unit
def test_accepted_candidate_stands_while_its_event_is_alive_and_returns_after_undo() -> None:
    entries = [entry("r:1", "accept", event_id="e:9")]

    alive = review_state(entries, ["e:9", "p:1"])
    undone = review_state(entries, ["p:1"])

    assert alive.decided == {"r:1"}
    assert alive.accepted == {"r:1"}
    assert undone.decided == frozenset()
    assert undone.accepted == frozenset()


@pytest.mark.unit
def test_deleted_event_stands_while_gone_and_returns_for_review_after_undo() -> None:
    entries = [entry("p:3", "delete")]

    gone = review_state(entries, [])
    restored = review_state(entries, ["p:3"])

    assert gone.decided == {"p:3"}
    assert restored.decided == frozenset()


@pytest.mark.unit
def test_the_last_decision_on_a_row_counts() -> None:
    entries = [entry("r:1", "reject", 0), entry("r:1", "accept", 1, event_id="e:7")]

    assert review_state(entries, ["e:7"]).accepted == {"r:1"}
    assert review_state(list(reversed(entries)), ["e:7"]).decided == {"r:1"}  # rejected last
    assert review_state(list(reversed(entries)), ["e:7"]).accepted == frozenset()


@pytest.mark.unit
def test_decisions_on_candidates_that_the_layer_no_longer_has_are_dropped() -> None:
    entries = [
        entry("r:1", "reject", t_v=100.0),  # the same candidate still exists
        entry("r:2", "reject", t_v=200.0),  # r:2 is now another detection
        entry("r:3", "accept", event_id="e:1", t_v=300.0),  # r:3 is gone
        entry("p:4", "confirm", t_v=999.0),  # decisions on events are not tied to the layer
    ]
    layer = [("r:1", EventType.SHOT, 100.0), ("r:2", EventType.SHOT, 250.0)]

    kept = entries_for_layer(entries, layer)

    assert [e.id for e in kept] == ["r:1", "p:4"]
    assert entries_for_layer(entries, []) == [entries[3]]
    assert entries_for_layer([], layer) == []
    # Another type at the same time is another detection too.
    other_type = [("r:1", EventType.FOUL, 100.0)]
    assert [e.id for e in entries_for_layer(entries[:1], other_type)] == []


@pytest.mark.unit
def test_deleted_ids_are_the_events_whose_last_decision_is_a_deletion() -> None:
    entries = [
        entry("p:1", "delete", 0),
        entry("p:2", "delete", 1),
        entry("p:2", "confirm", 2),
        entry("r:1", "reject", 3),
    ]

    assert deleted_ids(entries) == {"p:1"}
    assert deleted_ids([]) == frozenset()
