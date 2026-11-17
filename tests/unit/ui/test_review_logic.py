"""The pure logic of the review mode: rows, filters, progress, the next row."""

import pytest
from tests.fixtures.ui_synth import make_event

from aicoach.domain.events import EventType
from aicoach.i18n import ru
from aicoach.review.log import ReviewState
from aicoach.review.schemas import ReviewCandidate
from aicoach.scoreboard.schemas import OffsetSegment
from aicoach.storage import CurrentEvent
from aicoach.ui.logic import EventFilter
from aicoach.ui.review_logic import (
    ReviewRow,
    build_review_rows,
    filter_review_rows,
    next_row_id,
    review_progress,
)

NOTHING = ReviewState(frozenset(), frozenset())
SEGMENTS = [
    OffsetSegment(
        start=100.0, end=1000.0, clock_start_s=0, clock_end_s=900, offset_s=100.0, n_readings=9
    )
]


def cand(
    number: int, t_v: float, kind: EventType = EventType.SHOT, half: int = 1
) -> ReviewCandidate:
    return ReviewCandidate(
        id=f"r:{number}",
        type=kind,
        t_v=t_v,
        half=half,  # type: ignore[arg-type]  # 1 or 2
        minute=2,
        score=0.1,
    )


def user_event(event_id: str, t_v: float) -> CurrentEvent:
    return make_event(event_id, t_v=t_v, source=None, score=None, half=None, team=None)


def rows(
    events: list[CurrentEvent],
    candidates: list[ReviewCandidate],
    state: ReviewState = NOTHING,
    *,
    show_all: bool = False,
) -> list[ReviewRow]:
    return build_review_rows(events, candidates, state, SEGMENTS, 30.0, show_all=show_all)


def ids(items: list[ReviewRow]) -> list[str]:
    return [r.row_id for r in items]


@pytest.mark.unit
def test_rows_mix_events_and_candidates_sorted_by_time_events_first_on_a_tie() -> None:
    events = [make_event("p:1", t_v=300.0), make_event("p:2", t_v=100.0)]
    candidates = [cand(1, 300.0), cand(2, 200.0)]

    assert ids(rows(events, candidates)) == ["p:2", "r:2", "p:1", "r:1"]


@pytest.mark.unit
def test_candidate_row_is_marked_and_has_no_team_while_an_event_row_has_its_texts() -> None:
    result = rows([make_event("p:1", t_v=250.0, team="away", source="plates")], [cand(1, 300.0)])

    event_row, candidate_row = result
    assert (event_row.kind, event_row.team_text) == ("event", ru.TEAM_LABELS["away"])
    assert event_row.source_text == ru.SOURCE_LABELS["plates"]
    assert event_row.minute_text == "3'"  # 150 s of clock: minute 3
    assert (candidate_row.kind, candidate_row.source_text) == (
        "candidate",
        ru.REVIEW_CANDIDATE_MARK,
    )
    assert candidate_row.team_text == ""
    assert candidate_row.minute_text == "2'"


@pytest.mark.unit
def test_user_events_are_not_rows_by_default_but_show_in_show_all() -> None:
    events = [make_event("p:1", t_v=100.0), user_event("e:1", 200.0)]

    assert ids(rows(events, [])) == ["p:1"]
    assert ids(rows(events, [], show_all=True)) == ["p:1", "e:1"]


@pytest.mark.unit
def test_decided_rows_are_hidden_by_default_and_marked_in_show_all() -> None:
    events = [make_event("p:1", t_v=100.0), make_event("p:2", t_v=110.0)]
    candidates = [cand(1, 120.0), cand(2, 130.0)]
    state = ReviewState(frozenset({"p:1", "r:1"}), frozenset())

    assert ids(rows(events, candidates, state)) == ["p:2", "r:2"]
    shown = rows(events, candidates, state, show_all=True)
    assert ids(shown) == ["p:1", "p:2", "r:1", "r:2"]
    assert [r.decided for r in shown] == [True, False, True, False]


@pytest.mark.unit
def test_an_accepted_candidate_is_replaced_by_its_event_even_in_show_all() -> None:
    events = [make_event("p:1", t_v=100.0), user_event("e:5", 120.0)]
    state = ReviewState(frozenset({"r:1"}), frozenset({"r:1"}))

    shown = rows(events, [cand(1, 120.0), cand(2, 130.0)], state, show_all=True)

    assert ids(shown) == ["p:1", "e:5", "r:2"]


@pytest.mark.unit
def test_rows_without_events_candidates_or_a_clock_are_fine() -> None:
    assert rows([], []) == []
    unknown = build_review_rows(
        [make_event("p:1", t_v=5000.0)], [], NOTHING, [], 30.0, show_all=False
    )
    assert unknown[0].minute_text is None


@pytest.mark.unit
def test_filter_keeps_types_half_and_score_and_lets_a_row_without_a_score_pass() -> None:
    all_rows = rows(
        [make_event("p:1", t_v=100.0, score=None, half=1), make_event("p:2", t_v=150.0, half=2)],
        [cand(1, 120.0, EventType.FOUL), cand(2, 130.0)],
    )

    only_shots = EventFilter(types=frozenset({EventType.SHOT}), half=None, min_score=0.0)
    assert ids(filter_review_rows(all_rows, only_shots)) == ["r:2"]
    everything = EventFilter(types=frozenset(EventType), half=1, min_score=0.5)
    assert ids(filter_review_rows(all_rows, everything)) == ["p:1"]  # no score: passes
    nothing = EventFilter(types=frozenset(), half=None, min_score=0.0)
    assert filter_review_rows(all_rows, nothing) == []


@pytest.mark.unit
def test_progress_counts_decided_rows_of_the_review_only() -> None:
    events = [make_event("p:1"), make_event("p:2"), user_event("e:9", 50.0)]
    candidates = [cand(1, 10.0), cand(2, 20.0)]
    state = ReviewState(frozenset({"p:1", "r:1", "p:7", "e:9"}), frozenset())

    # Rows: p:1, p:2, r:1, r:2 and p:7 (deleted by the reviewer). e:9 is not a row.
    assert review_progress(events, candidates, state, {"p:7", "e:3"}) == (3, 5)


@pytest.mark.unit
def test_progress_of_an_empty_review_is_zero_of_zero() -> None:
    assert review_progress([], [], NOTHING, set()) == (0, 0)


@pytest.mark.unit
def test_a_deleted_event_that_is_alive_again_counts_once() -> None:
    state = ReviewState(frozenset(), frozenset())

    assert review_progress([make_event("p:1")], [], state, {"p:1"}) == (0, 1)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("order", "current", "expected"),
    [
        (["a", "b", "c"], "a", "b"),
        (["a", "b", "c"], "b", "c"),
        (["a", "b", "c"], "c", None),  # the last row: nothing after it
        (["a", "b", "c"], None, "a"),  # nothing chosen: the first
        (["a", "b", "c"], "gone", "a"),  # not in the list any more: the first
        ([], "a", None),
        ([], None, None),
    ],
)
def test_next_row_id(order: list[str], current: str | None, expected: str | None) -> None:
    assert next_row_id(order, current) == expected
