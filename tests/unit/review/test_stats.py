"""review-stats: the window, the counts and the time from the journal."""

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aicoach.domain.events import EventType
from aicoach.errors import ConfigError, EvalError, StorageError
from aicoach.review.log import Action, ReviewLogEntry, append_entry, make_entry
from aicoach.review.stats import (
    ReviewStats,
    format_stats,
    parse_window,
    review_stats,
    run_review_stats,
)
from tests.fixtures.review_synth import write_review_layer

T0 = datetime(2026, 11, 18, 12, 0, 0, tzinfo=UTC)


def entry(
    row_id: str, action: Action, t_v: float, seconds: float, event_id: str | None = None
) -> ReviewLogEntry:
    return make_entry(
        row_id,
        action,
        t_v,
        EventType.SHOT,
        event_id=event_id,
        now=T0 + timedelta(seconds=seconds),
    )


JOURNAL = [
    entry("r:1", "accept", 100.0, 0, event_id="e:1"),
    entry("p:2", "confirm", 110.0, 10),
    entry("r:3", "reject", 120.0, 25.5),
    entry("p:4", "delete", 130.0, 31),
    entry("p:5", "confirm", 500.0, 900),  # outside the window
]


@pytest.mark.unit
def test_window_counts_rows_decisions_actions_and_time() -> None:
    stats = review_stats(JOURNAL, [99.0, 100.0, 110.0, 125.0, 130.0, 131.0, 500.0], 100.0, 130.0)

    assert stats == ReviewStats(
        start_s=100.0,
        end_s=130.0,
        rows=4,  # the ends are included
        rows_decided=4,
        decisions=4,
        seconds=31.0,
        accepted=1,
        rejected=1,
        confirmed=1,
        deleted=1,
    )


@pytest.mark.unit
def test_time_runs_from_the_first_to_the_last_decision_by_the_clock_not_by_file_order() -> None:
    shuffled = [JOURNAL[3], JOURNAL[0]]  # 31 s, then 0 s

    assert review_stats(shuffled, [], 0.0, 1000.0).seconds == 31.0


@pytest.mark.unit
def test_two_decisions_on_one_row_are_two_decisions_and_one_row() -> None:
    journal = [
        entry("r:1", "reject", 100.0, 0),
        entry("r:1", "accept", 100.0, 4, event_id="e:2"),
    ]

    stats = review_stats(journal, [100.0], 0.0, 200.0)

    assert (stats.decisions, stats.rows_decided, stats.seconds) == (2, 1, 4.0)
    assert (stats.accepted, stats.rejected) == (1, 1)


@pytest.mark.unit
def test_one_decision_takes_zero_seconds_and_none_takes_no_time() -> None:
    assert review_stats(JOURNAL[:1], [], 100.0, 100.0).seconds == 0.0
    empty = review_stats([], [50.0], 0.0, 100.0)
    assert (empty.rows, empty.decisions, empty.seconds) == (1, 0, None)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("start", "end"), [(10.0, 5.0), (-1.0, 5.0), (math.nan, 5.0), (0.0, math.inf)]
)
def test_a_bad_window_is_refused(start: float, end: float) -> None:
    with pytest.raises(EvalError, match="window"):
        review_stats([], [], start, end)


@pytest.mark.unit
def test_parse_window_reads_mmss_and_allows_equal_ends() -> None:
    assert parse_window("12:30", "13:00") == (750.0, 780.0)
    assert parse_window("00:05", "00:05") == (5.0, 5.0)
    assert parse_window("125:03", "126:00") == (7503.0, 7560.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("start", "end"),
    [("12", "13:00"), ("12:30", "abc"), ("", ""), ("12:75", "13:00"), ("-1:00", "2:00")],
)
def test_parse_window_refuses_text_that_is_not_mmss(start: str, end: str) -> None:
    with pytest.raises(ConfigError, match="mm:ss"):
        parse_window(start, end)


@pytest.mark.unit
def test_parse_window_refuses_a_start_after_the_end() -> None:
    with pytest.raises(ConfigError, match="after"):
        parse_window("13:00", "12:30")


@pytest.mark.unit
def test_format_stats_has_every_number_and_says_none_without_time() -> None:
    text = format_stats(review_stats(JOURNAL, [100.0, 110.0], 100.0, 130.0))

    for line in (
        "window: 100-130 s of video",
        "rows in the window: 2",
        "rows with a decision: 4",
        "decisions (journal lines): 4",
        "seconds from the first to the last decision: 31.0",
        "added (candidates accepted): 1",
        "rejected (candidates): 1",
        "confirmed (events): 1",
        "deleted (events): 1",
    ):
        assert line in text
    assert "last decision: none" in format_stats(review_stats([], [], 0.0, 1.0))


@pytest.mark.unit
def test_run_review_stats_counts_the_frozen_predictions_and_the_candidates(
    tmp_path: Path,
) -> None:
    layer = write_review_layer(tmp_path, "m1")
    first = layer.candidates[0]
    append_entry(tmp_path, make_entry(first.id, "reject", first.t_v, first.type, now=T0))

    # The whole first half: the predictions there (4 events) and the candidates (3).
    stats = run_review_stats(tmp_path, "m1", tmp_path / "aicoach.db", 0.0, 1100.0)

    assert stats.rows == 4 + 3
    assert (stats.decisions, stats.rejected) == (1, 1)


@pytest.mark.unit
def test_run_review_stats_without_the_layer_counts_predictions_only(tmp_path: Path) -> None:
    write_review_layer(tmp_path, "m1")
    (tmp_path / "review_candidates.json").unlink()

    stats = run_review_stats(tmp_path, "m1", tmp_path / "aicoach.db", 0.0, 3000.0)

    assert stats.rows == 4  # the four predictions of the synthetic match
    assert stats.decisions == 0


@pytest.mark.unit
def test_run_review_stats_without_the_database_is_a_storage_error(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="does not exist"):
        run_review_stats(tmp_path, "m1", tmp_path / "aicoach.db", 0.0, 10.0)
