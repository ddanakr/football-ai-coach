"""Pure logic of the manual edits: time parsing, time checks, the edit to write."""

import math

import pytest
from tests.fixtures.ui_synth import make_event

from aicoach.domain.events import EventType
from aicoach.fusion.schemas import HalfSpan
from aicoach.ui.edits import (
    EditInputError,
    TimeBounds,
    check_time,
    parse_mmss,
    plan_add,
    plan_delete,
    plan_move,
    plan_retype,
    with_derived_half,
)

BOUNDS = TimeBounds(
    halves={1: HalfSpan(start=100.0, end=1000.0), 2: HalfSpan(start=1500.0, end=2500.0)},
    video_duration_s=3000.0,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "seconds"),
    [("00:00", 0.0), ("12:05", 725.0), (" 12:05 ", 725.0), ("125:03", 7503.0), ("0:59", 59.0)],
)
def test_parse_mmss_reads_minutes_and_seconds(text: str, seconds: float) -> None:
    assert parse_mmss(text) == seconds


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "12",
        "12:5",
        "12:60",
        "-1:00",
        "-01:30",
        "1:2:3",
        "ab:cd",
        "12:05.5",
        "12,05",
        "٣:٠٠",
    ],
)
def test_parse_mmss_rejects_anything_but_mm_ss(text: str) -> None:
    with pytest.raises(EditInputError, match="мм:сс"):
        parse_mmss(text)


@pytest.mark.unit
@pytest.mark.parametrize(("t_v", "half"), [(100.0, 1), (1000.0, 1), (1500.0, 2), (3000.0 - 500, 2)])
def test_check_time_gives_the_half_and_includes_the_ends(t_v: float, half: int) -> None:
    assert check_time(t_v, BOUNDS) == half


@pytest.mark.unit
def test_check_time_in_the_break_names_both_halves() -> None:
    with pytest.raises(EditInputError, match="вне таймов") as err:
        check_time(1200.0, BOUNDS)

    assert "1-й тайм 01:40–16:40" in str(err.value)
    assert "2-й тайм 25:00–41:40" in str(err.value)


@pytest.mark.unit
@pytest.mark.parametrize("t_v", [0.0, 99.9, 1000.1, 2500.1])
def test_check_time_before_between_and_after_halves_is_refused(t_v: float) -> None:
    with pytest.raises(EditInputError, match="вне таймов"):
        check_time(t_v, BOUNDS)


@pytest.mark.unit
def test_check_time_past_the_end_of_the_video_is_refused() -> None:
    bounds = TimeBounds(halves=BOUNDS.halves, video_duration_s=2000.0)

    with pytest.raises(EditInputError, match="дальше конца видео"):
        check_time(2100.0, bounds)


@pytest.mark.unit
@pytest.mark.parametrize("t_v", [-1.0, math.nan, math.inf])
def test_check_time_refuses_negative_and_non_finite(t_v: float) -> None:
    with pytest.raises(EditInputError, match="мм:сс"):
        check_time(t_v, BOUNDS)


@pytest.mark.unit
def test_plan_move_gives_a_move_edit_with_the_new_time_and_a_russian_message() -> None:
    event = make_event("p:7", EventType.FOUL, 700.4)

    plan = plan_move(event, "12:00", BOUNDS)

    assert (plan.action, plan.target, plan.t_v, plan.type) == ("move", "p:7", 720.0, None)
    assert plan.message == "Событие «Фол» сдвинуто на 12:00 (1-й тайм)."


@pytest.mark.unit
def test_plan_move_to_the_other_half_names_it() -> None:
    plan = plan_move(make_event(t_v=700.0), "30:00", BOUNDS)

    assert "2-й тайм" in plan.message


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("abc", "мм:сс"),
        ("20:00", "вне таймов"),  # 1200 s: the break
        ("60:00", "дальше конца видео"),  # 3600 s
    ],
)
def test_plan_move_refuses_bad_times(text: str, error: str) -> None:
    with pytest.raises(EditInputError, match=error):
        plan_move(make_event(t_v=700.0), text, BOUNDS)


@pytest.mark.unit
@pytest.mark.parametrize("t_v", [720.0, 720.9])
def test_plan_move_to_the_second_it_already_shows_is_not_a_move(t_v: float) -> None:
    with pytest.raises(EditInputError, match="не изменилось"):
        plan_move(make_event(t_v=t_v), "12:00", BOUNDS)


@pytest.mark.unit
def test_plan_retype_gives_a_retype_edit() -> None:
    plan = plan_retype(make_event("e:3", EventType.SHOT), EventType.CORNER)

    assert (plan.action, plan.target, plan.type, plan.t_v) == (
        "retype",
        "e:3",
        EventType.CORNER,
        None,
    )
    assert plan.message == "Тип события изменён: «Удар» → «Угловой»."


@pytest.mark.unit
def test_plan_retype_to_the_same_type_is_refused() -> None:
    with pytest.raises(EditInputError, match="тот же тип"):
        plan_retype(make_event(type=EventType.SHOT), EventType.SHOT)


@pytest.mark.unit
def test_plan_delete_targets_the_event() -> None:
    plan = plan_delete(make_event("p:2", EventType.GOAL, 745.0))

    assert (plan.action, plan.target, plan.type, plan.t_v) == ("delete", "p:2", None, None)
    assert "«Гол» в 12:25 удалено" in plan.message


@pytest.mark.unit
def test_plan_add_gives_an_add_edit_without_a_target() -> None:
    plan = plan_add(EventType.CORNER, "05:00", BOUNDS)

    assert (plan.action, plan.target, plan.type, plan.t_v) == ("add", None, EventType.CORNER, 300.0)
    assert plan.message == "Добавлено событие «Угловой» в 05:00 (1-й тайм)."


@pytest.mark.unit
@pytest.mark.parametrize("text", ["", "x", "20:00", "99:00"])
def test_plan_add_refuses_bad_times(text: str) -> None:
    with pytest.raises(EditInputError):
        plan_add(EventType.CORNER, text, BOUNDS)


@pytest.mark.unit
def test_with_derived_half_recomputes_it_for_user_events_and_edited_predictions() -> None:
    added = make_event("e:1", t_v=1600.0, half=None, source=None)
    moved = make_event("p:1", t_v=1600.0, half=1).model_copy(update={"n_edits": 1})

    assert with_derived_half(added, BOUNDS).half == 2
    assert with_derived_half(moved, BOUNDS).half == 2


@pytest.mark.unit
def test_with_derived_half_leaves_untouched_predictions_and_missing_bounds_alone() -> None:
    plain = make_event("p:1", t_v=1600.0, half=1)  # stored half is not the time's half

    assert with_derived_half(plain, BOUNDS) is plain
    edited = plain.model_copy(update={"n_edits": 2})
    assert with_derived_half(edited, None) is edited


@pytest.mark.unit
def test_with_derived_half_outside_the_halves_is_unknown() -> None:
    added = make_event("e:1", t_v=1200.0, half=None, source=None)

    assert with_derived_half(added, BOUNDS).half is None
