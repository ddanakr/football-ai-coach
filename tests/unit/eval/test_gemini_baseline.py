"""The Gemini baseline reader: good file, junk entries, empty array, broken file."""

import json

import pytest

from aicoach.domain.events import EventType
from aicoach.errors import EvalError
from aicoach.eval.gemini_baseline import parse_clock_time, parse_gemini_baseline

HEADER = "source and date\n"


def _text(items: object) -> str:
    return HEADER + json.dumps(items)


def test_parse_normal_file_returns_events_in_seconds() -> None:
    items = [
        {"type": "foul", "t": "01:18:12", "note": "somebody"},
        {"type": "corner", "t": "00:03:05", "note": ""},
    ]

    baseline = parse_gemini_baseline(_text(items))

    assert [(e.type, e.t_v) for e in baseline.events] == [
        (EventType.FOUL, 4692.0),
        (EventType.CORNER, 185.0),
    ]
    assert baseline.header == "source and date"
    assert baseline.skipped == {}


def test_parse_never_keeps_the_note() -> None:
    baseline = parse_gemini_baseline(_text([{"type": "goal", "t": "00:10:00", "note": "Name"}]))

    assert "Name" not in repr(baseline)


def test_parse_empty_array_gives_empty_baseline() -> None:
    baseline = parse_gemini_baseline(_text([]))

    assert baseline.events == ()
    assert baseline.skipped == {}


def test_parse_skips_and_counts_bad_entries() -> None:
    items = [
        "junk line",
        {"type": "offside", "t": "00:10:00"},
        {"type": 5, "t": "00:10:00"},
        {"type": "shot", "t": "ten past"},
        {"type": "shot"},
        {"type": "shot", "t": "00:10:00"},
    ]

    baseline = parse_gemini_baseline(_text(items))

    assert len(baseline.events) == 1
    assert baseline.skipped == {"not_an_object": 1, "unknown_type": 2, "bad_time": 2}


@pytest.mark.parametrize(
    "text",
    ["", "header only", HEADER + "not json", HEADER + '{"type": "shot"}', "\n[]"],
)
def test_parse_broken_file_raises(text: str) -> None:
    with pytest.raises(EvalError):
        parse_gemini_baseline(text)


def test_parse_tolerates_a_byte_order_mark() -> None:
    baseline = parse_gemini_baseline("﻿" + _text([{"type": "goal", "t": "00:01:00"}]))

    assert baseline.header == "source and date"
    assert len(baseline.events) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("01:02:03", 3723.0),
        ("02:03", 123.0),
        ("00:00:00", 0.0),
        ("00:60:00", None),
        ("1:2:3", None),
        ("", None),
        (None, None),
        (125, None),
    ],
)
def test_parse_clock_time_table(text: object, expected: float | None) -> None:
    assert parse_clock_time(text) == expected
