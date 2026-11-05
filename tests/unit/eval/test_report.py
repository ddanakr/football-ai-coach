"""Markdown rendering of an evaluation report."""

import pytest

from aicoach.eval.metrics import evaluate
from aicoach.eval.report import format_markdown
from tests.fixtures.factories import make_gt, make_pred


@pytest.mark.unit
def test_format_markdown_has_class_rows_and_metric_columns() -> None:
    report = evaluate(
        [make_pred("goal", 100.0), make_pred("shot", 400.0)],
        [make_gt("goal", 101.0), make_gt("corner", 700.0)],
    )

    text = format_markdown(report)

    lines = text.splitlines()
    header = next(line for line in lines if line.startswith("| Class"))
    for column in ("TP", "FP", "FN", "Precision", "Recall", "Median err", "P90 err"):
        assert column in header
    for row_name in ("goal", "shot", "corner", "foul", "subst", "all (micro)"):
        assert any(line.startswith(f"| {row_name} ") for line in lines)
    assert "±5" in text
    assert "±10" in text


@pytest.mark.unit
def test_format_markdown_shows_dash_for_undefined_values() -> None:
    text = format_markdown(evaluate([], []))

    foul_row = next(line for line in text.splitlines() if line.startswith("| foul "))
    cells = [c.strip() for c in foul_row.strip("|").split("|")]
    assert cells[1:4] == ["0", "0", "0"]
    assert all(cell == "—" for cell in cells[4:])


@pytest.mark.unit
def test_format_markdown_formats_numbers_and_intervals() -> None:
    report = evaluate([make_pred("goal", 103.0)], [make_gt("goal", 100.0)], tolerances_s=(5.0,))

    text = format_markdown(report)

    goal_row = next(line for line in text.splitlines() if line.startswith("| goal "))
    assert "1.000" in goal_row
    assert "[1.000, 1.000]" in goal_row
    assert "3.00" in goal_row


@pytest.mark.unit
def test_format_markdown_mentions_half_filter() -> None:
    assert "half 2" in format_markdown(evaluate([], [], half=2))
    assert "all halves" in format_markdown(evaluate([], []))


@pytest.mark.unit
def test_format_markdown_every_table_row_has_same_number_of_cells() -> None:
    report = evaluate([make_pred("shot", 10.0)], [make_gt("shot", 12.0)])

    rows = [line for line in format_markdown(report).splitlines() if line.startswith("|")]

    assert len({line.count("|") for line in rows}) == 1
