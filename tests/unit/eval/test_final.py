"""Pure helpers of the final evaluation: F1, guard, edits, Gemini shift, agent cost, tables."""

import json

import pytest

from aicoach.errors import EvalError
from aicoach.eval.final import (
    SystemRow,
    check_final_allowed,
    class_table,
    comparison_table,
    cost_table,
    count_edits,
    edits_table,
    f1_score,
    gemini_shift,
    stage_table,
    summarize_agent_log,
    total_edits,
    truth_counts,
)
from aicoach.eval.metrics import EvaluationReport, evaluate
from aicoach.status import StageStatus
from tests.fixtures.factories import make_gt, make_pred


@pytest.mark.parametrize(
    ("p", "r", "expected"),
    [(None, 0.5, None), (0.5, None, None), (0.0, 0.0, 0.0), (1.0, 0.5, pytest.approx(2 / 3))],
)
def test_f1_score_table(p: float | None, r: float | None, expected: float | None) -> None:
    assert f1_score(p, r) == expected


def test_check_final_allowed_half_one_always_passes() -> None:
    check_final_allowed(1, final=False)


def test_check_final_allowed_half_two_needs_the_switch() -> None:
    check_final_allowed(2, final=True)
    with pytest.raises(EvalError, match="--final"):
        check_final_allowed(2, final=False)


@pytest.mark.parametrize("half", [0, 3, -1])
def test_check_final_allowed_rejects_other_halves(half: int) -> None:
    with pytest.raises(EvalError):
        check_final_allowed(half, final=True)


def _report() -> EvaluationReport:
    truth = [make_gt("goal", 100.0), make_gt("foul", 200.0), make_gt("foul", 300.0)]
    preds = [
        make_pred("goal", 101.0, half=1),
        make_pred("foul", 200.0, half=1),
        make_pred("shot", 500.0, half=1),
    ]
    return evaluate(preds, truth, half=1, n_boot=10)


def test_count_edits_splits_deletes_and_adds_per_class() -> None:
    report = _report()

    counts = count_edits(report.result_for(5.0))

    assert counts["goal"] == (0, 0)
    assert counts["foul"] == (0, 1)
    assert counts["shot"] == (1, 0)
    assert counts["all"] == (1, 1)
    assert total_edits(report, 5.0) == 2


def test_tables_render_numbers_without_crashing_on_empty_classes() -> None:
    report = _report()

    assert "все (micro)" in class_table(report)
    assert "| all | 1 | 1 | 2 |" in edits_table(report)
    table = comparison_table(
        [SystemRow("x", report), SystemRow("y", None, "нет данных")], (5.0, 10.0)
    )
    assert "| y | нет данных |" in table


def test_truth_counts_counts_one_half() -> None:
    events = [make_gt("goal", 1.0), make_gt("goal", 2.0, half=2), make_gt("foul", 3.0, half=2)]

    assert truth_counts(events, 2) == {"goal": 1, "shot": 0, "corner": 0, "foul": 1, "subst": 0}


def test_gemini_shift_flags_a_systematic_offset_without_correcting() -> None:
    truth = [make_gt("shot", 100.0 * i, half=2) for i in range(1, 6)]
    preds = [make_pred("shot", 100.0 * i + 15.0) for i in range(1, 6)]

    shift = gemini_shift(preds, truth)

    assert shift.n_pairs == 5
    assert shift.median_signed_s == 15.0
    assert shift.systematic
    assert preds[0].t_v == 115.0


def test_gemini_shift_small_offset_is_not_systematic() -> None:
    shift = gemini_shift([make_pred("shot", 103.0)], [make_gt("shot", 100.0)])

    assert not shift.systematic
    assert shift.median_abs_s == 3.0


def test_gemini_shift_without_pairs_is_empty() -> None:
    shift = gemini_shift([make_pred("shot", 100.0)], [make_gt("foul", 100.0)])

    assert (shift.n_pairs, shift.median_signed_s, shift.systematic) == (0, None, False)


def test_summarize_agent_log_averages_and_survives_junk() -> None:
    first = {
        "tokens_in": 100,
        "tokens_out": 10,
        "cost_usd": 0.01,
        "duration_ms": 2000,
        "validator": "first_try",
    }
    second = {
        "tokens_in": 300,
        "tokens_out": 30,
        "cost_usd": None,
        "duration_ms": 4000,
        "validator": "retry",
    }
    lines = [json.dumps(first), json.dumps(second), "not json", "[1, 2]", ""]

    cost = summarize_agent_log(lines)

    assert cost.n_records == 2
    assert cost.n_bad_lines == 2
    assert cost.mean_tokens_in == 200.0
    assert cost.mean_cost_usd == 0.01
    assert cost.n_with_cost == 1
    assert cost.mean_duration_s == 3.0
    assert cost.first_try_share == 0.5
    assert "200" in cost_table(cost)


def test_summarize_agent_log_empty_is_all_undefined() -> None:
    cost = summarize_agent_log([])

    assert cost.n_records == 0
    assert cost.mean_tokens_in is None
    assert cost.first_try_share is None
    assert "—" in cost_table(cost)


def test_stage_table_shows_times_and_marks_missing_ones() -> None:
    stages = {
        "scoreboard": StageStatus(
            state="done",
            progress=1.0,
            started_at="2026-11-16T10:00:00.000+00:00",
            finished_at="2026-11-16T10:07:30.250+00:00",
            duration_s=450.25,
        ),
        "plate_names": StageStatus(state="skipped", progress=1.0, error="no tesseract"),
        "tdeed": StageStatus(state="waiting", progress=0.0),
        "untimed": StageStatus(
            state="done", progress=1.0, started_at="2026-11-16T10:00:00.000+00:00"
        ),
        "old": StageStatus(state="done", progress=1.0),
        "quick": StageStatus(state="done", progress=1.0, duration_s=0.0),
        "broken": StageStatus(state="failed", progress=0.3, error="boom"),
    }

    rows = {line.split(" | ")[0][2:]: line for line in stage_table(stages).splitlines()[2:]}

    assert "| 450.2 |" in rows["scoreboard"] or "| 450.3 |" in rows["scoreboard"]
    assert "2026-11-16T10:00:00.000+00:00" in rows["scoreboard"]
    assert rows["plate_names"].endswith("| — | — | — |")
    assert rows["tdeed"].endswith("| — | — | — |")
    assert rows["old"].endswith("| нет данных |")
    assert rows["untimed"].endswith("| — | — | нет данных |")  # a start alone is not shown
    assert rows["quick"].endswith("| 0.0 |")
    assert rows["broken"].endswith("| нет данных |")


def test_stage_table_empty_has_only_the_header() -> None:
    assert len(stage_table({}).splitlines()) == 2
