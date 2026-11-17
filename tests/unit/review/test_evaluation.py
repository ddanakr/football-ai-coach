"""review-eval: recall of events + candidates, the table, the section in the document."""

from pathlib import Path

import pytest

from aicoach.domain.events import EventType
from aicoach.errors import EvalError
from aicoach.eval.schemas import GroundTruth, GroundTruthEvent, HalfBounds, Prediction
from aicoach.review.evaluation import (
    BEGIN_MARKER,
    END_MARKER,
    ReviewEvalPaths,
    build_half_review,
    candidate_predictions,
    half_tables,
    render_section,
    replace_section,
    run_review_eval,
)
from aicoach.review.schemas import ReviewCandidate
from tests.fixtures.review_synth import write_review_match


def truth_event(kind: EventType, t_v: float, half: int = 1) -> GroundTruthEvent:
    return GroundTruthEvent(type=kind, t_v=t_v, half=half)  # type: ignore[arg-type]  # 1 or 2


def candidate(number: int, kind: EventType, t_v: float, half: int = 1) -> ReviewCandidate:
    return ReviewCandidate(id=f"r:{number}", type=kind, t_v=t_v, half=half, score=0.1)  # type: ignore[arg-type]  # 1 or 2


def event(kind: EventType, t_v: float, half: int = 1) -> Prediction:
    return Prediction(type=kind, t_v=t_v, score=0.9, half=half)  # type: ignore[arg-type]  # 1 or 2


TRUTH = GroundTruth(
    video="v.mp4",
    halves={"1": HalfBounds(start=0.0, end=1000.0), "2": HalfBounds(start=1500.0, end=2500.0)},
    events=(
        truth_event(EventType.SHOT, 100.0),
        truth_event(EventType.SHOT, 300.0),
        truth_event(EventType.FOUL, 500.0),
        truth_event(EventType.SHOT, 1600.0, 2),
    ),
)


@pytest.mark.unit
def test_union_finds_what_the_candidates_add_and_counts_the_rows() -> None:
    events = [event(EventType.SHOT, 101.0), event(EventType.FOUL, 700.0)]
    candidates = [
        candidate(1, EventType.SHOT, 298.0),  # finds the second shot
        candidate(2, EventType.SHOT, 900.0),  # a false row
        candidate(3, EventType.SHOT, 1599.0, 2),  # belongs to the other half
    ]

    review = build_half_review(1, TRUTH, events, candidates)

    events_5, union_5 = review.events.result_for(5.0), review.union.result_for(5.0)
    assert (events_5.per_class[EventType.SHOT].tp, events_5.per_class[EventType.SHOT].recall) == (
        1,
        0.5,
    )
    assert (union_5.per_class[EventType.SHOT].tp, union_5.per_class[EventType.SHOT].recall) == (
        2,
        1.0,
    )
    # The foul 200 s from the mark is a row but finds nothing at any tolerance here.
    assert union_5.per_class[EventType.FOUL].tp == 0
    assert review.n_truth["shot"] == 2
    # Rows to review: events 2 + candidates of this half 2 (the third is of the 2nd half).
    assert union_5.micro.tp + union_5.micro.fp == 4
    assert events_5.micro.tp + events_5.micro.fp == 2


@pytest.mark.unit
def test_tolerance_decides_what_a_row_finds() -> None:
    candidates = [candidate(1, EventType.SHOT, 108.0)]  # 8 s from the mark

    review = build_half_review(1, TRUTH, [], candidates)

    assert review.union.result_for(5.0).per_class[EventType.SHOT].tp == 0
    assert review.union.result_for(10.0).per_class[EventType.SHOT].tp == 1


@pytest.mark.unit
def test_no_events_and_no_candidates_give_zero_found_and_defined_numbers() -> None:
    review = build_half_review(2, TRUTH, [], [])

    micro = review.union.result_for(5.0).micro
    assert (micro.tp, micro.fp, micro.fn) == (0, 0, 1)
    assert micro.recall == 0.0
    text = half_tables(review)
    assert "| все (micro) | 1 | 0 | 0/1 | 0.000 | 0 | 0 | 0/1 | 0.000 |" in text


@pytest.mark.unit
def test_a_class_without_marks_has_a_dash_for_recall() -> None:
    review = build_half_review(1, TRUTH, [], [candidate(1, EventType.GOAL, 50.0)])

    text = half_tables(review)

    assert "| goal | 0 | 0 | 0/0 | — | 1 | 1 | 0/0 | — |" in text


@pytest.mark.unit
def test_candidate_predictions_keep_type_time_and_half() -> None:
    preds = candidate_predictions([candidate(1, EventType.CORNER, 12.0, 2)])

    assert [(p.type, p.t_v, p.half, p.status) for p in preds] == [
        (EventType.CORNER, 12.0, 2, "auto")
    ]


@pytest.mark.unit
def test_section_states_the_threshold_and_has_no_times_names_or_scores() -> None:
    reviews = [build_half_review(h, TRUTH, [], []) for h in (1, 2)]

    section = render_section(reviews, n_model=606, merge_radius_s=3.0)

    assert section.startswith(BEGIN_MARKER) and section.endswith(END_MARKER)
    assert "Порог 0.05 задан заранее" in section
    assert "не подбирался и под 2-й тайм не подбирается" in section
    assert "### 1-й тайм" in section and "### 2-й тайм" in section
    assert "Допуск ±5 с" in section and "Допуск ±10 с" in section
    assert "606" in section


DOC = "# Метрики\n\nцифры 0.889\n"


@pytest.mark.unit
def test_replace_section_appends_after_the_existing_text_and_keeps_it() -> None:
    section = f"{BEGIN_MARKER}\nnew\n{END_MARKER}"

    result = replace_section(DOC, section)

    assert result.startswith(DOC.rstrip("\n"))
    assert result.endswith(section + "\n")
    assert replace_section("", section) == section + "\n"


@pytest.mark.unit
def test_replace_section_replaces_only_between_the_markers_and_is_idempotent() -> None:
    old = f"{BEGIN_MARKER}\nold\n{END_MARKER}"
    document = f"{DOC}\n{old}\n\n## Дальше\n\nтекст\n"
    section = f"{BEGIN_MARKER}\nnew\n{END_MARKER}"

    once = replace_section(document, section)

    assert "old" not in once and "new" in once
    assert once.startswith(DOC) and once.endswith("\n\n## Дальше\n\nтекст\n")
    assert replace_section(once, section) == once


@pytest.mark.unit
@pytest.mark.parametrize("marker", [BEGIN_MARKER, END_MARKER])
def test_replace_section_refuses_a_document_with_one_marker_only(marker: str) -> None:
    with pytest.raises(EvalError, match="only one of the markers"):
        replace_section(f"text\n{marker}\nmore\n", "x")


def write_truth(path: Path) -> Path:
    path.write_text(TRUTH_WITH_EVENTS.model_dump_json(), encoding="utf-8")
    return path


# The synthetic match has a goal at 200 s, a shot at 400 s, a substitution at 600 s and a foul
# before the card plate; the marks lie near them, and one shot mark is only near a candidate.
TRUTH_WITH_EVENTS = GroundTruth(
    video="v.mp4",
    halves={"1": HalfBounds(start=0.0, end=1000.0), "2": HalfBounds(start=1500.0, end=2500.0)},
    events=(
        truth_event(EventType.GOAL, 200.0),
        truth_event(EventType.SHOT, 400.0),
        truth_event(EventType.SHOT, 601.0),  # found only through the candidate at 600 s
        truth_event(EventType.SHOT, 2001.0, 2),  # found only through the candidate at 2000 s
        truth_event(EventType.FOUL, 2001.0, 2),  # fouls get no candidates: not found
    ),
)


@pytest.mark.unit
def test_run_review_eval_writes_the_section_and_leaves_the_document_alone(
    tmp_path: Path,
) -> None:
    match_dir = tmp_path / "m1"
    match_dir.mkdir()
    write_review_match(match_dir, "m1")
    # `final-eval` reads the database named `aicoach_model.db`; here it is the stage's own.
    (match_dir / "aicoach_model.db").write_bytes((match_dir / "aicoach.db").read_bytes())
    output = tmp_path / "metrics.md"
    output.write_text(DOC, encoding="utf-8", newline="\n")

    section = run_review_eval(
        ReviewEvalPaths(
            match_dir=match_dir,
            ground_truth=write_truth(tmp_path / "truth.json"),
            model_db=match_dir / "aicoach_model.db",
            tdeed=match_dir / "tdeed_synth.json",
        ),
        output,
    )

    written = output.read_text(encoding="utf-8")
    assert written.startswith(DOC.rstrip("\n"))
    assert section in written
    assert b"\r\n" not in output.read_bytes()
    # Half 1: shots 2 marks, the event finds one, the candidate at 600 s the other.
    assert "| shot | 2 | 1 | 1/2 | 0.500 | 3 | 4 | 2/2 | 1.000 |" in written
    # Half 2: the shot mark is found only through the candidate; the foul mark by nothing.
    assert "| shot | 1 | 0 | 0/1 | 0.000 | 1 | 1 | 1/1 | 1.000 |" in written
    assert "| foul | 1 | 0 | 0/1 | 0.000 | 0 | 0 | 0/1 | 0.000 |" in written
    # A second run replaces the section: the document does not grow.
    run_review_eval(
        ReviewEvalPaths(
            match_dir,
            tmp_path / "truth.json",
            match_dir / "aicoach_model.db",
            match_dir / "tdeed_synth.json",
        ),
        output,
    )
    assert output.read_text(encoding="utf-8") == written


@pytest.mark.unit
def test_run_review_eval_refuses_a_database_that_is_not_the_frozen_run(tmp_path: Path) -> None:
    match_dir = tmp_path / "m1"
    match_dir.mkdir()
    write_review_match(match_dir, "m1")
    # The model file of this match is not the one that made the database.
    tdeed = match_dir / "tdeed_synth.json"
    tdeed.write_text('{"predictions": []}', encoding="utf-8")

    with pytest.raises(EvalError, match="do not match the frozen configuration"):
        run_review_eval(
            ReviewEvalPaths(
                match_dir, write_truth(tmp_path / "t.json"), match_dir / "aicoach.db", tdeed
            ),
            tmp_path / "out.md",
        )


@pytest.mark.unit
def test_run_review_eval_without_the_database_or_the_scoreboard_is_an_eval_error(
    tmp_path: Path,
) -> None:
    match_dir = tmp_path / "m1"
    match_dir.mkdir()
    paths = ReviewEvalPaths(
        match_dir, write_truth(tmp_path / "t.json"), match_dir / "none.db", match_dir / "x.json"
    )

    with pytest.raises(EvalError, match="scoreboard.json"):
        run_review_eval(paths, tmp_path / "out.md")

    write_review_match(match_dir, "m1")
    with pytest.raises(EvalError, match="database not found"):
        run_review_eval(
            ReviewEvalPaths(
                match_dir,
                tmp_path / "t.json",
                match_dir / "none.db",
                match_dir / "tdeed_synth.json",
            ),
            tmp_path / "out.md",
        )
