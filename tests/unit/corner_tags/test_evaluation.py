"""The truth file of the corners, the score per field and the section of the metrics file."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from aicoach.cli import main
from aicoach.corner_tags.evaluation import (
    BEGIN_MARKER,
    END_MARKER,
    FieldScore,
    filled_rows,
    load_truth,
    parse_truth,
    render_section,
    replace_section,
    rows_of_half,
    score_field,
    score_rows,
    score_table,
)
from aicoach.corner_tags.run import run_corner_eval
from aicoach.corner_tags.schemas import CornerFields, CornerTag, CornerTagsFile
from aicoach.errors import EvalError


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    """`main` points structlog at the captured stderr, which pytest closes after the test."""
    yield
    structlog.reset_defaults()


HEADER = "id,half,minute,video_time,is_corner,attacking_kit,delivery,zone,first_contact,outcome\n"


def truth(*lines: str, comments: bool = True) -> bytes:
    head = "# comment line, with, commas\n# another\n" if comments else ""
    return (head + HEADER + "\n".join(lines) + "\n").encode("utf-8-sig")


def tag(event_id: str, half: int | None = 1, **fields: str) -> CornerTag:
    base = dict.fromkeys(
        ("is_corner", "attacking_kit", "delivery", "zone", "first_contact", "outcome"), "unclear"
    )
    return CornerTag(
        event_id=event_id,
        t_v=10.0,
        half=half,
        fields=CornerFields.model_validate({**base, **fields}),
        status="ok",
        model="test/vlm",
        prompt_version=1,
        before_s=4.0,
        after_s=12.0,
        cost_usd=0.004,
    )


@pytest.mark.unit
def test_parse_truth_skips_comments_reads_russian_names_and_blank_cells() -> None:
    rows = parse_truth(
        truth("p:1,1,1,13:53,да,белый,навес,дальняя штанга,защита,выбили", "p:2,2,52,83:10,,,,,,")
    )

    assert [r.event_id for r in rows] == ["p:1", "p:2"]
    assert rows[0].half == 1
    assert rows[0].values["zone"] == "far_post"
    assert rows[0].values["outcome"] == "cleared"
    assert all(v is None for v in rows[1].values.values())
    assert filled_rows(rows) == 1


@pytest.mark.unit
def test_parse_truth_accepts_semicolons_and_no_comments() -> None:
    text = (
        HEADER.replace(",", ";") + "p:1;1;1;13:53;yes;white;cross;center;attack;goal\n"
    ).encode()

    rows = parse_truth(text)

    assert rows[0].values["outcome"] == "goal"


@pytest.mark.unit
def test_parse_truth_a_wrong_value_names_the_row_and_the_column() -> None:
    with pytest.raises(EvalError, match=r"row 3 \(p:2\), column zone: 'middle' is not allowed"):
        parse_truth(truth("p:1,1,1,1,yes,,,,,", "p:2,1,1,1,yes,,,middle,,"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "no header row"),
        (b"# only comments\n", "no header row"),
        (b"id,half\np:1,1\n", "lacks the columns"),
        (b"\xff\xfe\x00", "not UTF-8"),
        (truth("p:1,1,1,1,,,,,,", "p:1,1,1,1,,,,,,"), "repeated id"),
        (truth(",1,1,1,,,,,,"), "empty or repeated id"),
    ],
)
def test_parse_truth_rejects_a_broken_file(data: bytes, message: str) -> None:
    with pytest.raises(EvalError, match=message):
        parse_truth(data)


@pytest.mark.unit
def test_load_truth_of_a_missing_file_is_an_eval_error(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="cannot read the truth file"):
        load_truth(tmp_path / "nope.csv")


@pytest.mark.unit
def test_score_field_counts_hit_miss_unclear_untagged_and_skipped_apart() -> None:
    rows = parse_truth(
        truth(
            "p:1,1,1,1,yes,,cross,,,",  # hit
            "p:2,1,1,1,yes,,cross,,,",  # miss
            "p:3,1,1,1,yes,,cross,,,",  # the model says unclear: not a miss, not a hit
            "p:4,1,1,1,yes,,cross,,,",  # no tag at all
            "p:5,1,1,1,yes,,unclear,,,",  # truth cannot say: left out
            "p:6,1,1,1,yes,,,,,",  # empty truth: left out
            "p:7,1,1,1,no,,cross,,,",  # not a corner: left out
        )
    )
    tags = {
        "p:1": tag("p:1", delivery="cross"),
        "p:2": tag("p:2", delivery="short"),
        "p:3": tag("p:3", delivery="unclear"),
        "p:5": tag("p:5", delivery="cross"),
        "p:6": tag("p:6", delivery="cross"),
        "p:7": tag("p:7", delivery="cross"),
    }

    score = score_field("delivery", rows, tags)

    assert (score.hits, score.misses, score.model_unclear, score.untagged) == (1, 1, 1, 1)
    assert score.skipped == 3
    assert score.n_truth == 4
    assert score.answered == 2
    assert score.accuracy_of_answers == 0.5
    assert score.accuracy_of_all == 0.25


@pytest.mark.unit
def test_score_field_is_corner_is_scored_on_every_row_with_a_known_value() -> None:
    rows = parse_truth(truth("p:1,1,1,1,yes,,,,,", "p:2,1,1,1,no,,,,,"))
    tags = {"p:1": tag("p:1", is_corner="yes"), "p:2": tag("p:2", is_corner="yes")}

    score = score_field("is_corner", rows, tags)

    assert (score.hits, score.misses, score.skipped) == (1, 1, 0)


@pytest.mark.unit
def test_score_field_with_nothing_to_score_has_no_accuracy_and_no_division_by_zero() -> None:
    empty = score_field("zone", [], {})
    unclear_only = score_field("zone", parse_truth(truth("p:1,1,1,1,yes,,,,,")), {})

    assert empty.accuracy_of_answers is None
    assert empty.accuracy_of_all is None
    assert (unclear_only.n_truth, unclear_only.skipped) == (0, 1)


@pytest.mark.unit
def test_score_field_all_model_answers_unclear_has_no_accuracy_of_answers() -> None:
    rows = parse_truth(truth("p:1,1,1,1,yes,,,center,,"))
    score = score_field("zone", rows, {"p:1": tag("p:1")})

    assert score.accuracy_of_answers is None
    assert score.accuracy_of_all == 0.0
    assert score.model_unclear == 1


@pytest.mark.unit
def test_score_rows_has_one_entry_per_field_in_schema_order() -> None:
    names = [s.field for s in score_rows([], {})]

    assert names == ["is_corner", "attacking_kit", "delivery", "zone", "first_contact", "outcome"]


@pytest.mark.unit
def test_rows_of_half_takes_the_truth_column_then_the_tag_then_nothing() -> None:
    rows = parse_truth(
        truth("p:1,1,1,1,yes,,,,,", "p:2,,1,1,yes,,,,,", "p:3,,1,1,yes,,,,,", "p:4,2,1,1,yes,,,,,")
    )
    tags = {"p:2": tag("p:2", half=2), "p:4": tag("p:4", half=1)}

    assert [r.event_id for r in rows_of_half(rows, tags, 1)] == ["p:1"]
    assert [r.event_id for r in rows_of_half(rows, tags, 2)] == ["p:2", "p:4"]


@pytest.mark.unit
def test_score_table_shows_percent_or_a_dash() -> None:
    scores = [
        FieldScore("zone", 4, 1, 1, 1, 1, 0),
        FieldScore("outcome", 0, 0, 0, 0, 0, 3),
    ]

    table = score_table(scores)

    assert "| zone | 4 | 1 | 1 | 1 | 1 | 50 % | 25 % |" in table
    assert "| outcome | 0 | 0 | 0 | 0 | 0 | — | — |" in table


@pytest.mark.unit
def test_render_section_states_the_pilot_status_and_the_role_of_each_half() -> None:
    scores = [FieldScore("zone", 2, 1, 1, 0, 0, 0)]

    section = render_section(
        {1: scores, 2: scores},
        n_corners={1: 4, 2: 6},
        model="test/vlm",
        prompt_version=1,
        cost_usd=0.04,
    )

    assert section.startswith(BEGIN_MARKER)
    assert section.endswith(END_MARKER)
    assert "запрос подбирался на этих угловых" in section
    assert "запрос не менялся после подбора" in section
    assert "пилот" in section
    assert "0.0400 USD" in section
    assert "не ошибка и не попадание" in section


@pytest.mark.unit
def test_render_section_unknown_cost() -> None:
    section = render_section({1: []}, n_corners={}, model="m", prompt_version=1, cost_usd=None)

    assert "стоимость разметки всех угловых матча неизвестна" in section


@pytest.mark.unit
def test_replace_section_appends_replaces_and_leaves_the_rest_alone() -> None:
    section = f"{BEGIN_MARKER}\nnew\n{END_MARKER}"
    document = "# Metrics\n\nold numbers\n"

    appended = replace_section(document, section)
    replaced = replace_section(appended.replace("new", "old section"), section)

    assert appended == document + "\n" + section + "\n"
    assert replaced == appended
    assert replace_section("", section) == section + "\n"


@pytest.mark.unit
def test_replace_section_with_one_marker_only_is_an_error() -> None:
    with pytest.raises(EvalError, match="only one of the markers"):
        replace_section(f"text {BEGIN_MARKER} text", "x")


def write_files(tmp_path: Path, lines: list[str], tags: list[CornerTag]) -> tuple[Path, Path]:
    match_dir = tmp_path / "match"
    match_dir.mkdir()
    truth_path = match_dir / "corner_truth.csv"
    truth_path.write_bytes(truth(*lines))
    (match_dir / "corner_tags.json").write_text(
        CornerTagsFile(match_id="m", tags=tags).model_dump_json(), encoding="utf-8"
    )
    return match_dir, truth_path


@pytest.mark.unit
def test_run_corner_eval_without_final_scores_only_the_first_half(tmp_path: Path) -> None:
    match_dir, truth_path = write_files(
        tmp_path,
        ["p:1,1,1,1,yes,white,cross,,,", "p:2,2,52,1,yes,red,short,,,"],
        [
            tag("p:1", 1, attacking_kit="white", delivery="cross"),
            tag("p:2", 2, attacking_kit="blue", delivery="short"),
        ],
    )
    output = tmp_path / "metrics.md"
    output.write_text("# Metrics\n\nkeep me\n", encoding="utf-8")

    scores = run_corner_eval(match_dir, truth_path, output, final=False)

    assert list(scores) == [1]
    document = output.read_text(encoding="utf-8")
    assert document.startswith("# Metrics\n\nkeep me\n")
    assert "### 1-й тайм" in document
    assert "### 2-й тайм" not in document
    assert "\r" not in document


@pytest.mark.unit
def test_run_corner_eval_with_final_scores_both_halves_and_replaces_the_section(
    tmp_path: Path,
) -> None:
    match_dir, truth_path = write_files(
        tmp_path,
        ["p:1,1,1,1,yes,white,cross,,,", "p:2,2,52,1,yes,red,short,,,"],
        [
            tag("p:1", 1, attacking_kit="white", delivery="cross"),
            tag("p:2", 2, attacking_kit="blue", delivery="short"),
        ],
    )
    output = tmp_path / "metrics.md"

    run_corner_eval(match_dir, truth_path, output, final=False)
    scores = run_corner_eval(match_dir, truth_path, output, final=True)

    assert list(scores) == [1, 2]
    document = output.read_text(encoding="utf-8")
    assert document.count(BEGIN_MARKER) == 1
    assert "### 2-й тайм" in document
    kit = next(s for s in scores[2] if s.field == "attacking_kit")
    assert (kit.hits, kit.misses) == (0, 1)


@pytest.mark.unit
def test_run_corner_eval_an_unfilled_truth_file_stops_with_a_clear_message(tmp_path: Path) -> None:
    match_dir, truth_path = write_files(tmp_path, ["p:1,1,1,1,,,,,,"], [tag("p:1")])

    with pytest.raises(EvalError, match="not filled in yet"):
        run_corner_eval(match_dir, truth_path, tmp_path / "m.md", final=False)
    assert not (tmp_path / "m.md").exists()


@pytest.mark.unit
def test_run_corner_eval_without_tags_says_to_run_corner_tags(tmp_path: Path) -> None:
    match_dir, truth_path = write_files(tmp_path, ["p:1,1,1,1,yes,,,,,"], [])
    (match_dir / "corner_tags.json").unlink()

    with pytest.raises(EvalError, match="run corner-tags first"):
        run_corner_eval(match_dir, truth_path, tmp_path / "m.md", final=False)


@pytest.mark.unit
def test_corner_eval_command_prints_the_halves_and_an_error_exits_with_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    match_dir, truth_path = write_files(
        tmp_path, ["p:1,1,1,1,yes,white,,,,"], [tag("p:1", 1, attacking_kit="white")]
    )
    output = tmp_path / "metrics.md"

    code = main(["corner-eval", "--match-dir", str(match_dir), "--output", str(output)])
    out = capsys.readouterr().out

    assert code == 0
    assert "halves: 1" in out

    truth_path.write_bytes(truth("p:1,1,1,1,,,,,,"))
    code = main(["corner-eval", "--match-dir", str(match_dir), "--output", str(output)])

    assert code == 2
    assert "not filled in yet" in capsys.readouterr().err
