"""Glyph segmentation, template matching, parsing and clustering on synthetic scoreboards."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.board import crop_fraction
from aicoach.scoreboard.glyphs import (
    FieldReading,
    TemplateSet,
    cluster_patches,
    match_glyph,
    parse_added_text,
    parse_clock_text,
    parse_score_text,
    read_field,
    read_gray,
    save_template,
    segment_glyphs,
    templates_from_dir,
    write_png,
)
from aicoach.scoreboard.imaging import (
    CLOCK_ROI,
    SCORE_AWAY_ROI,
    SCORE_HOME_ROI,
    ImageParams,
)
from tests.fixtures.scoreboard_synth import board_frame, grass, synthetic_templates

PARAMS = ImageParams()


@pytest.fixture(scope="module")
def templates() -> tuple[TemplateSet, TemplateSet]:
    return synthetic_templates(PARAMS)


@pytest.mark.unit
def test_segment_glyphs_clock_gives_digits_and_one_colon_in_order() -> None:
    crop = crop_fraction(board_frame(clock="69:17"), CLOCK_ROI)
    glyphs = segment_glyphs(crop, PARAMS)
    assert len(glyphs) == 5
    assert [g.x0 for g in glyphs] == sorted(g.x0 for g in glyphs)


@pytest.mark.unit
def test_segment_glyphs_empty_field_gives_no_glyphs() -> None:
    crop = crop_fraction(grass(), CLOCK_ROI)
    assert segment_glyphs(crop, PARAMS) == []


@pytest.mark.unit
def test_segment_glyphs_ignores_specks_below_the_pixel_limit() -> None:
    crop = np.zeros((20, 30, 3), dtype=np.uint8)
    crop[5, 5] = 255
    crop[6, 5] = 255
    assert segment_glyphs(crop, PARAMS) == []


@pytest.mark.unit
@pytest.mark.parametrize("clock", ["00:00", "69:17", "79:58", "45:00", "90:59", "88:11"])
def test_read_field_clock_is_read_exactly(
    clock: str, templates: tuple[TemplateSet, TemplateSet]
) -> None:
    reading = read_field(crop_fraction(board_frame(clock=clock), CLOCK_ROI), templates[1], PARAMS)
    assert reading.text == clock
    assert reading.conf >= PARAMS.min_match


@pytest.mark.unit
@pytest.mark.parametrize(("home", "away"), [(0, 0), (2, 1), (9, 3), (5, 6), (1, 7)])
def test_read_field_score_is_read_exactly(
    home: int, away: int, templates: tuple[TemplateSet, TemplateSet]
) -> None:
    frame = board_frame((home, away))
    got_home = read_field(crop_fraction(frame, SCORE_HOME_ROI), templates[0], PARAMS)
    got_away = read_field(crop_fraction(frame, SCORE_AWAY_ROI), templates[0], PARAMS)
    assert (got_home.text, got_away.text) == (str(home), str(away))


@pytest.mark.unit
def test_read_field_survives_noise_and_blur(templates: tuple[TemplateSet, TemplateSet]) -> None:
    frame = np.asarray(
        cv2.GaussianBlur(board_frame((2, 1), "69:17", seed=3), (3, 3), 0.8), dtype=np.uint8
    )
    clock = read_field(crop_fraction(frame, CLOCK_ROI), templates[1], PARAMS)
    assert clock.text in ("69:17", None)  # blur may cost the reading but never gives a wrong one


@pytest.mark.unit
def test_read_field_empty_field_is_unreadable(templates: tuple[TemplateSet, TemplateSet]) -> None:
    reading = read_field(crop_fraction(grass(), CLOCK_ROI), templates[1], PARAMS)
    assert reading == FieldReading(None, 0.0)


@pytest.mark.unit
def test_read_field_foreign_symbol_is_rejected(templates: tuple[TemplateSet, TemplateSet]) -> None:
    crop = np.zeros((20, 30, 3), dtype=np.uint8)
    cv2.putText(crop, "K", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    assert read_field(crop, templates[0], PARAMS).text is None


@pytest.mark.unit
def test_read_field_without_templates_is_unreadable() -> None:
    empty = TemplateSet([], np.zeros((0, PARAMS.patch_height, PARAMS.patch_width), np.float32))
    reading = read_field(crop_fraction(board_frame(clock="69:17"), CLOCK_ROI), empty, PARAMS)
    assert reading.text is None


@pytest.mark.unit
def test_match_glyph_ambiguous_match_is_rejected() -> None:
    glyph = segment_glyphs(crop_fraction(board_frame((3, 0)), SCORE_HOME_ROI), PARAMS)[0]
    # Two labels with the same patch: no margin, so the glyph must not be accepted.
    twin = TemplateSet(["3", "8"], np.stack([glyph.patch, glyph.patch]))
    assert match_glyph(glyph, twin, PARAMS).label is None


@pytest.mark.unit
def test_match_glyph_extra_exemplars_of_one_label_are_not_rivals() -> None:
    glyph = segment_glyphs(crop_fraction(board_frame((3, 0)), SCORE_HOME_ROI), PARAMS)[0]
    many = TemplateSet(["3", "3", "3"], np.stack([glyph.patch] * 3))
    assert match_glyph(glyph, many, PARAMS).label == "3"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("69:17", 4157),
        ("00:00", 0),
        ("9:05", 545),
        ("105:30", 6330),
        ("45:60", None),
        ("131:00", None),
        ("6917", None),
        ("69:1", None),
        ("", None),
        (None, None),
        ("ab:cd", None),
    ],
)
def test_parse_clock_text(text: str | None, expected: int | None) -> None:
    assert parse_clock_text(text, max_minutes=130) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", 0),
        ("2", 2),
        ("12", 12),
        ("20", None),
        ("123", None),
        ("", None),
        (None, None),
        ("x", None),
    ],
)
def test_parse_score_text(text: str | None, expected: int | None) -> None:
    assert parse_score_text(text, max_score=19) == expected


@pytest.mark.unit
def test_templates_round_trip_through_png_files(
    tmp_path: Path, templates: tuple[TemplateSet, TemplateSet]
) -> None:
    score = templates[0]
    for i, (label, patch) in enumerate(zip(score.labels, score.patches, strict=True)):
        save_template(tmp_path, label, i, patch)
    loaded = templates_from_dir(tmp_path, PARAMS)
    reading = read_field(crop_fraction(board_frame((7, 4)), SCORE_HOME_ROI), loaded, PARAMS)
    assert reading.text == "7"


@pytest.mark.unit
def test_templates_from_dir_errors_are_explicit(tmp_path: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="not found"):
        templates_from_dir(tmp_path / "missing", PARAMS)
    with pytest.raises(ScoreboardParseError, match="no templates"):
        templates_from_dir(tmp_path, PARAMS)
    (tmp_path / "x__0.png").write_bytes(b"png?")
    with pytest.raises(ScoreboardParseError, match="unexpected template label"):
        templates_from_dir(tmp_path, PARAMS)
    (tmp_path / "x__0.png").unlink()
    (tmp_path / "3__0.png").write_bytes(b"not an image")
    with pytest.raises(ScoreboardParseError, match="cannot read"):
        templates_from_dir(tmp_path, PARAMS)


@pytest.mark.unit
def test_save_template_rejects_unknown_label(tmp_path: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="unknown template label"):
        save_template(tmp_path, "x", 0, np.zeros((14, 10), dtype=np.float32))


@pytest.mark.unit
def test_cluster_patches_groups_equal_digits_and_orders_by_size() -> None:
    digits = [4, 7, 4, 4, 7, 1]
    patches = [
        segment_glyphs(crop_fraction(board_frame((d, 0), seed=i), SCORE_HOME_ROI), PARAMS)[0].patch
        for i, d in enumerate(digits)
    ]
    clusters = cluster_patches(patches, PARAMS.cluster_similarity)
    assert clusters == [[0, 2, 3], [1, 4], [5]]


@pytest.mark.unit
def test_cluster_patches_empty_input_returns_nothing() -> None:
    assert cluster_patches([], 0.9) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [("+3", 3), ("+12", 12), ("3", None), ("+", None), ("++3", None), ("", None), (None, None)],
)
def test_parse_added_text(text: str | None, expected: int | None) -> None:
    assert parse_added_text(text) == expected


@pytest.mark.unit
def test_segment_glyphs_zero_size_crop_gives_no_glyphs() -> None:
    assert segment_glyphs(np.zeros((0, 10, 3), dtype=np.uint8), PARAMS) == []
    assert segment_glyphs(np.zeros((10, 0, 3), dtype=np.uint8), PARAMS) == []


@pytest.mark.unit
@pytest.mark.parametrize("value", [0, 255])
def test_read_field_uniform_crop_is_unreadable(
    value: int, templates: tuple[TemplateSet, TemplateSet]
) -> None:
    crop = np.full((20, 40, 3), value, dtype=np.uint8)
    assert read_field(crop, templates[1], PARAMS).text is None


@pytest.mark.unit
def test_read_field_heavy_noise_never_gives_digits(
    templates: tuple[TemplateSet, TemplateSet],
) -> None:
    crop = np.random.default_rng(1).integers(0, 256, size=(20, 50, 3), dtype=np.uint8)
    assert read_field(crop, templates[1], PARAMS).text is None


@pytest.mark.unit
def test_read_field_touching_digits_are_unreadable_not_wrong(
    templates: tuple[TemplateSet, TemplateSet],
) -> None:
    crop = np.zeros((14, 24, 3), dtype=np.uint8)
    crop[2:12, 2:22] = 255  # two digits fused into one wide blob
    assert read_field(crop, templates[1], PARAMS).text is None


@pytest.mark.unit
def test_png_round_trip_works_in_a_directory_with_kazakh_letters(tmp_path: Path) -> None:
    folder = tmp_path / "Қазақстан_Ұ"
    folder.mkdir()
    image = np.arange(140, dtype=np.uint8).reshape(14, 10)
    write_png(folder / "a.png", image)
    assert np.array_equal(read_gray(folder / "a.png"), image)


@pytest.mark.unit
def test_write_png_into_a_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="cannot write image"):
        write_png(tmp_path / "missing" / "a.png", np.zeros((4, 4), dtype=np.uint8))


@pytest.mark.unit
def test_read_gray_rejects_missing_empty_and_non_image_files(tmp_path: Path) -> None:
    (tmp_path / "empty.png").write_bytes(b"")
    (tmp_path / "text.png").write_bytes(b"hello")
    for name in ("missing.png", "empty.png", "text.png"):
        with pytest.raises(ScoreboardParseError, match="cannot read image"):
            read_gray(tmp_path / name)


@pytest.mark.unit
def test_template_errors_tell_how_to_calibrate(tmp_path: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="scoreboard-calibrate"):
        templates_from_dir(tmp_path / "missing", PARAMS)
    with pytest.raises(ScoreboardParseError, match="scoreboard-calibrate"):
        templates_from_dir(tmp_path, PARAMS)
