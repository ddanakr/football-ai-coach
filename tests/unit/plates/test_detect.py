"""Per-frame plate detection on synthetic frames."""

import numpy as np
import pytest

from aicoach.errors import ScoreboardParseError
from aicoach.plates.config import DetectParams
from aicoach.plates.detect import (
    halves_count,
    light_share,
    panel_signature,
    read_frame,
    red_header_share,
    ring_blue_share,
    share,
    signature_distance,
)
from aicoach.plates.layout import PlateLayout
from aicoach.scoreboard.video import Frame
from tests.fixtures.plates_synth import (
    BLUE,
    GREEN,
    ORANGE,
    RED,
    PlateKind,
    background,
    blue_object_frame,
    box,
    plate_frame,
    triangle,
)

LAYOUT = PlateLayout()
PARAMS = DetectParams()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kind", "color"),
    [("goal", None), ("card", "yellow"), ("subst", None), ("card_red", "red")],
)
def test_read_frame_plate_gives_its_type_and_high_confidence(
    kind: PlateKind, color: str | None
) -> None:
    reading = read_frame(plate_frame(kind), LAYOUT, PARAMS)

    assert reading.kind == ("card" if kind == "card_red" else kind)
    assert reading.card_color == color
    assert reading.reject is None
    assert reading.conf >= 0.8
    assert reading.signature is not None
    assert reading.logo is not None


@pytest.mark.unit
@pytest.mark.parametrize("size", [(640, 360), (1280, 720), (320, 180)])
def test_read_frame_works_at_other_resolutions(size: tuple[int, int]) -> None:
    reading = read_frame(plate_frame("goal", width=size[0], height=size[1]), LAYOUT, PARAMS)

    assert reading.kind == "goal"


@pytest.mark.unit
def test_read_frame_plate_without_marker_is_rejected_as_no_marker() -> None:
    reading = read_frame(plate_frame("coach"), LAYOUT, PARAMS)

    assert reading.kind is None
    assert reading.reject == "no_marker"
    assert reading.is_candidate


@pytest.mark.unit
def test_read_frame_blue_object_without_panel_is_rejected_as_no_panel() -> None:
    reading = read_frame(blue_object_frame(), LAYOUT, PARAMS)

    assert reading.reject == "no_panel"
    assert reading.kind is None


@pytest.mark.unit
def test_read_frame_round_dots_instead_of_triangles_are_not_a_substitution() -> None:
    reading = read_frame(plate_frame("round_dots"), LAYOUT, PARAMS)

    assert reading.kind is None
    assert reading.reject == "no_marker"


@pytest.mark.unit
def test_read_frame_flipped_triangles_are_not_a_substitution() -> None:
    frame = plate_frame("coach")
    # green pointing down and red pointing up: the wrong way round
    triangle(frame, LAYOUT.arrow_up, GREEN, up=False)
    triangle(frame, LAYOUT.arrow_down, RED, up=True)

    assert read_frame(frame, LAYOUT, PARAMS).kind is None


@pytest.mark.unit
def test_read_frame_orange_tag_wins_over_a_yellow_header() -> None:
    frame = plate_frame("card")
    x0, y0, x1, y1 = box(frame, LAYOUT.tag)
    frame[y0:y1, x0:x1] = ORANGE

    assert read_frame(frame, LAYOUT, PARAMS).kind == "goal"


@pytest.mark.unit
def test_read_frame_a_small_orange_patch_is_below_the_tag_threshold() -> None:
    frame = plate_frame("coach")
    x0, y0, x1, y1 = box(frame, LAYOUT.tag)
    frame[y0:y1, x0 : x0 + (x1 - x0) // 4] = ORANGE  # about a quarter of the tag region

    assert read_frame(frame, LAYOUT, PARAMS).kind is None


@pytest.mark.unit
def test_read_frame_empty_frame_raises_the_scoreboard_parse_error() -> None:
    with pytest.raises(ScoreboardParseError, match="no pixels"):
        read_frame(np.zeros((0, 0, 3), dtype=np.uint8), LAYOUT, PARAMS)


@pytest.mark.unit
def test_read_frame_background_only_is_not_a_candidate() -> None:
    reading = read_frame(background(), LAYOUT, PARAMS)

    assert not reading.is_candidate
    assert reading.kind is None
    assert reading.conf == 0.0
    assert reading.signature is None


@pytest.mark.unit
def test_read_frame_orange_banner_without_blue_block_is_not_a_candidate() -> None:
    """Other broadcast graphics (for example an orange goal banner) have no blue block."""
    frame = background()
    frame[36:72, 115:320] = (20, 115, 235)

    assert not read_frame(frame, LAYOUT, PARAMS).is_candidate


@pytest.mark.unit
def test_read_frame_faded_plate_has_lower_confidence() -> None:
    clean = read_frame(plate_frame("goal"), LAYOUT, PARAMS)
    solid = plate_frame("goal").astype(np.float32)
    mixed = (solid * 0.5 + background(seed=5).astype(np.float32) * 0.5).astype(np.uint8)

    faded = read_frame(mixed, LAYOUT, PARAMS)

    assert faded.conf < clean.conf


@pytest.mark.unit
def test_read_frame_tiny_frame_does_not_crash() -> None:
    tiny = np.zeros((4, 4, 3), dtype=np.uint8)

    assert not read_frame(tiny, LAYOUT, PARAMS).is_candidate


@pytest.mark.unit
def test_read_frame_logo_crop_has_the_logo_colour() -> None:
    reading = read_frame(plate_frame("goal", logo=(30, 220, 240)), LAYOUT, PARAMS)

    assert reading.logo is not None
    centre = reading.logo[reading.logo.shape[0] // 2, reading.logo.shape[1] // 2]
    assert tuple(int(v) for v in centre) == (30, 220, 240)


@pytest.mark.unit
def test_ring_blue_share_ignores_the_middle_of_the_block() -> None:
    block = np.zeros((40, 40, 3), dtype=np.uint8)
    block[:] = BLUE
    block[10:30, 10:30] = (0, 255, 255)  # the logo

    assert ring_blue_share(block, 0.12) == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(0, 0, 3), (0, 5, 3), (5, 0, 3)])
def test_helpers_empty_crop_gives_zero(shape: tuple[int, int, int]) -> None:
    empty: Frame = np.zeros(shape, dtype=np.uint8)

    assert ring_blue_share(empty, 0.12) == 0.0
    assert light_share(empty, 180) == 0.0
    assert red_header_share(empty) == 0.0
    assert share(empty, lambda r, g, b: r > g + b) == 0.0
    assert halves_count(empty, lambda r, g, b: r > g + b) == (0, 0)
    assert not panel_signature(empty).any()


@pytest.mark.unit
def test_halves_count_splits_the_mask_by_rows() -> None:
    crop = np.zeros((10, 4, 3), dtype=np.uint8)
    crop[6:, :] = 255

    assert halves_count(crop, lambda r, g, b: (r > 200) & (g > 200) & (b > 200)) == (0, 16)


@pytest.mark.unit
def test_light_share_counts_pixels_above_the_level() -> None:
    crop = np.zeros((2, 2, 3), dtype=np.uint8)
    crop[0, :] = 200

    assert light_share(crop, 180) == pytest.approx(0.5)


@pytest.mark.unit
def test_signature_same_plate_is_near_and_another_text_is_far() -> None:
    first = read_frame(plate_frame("card", text_seed=1, seed=1), LAYOUT, PARAMS).signature
    again = read_frame(plate_frame("card", text_seed=1, seed=2), LAYOUT, PARAMS).signature
    other = read_frame(plate_frame("card", text_seed=2, seed=1), LAYOUT, PARAMS).signature

    assert first is not None
    assert again is not None
    assert other is not None
    assert signature_distance(first, again) < 0.02
    assert signature_distance(first, other) > 0.02


@pytest.mark.unit
def test_read_frame_header_of_a_non_red_non_yellow_colour_is_no_marker() -> None:
    frame = plate_frame("card")
    x0, y0, x1, y1 = box(frame, LAYOUT.header)
    frame[y0:y1, x0:x1] = (200, 60, 0)  # a strongly saturated blue header: not a card

    reading = read_frame(frame, LAYOUT, PARAMS)

    assert (reading.kind, reading.reject) == (None, "no_marker")
