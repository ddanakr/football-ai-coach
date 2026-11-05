"""Board visibility: reference building and masked correlation on synthetic frames."""

from pathlib import Path

import numpy as np
import pytest

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.board import (
    board_blue_fraction,
    build_reference,
    crop_fraction,
    load_reference,
    save_reference,
    similarity,
)
from aicoach.scoreboard.imaging import BOARD_ROI, ImageParams
from aicoach.scoreboard.schemas import Roi
from tests.fixtures.scoreboard_synth import board_frame, grass

PARAMS = ImageParams()


def _board_crops(n: int) -> list[np.ndarray]:
    scores = [(0, 0), (1, 0), (2, 1), (3, 3), (0, 2)]
    clocks = ["10:01", "45:00", "69:17", "88:42", "03:59"]
    return [
        crop_fraction(board_frame(scores[i % 5], clocks[i % 5], seed=i), BOARD_ROI)
        for i in range(n)
    ]


@pytest.mark.unit
def test_crop_fraction_never_returns_an_empty_region() -> None:
    frame = grass(width=100, height=60)
    crop = crop_fraction(frame, Roi(x0=0.999, y0=0.999, x1=1.0, y1=1.0))
    assert crop.shape[0] >= 1
    assert crop.shape[1] >= 1


@pytest.mark.unit
def test_crop_fraction_scales_with_the_frame_size() -> None:
    big = crop_fraction(grass(width=1280, height=720), BOARD_ROI)
    small = crop_fraction(grass(width=640, height=360), BOARD_ROI)
    assert big.shape[1] == pytest.approx(2 * small.shape[1], abs=2)


@pytest.mark.unit
def test_board_blue_fraction_tells_the_board_from_grass() -> None:
    assert board_blue_fraction(crop_fraction(board_frame(), BOARD_ROI)) > 0.2
    assert board_blue_fraction(crop_fraction(grass(), BOARD_ROI)) < 0.01


@pytest.mark.unit
def test_board_blue_fraction_cyan_advertising_is_not_the_board() -> None:
    cyan = np.zeros((50, 200, 3), dtype=np.uint8)
    cyan[:] = (36, 184, 178)
    assert board_blue_fraction(cyan) == 0.0


@pytest.mark.unit
def test_similarity_board_with_other_digits_and_background_stays_high() -> None:
    reference = build_reference(_board_crops(12), PARAMS.static_std)
    crop = crop_fraction(board_frame((5, 5), "77:31", seed=99), BOARD_ROI)
    assert similarity(crop, reference) > 0.9


@pytest.mark.unit
def test_similarity_picture_without_a_board_is_low() -> None:
    reference = build_reference(_board_crops(12), PARAMS.static_std)
    assert similarity(crop_fraction(grass(seed=5), BOARD_ROI), reference) < 0.3


@pytest.mark.unit
def test_similarity_flat_region_returns_zero_instead_of_nan() -> None:
    reference = build_reference(_board_crops(6), PARAMS.static_std)
    flat = np.zeros_like(_board_crops(1)[0])
    assert similarity(flat, reference) == 0.0


@pytest.mark.unit
def test_similarity_resizes_a_crop_of_another_resolution() -> None:
    reference = build_reference(_board_crops(6), PARAMS.static_std)
    half = crop_fraction(board_frame((1, 1), "12:34"), BOARD_ROI)[::2, ::2]
    assert similarity(half, reference) > 0.7


@pytest.mark.unit
def test_build_reference_without_crops_raises() -> None:
    with pytest.raises(ScoreboardParseError, match="no frames"):
        build_reference([], PARAMS.static_std)


@pytest.mark.unit
def test_build_reference_mixed_sizes_raise() -> None:
    crops = _board_crops(2)
    crops.append(crops[0][:-1])
    with pytest.raises(ScoreboardParseError, match="same size"):
        build_reference(crops, PARAMS.static_std)


@pytest.mark.unit
def test_build_reference_from_unrelated_pictures_raises() -> None:
    shape = _board_crops(1)[0].shape
    crops = [
        np.random.default_rng(i).integers(0, 256, size=shape, dtype=np.uint8) for i in range(6)
    ]
    with pytest.raises(ScoreboardParseError, match="too few static pixels"):
        build_reference(crops, PARAMS.static_std)


@pytest.mark.unit
def test_reference_round_trip(tmp_path: Path) -> None:
    reference = build_reference(_board_crops(6), PARAMS.static_std)
    save_reference(reference, tmp_path)
    loaded = load_reference(tmp_path)
    assert np.array_equal(loaded.mask, reference.mask)
    assert np.allclose(loaded.pixels, reference.pixels)


@pytest.mark.unit
def test_load_reference_missing_file_hints_at_calibration(tmp_path: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="scoreboard-calibrate"):
        load_reference(tmp_path)


@pytest.mark.unit
def test_load_reference_broken_file_raises(tmp_path: Path) -> None:
    (tmp_path / "board_reference.npz").write_bytes(b"garbage")
    with pytest.raises(ScoreboardParseError, match="broken"):
        load_reference(tmp_path)


@pytest.mark.unit
def test_colour_fractions_of_an_empty_crop_are_zero_not_nan() -> None:
    from aicoach.scoreboard.board import red_fraction

    empty = np.zeros((0, 10, 3), dtype=np.uint8)
    assert board_blue_fraction(empty) == 0.0
    assert red_fraction(empty) == 0.0


@pytest.mark.unit
def test_similarity_of_an_empty_crop_is_zero() -> None:
    reference = build_reference(_board_crops(6), PARAMS.static_std)
    assert similarity(np.zeros((0, 0, 3), dtype=np.uint8), reference) == 0.0


@pytest.mark.unit
def test_crop_fraction_of_a_frame_without_pixels_raises() -> None:
    with pytest.raises(ScoreboardParseError, match="no pixels"):
        crop_fraction(np.zeros((0, 0, 3), dtype=np.uint8), BOARD_ROI)
