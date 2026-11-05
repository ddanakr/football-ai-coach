"""read_frame on synthetic frames: visibility, exact reading, graceful degradation."""

import cv2
import numpy as np
import pytest

from aicoach.scoreboard.board import build_reference, crop_fraction
from aicoach.scoreboard.glyphs import TemplateSet, segment_glyphs
from aicoach.scoreboard.imaging import ADDED_ROI, BOARD_ROI, ImageParams
from aicoach.scoreboard.reader import ReaderContext, read_frame
from tests.fixtures.scoreboard_synth import board_frame, grass, synthetic_templates

PARAMS = ImageParams()


@pytest.fixture(scope="module")
def ctx() -> ReaderContext:
    score, clock = synthetic_templates(PARAMS)
    crops = [
        crop_fraction(board_frame((i % 4, i % 3), f"{10 + i}:{i:02d}", seed=i), BOARD_ROI)
        for i in range(10)
    ]
    return ReaderContext(build_reference(crops, PARAMS.static_std), score, clock, PARAMS)


@pytest.mark.unit
def test_read_frame_reads_clock_and_score_exactly(ctx: ReaderContext) -> None:
    reading = read_frame(board_frame((2, 1), "69:17", seed=40), ctx)
    assert reading.visible is True
    assert reading.clock_s == 69 * 60 + 17
    assert reading.score == (2, 1)
    assert reading.conf >= PARAMS.min_match


@pytest.mark.unit
def test_read_frame_picture_without_a_board_is_not_visible(ctx: ReaderContext) -> None:
    reading = read_frame(grass(seed=7), ctx)
    assert reading.visible is False
    assert reading.clock_s is None
    assert reading.score is None


@pytest.mark.unit
def test_read_frame_visible_board_with_unknown_digits_gives_none_not_a_guess(
    ctx: ReaderContext,
) -> None:
    empty = TemplateSet([], np.zeros((0, PARAMS.patch_height, PARAMS.patch_width), np.float32))
    blind = ReaderContext(ctx.reference, empty, empty, PARAMS)
    reading = read_frame(board_frame((2, 1), "69:17"), blind)
    assert reading.visible is True
    assert reading.clock_s is None
    assert reading.score is None
    assert reading.conf == 0.0


@pytest.mark.unit
def test_read_frame_clock_with_impossible_seconds_is_dropped(ctx: ReaderContext) -> None:
    reading = read_frame(board_frame((0, 0), "45:67"), ctx)
    assert reading.visible is True
    assert reading.clock_s is None
    assert reading.score == (0, 0)


@pytest.mark.unit
def test_read_frame_blurred_board_never_gives_a_wrong_clock(ctx: ReaderContext) -> None:
    frame = np.asarray(
        cv2.GaussianBlur(board_frame((1, 0), "33:21", seed=2), (5, 5), 1.5), dtype=np.uint8
    )
    reading = read_frame(frame, ctx)
    assert reading.clock_s in (None, 33 * 60 + 21)


@pytest.mark.unit
def test_read_frame_smaller_frame_of_the_same_layout_is_still_detected(ctx: ReaderContext) -> None:
    # Fractions make the regions resolution-independent; glyphs are resized to a fixed patch.
    small = np.asarray(
        cv2.resize(board_frame((2, 1), "69:17"), (960, 540), interpolation=cv2.INTER_AREA),
        dtype=np.uint8,
    )
    reading = read_frame(small, ctx)
    assert reading.visible is True


def _added_templates() -> TemplateSet:
    labels: list[str] = []
    patches = []
    for text, chars in (("+1", ["plus", "1"]), ("+3", ["plus", "3"])):
        crop = crop_fraction(board_frame(clock="45:30", added=text), ADDED_ROI)
        glyphs = segment_glyphs(crop, PARAMS)
        assert len(glyphs) == 2
        labels += chars
        patches += [g.patch for g in glyphs]
    return TemplateSet(labels, np.stack(patches))


@pytest.mark.unit
def test_read_frame_added_block_without_templates_gives_the_flag_only(ctx: ReaderContext) -> None:
    reading = read_frame(board_frame((0, 0), "45:30", added="+1"), ctx)
    assert reading.added_visible is True
    assert reading.added is None


@pytest.mark.unit
def test_read_frame_added_block_is_read_with_templates(ctx: ReaderContext) -> None:
    with_added = ReaderContext(
        ctx.reference, ctx.score_templates, ctx.clock_templates, PARAMS, _added_templates()
    )
    reading = read_frame(board_frame((0, 0), "45:30", seed=11, added="+3"), with_added)
    assert (reading.added_visible, reading.added) == (True, 3)
    assert read_frame(board_frame((0, 0), "45:30", seed=12), with_added).added_visible is False
