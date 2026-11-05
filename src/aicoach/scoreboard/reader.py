"""Read one frame: is the scoreboard on screen, and what do the clock and the score say."""

from dataclasses import dataclass

from aicoach.scoreboard.assemble import RawReading
from aicoach.scoreboard.board import BoardReference, crop_fraction, red_fraction, similarity
from aicoach.scoreboard.glyphs import (
    FieldReading,
    TemplateSet,
    parse_added_text,
    parse_clock_text,
    parse_score_text,
    read_field,
)
from aicoach.scoreboard.imaging import (
    ADDED_ROI,
    BOARD_ROI,
    CLOCK_ROI,
    SCORE_AWAY_ROI,
    SCORE_HOME_ROI,
    ImageParams,
)
from aicoach.scoreboard.video import Frame


@dataclass(frozen=True, eq=False)
class ReaderContext:
    """Everything calibrated from the match: the board reference and the digit templates."""

    reference: BoardReference
    score_templates: TemplateSet
    clock_templates: TemplateSet
    params: ImageParams
    # Optional: without them the red "+N" block is still detected but N stays unknown.
    added_templates: TemplateSet | None = None


def _confidence(readings: list[FieldReading]) -> float:
    """Lowest correlation among the fields that were read; 0.0 if none was."""
    confs = [r.conf for r in readings if r.text is not None]
    return min(confs) if confs else 0.0


def read_frame(frame: Frame, ctx: ReaderContext) -> RawReading:
    """Read the scoreboard of one frame.

    If the board is not recognised the reading is `visible=False` and nothing else is read.
    A visible board whose digits cannot be matched gives `None` for that field, never a guess.
    """
    params = ctx.params
    board = crop_fraction(frame, BOARD_ROI)
    if similarity(board, ctx.reference) < params.visible_threshold:
        return RawReading(visible=False)
    clock = read_field(crop_fraction(frame, CLOCK_ROI), ctx.clock_templates, params)
    home = read_field(crop_fraction(frame, SCORE_HOME_ROI), ctx.score_templates, params)
    away = read_field(crop_fraction(frame, SCORE_AWAY_ROI), ctx.score_templates, params)
    home_goals = parse_score_text(home.text, params.max_score)
    away_goals = parse_score_text(away.text, params.max_score)
    score = (home_goals, away_goals) if home_goals is not None and away_goals is not None else None
    added_crop = crop_fraction(frame, ADDED_ROI)
    added_visible = red_fraction(added_crop) >= params.red_block_min_share
    added = None
    if added_visible and ctx.added_templates is not None:
        added = parse_added_text(read_field(added_crop, ctx.added_templates, params).text)
    return RawReading(
        visible=True,
        added_visible=added_visible,
        added=added,
        clock_s=parse_clock_text(clock.text, params.max_clock_minutes),
        score=score,
        conf=_confidence([clock, home, away]),
    )
