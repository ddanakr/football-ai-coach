"""Where the graphics are on the frame and how the pixels are read (image parameters).

Debt: thresholds belong in `configs/*.yaml` once `Settings` exists.
"""

from pydantic import BaseModel, ConfigDict, Field

from aicoach.scoreboard.schemas import Roi


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


# Calibrated on one KFF match (1280x720, scoreboard at the top left).
# All pixel positions of the broadcast graphics enter the code only here, as frame fractions.
BOARD_ROI = Roi(x0=0.0516, y0=0.0806, x1=0.2484, y1=0.1570)  # white boxes, score, clock
SCORE_HOME_ROI = Roi(x0=0.1156, y0=0.0847, x1=0.1375, y1=0.1194)
SCORE_AWAY_ROI = Roi(x0=0.1633, y0=0.0847, x1=0.1859, y1=0.1194)
ADDED_ROI = Roi(x0=0.1773, y0=0.1208, x1=0.1977, y1=0.1542)  # red "+N" block right of the clock
CLOCK_ROI = Roi(x0=0.1312, y0=0.1278, x1=0.1703, y1=0.1528)


class ImageParams(_Frozen):
    """Thresholds of image reading (visibility, glyph segmentation, template matching)."""

    # Pixels brighter than this in the darkest channel are text (white on the blue blocks).
    text_threshold: int = Field(default=140, ge=1, le=254)
    # Glyph patches are resized to this size before matching (width, height).
    patch_width: int = Field(default=10, ge=4)
    patch_height: int = Field(default=14, ge=4)
    # A column group with fewer text pixels than this is noise.
    min_glyph_pixels: int = Field(default=4, ge=1)
    # A glyph is accepted if its best correlation reaches this and beats the runner-up.
    min_match: float = Field(default=0.75, ge=-1, le=1)
    min_margin: float = Field(default=0.03, ge=0, le=1)
    # The red "+N" block is on screen if this share of its region is red.
    red_block_min_share: float = Field(default=0.35, gt=0, le=1)
    # Calibration only: a frame has the board if this share of its pixels is the board's blue.
    candidate_blue_min: float = Field(default=0.15, gt=0, le=1)
    # Visibility: correlation of the board region with the reference, with a margin.
    visible_threshold: float = Field(default=0.7, ge=-1, le=1)
    # Static pixels of the board: standard deviation across reference frames below this.
    static_std: float = Field(default=14.0, gt=0)
    # Glyph clusters in calibration: patches this similar belong to one cluster.
    cluster_similarity: float = Field(default=0.92, ge=-1, le=1)
    # Plausible values; anything else is an OCR confusion and is dropped.
    max_clock_minutes: int = Field(default=130, ge=1)
    max_score: int = Field(default=19, ge=1)
