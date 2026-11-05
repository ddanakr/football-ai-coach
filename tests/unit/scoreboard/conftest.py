"""Fixtures for the stage tests: a match directory with a tiny video and calibrated templates."""

import json
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pytest

from aicoach.scoreboard.board import BoardReference, build_reference, crop_fraction, save_reference
from aicoach.scoreboard.glyphs import TemplateSet, save_template
from aicoach.scoreboard.imaging import BOARD_ROI, ImageParams
from aicoach.scoreboard.schemas import DecodeMode
from aicoach.scoreboard.stage import TEMPLATES_DIR
from aicoach.scoreboard.video import Frame
from tests.fixtures.scoreboard_synth import board_frame, synthetic_templates

PARAMS = ImageParams()
CLIP_FPS = 5
CLIP_FRAMES = 60  # 12 s at 5 fps, so the stage samples 12 frames at 1 fps


@cache
def _assets() -> tuple[BoardReference, TemplateSet, TemplateSet]:
    """Reference and templates drawn once per test session (drawing is the slow part)."""
    crops = [
        crop_fraction(board_frame((i % 3, i % 2), f"{10 + i}:{i:02d}", seed=i), BOARD_ROI)
        for i in range(10)
    ]
    reference = build_reference(crops, PARAMS.static_std)
    score, clock = synthetic_templates(PARAMS)
    return reference, score, clock


def write_calibration(match_dir: Path, mode: DecodeMode = "opencv_gpu") -> Path:
    """Write board reference, digit templates and the calibration record for synthetic frames."""
    templates_dir = match_dir / TEMPLATES_DIR
    reference, score, clock = _assets()
    save_reference(reference, templates_dir)
    for kind, tset in (("score", score), ("clock", clock)):
        for i, (label, patch) in enumerate(zip(tset.labels, tset.patches, strict=True)):
            save_template(templates_dir / kind, label, i, patch)
    record: dict[str, object] = {"decode_mode": mode, "video": "match.mp4", "templates": {}}
    (templates_dir / "calibration.json").write_text(json.dumps(record), encoding="utf-8")
    return templates_dir


@cache
def match_frames() -> list[Frame]:
    """12 one-second frames: clock 10:00.., score 0:0 for 6 s, then 1:0 for 6 s (read-only)."""
    return [
        board_frame((0, 0) if k < 6 else (1, 0), f"10:{k:02d}", seed=100 + k) for k in range(12)
    ]


@pytest.fixture
def match_dir(tmp_path: Path) -> Path:
    """A match directory with a tiny `match.mp4` (12 s at 5 fps)."""
    directory = tmp_path / "match01"
    directory.mkdir()
    writer = cv2.VideoWriter(
        str(directory / "match.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), CLIP_FPS, (64, 48)
    )
    for _ in range(CLIP_FRAMES):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return directory
