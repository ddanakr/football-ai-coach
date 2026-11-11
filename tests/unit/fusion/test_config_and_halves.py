"""FuseConfig validation and half bounds."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aicoach.errors import ConfigError, FusionError
from aicoach.fusion.config import FuseConfig, load_fuse_config
from aicoach.fusion.halves import half_at, half_bounds
from aicoach.scoreboard.schemas import OffsetSegment


def seg(start: float, end: float) -> OffsetSegment:
    """A clock segment from `start` to `end`."""
    return OffsetSegment(
        start=start,
        end=end,
        clock_start_s=0,
        clock_end_s=int(end - start),
        offset_s=start,
        n_readings=2,
    )


@pytest.mark.unit
def test_config_defaults_use_the_starting_values() -> None:
    config = FuseConfig()

    assert config.goal_plate_window_s == 120.0
    assert config.goal_plate_only_lead_s == 30.0
    assert config.shot_goal_radius_s == 10.0
    assert config.card_foul_window_before_s == 40.0
    assert config.card_foul_window_after_s == 5.0
    assert config.card_foul_lead_s == 22.0
    assert config.merge_radius_s == 3.0
    assert config.model_thresholds == {"shot": 0.2, "corner": 0.05, "foul": 0.2}


@pytest.mark.unit
@pytest.mark.parametrize(
    "values",
    [
        {"goal_plate_window_s": -1.0},
        {"goal_plate_window_s": float("nan")},
        {"goal_unconfirmed_score": 1.5},
        {"model_fps": 0.0},
        {"unknown_field": 1},
        {"model_thresholds": {"shot": 0.2}},
        {"model_thresholds": {"shot": 0.2, "corner": 0.2, "foul": 0.2, "goal": 0.2}},
        {"model_thresholds": {"shot": 0.2, "corner": 0.2, "foul": -0.1}},
    ],
)
def test_config_rejects_invalid_values(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        FuseConfig.model_validate(values)


@pytest.mark.unit
def test_half_bounds_split_at_the_widest_gap_and_apply_margins() -> None:
    bounds = half_bounds([seg(1500, 2500), seg(100, 600), seg(650, 1000)], 15.0, 10.0)

    assert (bounds[1].start, bounds[1].end) == (85.0, 1010.0)
    assert (bounds[2].start, bounds[2].end) == (1485.0, 2510.0)


@pytest.mark.unit
def test_half_bounds_start_never_goes_below_zero() -> None:
    assert half_bounds([seg(5, 100), seg(500, 900)], 15.0, 15.0)[1].start == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("segments", [[], [seg(100, 200)]])
def test_half_bounds_without_two_segments_raises_fusion_error(
    segments: list[OffsetSegment],
) -> None:
    with pytest.raises(FusionError, match="two are needed"):
        half_bounds(segments, 15.0, 15.0)


@pytest.mark.unit
def test_half_at_the_break_is_none() -> None:
    bounds = half_bounds([seg(100, 200), seg(500, 900)], 0.0, 0.0)

    times = (99.0, 100.0, 200.0, 300.0, 500.0, 900.0, 901.0)
    assert [half_at(t, bounds) for t in times] == [None, 1, 1, None, 2, 2, None]


@pytest.mark.unit
def test_load_fuse_config_none_gives_defaults_and_a_file_overrides_fields(tmp_path: Path) -> None:
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"merge_radius_s": 4.5}), encoding="utf-8")

    assert load_fuse_config(None) == FuseConfig()
    assert load_fuse_config(cfg).merge_radius_s == 4.5


@pytest.mark.unit
@pytest.mark.parametrize("content", ["not json", '{"nope": 1}', "[]"])
def test_load_fuse_config_invalid_file_raises_config_error(tmp_path: Path, content: str) -> None:
    cfg = tmp_path / "c.json"
    cfg.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid fusion config"):
        load_fuse_config(cfg)


@pytest.mark.unit
def test_load_fuse_config_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_fuse_config(tmp_path / "nope.json")
