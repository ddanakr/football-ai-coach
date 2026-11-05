"""Calibration: gathering glyphs, clustering, applying labels, calibration record."""

import json
from pathlib import Path

import pytest

from aicoach.errors import ScoreboardParseError
from aicoach.scoreboard.board import crop_fraction, load_reference
from aicoach.scoreboard.calibrate import (
    CALIBRATION_FILE,
    CalibrationSample,
    apply_labels,
    build_clusters,
    collect,
    read_calibrated_mode,
    render_sheet,
    save_board_reference,
    save_workdir,
)
from aicoach.scoreboard.glyphs import (
    Glyph,
    TemplateSet,
    match_glyph,
    read_field,
    templates_from_dir,
)
from aicoach.scoreboard.imaging import CLOCK_ROI, ImageParams
from aicoach.scoreboard.video import Frame
from tests.fixtures.scoreboard_synth import board_frame, grass, synthetic_templates

PARAMS = ImageParams()


def _frames() -> list[Frame]:
    """10 board frames showing every digit in the score, three clocks, 2 added blocks, 3 grass."""
    clocks = ["01:23", "45:67", "89:01"]
    frames = [
        board_frame((d, (d + 1) % 10), clocks[d % 3], seed=d, added="+1" if d == 4 else None)
        for d in range(10)
    ]
    frames.append(board_frame((0, 0), "45:00", seed=20, added="+3"))
    frames += [grass(seed=30 + i) for i in range(3)]
    return frames


@pytest.fixture(scope="module")
def sample() -> CalibrationSample:
    return collect(_frames(), PARAMS)


@pytest.fixture(scope="module")
def clusters(sample: CalibrationSample) -> dict[str, list[list[int]]]:
    return build_clusters(sample, PARAMS)


def _auto_labels(
    sample: CalibrationSample,
    clusters: dict[str, list[list[int]]],
    templates: tuple[TemplateSet, TemplateSet],
) -> dict[str, dict[str, str]]:
    """Label clusters by comparing with the known-good synthetic templates."""
    labels: dict[str, dict[str, str]] = {"score": {}, "clock": {}}
    for kind, tset in (("score", templates[0]), ("clock", templates[1])):
        for cid, members in enumerate(clusters[kind]):
            match = match_glyph(Glyph(0, 0, sample.patches[kind][members[0]]), tset, PARAMS)
            if match.label is not None:
                labels[kind][str(cid)] = match.label
    return labels


@pytest.mark.unit
def test_collect_counts_frames_and_keeps_only_board_frames(sample: CalibrationSample) -> None:
    assert sample.n_frames == 14
    assert sample.n_board_frames == 11
    assert len(sample.board_crops) == 11


@pytest.mark.unit
def test_collect_gathers_glyphs_of_every_kind(sample: CalibrationSample) -> None:
    assert len(sample.patches["score"]) == 22  # two digits on each of 11 frames
    assert len(sample.patches["clock"]) == 55  # five symbols on each of 11 frames
    assert len(sample.patches["added"]) >= 2  # only frames with the red block


@pytest.mark.unit
def test_collect_without_board_frames_gathers_nothing() -> None:
    empty = collect([grass(seed=1)], PARAMS)
    assert empty.n_board_frames == 0
    assert all(len(p) == 0 for p in empty.patches.values())


@pytest.mark.unit
def test_collect_empty_input_is_an_empty_sample() -> None:
    assert collect([], PARAMS).n_frames == 0


@pytest.mark.unit
def test_build_clusters_finds_one_cluster_per_distinct_symbol(
    clusters: dict[str, list[list[int]]],
) -> None:
    assert len(clusters["score"]) == 10  # digits 0-9
    assert len(clusters["clock"]) == 11  # digits 0-9 and the colon


@pytest.mark.unit
def test_render_sheet_has_a_row_per_cluster(
    sample: CalibrationSample, clusters: dict[str, list[list[int]]]
) -> None:
    sheet = render_sheet(sample.patches["score"], clusters["score"])
    assert sheet.ndim == 2
    assert sheet.shape[0] == 10 * PARAMS.patch_height * 4


@pytest.mark.unit
def test_render_sheet_without_clusters_returns_a_small_blank_image() -> None:
    assert render_sheet([], []).shape == (8, 8)


@pytest.mark.unit
def test_calibration_round_trip_gives_templates_that_read_the_clock(
    sample: CalibrationSample, clusters: dict[str, list[list[int]]], tmp_path: Path
) -> None:
    work, templates_dir = tmp_path / "work", tmp_path / "templates"
    save_workdir(work, sample, clusters, "opencv_gpu", "match.mp4")
    save_board_reference(sample, PARAMS, templates_dir)
    labels = _auto_labels(sample, clusters, synthetic_templates(PARAMS))
    report = apply_labels(work, labels, templates_dir)
    assert report.templates["clock"] >= 11
    assert report.missing["clock"] == []
    assert report.missing["score"] == []
    assert report.missing["added"] != []
    assert (work / "sheet_clock.png").is_file()
    assert read_calibrated_mode(templates_dir) == "opencv_gpu"
    assert load_reference(templates_dir).mask.any()
    clock_templates = templates_from_dir(templates_dir / "clock", PARAMS)
    reading = read_field(
        crop_fraction(board_frame(clock="23:41"), CLOCK_ROI), clock_templates, PARAMS
    )
    assert reading.text == "23:41"


@pytest.mark.unit
def test_apply_labels_replaces_old_templates(
    sample: CalibrationSample, clusters: dict[str, list[list[int]]], tmp_path: Path
) -> None:
    work, templates_dir = tmp_path / "work", tmp_path / "templates"
    save_workdir(work, sample, clusters, "opencv_cpu", "match.mp4")
    labels = _auto_labels(sample, clusters, synthetic_templates(PARAMS))
    apply_labels(work, labels, templates_dir)
    stale = templates_dir / "score" / "9__999.png"
    stale.write_bytes(b"old")
    apply_labels(work, labels, templates_dir)
    assert not stale.exists()


@pytest.mark.unit
def test_apply_labels_unknown_label_raises(
    sample: CalibrationSample, clusters: dict[str, list[list[int]]], tmp_path: Path
) -> None:
    save_workdir(tmp_path / "w", sample, clusters, "opencv_gpu", "m.mp4")
    with pytest.raises(ScoreboardParseError, match="unknown label"):
        apply_labels(tmp_path / "w", {"clock": {"0": "x"}}, tmp_path / "t")


@pytest.mark.unit
@pytest.mark.parametrize("cid", ["99", "-1", "abc"])
def test_apply_labels_unknown_cluster_id_raises(
    cid: str, sample: CalibrationSample, clusters: dict[str, list[list[int]]], tmp_path: Path
) -> None:
    save_workdir(tmp_path / "w", sample, clusters, "opencv_gpu", "m.mp4")
    with pytest.raises(ScoreboardParseError, match="unknown cluster id"):
        apply_labels(tmp_path / "w", {"score": {cid: "1"}}, tmp_path / "t")


@pytest.mark.unit
def test_apply_labels_without_templates_for_a_required_kind_raises(
    sample: CalibrationSample, clusters: dict[str, list[list[int]]], tmp_path: Path
) -> None:
    save_workdir(tmp_path / "w", sample, clusters, "opencv_gpu", "m.mp4")
    with pytest.raises(ScoreboardParseError, match="no templates for score"):
        apply_labels(tmp_path / "w", {"clock": {"0": "1"}}, tmp_path / "t")


@pytest.mark.unit
def test_apply_labels_without_calibration_data_raises(tmp_path: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="no usable calibration data"):
        apply_labels(tmp_path / "empty", {}, tmp_path / "t")


@pytest.mark.unit
def test_save_board_reference_without_board_frames_raises(tmp_path: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="no frames"):
        save_board_reference(CalibrationSample(), PARAMS, tmp_path)


@pytest.mark.unit
def test_read_calibrated_mode_errors_hint_at_calibration(tmp_path: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="scoreboard-calibrate"):
        read_calibrated_mode(tmp_path)
    (tmp_path / CALIBRATION_FILE).write_text("{broken", encoding="utf-8")
    with pytest.raises(ScoreboardParseError, match="broken calibration record"):
        read_calibrated_mode(tmp_path)
    (tmp_path / CALIBRATION_FILE).write_text(json.dumps({"decode_mode": "magic"}), encoding="utf-8")
    with pytest.raises(ScoreboardParseError, match="broken calibration record"):
        read_calibrated_mode(tmp_path)


@pytest.mark.unit
def test_save_workdir_skips_patch_files_of_empty_kinds(tmp_path: Path) -> None:
    empty = CalibrationSample()
    save_workdir(tmp_path, empty, build_clusters(empty, PARAMS), "opencv_gpu", "m.mp4")
    assert not (tmp_path / "patches_score.npy").exists()
    assert (tmp_path / "clusters.json").is_file()


def _prepared(
    sample: CalibrationSample, clusters: dict[str, list[list[int]]], tmp_path: Path
) -> tuple[Path, Path, dict[str, dict[str, str]]]:
    work, templates_dir = tmp_path / "work", tmp_path / "templates"
    save_workdir(work, sample, clusters, "opencv_gpu", "m.mp4")
    labels = _auto_labels(sample, clusters, synthetic_templates(PARAMS))
    return work, templates_dir, labels


@pytest.mark.unit
def test_apply_labels_error_leaves_the_old_templates_untouched(
    sample: CalibrationSample, clusters: dict[str, list[list[int]]], tmp_path: Path
) -> None:
    work, templates_dir, labels = _prepared(sample, clusters, tmp_path)
    apply_labels(work, labels, templates_dir)
    before = {p.name: p.read_bytes() for p in templates_dir.rglob("*") if p.is_file()}
    bad = {"score": labels["score"], "clock": {**labels["clock"], "0": "not-a-label"}}
    with pytest.raises(ScoreboardParseError, match="unknown label"):
        apply_labels(work, bad, templates_dir)
    after = {p.name: p.read_bytes() for p in templates_dir.rglob("*") if p.is_file()}
    assert after == before


@pytest.mark.unit
def test_apply_labels_missing_patches_file_leaves_old_templates_untouched(
    sample: CalibrationSample, clusters: dict[str, list[list[int]]], tmp_path: Path
) -> None:
    work, templates_dir, labels = _prepared(sample, clusters, tmp_path)
    apply_labels(work, labels, templates_dir)
    before = sorted(p.name for p in templates_dir.rglob("*.png"))
    (work / "patches_clock.npy").unlink()
    with pytest.raises(ScoreboardParseError, match="missing or broken"):
        apply_labels(work, labels, templates_dir)
    assert sorted(p.name for p in templates_dir.rglob("*.png")) == before


@pytest.mark.unit
@pytest.mark.parametrize("content", ["{broken", "[]", "{}", '{"clusters": {"score": 5}}'])
def test_apply_labels_garbage_clusters_file_raises_with_a_hint(
    content: str, sample: CalibrationSample, clusters: dict[str, list[list[int]]], tmp_path: Path
) -> None:
    work, templates_dir, labels = _prepared(sample, clusters, tmp_path)
    (work / "clusters.json").write_text(content, encoding="utf-8")
    with pytest.raises(ScoreboardParseError, match="scoreboard-calibrate"):
        apply_labels(work, labels, templates_dir)


@pytest.mark.unit
@pytest.mark.parametrize("cid", ["\u00b2", "\u0663", " 1", "1.0", ""])
def test_apply_labels_odd_cluster_ids_are_rejected_not_crashed(
    cid: str, sample: CalibrationSample, clusters: dict[str, list[list[int]]], tmp_path: Path
) -> None:
    work, templates_dir, _ = _prepared(sample, clusters, tmp_path)
    with pytest.raises(ScoreboardParseError, match="unknown cluster id"):
        apply_labels(work, {"score": {cid: "1"}}, templates_dir)


@pytest.mark.unit
def test_apply_labels_writes_the_record_atomically(
    sample: CalibrationSample, clusters: dict[str, list[list[int]]], tmp_path: Path
) -> None:
    work, templates_dir, labels = _prepared(sample, clusters, tmp_path)
    apply_labels(work, labels, templates_dir)
    assert not list(templates_dir.glob("*.tmp"))
    assert not list(work.glob("*.tmp"))
