"""The plates stage end to end on a tiny clip with a fake frame source (no real footage)."""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from structlog.testing import capture_logs

from aicoach.errors import PlatesError, ScoreboardParseError, StorageError
from aicoach.logging_setup import stage_logger
from aicoach.plates.config import PlatesConfig
from aicoach.plates.schemas import PlatesResult
from aicoach.plates.stage import (
    RESULT_FILE,
    _encode,
    _logo_of,
    cache_key,
    collect_hits,
    load_scoreboard,
    run_plates,
    write_crops,
)
from aicoach.plates.tracks import Track
from aicoach.scoreboard.schemas import DecodeMode
from aicoach.scoreboard.stage import RESULT_FILE as SCOREBOARD_FILE
from aicoach.scoreboard.stage import SourceFactory
from aicoach.scoreboard.video import Frame, probe_video
from aicoach.stage_common import Progress
from aicoach.status import read_status
from tests.fixtures.plates_synth import (
    AWAY_LOGO,
    HOME_LOGO,
    PlateKind,
    change,
    hit,
    plate_frame,
    scoreboard_result,
)

CONFIG = PlatesConfig(sample_fps=1.0)


class FakeSource:
    """Stands in for SampledReader: yields prepared frames and counts how often it was used."""

    used = 0

    def __init__(
        self, frames: list[Frame], mode: DecodeMode = "opencv_gpu", decoded: int | None = None
    ) -> None:
        self._frames = frames
        self.decode_mode: DecodeMode = mode
        self.frames_decoded = 60 if decoded is None else decoded

    def __iter__(self) -> Iterator[tuple[int, Frame]]:
        FakeSource.used += 1
        yield from enumerate(self._frames)


class BrokenSource(FakeSource):
    """A source that dies in the middle of the pass."""

    def __iter__(self) -> Iterator[tuple[int, Frame]]:
        yield from enumerate(self._frames[:3])
        msg = "decoder crashed"
        raise RuntimeError(msg)


def _factory(frames: list[Frame], mode: DecodeMode = "opencv_gpu") -> SourceFactory:
    return lambda _path, _step, _hw: FakeSource(frames, mode)


def match_frames() -> list[Frame]:
    """12 one-second frames: a home goal plate, an away goal plate and a card plate (away logo)."""
    kinds: list[tuple[PlateKind | None, tuple[int, int, int], int]] = [
        (None, HOME_LOGO, 0),
        ("goal", HOME_LOGO, 1),
        ("goal", HOME_LOGO, 1),
        ("goal", HOME_LOGO, 1),
        (None, HOME_LOGO, 0),
        (None, HOME_LOGO, 0),
        ("goal", AWAY_LOGO, 2),
        ("goal", AWAY_LOGO, 2),
        ("goal", AWAY_LOGO, 2),
        (None, HOME_LOGO, 0),
        ("card", AWAY_LOGO, 3),
        ("card", AWAY_LOGO, 3),
    ]
    return [
        plate_frame(kind, logo=logo, text_seed=seed, seed=100 + k)
        for k, (kind, logo, seed) in enumerate(kinds)
    ]


def write_scoreboard(match_dir: Path, mode: DecodeMode = "opencv_gpu") -> None:
    """scoreboard.json: a home goal before the first plate, an away goal before the second."""
    board = scoreboard_result(
        [change((0, 0), (1, 0), 0.0, 1.0), change((1, 0), (1, 1), 4.0, 5.0)],
        decode_mode=mode,
        segments=[(0.0, 5.0), (7.0, 11.0)],
    )
    (match_dir / SCOREBOARD_FILE).write_text(board.model_dump_json(), encoding="utf-8")


@pytest.fixture(autouse=True)
def _reset_counter() -> None:
    FakeSource.used = 0


@pytest.fixture
def ready(match_dir: Path) -> Path:
    write_scoreboard(match_dir)
    return match_dir


def _run(match_dir: Path, frames: list[Frame] | None = None, force: bool = False) -> PlatesResult:
    return run_plates(
        match_dir / "match.mp4",
        "match01",
        CONFIG,
        force=force,
        source_factory=_factory(match_frames() if frames is None else frames),
    )


@pytest.mark.unit
def test_run_plates_finds_the_plates_and_writes_the_result(ready: Path) -> None:
    result = _run(ready)

    stored = PlatesResult.model_validate_json((ready / RESULT_FILE).read_text("utf-8"))
    assert stored == result
    assert [(p.type, p.t_first_seen, p.t_last_seen) for p in result.plates] == [
        ("goal", 1.0, 3.0),
        ("goal", 6.0, 8.0),
        ("card", 10.0, 11.0),
    ]
    assert result.decode_mode == "opencv_gpu"
    assert result.sample_fps == pytest.approx(1.0)
    assert result.stats.n_samples == 12
    assert result.stats.n_by_type == {"subst": 0, "goal": 2, "card": 1}
    assert result.cache_key != ""


@pytest.mark.unit
def test_run_plates_assigns_teams_from_the_score_changes(ready: Path) -> None:
    result = _run(ready)

    assert [p.team for p in result.plates] == ["home", "away", "away"]
    assert [p.score_change for p in result.plates] == [0, 1, None]
    assert [(c.team, c.n_plates) for c in result.logo_clusters] == [("away", 2), ("home", 1)]
    assert result.stats.warnings == []


@pytest.mark.unit
def test_run_plates_writes_crops_for_every_plate(ready: Path) -> None:
    result = _run(ready)

    for plate in result.plates:
        assert (ready / plate.crop).is_file()
        assert (ready / plate.logo_crop).is_file()
    assert (ready / result.plates[-1].crop).read_bytes()[:4] == b"\x89PNG"


@pytest.mark.unit
def test_run_plates_writes_status_done_and_keeps_the_scoreboard_entry(ready: Path) -> None:
    (ready / "status.json").write_text(
        json.dumps({"stages": {"scoreboard": {"state": "done", "progress": 1.0}}}),
        encoding="utf-8",
    )

    _run(ready)

    stages = read_status(ready).stages
    assert (stages["plates"].state, stages["plates"].progress) == ("done", 1.0)
    assert stages["scoreboard"].state == "done"


@pytest.mark.unit
def test_run_plates_logs_stage_match_id_and_duration(ready: Path) -> None:
    with capture_logs() as logs:
        _run(ready)

    done = [entry for entry in logs if entry["event"] == "stage_done"]
    assert done[0]["stage"] == "plates"
    assert done[0]["match_id"] == "match01"
    assert isinstance(done[0]["duration_ms"], int)
    assert done[0]["n_plates"] == 3


@pytest.mark.unit
def test_run_plates_second_run_uses_the_cache(ready: Path) -> None:
    first = _run(ready)
    with capture_logs() as logs:
        second = _run(ready)

    assert second == first
    assert FakeSource.used == 1
    assert [e["event"] for e in logs] == ["stage_skipped"]


@pytest.mark.unit
def test_run_plates_force_reads_the_video_again(ready: Path) -> None:
    _run(ready)
    again = _run(ready, force=True)

    assert FakeSource.used == 2
    assert len(again.plates) == 3


@pytest.mark.unit
def test_run_plates_changed_scoreboard_json_invalidates_the_cache(ready: Path) -> None:
    _run(ready)
    board = scoreboard_result(
        [change((0, 0), (1, 0), 0.0, 1.0)], segments=[(0.0, 5.0), (7.0, 11.0)]
    )
    (ready / SCOREBOARD_FILE).write_text(board.model_dump_json(), encoding="utf-8")

    _run(ready)

    assert FakeSource.used == 2


@pytest.mark.unit
def test_run_plates_missing_crop_invalidates_the_cache(ready: Path) -> None:
    result = _run(ready)
    (ready / result.plates[0].crop).unlink()

    with capture_logs() as logs:
        _run(ready)

    assert FakeSource.used == 2
    assert any(entry["event"] == "cache_crops_missing" for entry in logs)
    assert (ready / result.plates[0].crop).is_file()


@pytest.mark.unit
def test_run_plates_unreadable_result_file_is_recomputed(ready: Path) -> None:
    _run(ready)
    (ready / RESULT_FILE).write_text("{not json", encoding="utf-8")

    with capture_logs() as logs:
        _run(ready)

    assert FakeSource.used == 2
    assert any(entry["event"] == "cache_unreadable" for entry in logs)


@pytest.mark.unit
def test_run_plates_without_scoreboard_json_says_how_to_run_the_stage(match_dir: Path) -> None:
    with pytest.raises(PlatesError, match="scoreboard"):
        _run(match_dir)

    status = read_status(match_dir).stages["plates"]
    assert status.state == "failed"
    assert "python -m aicoach.cli scoreboard" in (status.error or "")


@pytest.mark.unit
def test_run_plates_invalid_scoreboard_json_raises_plates_error(match_dir: Path) -> None:
    (match_dir / SCOREBOARD_FILE).write_text('{"video": 1}', encoding="utf-8")

    with pytest.raises(PlatesError, match="not a valid scoreboard result"):
        _run(match_dir)


@pytest.mark.unit
def test_run_plates_decode_mode_differing_from_scoreboard_json_fails(ready: Path) -> None:
    factory = _factory(match_frames(), mode="opencv_cpu")

    with pytest.raises(PlatesError, match="differs from scoreboard.json"):
        run_plates(ready / "match.mp4", "match01", CONFIG, source_factory=factory)

    assert read_status(ready).stages["plates"].state == "failed"
    assert not (ready / RESULT_FILE).exists()


@pytest.mark.unit
def test_run_plates_decoder_crash_keeps_the_previous_result(ready: Path) -> None:
    _run(ready)
    before = (ready / RESULT_FILE).read_bytes()

    with pytest.raises(RuntimeError, match="decoder crashed"):
        run_plates(
            ready / "match.mp4",
            "match01",
            CONFIG,
            force=True,
            source_factory=lambda _p, _s, _h: BrokenSource(match_frames()),
        )

    assert (ready / RESULT_FILE).read_bytes() == before
    status = read_status(ready).stages["plates"]
    assert status.state == "failed"
    assert "decoder crashed" in (status.error or "")


@pytest.mark.unit
def test_run_plates_video_without_plates_gives_an_empty_result(ready: Path) -> None:
    frames = [plate_frame(None, seed=k) for k in range(12)]

    result = _run(ready, frames)

    assert result.plates == []
    assert result.rejected == []
    assert result.stats.n_candidate_frames == 0
    assert read_status(ready).stages["plates"].state == "done"


@pytest.mark.unit
def test_run_plates_candidates_that_are_not_plates_are_listed_as_rejected(ready: Path) -> None:
    frames = [plate_frame("coach", text_seed=5, seed=k) for k in range(4)] + [
        plate_frame(None, seed=k) for k in range(8)
    ]

    result = _run(ready, frames)

    assert result.plates == []
    [rejected] = result.rejected
    assert (rejected.reason, rejected.n_frames) == ("no_marker", 4)
    assert (ready / rejected.crop).is_file()


@pytest.mark.unit
def test_run_plates_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(ScoreboardParseError, match="not found"):
        run_plates(tmp_path / "nope.mp4", "m", CONFIG)


@pytest.mark.unit
def test_run_plates_duration_mismatch_is_logged_and_stored(ready: Path) -> None:
    factory: SourceFactory = lambda _p, _s, _h: FakeSource(match_frames(), decoded=30)  # noqa: E731

    with capture_logs() as logs:
        result = run_plates(ready / "match.mp4", "match01", CONFIG, source_factory=factory)

    assert result.stats.duration_warning is not None
    assert any(entry["event"] == "duration_mismatch" for entry in logs)


@pytest.mark.unit
def test_collect_hits_gives_video_time_from_the_frame_index(ready: Path) -> None:
    info = probe_video(ready / "match.mp4")
    progress = Progress(ready, "plates", stage_logger("plates", "match01"))

    hits, crops, n = collect_hits(
        FakeSource(match_frames()), info, 5, CONFIG, "opencv_gpu", progress
    )

    assert n == 12
    assert [h.k for h in hits][:3] == [1, 2, 3]
    assert hits[0].t_v == pytest.approx(1.0)  # 1 * 5 frames / 5 fps
    assert set(crops) == {h.k for h in hits}


@pytest.mark.unit
def test_cache_key_changes_with_every_input(ready: Path) -> None:
    video = ready / "match.mp4"
    base = cache_key(video, "v1", "opencv_gpu", CONFIG)

    assert cache_key(video, "v1", "opencv_gpu", CONFIG) == base
    assert cache_key(video, "v2", "opencv_gpu", CONFIG) != base
    assert cache_key(video, "v1", "opencv_cpu", CONFIG) != base
    assert cache_key(video, "v1", "opencv_gpu", PlatesConfig(sample_fps=2.0)) != base


@pytest.mark.unit
def test_load_scoreboard_returns_a_version_that_changes_with_the_file(ready: Path) -> None:
    _, first = load_scoreboard(ready)
    (ready / SCOREBOARD_FILE).write_text(
        scoreboard_result(n_samples=3).model_dump_json(), encoding="utf-8"
    )

    _, second = load_scoreboard(ready)

    assert first != second
    assert len(first) == 64


@pytest.mark.unit
def test_load_scoreboard_unreadable_path_raises_plates_error(tmp_path: Path) -> None:
    (tmp_path / SCOREBOARD_FILE).mkdir()

    with pytest.raises(PlatesError, match="cannot read"):
        load_scoreboard(tmp_path)


@pytest.mark.unit
def test_write_crops_replaces_the_folder_and_removes_stale_files(tmp_path: Path) -> None:
    write_crops(tmp_path, {"a.png": b"1", "b.png": b"2"})
    write_crops(tmp_path, {"a.png": b"3"})

    assert sorted(p.name for p in (tmp_path / "plate_crops").iterdir()) == ["a.png"]
    assert (tmp_path / "plate_crops" / "a.png").read_bytes() == b"3"
    assert not (tmp_path / "plate_crops.tmp").exists()


@pytest.mark.unit
def test_write_crops_failure_raises_storage_error_and_keeps_the_old_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_crops(tmp_path, {"a.png": b"1"})
    original = Path.replace

    def failing_replace(self: Path, target: Path) -> Path:
        if self.name == "plate_crops.tmp":
            msg = "locked"
            raise PermissionError(msg)
        return original(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(StorageError, match="cannot write"):
        write_crops(tmp_path, {"b.png": b"2"})

    assert not (tmp_path / "plate_crops.tmp").exists()
    assert (tmp_path / "plate_crops" / "a.png").read_bytes() == b"1"
    assert [p.name for p in (tmp_path / "plate_crops").iterdir()] == ["a.png"]


@pytest.mark.unit
def test_write_crops_failure_while_swapping_in_the_new_folder_restores_the_old_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_crops(tmp_path, {"a.png": b"1"})
    original = Path.replace
    calls: list[str] = []

    def failing_replace(self: Path, target: Path) -> Path:
        calls.append(self.name)
        if self.name == "plate_crops.tmp":
            msg = "locked"
            raise PermissionError(msg)
        return original(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(StorageError):
        write_crops(tmp_path, {"b.png": b"2"})

    assert calls == ["plate_crops", "plate_crops.tmp", "plate_crops.old"]
    assert (tmp_path / "plate_crops" / "a.png").read_bytes() == b"1"
    assert not (tmp_path / "plate_crops.old").exists()


@pytest.mark.unit
def test_logo_of_plate_without_a_logo_crop_raises() -> None:
    track = Track("goal", None, [hit(1.0, "goal"), hit(1.5, "goal")])

    with pytest.raises(PlatesError, match="without a logo crop"):
        _logo_of(track)


@pytest.mark.unit
def test_encode_empty_image_raises_plates_error() -> None:
    with pytest.raises(PlatesError, match="cannot encode"):
        _encode(np.zeros((0, 0, 3), dtype=np.uint8))
