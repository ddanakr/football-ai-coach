"""The ingest command end to end with fakes: download, audio, stages, T-DEED wait, resume."""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aicoach.errors import IngestError, PlatesError
from aicoach.ingest.download import VIDEO_FILE, ProgressCallback
from aicoach.ingest.lock import LOCK_FILE
from aicoach.ingest.media import AUDIO_FILE
from aicoach.ingest.stage import (
    STAGE,
    IngestResult,
    IngestTools,
    ensure_templates,
    find_tdeed_file,
    match_directory,
    run_ingest,
)
from aicoach.scoreboard.stage import TEMPLATES_DIR
from aicoach.scoreboard.video import VideoInfo
from aicoach.status import read_status
from tests.fixtures.fusion_synth import write_tdeed
from tests.fixtures.pipeline_fakes import Recorder

VID = "abcDEF_-123"  # invented id
URL = f"https://youtu.be/{VID}?t=40"


@dataclass
class FakeDownloader:
    """Writes `src.mp4`, or fails; records the links it was given."""

    error: IngestError | None = None
    urls: list[str] = field(default_factory=list)

    def download(self, url: str, target_dir: Path, on_progress: ProgressCallback) -> Path:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        on_progress(0.5)
        path = target_dir / VIDEO_FILE
        path.write_bytes(b"video")
        return path


@dataclass
class FakeFfmpeg:
    """Writes the output file it was asked for; counts its calls."""

    calls: int = 0

    def __call__(
        self, command: list[str], _timeout_s: float
    ) -> "subprocess.CompletedProcess[bytes]":
        self.calls += 1
        Path(command[-1]).write_bytes(b"RIFF")
        return subprocess.CompletedProcess(command, 0, b"", b"")


@dataclass
class Fakes:
    downloader: FakeDownloader = field(default_factory=FakeDownloader)
    ffmpeg: FakeFfmpeg = field(default_factory=FakeFfmpeg)
    stages: Recorder = field(default_factory=Recorder)
    size: tuple[int, int] = (1280, 720)

    def probe(self, path: Path) -> VideoInfo:
        assert path.name == VIDEO_FILE
        return VideoInfo(fps=60.0, frame_count=6000, width=self.size[0], height=self.size[1])

    def tools(self) -> IngestTools:
        return IngestTools(
            downloader=self.downloader,
            ffmpeg=self.ffmpeg,
            probe=self.probe,
            stages=self.stages.stages(),
        )


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()


def make_templates(root: Path) -> Path:
    """A minimal `scoreboard_templates` directory of a calibrated match."""
    templates = root / "source_templates"
    (templates / "score").mkdir(parents=True, exist_ok=True)
    (templates / "calibration.json").write_text('{"decode_mode": "opencv_cpu"}')
    (templates / "score" / "0_a.png").write_bytes(b"png")
    return templates


def ingest(
    tmp_path: Path, fakes: Fakes, match_id: str | None = None, *, templates: bool = True
) -> IngestResult:
    return run_ingest(
        URL,
        tmp_path / "data",
        match_id=match_id,
        templates=make_templates(tmp_path) if templates else None,
        tools=fakes.tools(),
    )


def states(match_dir: Path) -> dict[str, str]:
    return {name: s.state for name, s in read_status(match_dir).stages.items()}


@pytest.mark.unit
def test_run_ingest_with_tdeed_file_runs_every_step_and_ends_done(
    tmp_path: Path, fakes: Fakes
) -> None:
    match_dir = tmp_path / "data" / VID
    match_dir.mkdir(parents=True)
    write_tdeed(match_dir / "tdeed_small.json", [])

    result = ingest(tmp_path, fakes)

    assert result.match_dir == match_dir
    assert (result.video_id, result.match_id) == (VID, VID)
    assert not result.waiting_for_tdeed
    assert fakes.downloader.urls == [f"https://www.youtube.com/watch?v={VID}"]
    assert (match_dir / AUDIO_FILE).is_file()
    assert (match_dir / TEMPLATES_DIR / "score" / "0_a.png").is_file()
    assert [c[0] for c in fakes.stages.calls] == ["scoreboard", "plates", "fusion"]
    fusion_args = fakes.stages.calls[2][1]
    assert fusion_args[0] == match_dir
    assert fusion_args[2] == match_dir / "tdeed_small.json"
    assert states(match_dir) == {
        "download": "done",
        "audio": "done",
        "tdeed": "done",
        STAGE: "done",
    }
    assert not (match_dir / LOCK_FILE).exists()


@pytest.mark.unit
def test_run_ingest_without_tdeed_waits_then_resumes_without_redoing_ready_steps(
    tmp_path: Path, fakes: Fakes
) -> None:
    first = ingest(tmp_path, fakes)

    assert first.waiting_for_tdeed
    assert first.fusion is None
    assert [c[0] for c in fakes.stages.calls] == ["scoreboard", "plates"]
    assert states(first.match_dir)["tdeed"] == "waiting"
    assert states(first.match_dir)[STAGE] == "waiting"
    assert not (first.match_dir / LOCK_FILE).exists()

    write_tdeed(first.match_dir / "tdeed_SoccerNet_small.json", [])
    again = Fakes()
    second = ingest(tmp_path, again, templates=False)  # the templates were copied the first time

    assert not second.waiting_for_tdeed
    assert again.downloader.urls == []  # src.mp4 is there
    assert again.ffmpeg.calls == 0  # audio.wav is there
    assert [c[0] for c in again.stages.calls] == ["scoreboard", "plates", "fusion"]
    assert states(second.match_dir)["tdeed"] == "done"
    assert states(second.match_dir)[STAGE] == "done"


@pytest.mark.unit
def test_run_ingest_wrong_frame_size_stops_before_the_stages(tmp_path: Path, fakes: Fakes) -> None:
    fakes.size = (1920, 1080)

    with pytest.raises(IngestError, match=r"stage download failed.*1920x1080.*1280x720"):
        ingest(tmp_path, fakes)

    match_dir = tmp_path / "data" / VID
    assert fakes.stages.calls == []
    assert fakes.ffmpeg.calls == 0
    assert states(match_dir)["download"] == "failed"
    status = read_status(match_dir).stages[STAGE]
    assert status.state == "failed"
    assert status.error is not None
    assert "1920x1080" in status.error
    assert not (match_dir / LOCK_FILE).exists()


@pytest.mark.unit
def test_run_ingest_unavailable_video_fails_the_download_step(tmp_path: Path, fakes: Fakes) -> None:
    fakes.downloader.error = IngestError("YouTube download failed: Video unavailable")

    with pytest.raises(IngestError, match="Video unavailable"):
        ingest(tmp_path, fakes)

    assert states(tmp_path / "data" / VID)["download"] == "failed"
    assert fakes.ffmpeg.calls == 0
    assert fakes.stages.calls == []


@pytest.mark.unit
def test_run_ingest_failing_stage_is_named_and_later_steps_do_not_run(
    tmp_path: Path, fakes: Fakes
) -> None:
    fakes.stages = Recorder(fail_at="plates", error=PlatesError("broken"))

    with pytest.raises(PlatesError, match="stage plates failed: broken"):
        ingest(tmp_path, fakes)

    assert [c[0] for c in fakes.stages.calls] == ["scoreboard", "plates"]
    assert states(tmp_path / "data" / VID)[STAGE] == "failed"


@pytest.mark.unit
def test_run_ingest_without_templates_fails_before_downloading(
    tmp_path: Path, fakes: Fakes
) -> None:
    with pytest.raises(IngestError, match="no scoreboard templates"):
        ingest(tmp_path, fakes, templates=False)

    assert fakes.downloader.urls == []


@pytest.mark.unit
def test_run_ingest_bad_link_writes_nothing(tmp_path: Path, fakes: Fakes) -> None:
    with pytest.raises(IngestError, match="not a YouTube"):
        run_ingest("https://vimeo.com/123", tmp_path / "data", tools=fakes.tools())

    assert not (tmp_path / "data").exists()


@pytest.mark.unit
def test_run_ingest_match_id_names_the_directory(tmp_path: Path, fakes: Fakes) -> None:
    result = ingest(tmp_path, fakes, "match01_ingest")

    assert result.match_dir == tmp_path / "data" / "match01_ingest"
    assert fakes.stages.calls[0][1][1] == "match01_ingest"


@pytest.mark.unit
def test_run_ingest_refuses_while_another_live_process_holds_the_lock(
    tmp_path: Path, fakes: Fakes
) -> None:
    match_dir = tmp_path / "data" / VID
    match_dir.mkdir(parents=True)
    holder = str(os.getppid())  # the parent of the test runner is alive
    (match_dir / LOCK_FILE).write_text(holder)

    with pytest.raises(IngestError, match="already being processed"):
        ingest(tmp_path, fakes)

    assert fakes.downloader.urls == []
    assert (match_dir / LOCK_FILE).read_text() == holder


@pytest.mark.unit
def test_run_ingest_unwritable_data_root_raises(tmp_path: Path, fakes: Fakes) -> None:
    (tmp_path / "data").write_text("a file, not a directory")

    with pytest.raises(IngestError, match="cannot create the match directory"):
        ingest(tmp_path, fakes)


@pytest.mark.unit
def test_run_ingest_broken_tdeed_file_fails_the_tdeed_step(tmp_path: Path, fakes: Fakes) -> None:
    match_dir = tmp_path / "data" / VID
    match_dir.mkdir(parents=True)
    (match_dir / "tdeed_x.json").write_text("{not json")

    with pytest.raises(IngestError, match="T-DEED file is not valid"):
        ingest(tmp_path, fakes)

    assert states(match_dir)["tdeed"] == "failed"
    assert [c[0] for c in fakes.stages.calls] == ["scoreboard", "plates"]


@pytest.mark.unit
def test_match_directory_empty_match_id_means_the_video_id(tmp_path: Path) -> None:
    assert match_directory(tmp_path, VID, "") == tmp_path / VID
    assert match_directory(tmp_path, VID, None) == tmp_path / VID
    assert match_directory(tmp_path, VID, "m_1.a") == tmp_path / "m_1.a"


@pytest.mark.unit
@pytest.mark.parametrize("name", ["..", ".", "a/b", "../up", "a\\b", "x" * 65, "-dash", "имя"])
def test_match_directory_refuses_unsafe_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(IngestError, match="match id"):
        match_directory(tmp_path, VID, name)


@pytest.mark.unit
@pytest.mark.parametrize("video_id", ["-bcDEF_-123", "_bcDEF_-123", "-" * 11])
def test_run_ingest_video_id_starting_with_dash_or_underscore_names_the_directory(
    tmp_path: Path, fakes: Fakes, video_id: str
) -> None:
    result = run_ingest(
        f"https://youtu.be/{video_id}",
        tmp_path / "data",
        templates=make_templates(tmp_path),
        tools=fakes.tools(),
    )

    assert result.match_dir == tmp_path / "data" / video_id
    assert result.waiting_for_tdeed
    assert (result.match_dir / VIDEO_FILE).is_file()


@pytest.mark.unit
def test_run_ingest_without_templates_creates_no_directory(tmp_path: Path, fakes: Fakes) -> None:
    with pytest.raises(IngestError, match="no scoreboard templates"):
        ingest(tmp_path, fakes, templates=False)

    assert not (tmp_path / "data").exists()


@pytest.mark.unit
def test_run_ingest_two_tdeed_files_fail_the_tdeed_step(tmp_path: Path, fakes: Fakes) -> None:
    match_dir = tmp_path / "data" / VID
    match_dir.mkdir(parents=True)
    (match_dir / "tdeed_a.json").write_text("{}")
    (match_dir / "tdeed_b.json").write_text("{}")

    with pytest.raises(IngestError, match="several T-DEED files"):
        ingest(tmp_path, fakes)

    assert states(match_dir)["tdeed"] == "failed"
    assert states(match_dir)[STAGE] == "failed"


@pytest.mark.unit
def test_find_tdeed_file_none_one_and_several(tmp_path: Path) -> None:
    assert find_tdeed_file(tmp_path) is None
    (tmp_path / "eval_tdeed_a_half1.json").write_text("{}")  # an evaluation, not a T-DEED file
    (tmp_path / "tdeed_dir.json").mkdir()
    assert find_tdeed_file(tmp_path) is None

    (tmp_path / "tdeed_a.json").write_text("{}")
    assert find_tdeed_file(tmp_path) == tmp_path / "tdeed_a.json"

    (tmp_path / "tdeed_b.json").write_text("{}")
    with pytest.raises(IngestError, match=r"several T-DEED files.*tdeed_a.json, tdeed_b.json"):
        find_tdeed_file(tmp_path)


@pytest.mark.unit
def test_ensure_templates_keeps_existing_and_refuses_a_wrong_source(tmp_path: Path) -> None:
    match_dir = tmp_path / "m"
    (match_dir / TEMPLATES_DIR).mkdir(parents=True)
    ensure_templates(match_dir, None)  # already there: nothing to do

    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(IngestError, match="no calibration.json"):
        ensure_templates(other, tmp_path / "not_templates")


@pytest.mark.unit
def test_ensure_templates_copy_failure_leaves_no_partial_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_templates(tmp_path)
    match_dir = tmp_path / "m"
    match_dir.mkdir()

    def broken_copy(_src: Path, dst: Path) -> None:
        Path(dst).mkdir()
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr("aicoach.ingest.stage.shutil.copytree", broken_copy)

    with pytest.raises(IngestError, match="cannot copy the templates"):
        ensure_templates(match_dir, source)
    assert list(match_dir.iterdir()) == []
