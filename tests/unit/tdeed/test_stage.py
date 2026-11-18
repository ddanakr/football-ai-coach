"""The `tdeed` stage with fakes: no git, FFmpeg, torch, video card or network."""

import json
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from aicoach.errors import SpottingError
from aicoach.spotting.schemas import load_tdeed_file
from aicoach.status import read_status
from aicoach.tdeed.config import CHECKPOINT_URL, NOTEBOOK_URL, REPO_COMMIT
from aicoach.tdeed.stage import (
    CHECKPOINT_FILE,
    REPO_DIR,
    ensure_checkpoint,
    find_existing,
    run_tdeed,
)
from tests.fixtures.tdeed_synth import CHECKPOINT_BYTES, Fakes, raw_output

WHOLE = "tdeed_SoccerNet_small_thr0.05_bs2.json"
SEGMENT = "tdeedseg_SoccerNet_small_thr0.05_bs2_792-1392s.json"


def _run(tmp_path: Path, fakes: Fakes, **kwargs: object) -> object:
    match_dir = tmp_path / "match"
    match_dir.mkdir(exist_ok=True)
    if not (match_dir / "src.mp4").exists():  # a rewrite would change the mtime in the key
        (match_dir / "src.mp4").write_bytes(b"source")
    return run_tdeed(
        match_dir,
        "m",
        cache_dir=tmp_path / "cache",
        tools=fakes.tools(),
        **kwargs,  # type: ignore[arg-type]  # forwarded options
    )


@pytest.mark.unit
def test_run_tdeed_whole_match_writes_the_file_that_fuse_looks_for(
    tmp_path: Path, fakes: Fakes
) -> None:
    fakes.inference.text = raw_output([(12, "Foul", 0.138427734375), (24, "Corner", 0.3)])

    result = _run(tmp_path, fakes)

    output = tmp_path / "match" / WHOLE
    assert result.output == output  # type: ignore[attr-defined]  # the result type of _run
    assert result.n_predictions == 2  # type: ignore[attr-defined]
    assert not result.cached  # type: ignore[attr-defined]
    assert [p.frame for p in load_tdeed_file(output).predictions] == [12, 24]
    assert find_existing(tmp_path / "match") == [output]


@pytest.mark.unit
def test_run_tdeed_fetches_the_pinned_commit_once_and_reuses_the_cache(
    tmp_path: Path, fakes: Fakes
) -> None:
    _run(tmp_path, fakes)
    first = list(fakes.commands.calls)
    (tmp_path / "match" / WHOLE).unlink()

    _run(tmp_path, fakes)

    fetches = [c for c in first if "fetch" in c]
    assert len(fetches) == 1
    assert REPO_COMMIT in fetches[0]
    second = fakes.commands.calls[len(first) :]
    assert not [c for c in second if "fetch" in c or "init" in c]  # reused
    assert not [c for c in second if c[0] == "ffmpeg"]  # the model input is cached too
    assert fakes.download.urls == [CHECKPOINT_URL]  # the checkpoint was downloaded once


@pytest.mark.unit
def test_run_tdeed_runs_inference_in_a_work_directory_with_the_patched_config(
    tmp_path: Path, fakes: Fakes
) -> None:
    _run(tmp_path, fakes)

    (run,) = fakes.inference.runs
    assert run["config"]["batch_size"] == 2
    assert run["config"]["num_workers"] == 1
    assert run["config"]["clip_len"] == 100  # the rest of the model's config is kept
    assert run["checkpoint"] == CHECKPOINT_BYTES
    assert run["classes"]
    assert run["cwd"].parent == tmp_path / "cache" / "runs"
    assert not run["cwd"].exists()  # removed after the run
    assert run["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert run["env"]["HF_HOME"] == str(tmp_path / "cache" / "hf")
    repo = tmp_path / "cache" / REPO_DIR
    assert run["command"][2] == str(repo / "inference.py")  # the script of the cached copy
    assert (
        json.loads((repo / "config" / "SoccerNet" / "SoccerNet_small.json").read_text())[
            "batch_size"
        ]
        == 8
    )  # the cached copy is untouched


@pytest.mark.unit
def test_run_tdeed_ffmpeg_gets_the_whole_video_without_a_cut(tmp_path: Path, fakes: Fakes) -> None:
    _run(tmp_path, fakes)

    (ffmpeg,) = [c for c in fakes.commands.calls if c[0] == "ffmpeg"]
    assert "-ss" not in ffmpeg
    assert "-t" not in ffmpeg


@pytest.mark.unit
def test_run_tdeed_status_has_running_done_and_timing(tmp_path: Path, fakes: Fakes) -> None:
    _run(tmp_path, fakes)

    entry = read_status(tmp_path / "match").stages["tdeed"]
    assert entry.state == "done"
    assert entry.progress == 1.0
    assert entry.started_at is not None
    assert entry.finished_at is not None
    assert entry.duration_s is not None
    assert entry.duration_s >= 0


@pytest.mark.unit
def test_run_tdeed_with_an_existing_file_does_nothing(tmp_path: Path, fakes: Fakes) -> None:
    match_dir = tmp_path / "match"
    match_dir.mkdir()
    existing = match_dir / "tdeed_frozen.json"
    existing.write_text(raw_output([(12, "Goal", 0.9)]))

    result = _run(tmp_path, fakes)

    assert result.cached  # type: ignore[attr-defined]
    assert result.output == existing  # type: ignore[attr-defined]
    assert fakes.commands.calls == []
    assert fakes.inference.runs == []
    assert existing.read_text() == raw_output([(12, "Goal", 0.9)])  # not touched


@pytest.mark.unit
@pytest.mark.parametrize("name", ["tdeed_frozen.json", WHOLE])
def test_run_tdeed_force_never_replaces_a_whole_match_file(
    tmp_path: Path, fakes: Fakes, name: str
) -> None:
    match_dir = tmp_path / "match"
    match_dir.mkdir()
    (match_dir / name).write_text(raw_output([(12, "Goal", 0.9)]))

    with pytest.raises(SpottingError, match="delete it by hand"):
        _run(tmp_path, fakes, force=True)

    assert fakes.inference.runs == []
    assert (match_dir / name).read_text() == raw_output([(12, "Goal", 0.9)])


@pytest.mark.unit
def test_run_tdeed_existing_file_without_cuda_says_so_instead_of_failing(
    tmp_path: Path, fakes: Fakes
) -> None:
    match_dir = tmp_path / "match"
    match_dir.mkdir()
    (match_dir / WHOLE).write_text(raw_output([(12, "Goal", 0.9)]))
    fakes.cuda = False

    assert _run(tmp_path, fakes).cached  # type: ignore[attr-defined]


@pytest.mark.unit
def test_run_tdeed_child_environment_has_no_keys_or_tokens(
    tmp_path: Path, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")  # pragma: allowlist secret
    monkeypatch.setenv("HF_TOKEN", "y")  # pragma: allowlist secret
    monkeypatch.setenv("KEEP_ME", "z")

    _run(tmp_path, fakes)

    env = fakes.inference.runs[0]["env"]
    assert "OPENROUTER_API_KEY" not in env
    assert "HF_TOKEN" not in env
    assert env["KEEP_ME"] == "z"


@pytest.mark.unit
def test_run_tdeed_segment_writes_its_own_file_and_leaves_the_whole_match_file_alone(
    tmp_path: Path, fakes: Fakes
) -> None:
    match_dir = tmp_path / "match"
    match_dir.mkdir()
    frozen = match_dir / WHOLE
    frozen.write_text(raw_output([(12, "Goal", 0.9)]))
    fakes.inference.text = raw_output([(0, "Foul", 0.5), (12, "Corner", 0.4)])

    result = _run(tmp_path, fakes, t_from=800.0, t_to=1392.0)

    segment_file = match_dir / SEGMENT
    assert result.output == segment_file  # type: ignore[attr-defined]
    assert result.segment.start_s == 792.0  # type: ignore[attr-defined]
    assert [p.frame for p in load_tdeed_file(segment_file).predictions] == [19800, 19812]
    assert frozen.read_text() == raw_output([(12, "Goal", 0.9)])
    assert find_existing(match_dir) == [frozen]  # `fuse` and `ingest` do not see the segment
    stages = read_status(match_dir).stages
    assert "tdeed" not in stages  # the entry of the real stage is not set by a segment run
    assert stages["tdeed_segment"].state == "done"


@pytest.mark.unit
def test_run_tdeed_segment_cuts_the_video_with_the_tail_and_keeps_its_own_input(
    tmp_path: Path, fakes: Fakes
) -> None:
    _run(tmp_path, fakes, t_from=800.0, t_to=1392.0)

    (ffmpeg,) = [c for c in fakes.commands.calls if c[0] == "ffmpeg"]
    assert ffmpeg[ffmpeg.index("-ss") + 1] == "792.000"
    assert ffmpeg[ffmpeg.index("-t") + 1] == "660.000"  # 600 s and the 60 s tail


@pytest.mark.unit
def test_run_tdeed_segment_run_twice_is_cached_and_force_runs_again(
    tmp_path: Path, fakes: Fakes
) -> None:
    _run(tmp_path, fakes, t_from=792.0, t_to=1392.0)
    again = _run(tmp_path, fakes, t_from=792.0, t_to=1392.0)

    assert again.cached  # type: ignore[attr-defined]
    assert len(fakes.inference.runs) == 1

    _run(tmp_path, fakes, t_from=792.0, t_to=1392.0, force=True)
    assert len(fakes.inference.runs) == 2


@pytest.mark.unit
def test_run_tdeed_segment_outside_the_video_raises_before_any_work(
    tmp_path: Path, fakes: Fakes
) -> None:
    fakes.duration_s = 300.0

    with pytest.raises(SpottingError, match="shorter than"):
        _run(tmp_path, fakes, t_from=792.0, t_to=1392.0)

    assert fakes.commands.calls == []


@pytest.mark.unit
def test_run_tdeed_auto_without_cuda_fails_with_the_notebook_and_does_no_work(
    tmp_path: Path, fakes: Fakes
) -> None:
    fakes.cuda = False

    with pytest.raises(SpottingError) as caught:
        _run(tmp_path, fakes)

    assert NOTEBOOK_URL in str(caught.value)
    assert fakes.commands.calls == []
    assert fakes.download.urls == []


@pytest.mark.unit
def test_run_tdeed_cpu_runs_without_cuda_through_the_launcher_and_warns(
    tmp_path: Path, fakes: Fakes
) -> None:
    fakes.cuda = False

    with capture_logs() as logs:
        result = _run(tmp_path, fakes, device="cpu")

    assert result.device == "cpu"  # type: ignore[attr-defined]
    assert "cpu_runner.py" in fakes.inference.runs[0]["command"][2]
    assert any(log["event"] == "tdeed_on_cpu" and log["log_level"] == "warning" for log in logs)


@pytest.mark.unit
def test_run_tdeed_unreadable_video_raises(tmp_path: Path, fakes: Fakes) -> None:
    from aicoach.errors import ScoreboardParseError  # noqa: PLC0415

    def broken(_path: Path) -> object:
        msg = "cannot open video"
        raise ScoreboardParseError(msg)

    tools = fakes.tools()
    object.__setattr__(tools, "probe", broken)
    (tmp_path / "match").mkdir()

    with pytest.raises(SpottingError, match="cannot read the video"):
        run_tdeed(tmp_path / "match", "m", cache_dir=tmp_path / "cache", tools=tools)


@pytest.mark.unit
def test_run_tdeed_inference_failure_is_recorded_and_leaves_no_output_or_work_dir(
    tmp_path: Path, fakes: Fakes
) -> None:
    fakes.inference.code = 1

    with pytest.raises(SpottingError, match=r"inference\.py failed \(exit code 1\)"):
        _run(tmp_path, fakes)

    entry = read_status(tmp_path / "match").stages["tdeed"]
    assert entry.state == "failed"
    assert "inference.py failed" in (entry.error or "")
    assert not (tmp_path / "match" / WHOLE).exists()
    assert list((tmp_path / "cache" / "runs").iterdir()) == []


@pytest.mark.unit
def test_run_tdeed_garbage_output_is_not_written(tmp_path: Path, fakes: Fakes) -> None:
    fakes.inference.text = '{"predictions": [{"frame": -4, "label": "Goal", "confidence": 1}]}'

    with pytest.raises(SpottingError):
        _run(tmp_path, fakes)

    assert not (tmp_path / "match" / WHOLE).exists()


@pytest.mark.unit
def test_run_tdeed_ffmpeg_failure_leaves_no_partial_input(tmp_path: Path, fakes: Fakes) -> None:
    fakes.commands.fail_on = "ffmpeg"

    with pytest.raises(SpottingError, match="preparing the model input"):
        _run(tmp_path, fakes)

    assert list((tmp_path / "cache" / "inputs").iterdir()) == []
    assert fakes.inference.runs == []


@pytest.mark.unit
def test_run_tdeed_git_failure_is_a_spotting_error(tmp_path: Path, fakes: Fakes) -> None:
    fakes.commands.fail_on = "git"

    with pytest.raises(SpottingError, match="git"):
        _run(tmp_path, fakes)


@pytest.mark.unit
def test_run_tdeed_cached_copy_at_another_commit_or_modified_is_refused(
    tmp_path: Path, fakes: Fakes
) -> None:
    _run(tmp_path, fakes)
    (tmp_path / "match" / WHOLE).unlink()

    fakes.commands.head = "0" * 40
    with pytest.raises(SpottingError, match="not d953b4d"):
        _run(tmp_path, fakes)

    fakes.commands.head = REPO_COMMIT
    fakes.commands.modified = " M inference.py"
    with pytest.raises(SpottingError, match="was modified"):
        _run(tmp_path, fakes)


@pytest.mark.unit
def test_run_tdeed_half_fetched_copy_is_fetched_again(tmp_path: Path, fakes: Fakes) -> None:
    broken = tmp_path / "cache" / REPO_DIR
    broken.mkdir(parents=True)
    (broken / "leftover").write_text("x")

    _run(tmp_path, fakes)

    assert not (broken / "leftover").exists()
    assert (broken / "inference.py").is_file()


@pytest.mark.unit
def test_ensure_checkpoint_known_file_is_downloaded_and_checked(
    tmp_path: Path, fakes: Fakes
) -> None:
    from structlog import get_logger  # noqa: PLC0415

    path, warnings = ensure_checkpoint(tmp_path, None, fakes.tools(), get_logger())

    assert path == tmp_path / "checkpoints" / "SoccerNet_small" / CHECKPOINT_FILE
    assert path.read_bytes() == CHECKPOINT_BYTES
    assert warnings == []


@pytest.mark.unit
def test_ensure_checkpoint_wrong_download_is_removed_and_explained(
    tmp_path: Path, fakes: Fakes
) -> None:
    from structlog import get_logger  # noqa: PLC0415

    fakes.download.content = b"<html>quota exceeded</html>"

    with pytest.raises(SpottingError, match="--checkpoint"):
        ensure_checkpoint(tmp_path, None, fakes.tools(), get_logger())

    assert not (tmp_path / "checkpoints" / "SoccerNet_small" / CHECKPOINT_FILE).exists()


@pytest.mark.unit
def test_ensure_checkpoint_given_file_is_used_and_an_unknown_one_warns(
    tmp_path: Path, fakes: Fakes
) -> None:
    from structlog import get_logger  # noqa: PLC0415

    known = tmp_path / "known.pt"
    known.write_bytes(CHECKPOINT_BYTES)
    other = tmp_path / "other.pt"
    other.write_bytes(b"something else")

    assert ensure_checkpoint(tmp_path, known, fakes.tools(), get_logger()) == (known, [])
    path, warnings = ensure_checkpoint(tmp_path, other, fakes.tools(), get_logger())
    assert path == other
    assert "not the known SoccerNet_small" in warnings[0]
    with pytest.raises(SpottingError, match="checkpoint not found"):
        ensure_checkpoint(tmp_path, tmp_path / "missing.pt", fakes.tools(), get_logger())
    assert fakes.download.urls == []  # a given file is never downloaded over


@pytest.mark.unit
def test_run_tdeed_with_a_given_checkpoint_skips_the_download(tmp_path: Path, fakes: Fakes) -> None:
    known = tmp_path / "known.pt"
    known.write_bytes(CHECKPOINT_BYTES)

    result = _run(tmp_path, fakes, checkpoint=known)

    assert fakes.download.urls == []
    assert result.warnings == []  # type: ignore[attr-defined]
    assert fakes.inference.runs[0]["checkpoint"] == CHECKPOINT_BYTES
