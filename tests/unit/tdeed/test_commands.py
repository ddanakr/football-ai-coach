"""The command lines of the T-DEED run."""

import sys
from pathlib import Path

import pytest

from aicoach.tdeed.commands import (
    CPU_RUNNER,
    build_inference_command,
    build_input_command,
    git_fetch_commands,
    git_head_command,
    git_modified_command,
)
from aicoach.tdeed.config import REPO_COMMIT, REPO_URL


@pytest.mark.unit
def test_git_fetch_commands_fetch_the_pinned_commit_without_history(tmp_path: Path) -> None:
    commands = git_fetch_commands(tmp_path / "T-DEED")

    flat = [" ".join(c) for c in commands]
    assert flat[0].startswith("git init")
    assert f"remote add origin {REPO_URL}" in flat[1]
    assert f"fetch --quiet --depth 1 origin {REPO_COMMIT}" in flat[2]
    assert "checkout --quiet --detach FETCH_HEAD" in flat[3]
    assert REPO_COMMIT.startswith("d953b4d")  # the commit of D-042


@pytest.mark.unit
def test_git_read_commands_name_the_directory(tmp_path: Path) -> None:
    assert git_head_command(tmp_path)[-2:] == ["rev-parse", "HEAD"]
    assert "--untracked-files=no" in git_modified_command(tmp_path)


@pytest.mark.unit
def test_build_input_command_whole_video_has_no_cut(tmp_path: Path) -> None:
    command = build_input_command(tmp_path / "src.mp4", tmp_path / "out.mp4")

    assert "-ss" not in command
    assert "-t" not in command
    assert command[command.index("-vf") + 1] == "fps=25,scale=796:448"
    assert command[command.index("-i") + 1] == str(tmp_path / "src.mp4")
    assert "-an" in command
    assert command[-1] == str(tmp_path / "out.mp4")
    assert command[command.index("-crf") + 1] == "18"


@pytest.mark.unit
def test_build_input_command_segment_seeks_before_the_input_and_limits_the_length(
    tmp_path: Path,
) -> None:
    command = build_input_command(
        tmp_path / "src.mp4", tmp_path / "out.mp4", start_s=792.0, length_s=660.0
    )

    assert command.index("-ss") < command.index("-i") < command.index("-t")
    assert command[command.index("-ss") + 1] == "792.000"
    assert command[command.index("-t") + 1] == "660.000"


@pytest.mark.unit
def test_build_inference_command_runs_the_script_by_its_path(tmp_path: Path) -> None:
    command = build_inference_command(tmp_path / "T-DEED", tmp_path / "in.mp4", cpu=False)

    assert command[:3] == [sys.executable, "-u", str(tmp_path / "T-DEED" / "inference.py")]
    assert command[command.index("--model") + 1] == "SoccerNet_small"
    assert command[command.index("--inference_threshold") + 1] == "0.05"
    assert command[command.index("--frame_width") + 1] == "796"
    assert command[command.index("--frame_height") + 1] == "448"
    assert command[command.index("--video_path") + 1] == str(tmp_path / "in.mp4")


@pytest.mark.unit
def test_build_inference_command_cpu_goes_through_the_launcher(tmp_path: Path) -> None:
    command = build_inference_command(tmp_path / "T-DEED", tmp_path / "in.mp4", cpu=True)

    assert command[2] == str(CPU_RUNNER)
    assert CPU_RUNNER.is_file()
    assert command[3] == str(tmp_path / "T-DEED" / "inference.py")
