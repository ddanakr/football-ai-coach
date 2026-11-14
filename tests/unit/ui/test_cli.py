"""The `ui` command: arguments, defaults and the command it starts (Streamlit is not run)."""

import subprocess
from pathlib import Path

import pytest

from aicoach.cli import main
from aicoach.ui.launch import app_script

VID = "abcDEF_-123"


class FakeRun:
    """Stands in for `subprocess.run`: records the command, returns a canned exit code."""

    def __init__(self, returncode: int = 0, raises: BaseException | None = None) -> None:
        self.returncode = returncode
        self.raises = raises
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], *, check: bool) -> subprocess.CompletedProcess[bytes]:
        assert check is False
        self.commands.append(command)
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(command, self.returncode)


@pytest.mark.unit
def test_ui_starts_streamlit_with_defaults_from_the_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeRun()
    monkeypatch.setattr("aicoach.cli.subprocess.run", fake)
    video = tmp_path / "match01" / "src.mp4"

    code = main(["ui", "--video", str(video), "--youtube-id", VID])

    assert code == 0
    (command,) = fake.commands
    assert command[1:5] == ["-m", "streamlit", "run", str(app_script())]
    assert command[5:] == [
        "--",
        "--db",
        str(video.parent / "aicoach.db"),
        "--match-id",
        "match01",
        f"--youtube-id={VID}",
        "--match-dir",
        str(video.parent),
        "--video",
        str(video),
    ]


@pytest.mark.unit
def test_ui_uses_given_match_id_and_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeRun()
    monkeypatch.setattr("aicoach.cli.subprocess.run", fake)
    db = tmp_path / "other.db"

    main(["ui", "--video", "v.mp4", "--youtube-id", VID, "--match-id", "m9", "--db", str(db)])

    command = fake.commands[0]
    assert command[command.index("--db") + 1] == str(db)
    assert command[command.index("--match-id") + 1] == "m9"


@pytest.mark.unit
def test_ui_accepts_an_id_that_starts_with_a_dash(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRun()
    monkeypatch.setattr("aicoach.cli.subprocess.run", fake)

    code = main(["ui", "--video", "v.mp4", f"--youtube-id=-{VID[1:]}"])

    assert code == 0
    assert f"--youtube-id=-{VID[1:]}" in fake.commands[0]


@pytest.mark.unit
def test_ui_bad_id_exits_with_2_and_does_not_start(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeRun()
    monkeypatch.setattr("aicoach.cli.subprocess.run", fake)

    code = main(["ui", "--video", "v.mp4", "--youtube-id", "not-an-id"])

    assert code == 2
    assert "11 characters" in capsys.readouterr().err
    assert fake.commands == []


@pytest.mark.unit
def test_ui_passes_on_the_exit_code_of_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aicoach.cli.subprocess.run", FakeRun(returncode=3))

    assert main(["ui", "--video", "v.mp4", "--youtube-id", VID]) == 3


@pytest.mark.unit
def test_ui_ctrl_c_is_a_normal_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aicoach.cli.subprocess.run", FakeRun(raises=KeyboardInterrupt()))

    assert main(["ui", "--video", "v.mp4", "--youtube-id", VID]) == 0
