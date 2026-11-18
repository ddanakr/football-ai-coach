"""Child processes, the checkpoint download and file hashes of the T-DEED run (all faked)."""

import hashlib
import io
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from aicoach.errors import SpottingError
from aicoach.tdeed.runner import (
    checked,
    download_file,
    run_command,
    run_inference,
    sha256_of,
    stderr_tail,
)


def _fixed(code: int, out: bytes = b"", err: bytes = b"") -> Any:
    def run(command: list[str], _timeout_s: float) -> "subprocess.CompletedProcess[bytes]":
        return subprocess.CompletedProcess(command, code, out, err)

    return run


@pytest.mark.unit
def test_checked_returns_the_output_text() -> None:
    assert checked(["git"], _fixed(0, b"abc\n"), 1.0, "git x") == "abc\n"


@pytest.mark.unit
def test_checked_nonzero_exit_names_the_step_and_the_error_tail() -> None:
    with pytest.raises(SpottingError, match=r"git x failed \(exit code 128\): fatal: no"):
        checked(["git"], _fixed(128, b"", b"fatal: no"), 1.0, "git x")


@pytest.mark.unit
def test_checked_missing_program_says_to_install_it() -> None:
    def missing(_command: list[str], _timeout_s: float) -> Any:
        raise FileNotFoundError

    with pytest.raises(SpottingError, match="git not found"):
        checked(["git"], missing, 1.0, "git x")


@pytest.mark.unit
def test_checked_timeout_and_os_error_are_wrapped() -> None:
    def slow(command: list[str], timeout_s: float) -> Any:
        raise subprocess.TimeoutExpired(command, timeout_s)

    def broken(_command: list[str], _timeout_s: float) -> Any:
        raise PermissionError("denied")

    with pytest.raises(SpottingError, match="did not finish in 5 s"):
        checked(["ffmpeg"], slow, 5.0, "step")
    with pytest.raises(SpottingError, match="cannot start ffmpeg"):
        checked(["ffmpeg"], broken, 5.0, "step")


@pytest.mark.unit
def test_stderr_tail_keeps_the_last_lines_and_handles_empty_and_garbage() -> None:
    long = "\n".join(f"line {i}" for i in range(100)).encode()

    tail = stderr_tail(long)

    assert tail.splitlines()[-1] == "line 99"
    assert len(tail.splitlines()) == 20
    assert stderr_tail(b"") == "no error output"
    assert "�" in stderr_tail(b"\xff\xfe broken")


@pytest.mark.unit
def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 3_000_000)

    assert sha256_of(path) == hashlib.sha256(b"x" * 3_000_000).hexdigest()


@pytest.mark.unit
def test_run_command_and_run_inference_run_a_real_python(tmp_path: Path) -> None:
    done = run_command([sys.executable, "-c", "print('hi')"], 60.0)
    code = run_inference(
        [sys.executable, "-c", "import sys; sys.exit(3)"], tmp_path, {"PATH": ""}, 60.0
    )

    assert done.returncode == 0
    assert done.stdout.strip() == b"hi"
    assert code == 3


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@pytest.mark.unit
def test_download_file_writes_the_file_and_leaves_no_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> _Response:
        seen.append(url)
        assert timeout > 0
        return _Response(b"payload")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    target = tmp_path / "sub" / "ckpt.pt"

    download_file("https://example.org/ckpt", target)

    assert target.read_bytes() == b"payload"
    assert not (tmp_path / "sub" / "ckpt.pt.part").exists()
    assert seen == ["https://example.org/ckpt"]


@pytest.mark.unit
def test_download_file_refuses_a_non_https_address(tmp_path: Path) -> None:
    with pytest.raises(SpottingError, match="non-https"):
        download_file("http://example.org/x", tmp_path / "x")
    with pytest.raises(SpottingError, match="non-https"):
        download_file("file:///etc/passwd", tmp_path / "x")


@pytest.mark.unit
def test_download_file_network_failure_is_wrapped_and_leaves_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def failing(_url: str, timeout: float) -> Any:
        assert timeout > 0
        reason = "no route"
        raise urllib.error.URLError(reason)

    monkeypatch.setattr(urllib.request, "urlopen", failing)

    with pytest.raises(SpottingError, match="cannot download the checkpoint"):
        download_file("https://example.org/x", tmp_path / "x.pt")

    assert list(tmp_path.iterdir()) == []
