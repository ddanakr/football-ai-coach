"""Fakes for the `tdeed` stage: git, FFmpeg, `inference.py` and the checkpoint download, with no
network, no video card and no T-DEED code."""

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aicoach.scoreboard.video import VideoInfo
from aicoach.tdeed.config import REPO_COMMIT
from aicoach.tdeed.stage import TdeedTools

CHECKPOINT_BYTES = b"fake checkpoint"
CHECKPOINT_SHA = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()
FAKE_CONFIG = {"batch_size": 8, "num_workers": 4, "clip_len": 100, "num_classes": 17}


def raw_output(detections: list[tuple[int, str, float]]) -> str:
    """The text of a raw `results_inference.json` with `(frame, label, confidence)` items."""
    items = [{"frame": f, "label": label, "confidence": c} for f, label, c in detections]
    return json.dumps({"predictions": items}, indent=4)


@dataclass
class FakeCommands:
    """git, FFmpeg and anything else run with captured output; records every command."""

    head: str = REPO_COMMIT
    modified: str = ""
    fail_on: str | None = None  # the program name (`git`, `ffmpeg`) that exits with 1
    calls: list[list[str]] = field(default_factory=list)

    def __call__(
        self, command: list[str], _timeout_s: float
    ) -> "subprocess.CompletedProcess[bytes]":
        self.calls.append(command)
        if command[0] == self.fail_on:
            return subprocess.CompletedProcess(command, 1, b"", b"boom: fake failure")
        out = b""
        if command[0] == "git":
            out = self._git(command)
        elif command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"video")
        return subprocess.CompletedProcess(command, 0, out, b"")

    def _git(self, command: list[str]) -> bytes:
        if "init" in command:
            repo = Path(command[-1])
            (repo / ".git").mkdir(parents=True)
            (repo / "inference.py").write_text("# the unmodified script", encoding="utf-8")
            config = repo / "config" / "SoccerNet" / "SoccerNet_small.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps(FAKE_CONFIG), encoding="utf-8")
            classes = repo / "data" / "soccernet" / "class.txt"
            classes.parent.mkdir(parents=True)
            classes.write_text("Penalty\nGoal\n", encoding="utf-8")
            return b""
        if "rev-parse" in command:
            return self.head.encode()
        if "status" in command:
            return self.modified.encode()
        return b""

    def programs(self) -> list[str]:
        """The program names in the order they were run."""
        return [c[0] for c in self.calls]


@dataclass
class FakeInference:
    """Stands in for `inference.py`: records what the work directory held and writes an output."""

    text: str = field(default_factory=lambda: raw_output([(12, "Foul", 0.5)]))
    code: int = 0
    runs: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self, command: list[str], cwd: Path, env: Mapping[str, str], _timeout_s: float
    ) -> int:
        config = cwd / "config" / "SoccerNet" / "SoccerNet_small.json"
        self.runs.append(
            {
                "command": command,
                "cwd": cwd,
                "env": dict(env),
                "config": json.loads(config.read_text(encoding="utf-8")),
                "checkpoint": (
                    cwd / "checkpoints" / "SoccerNet" / "SoccerNet_small" / "checkpoint_best.pt"
                ).read_bytes(),
                "classes": (cwd / "data" / "soccernet" / "class.txt").is_file(),
            }
        )
        if self.code == 0:
            out = cwd / "inference_output" / "results_inference.json"
            out.parent.mkdir(parents=True)
            out.write_text(self.text, encoding="utf-8")
        return self.code


@dataclass
class FakeDownload:
    """Writes the fake checkpoint (or other bytes) where it is told to."""

    content: bytes = CHECKPOINT_BYTES
    urls: list[str] = field(default_factory=list)

    def __call__(self, url: str, destination: Path) -> None:
        self.urls.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.content)


@dataclass
class Fakes:
    """All fakes of one test, and the tools built from them."""

    commands: FakeCommands = field(default_factory=FakeCommands)
    inference: FakeInference = field(default_factory=FakeInference)
    download: FakeDownload = field(default_factory=FakeDownload)
    cuda: bool = True
    duration_s: float = 3000.0

    def probe(self, _path: Path) -> VideoInfo:
        return VideoInfo(fps=25.0, frame_count=round(self.duration_s * 25), width=1280, height=720)

    def tools(self) -> TdeedTools:
        return TdeedTools(
            command=self.commands,
            inference=self.inference,
            download=self.download,
            probe=self.probe,
            cuda_available=lambda: self.cuda,
        )
