"""Synthetic events and a fake FFmpeg for the reel tests: no video, no FFmpeg."""

import json
import subprocess
from pathlib import Path
from typing import Literal

from aicoach.agent.context import EventFacts, minute_text
from aicoach.domain.events import EventType
from aicoach.scoreboard.video import VideoInfo
from tests.fixtures.ui_synth import make_event

INFO = VideoInfo(fps=25.0, frame_count=25 * 4000, width=64, height=36)  # 4000 s


def fake_probe(_video: Path) -> VideoInfo:
    return INFO


def make_facts(
    event_id: str,
    event_type: EventType,
    t_v: float,
    *,
    half: Literal[1, 2] | None = 1,
    team: Literal["home", "away"] | None = "home",
    minute: int | None = None,
    card: str | None = None,
) -> EventFacts:
    """An event with its game minute and (for a foul) the colour of its card plate."""
    event = make_event(event_id, event_type, t_v, half, team=team)
    return EventFacts(event, minute, minute_text(minute, half), (), card)


def match_facts() -> list[EventFacts]:
    """Goals in both halves, a foul with a card, a plain foul, a corner with an unknown team,
    and events given out of time order."""
    return [
        make_facts("p:9", EventType.GOAL, 3000.0, half=2, team="away", minute=60),
        make_facts("p:1", EventType.GOAL, 600.0, team="home", minute=5),
        make_facts("p:2", EventType.FOUL, 700.0, team="home", minute=7, card="yellow"),
        make_facts("p:3", EventType.FOUL, 800.0, team="away", minute=9),
        make_facts("p:4", EventType.CORNER, 900.0, team=None, minute=11),
        make_facts("e:5", EventType.GOAL, 1000.0, half=None, team=None, minute=None),
    ]


class ReelRunner:
    """Stands in for `subprocess.run` for FFmpeg and FFprobe.

    A cut or a join writes its output file (the last argument); FFprobe answers with the stream
    parameters (`signatures`, chosen by the start of the file name; the default is one common
    signature) and with a length of 40 s.
    """

    def __init__(
        self,
        *,
        signatures: dict[str, list[dict[str, str]]] | None = None,
        ffprobe_missing: bool = False,
        fail_join: bool = False,
        fail_cut_number: int | None = None,
    ) -> None:
        self.signatures = signatures or {}
        self.ffprobe_missing = ffprobe_missing
        self.fail_join = fail_join
        self.fail_cut_number = fail_cut_number
        self.calls: list[list[str]] = []

    def cuts(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "ffmpeg" and "concat" not in c]

    def joins(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "ffmpeg" and "concat" in c]

    def __call__(
        self, command: list[str], _timeout_s: float
    ) -> "subprocess.CompletedProcess[bytes]":
        self.calls.append(command)
        if command[0] == "ffprobe":
            return self._probe(command)
        if "concat" in command and self.fail_join:
            Path(command[-1]).write_bytes(b"half a reel")  # FFmpeg creates the file first
            return subprocess.CompletedProcess(command, 1, b"", b"join failed")
        if "concat" not in command and len(self.cuts()) == self.fail_cut_number:
            return subprocess.CompletedProcess(command, 1, b"", b"cut failed")
        Path(command[-1]).write_bytes(b"fake mp4")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    def _probe(self, command: list[str]) -> "subprocess.CompletedProcess[bytes]":
        if self.ffprobe_missing:
            raise FileNotFoundError(command[0])
        if "format=duration" in command:
            return subprocess.CompletedProcess(command, 0, b'{"format": {"duration": "40.0"}}', b"")
        name = Path(command[-1]).name
        # A signature is chosen by the start of the file name; the rest share the default one.
        streams = next(
            (s for prefix, s in self.signatures.items() if name.startswith(prefix)),
            [
                {"codec_type": "video", "codec_name": "h264", "width": "64", "height": "36"},
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
            ],
        )
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"streams": streams}).encode(), b""
        )
