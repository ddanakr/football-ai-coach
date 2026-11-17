"""Fakes for the episode description: a scripted video model and a clip cutter. No network, no
FFmpeg. The words of the descriptions are invented."""

from collections.abc import Sequence
from pathlib import Path

from aicoach.agent.messages import Usage
from aicoach.storage import CurrentEvent
from aicoach.vlm import VideoReply

TEXT = "Игрок в красной форме бьёт по воротам, вратарь отбивает мяч."


def reply(text: str = TEXT, cost: float | None = 0.0031, *, truncated: bool = False) -> VideoReply:
    """A scripted answer of the video model."""
    return VideoReply(text, Usage(2100, 480, cost), truncated)


class FakeVideoModel:
    """Plays a script of replies (or exceptions); with no script it answers `reply()` forever."""

    def __init__(
        self, script: Sequence[VideoReply | Exception] = (), model: str = "test/vlm-1"
    ) -> None:
        self.model = model
        self._script = list(script)
        self.requests: list[tuple[str, str, bytes, float | None]] = []

    def describe(
        self, system: str, prompt: str, video: bytes, *, video_seconds: float | None = None
    ) -> VideoReply:
        """Record the request and play the next step."""
        self.requests.append((system, prompt, video, video_seconds))
        step = self._script.pop(0) if self._script else reply()
        if isinstance(step, Exception):
            raise step
        return step


class FakeClipper:
    """Stands in for `export_clip`: writes a small file next to the video, or fails."""

    def __init__(self, raises: Exception | None = None, content: bytes = b"fake mp4") -> None:
        self.raises = raises
        self.content = content
        self.calls: list[tuple[Path, CurrentEvent, float, float]] = []

    def __call__(self, video: Path, event: CurrentEvent, before: float, after: float) -> Path:
        self.calls.append((video, event, before, after))
        if self.raises is not None:
            raise self.raises
        path = video.parent / "clips" / f"{event.event_id.replace(':', '')}.mp4"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(self.content)
        return path
