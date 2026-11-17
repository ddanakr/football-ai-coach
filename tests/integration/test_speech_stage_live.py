"""The speech stage against the real OpenRouter on 50 seconds of the match sound. Skipped unless
`AICOACH_LIVE_LLM=1` and `AICOACH_REAL_MATCH_VIDEO` is set (the match video, FFmpeg, an OpenRouter
key and the network are needed). Two chunks, a few thousandths of a dollar."""

import os
import subprocess
from pathlib import Path

import pytest

from aicoach.config import load_settings
from aicoach.speech.stage import run_transcribe
from aicoach.speech.transcript import TranscribeConfig

SPEECH_START_S = 1113.0  # Russian commentary in match01; change for another match

pytestmark = [pytest.mark.live_llm, pytest.mark.enable_socket]


@pytest.mark.skipif(os.environ.get("AICOACH_LIVE_LLM") != "1", reason="AICOACH_LIVE_LLM is not 1")
@pytest.mark.skipif(not os.environ.get("AICOACH_REAL_MATCH_VIDEO"), reason="no match video")
def test_real_stage_transcribes_two_chunks_into_segments_with_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = Path(os.environ["AICOACH_REAL_MATCH_VIDEO"])
    monkeypatch.undo()  # the real settings, not the isolated ones
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(SPEECH_START_S), "-i", str(video), "-t", "50",
        "-vn", "-ac", "1", "-ar", "16000", str(tmp_path / "audio.wav"),
    ]  # fmt: skip
    subprocess.run(command, check=True, timeout=300)  # noqa: S603 - fixed program, our arguments

    result = run_transcribe(
        tmp_path, "live", config=TranscribeConfig(chunk_s=30, overlap_s=5), settings=load_settings()
    )

    assert result is not None
    assert result.stats.n_chunks == 2
    assert result.segments
    assert result.stats.cost_usd is not None
    assert (tmp_path / "transcript.json").is_file()
