"""One short transcription with the real Gemini model. Skipped unless `AICOACH_LIVE_LLM=1`.

Run: `$env:AICOACH_LIVE_LLM = "1"; $env:AICOACH_REAL_MATCH_VIDEO = "<src.mp4>"; pytest
tests/integration/test_speech_live.py --no-cov` (a key, FFmpeg, the network and the video needed).
"""

import os
from pathlib import Path

import pytest

from aicoach.config import load_settings
from aicoach.speech.audio import AUDIO_MIME, cut_audio
from aicoach.speech.transcriber import GeminiTranscriber

SPEECH_START_S = 1113.0  # a second of match01 with Russian commentary; change for another match

pytestmark = [pytest.mark.live_llm, pytest.mark.enable_socket]


@pytest.mark.skipif(os.environ.get("AICOACH_LIVE_LLM") != "1", reason="AICOACH_LIVE_LLM is not 1")
@pytest.mark.skipif(not os.environ.get("AICOACH_REAL_MATCH_VIDEO"), reason="no match video")
def test_real_model_returns_timed_words_for_twenty_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = Path(os.environ["AICOACH_REAL_MATCH_VIDEO"])
    monkeypatch.undo()  # the real settings, not the isolated ones
    settings = load_settings()
    client = GeminiTranscriber(settings.require_gemini_api_key(), settings.gemini_transcribe_model)
    audio = cut_audio(video, tmp_path / "s.mp3", SPEECH_START_S, 20.0)

    raw = client.transcribe(audio, AUDIO_MIME)

    assert raw.input_tokens > 0
    assert raw.text
    assert raw.words
