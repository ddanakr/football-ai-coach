"""Short requests to the real OpenRouter. Skipped unless `AICOACH_LIVE_LLM=1`.

Run: `$env:AICOACH_LIVE_LLM = "1"; pytest tests/integration/test_openrouter_live.py --no-cov` (an
OpenRouter key and the network are needed; the speech test also needs FFmpeg and the match video
in `AICOACH_REAL_MATCH_VIDEO`). Each request costs a fraction of a cent.
"""

import os
from pathlib import Path

import pytest

from aicoach.agent.messages import ToolSpec, UserMessage
from aicoach.agent.openrouter_llm import OpenRouterClient
from aicoach.config import DEFAULT_AGENT_MODEL, load_settings
from aicoach.openrouter import OpenRouterHttp
from aicoach.speech.audio import AUDIO_MIME, cut_audio
from aicoach.speech.openrouter_stt import OpenRouterTranscriber

SPEECH_START_S = (
    1113.0  # 20 seconds from here of match01 have Russian commentary; change for another match
)

pytestmark = [pytest.mark.live_llm, pytest.mark.enable_socket]

LIVE = pytest.mark.skipif(
    os.environ.get("AICOACH_LIVE_LLM") != "1", reason="AICOACH_LIVE_LLM is not 1"
)


@LIVE
def test_real_model_calls_a_tool_and_reports_its_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # the real settings, not the isolated ones
    key = load_settings().require_openrouter_api_key()
    client = OpenRouterClient(OpenRouterHttp(key), DEFAULT_AGENT_MODEL)
    spec = ToolSpec("count_goals", "Число голов в матче.", {"type": "object", "properties": {}})

    turn = client.generate(
        "Отвечай, вызвав инструмент.", [UserMessage("Сколько голов в матче?")], [spec]
    )

    assert [c.name for c in turn.calls] == ["count_goals"]
    assert turn.usage.input_tokens > 0
    assert turn.usage.cost_usd is not None
    assert turn.usage.cost_usd > 0


@LIVE
@pytest.mark.skipif(not os.environ.get("AICOACH_REAL_MATCH_VIDEO"), reason="no match video")
def test_real_scribe_returns_timed_words_for_twenty_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = Path(os.environ["AICOACH_REAL_MATCH_VIDEO"])
    monkeypatch.undo()
    key = load_settings().require_openrouter_api_key()
    client = OpenRouterTranscriber(OpenRouterHttp(key), "elevenlabs/scribe-v2")
    audio = cut_audio(video, tmp_path / "s.mp3", SPEECH_START_S, 20.0)

    raw = client.transcribe(audio, AUDIO_MIME, duration_s=20.0)

    assert raw.text
    assert raw.words
    assert raw.cost_usd is not None
    assert raw.audio_seconds is not None
