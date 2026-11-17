"""One real request with a clip to the video model. Skipped unless `AICOACH_LIVE_LLM=1`.

Run: `$env:AICOACH_LIVE_LLM = "1"; $env:AICOACH_REAL_MATCH_VIDEO = "<src.mp4>"; pytest
tests/integration/test_episode_live.py --no-cov` (an OpenRouter key, the network, FFmpeg and the
match video are needed). One request costs about a third of a cent; the clip is cut into the
`clips/` directory next to the video (outside git), nothing else is written.
"""

import os
from pathlib import Path

import pytest

from aicoach.clips import export_clip
from aicoach.config import DEFAULT_VLM_MODEL, load_settings
from aicoach.domain.events import EventType
from aicoach.episode import SYSTEM_PROMPT, build_prompt
from aicoach.openrouter import OpenRouterHttp
from aicoach.vlm import OpenRouterVideoModel
from tests.fixtures.ui_synth import make_event

GOAL_T_V = 1055.0  # the first goal of match01 (video seconds); change for another match

pytestmark = [pytest.mark.live_llm, pytest.mark.enable_socket]

LIVE = pytest.mark.skipif(
    os.environ.get("AICOACH_LIVE_LLM") != "1" or not os.environ.get("AICOACH_REAL_MATCH_VIDEO"),
    reason="AICOACH_LIVE_LLM is not 1, or AICOACH_REAL_MATCH_VIDEO is not set",
)


@LIVE
def test_real_video_model_describes_a_real_clip_and_reports_its_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = Path(os.environ["AICOACH_REAL_MATCH_VIDEO"])
    monkeypatch.undo()  # the real settings, not the isolated ones
    key = load_settings().require_openrouter_api_key()
    model = OpenRouterVideoModel(OpenRouterHttp(key), DEFAULT_VLM_MODEL)
    event = make_event("p:5", EventType.GOAL, GOAL_T_V)
    clip = export_clip(video, event)

    reply = model.describe(
        SYSTEM_PROMPT,
        build_prompt(8.0),
        clip.read_bytes(),
        video_seconds=20.0,
    )

    assert reply.text.strip()
    assert not reply.truncated
    assert reply.usage.input_tokens > 0
    assert reply.usage.cost_usd is not None
    assert reply.usage.cost_usd > 0
