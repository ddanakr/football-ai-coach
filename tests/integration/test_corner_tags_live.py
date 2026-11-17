"""Two real requests with clips to the video model: a kit colour and the tags of a corner.
Skipped unless `AICOACH_LIVE_LLM=1`.

Run: `$env:AICOACH_LIVE_LLM = "1"; $env:AICOACH_REAL_MATCH_VIDEO = "<src.mp4>"; pytest
tests/integration/test_corner_tags_live.py --no-cov` (an OpenRouter key, the network, FFmpeg and
the match video are needed). Two requests cost about a third of a cent each; the clips are cut
into the `clips/` directory next to the video (outside git), nothing else is written. The
assertions are about the form of the answers (valid JSON of the schema, allowed values), not about
what is on the pitch: the model may be wrong, the contract must hold.
"""

import os
from pathlib import Path

import pytest

from aicoach.clips import export_clip
from aicoach.config import DEFAULT_VLM_MODEL, load_settings
from aicoach.corner_tags.tagger import SYSTEM_PROMPT as CORNER_PROMPT
from aicoach.corner_tags.tagger import build_prompt as corner_prompt
from aicoach.corner_tags.tagger import parse_corner_answer
from aicoach.domain.events import EventType
from aicoach.openrouter import OpenRouterHttp
from aicoach.teams.kits import SYSTEM_PROMPT as KIT_PROMPT
from aicoach.teams.kits import build_prompt as kit_prompt
from aicoach.teams.kits import parse_kit_answer
from aicoach.vlm import OpenRouterVideoModel, ask_json
from tests.fixtures.ui_synth import make_event

GOAL_T_V = 1055.0  # the first goal of match01 (video seconds); change for another match
CORNER_T_V = 833.28  # the first corner of match01

pytestmark = [pytest.mark.live_llm, pytest.mark.enable_socket]

LIVE = pytest.mark.skipif(
    os.environ.get("AICOACH_LIVE_LLM") != "1" or not os.environ.get("AICOACH_REAL_MATCH_VIDEO"),
    reason="AICOACH_LIVE_LLM is not 1, or AICOACH_REAL_MATCH_VIDEO is not set",
)


def real_model(monkeypatch: pytest.MonkeyPatch) -> OpenRouterVideoModel:
    monkeypatch.undo()  # the real settings, not the isolated ones
    key = load_settings().require_openrouter_api_key()
    return OpenRouterVideoModel(OpenRouterHttp(key), DEFAULT_VLM_MODEL)


@LIVE
def test_real_model_answers_the_kit_question_with_an_allowed_colour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = Path(os.environ["AICOACH_REAL_MATCH_VIDEO"])
    model = real_model(monkeypatch)
    clip = export_clip(video, make_event("p:5", EventType.GOAL, GOAL_T_V), 4.0, 12.0)

    answer = ask_json(
        model, KIT_PROMPT, kit_prompt("goal", 4.0), clip.read_bytes(), parse_kit_answer
    )

    assert answer.value is not None
    assert answer.usage.cost_usd is not None
    assert answer.usage.cost_usd > 0


@LIVE
def test_real_model_answers_the_corner_question_in_the_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = Path(os.environ["AICOACH_REAL_MATCH_VIDEO"])
    model = real_model(monkeypatch)
    clip = export_clip(video, make_event("p:1", EventType.CORNER, CORNER_T_V), 4.0, 12.0)

    answer = ask_json(
        model, CORNER_PROMPT, corner_prompt(4.0), clip.read_bytes(), parse_corner_answer
    )

    assert answer.value is not None
    assert answer.value.is_corner in ("yes", "no", "unclear")
    assert answer.usage.cost_usd is not None
    assert answer.usage.cost_usd > 0
