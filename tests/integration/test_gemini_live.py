"""One question to the real Gemini API. Skipped unless `AICOACH_LIVE_LLM=1` and a key is set.

Run: `$env:AICOACH_LIVE_LLM = "1"; pytest tests/integration/test_gemini_live.py --no-cov`.
The socket block of the test run is lifted for this test only.
"""

import os

import pytest

from aicoach.agent.llm import GeminiClient
from aicoach.agent.messages import UserMessage
from aicoach.agent.tools import QUERY_EVENTS
from aicoach.config import load_settings

pytestmark = [pytest.mark.live_llm, pytest.mark.enable_socket]


@pytest.mark.skipif(os.environ.get("AICOACH_LIVE_LLM") != "1", reason="AICOACH_LIVE_LLM is not 1")
def test_real_gemini_answers_a_plain_question_and_calls_a_tool_when_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The autouse fixture hides the developer's settings; the live test needs the real ones.
    monkeypatch.undo()
    settings = load_settings()
    client = GeminiClient(settings.require_gemini_api_key(), settings.gemini_model)

    turn = client.generate(
        "Ты помощник. Для вопросов о событиях матча вызывай query_events.",
        [UserMessage("Покажи все голы матча.")],
        [QUERY_EVENTS],
    )

    assert turn.usage.input_tokens > 0
    assert turn.calls or turn.text
