"""Usage sums, the price tables and the estimate before a request."""

import pytest

from aicoach.agent.messages import Usage
from aicoach.agent.pricing import (
    CHARS_PER_TOKEN,
    ESTIMATED_OUTPUT_TOKENS,
    cost_of,
    cost_usd,
    estimate_request_usd,
)
from aicoach.speech.pricing import estimate_transcription_usd


@pytest.mark.unit
def test_usage_sums_tokens_and_keeps_the_cost_when_every_part_has_one() -> None:
    assert Usage(10, 1, 0.5) + Usage(20, 2, 0.25) == Usage(30, 3, 0.75)


@pytest.mark.unit
def test_usage_sum_has_no_cost_when_a_part_lacks_one() -> None:
    assert Usage(10, 1, 0.5) + Usage(20, 2) == Usage(30, 3, None)


@pytest.mark.unit
def test_an_empty_usage_is_the_identity_of_the_sum() -> None:
    reported = Usage(10, 1, 0.5)

    assert Usage() + reported == reported
    assert reported + Usage() == reported
    assert Usage() + Usage() == Usage()


@pytest.mark.unit
def test_cost_of_prefers_what_the_service_charged() -> None:
    model = "google/gemini-3.5-flash-lite"

    assert cost_of(model, Usage(1_000_000, 1_000_000, 0.123)) == 0.123
    assert cost_of(model, Usage(1_000_000, 1_000_000)) == pytest.approx(2.8)
    assert cost_of("vendor/unknown", Usage(5, 5)) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("google/gemini-3.5-flash-lite", 0.30 + 2.50),
        ("google/gemini-3.7-flash", 0.75 + 3.75),
        ("openai/gpt-6-luna", 0.10 + 0.50),
        ("anthropic/claude-haiku-5.5", 0.10 + 0.50),
    ],
)
def test_the_four_openrouter_models_have_table_prices(model: str, expected: float) -> None:
    assert cost_usd(model, Usage(1_000_000, 1_000_000)) == pytest.approx(expected)


@pytest.mark.unit
def test_request_estimate_counts_the_prompt_and_an_assumed_answer() -> None:
    chars = 10_000 * CHARS_PER_TOKEN
    expected = (10_000 * 0.30 + ESTIMATED_OUTPUT_TOKENS * 2.50) / 1_000_000

    assert estimate_request_usd("google/gemini-3.5-flash-lite", chars) == pytest.approx(expected)


@pytest.mark.unit
def test_request_estimate_rounds_tokens_up_and_handles_empty_and_negative_input() -> None:
    one = estimate_request_usd("openai/gpt-6-luna", 1)
    zero = estimate_request_usd("openai/gpt-6-luna", 0)

    assert one is not None
    assert zero is not None
    assert one > zero  # one character is one token, not zero
    assert estimate_request_usd("openai/gpt-6-luna", -5) == zero


@pytest.mark.unit
def test_request_estimate_of_an_unknown_model_is_none_not_a_guess() -> None:
    assert estimate_request_usd("vendor/unknown", 1000) is None


@pytest.mark.unit
def test_transcription_estimate_is_seconds_times_the_price_and_none_when_unknown() -> None:
    assert estimate_transcription_usd("elevenlabs/scribe-v2", 100) == pytest.approx(0.00611111)
    assert estimate_transcription_usd("elevenlabs/scribe-v2", -5) == 0.0
    assert estimate_transcription_usd("vendor/unknown", 100) is None
