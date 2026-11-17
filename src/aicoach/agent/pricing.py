"""Price of the language-model requests, for the cost of one question (D-014).

The prices are a table in the code, not a setting in `.env` (D-060), with their source. A model
that is not in the table has no price: the cost is `None`, never a guess.
"""

from aicoach.agent.messages import Usage

# US dollars per 1M tokens (input, output), Standard tier, text. Output includes thinking
# tokens. Source: the pricing page of the Gemini API docs (https://ai.google.dev/gemini-api/docs/
# pricing), read on 2026-11-14. These are the prices of the PAID tier; the free tier is free of
# charge, so the figures are what the same traffic would cost once billing is on. The
# gemini-3.6-flash and gemini-3.8-flash prices change on 2027-01-01 (0.75 / 3.75 until then), so
# they are left out: such a model has no price rather than a price that goes stale.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}
_PER_MILLION = 1_000_000


def cost_usd(model: str, usage: Usage) -> float | None:
    """The cost of `usage` at the table price of `model` (Gemini paid tier or OpenRouter);
    `None` for an unknown model."""
    prices = PRICES_USD_PER_MTOK.get(model) or OPENROUTER_PRICES_USD_PER_MTOK.get(model)
    if prices is None:
        return None
    price_in, price_out = prices
    return (usage.input_tokens * price_in + usage.output_tokens * price_out) / _PER_MILLION


def cost_of(model: str, usage: Usage) -> float | None:
    """The cost the service charged (`usage.cost_usd`), else the table price, else `None`."""
    return usage.cost_usd if usage.cost_usd is not None else cost_usd(model, usage)


# OpenRouter, US dollars per 1M tokens (input, output): `pricing.prompt` and `pricing.completion`
# of each model in https://openrouter.ai/api/v1/models multiplied by 1e6, read on 2026-11-17.
# Used only for the estimate BEFORE a request (the budget check); the real cost is what
# OpenRouter reports in the response. Two models have a dearer tier for very long prompts (over
# 272000 tokens for gpt-6-luna, over 100000 for claude-haiku-5.5): the questions are far below that.
OPENROUTER_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "google/gemini-3.5-flash-lite": (0.30, 2.50),
    "google/gemini-3.7-flash": (0.75, 3.75),
    "openai/gpt-6-luna": (0.10, 0.50),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    # Video models (FR-11, D-073), same source, read 2026-11-18.
    "qwen/qwen3.8-omni-flash": (0.15, 0.47),
}
# The estimate of a request before it is sent. Our prompts are mostly Russian text: about two
# characters per token is a deliberately high count (more tokens, a higher estimate). The answer
# is assumed to be at most this long; both are assumptions, not measurements.
CHARS_PER_TOKEN = 2
ESTIMATED_OUTPUT_TOKENS = 1000


# A video costs input tokens by its length. Gemini counts about 260 tokens per second of video at
# its default sampling (Gemini API docs, video understanding); the figure is rounded up and used for
# every model, which is an assumption for the others, not a measurement. The estimate only decides
# whether the budget allows the request; the ledger records what the service charged. Measured on
# a 20 s clip (D-073, one run each): about 105 tokens per second at Gemini 3.7 Flash, about 600 at
# Qwen 3.8 Omni Flash, so the estimate is high for the first and low for the second.
VIDEO_TOKENS_PER_S = 300


def estimate_video_request_usd(model: str, prompt_chars: int, video_seconds: float) -> float | None:
    """The estimated cost of one request with a video clip before it is sent.

    Returns `None` for a model that is not in the table (no price, no guess).
    """
    prices = OPENROUTER_PRICES_USD_PER_MTOK.get(model)
    if prices is None:
        return None
    tokens_in = -(-max(prompt_chars, 0) // CHARS_PER_TOKEN) + round(
        max(video_seconds, 0.0) * VIDEO_TOKENS_PER_S
    )
    return (tokens_in * prices[0] + ESTIMATED_OUTPUT_TOKENS * prices[1]) / _PER_MILLION


def estimate_request_usd(model: str, prompt_chars: int) -> float | None:
    """The estimated cost of one OpenRouter chat request before it is sent.

    Returns `None` for a model that is not in the table (no price, no guess).
    """
    prices = OPENROUTER_PRICES_USD_PER_MTOK.get(model)
    if prices is None:
        return None
    tokens_in = -(-max(prompt_chars, 0) // CHARS_PER_TOKEN)  # ceiling
    return (tokens_in * prices[0] + ESTIMATED_OUTPUT_TOKENS * prices[1]) / _PER_MILLION
