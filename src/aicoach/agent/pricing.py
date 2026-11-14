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
    """The cost of `usage` at the paid-tier price of `model`; `None` for an unknown model."""
    prices = PRICES_USD_PER_MTOK.get(model)
    if prices is None:
        return None
    price_in, price_out = prices
    return (usage.input_tokens * price_in + usage.output_tokens * price_out) / _PER_MILLION
