"""Price of speech-to-text requests per second of sound, for the estimate before a request.

The real cost is what OpenRouter reports in the response; this table only feeds the budget check
(D-067). A model that is not in the table has no estimate (`None`), never a guess.
"""

# US dollars per second of audio. Sources, read on 2026-11-17:
# - elevenlabs/scribe-v2: `pricing.prompt` 0.0000611111 per second in the OpenRouter model list
#   (https://openrouter.ai/api/v1/models, the model's description lists per-second billing).
# - google/gemini-3.5-transcribe: OpenRouter lists it per token (2 USD in, 12 USD out per 1M
#   tokens). The D-066 runs measured about 25 input and 4 output tokens per second of sound,
#   which is 25 * 0.000002 + 4 * 0.000012 = 0.000098 per second (a rough figure from 3 runs).
TRANSCRIBE_USD_PER_SECOND: dict[str, float] = {
    "elevenlabs/scribe-v2": 0.0000611111,
    "google/gemini-3.5-transcribe": 0.000098,
}


def estimate_transcription_usd(model: str, seconds: float) -> float | None:
    """The estimated cost of transcribing `seconds` of sound; `None` for an unknown model."""
    price = TRANSCRIBE_USD_PER_SECOND.get(model)
    if price is None:
        return None
    return max(seconds, 0.0) * price
