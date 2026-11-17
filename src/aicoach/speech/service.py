"""Wiring of the transcription model: the provider, the ledger and the budget."""

from pathlib import Path

from aicoach.config import DEFAULT_OPENROUTER_TRANSCRIBE_MODEL, Settings
from aicoach.openrouter import OpenRouterHttp
from aicoach.speech.openrouter_stt import OpenRouterTranscriber
from aicoach.speech.transcriber import GeminiTranscriber, Transcriber
from aicoach.spend import MeteredTranscriber, SpendLedger, ledger_path


def make_transcriber(
    settings: Settings, match_dir: Path, model: str | None = None
) -> MeteredTranscriber:
    """The speech-to-text client of the provider in the settings, wrapped in the ledger.

    Args:
        settings: Provider, keys, budget, data root.
        match_dir: The match directory (the ledger goes to the data root, or above this one).
        model: The model; the default of the provider when `None` (for Gemini
            `GEMINI_TRANSCRIBE_MODEL`, for OpenRouter `google/gemini-3.5-transcribe`).

    Raises:
        ConfigError: The key of the chosen provider is not set.
    """
    provider = settings.aicoach_llm_provider
    inner: Transcriber
    if provider == "gemini":
        inner = GeminiTranscriber(
            settings.require_gemini_api_key(), model or settings.gemini_transcribe_model
        )
    else:
        http = OpenRouterHttp(settings.require_openrouter_api_key())
        inner = OpenRouterTranscriber(http, model or DEFAULT_OPENROUTER_TRANSCRIBE_MODEL)
    ledger = SpendLedger(ledger_path(settings, match_dir), settings.aicoach_budget_usd)
    return MeteredTranscriber(inner, ledger, provider)
